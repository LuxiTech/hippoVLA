"""推理环境:Agilex Piper 双臂 + 5 相机 + 内外参,适配 openpi 服务端 obs 协议。

obs 结构(与服务端约定):
    {
        "obs": {
            "env_cameras":        [V=2, 3, H, W] uint8  (front + agent)
            "left_wrist_camera":  (3, H, W) uint8
            "right_wrist_camera": (3, H, W) uint8
            "anchor_camera":      (3, H, W) uint8       (head)
        },
        "state":   (14,) float32                          (puppet_left.pos + puppet_right.pos)
        "actions": (50, 14) float32                       (占位,RepackTransform 会读)
        "prompt":  str
        "intrinsic_cv": dict 各相机的 K
        "cam2world_cv": dict 静态相机来自标定,wrist 实时计算
    }
"""
import dataclasses
import logging
import os
import sys
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

import einops
import numpy as np
import rclpy
from cv_bridge import CvBridge

_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _ROOT.parents[1]
for _path in (_REPO_ROOT, _ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

_ROBOMME_ROOTS = []
if os.environ.get("ROBOMME_ROOT"):
    _ROBOMME_ROOTS.append(Path(os.environ["ROBOMME_ROOT"]).expanduser())
_ROBOMME_ROOTS.extend([
    _ROOT / "robomme_policy_learning",
    _REPO_ROOT / "robomme_policy_learning",
    Path("/home/agilex/YueBo/robomme_policy_learning"),
])
for _robomme_root in _ROBOMME_ROOTS:
    for _extra_path in (
        _robomme_root / "packages" / "openpi-client" / "src",
        _robomme_root / "src",
    ):
        if _extra_path.exists() and str(_extra_path) not in sys.path:
            sys.path.insert(0, str(_extra_path))

from openpi_client import image_tools
from openpi_client.runtime import environment as _environment
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from typing_extensions import override

from collect_data import load_camera_poses, poscmd_to_matrix
from controller import AgilexController
from ros_bridge import RosOperatorForRos2

try:
    from piper_msgs.msg import PosCmd
except ImportError:
    PosCmd = None

logger = logging.getLogger(__name__)


TASK_INSTRUCTIONS: Dict[str, str] = {
    "PickCube":  "Use your right arm to center the red cube, then use your left arm to place the red cube into the bowl.",
    "Place":     "Pick up the test tube with the right arm, hand it over to the left arm, and then insert it into the orange test tube rack.",
    "StackCube": "Use arms to centralize red block, green block, yellow block and stack yellow block above green block, then green block above red block.",
}

# 5 路相机:前 3 个为静态环境/锚点相机(需要标定 cam2world),后 2 个为腕部(动态计算)
STATIC_CAMERAS: List[str] = ["head_camera", "agent_camera", "front_camera"]
WRIST_CAMERAS: List[str] = ["left_wrist_camera", "right_wrist_camera"]
ALL_CAMERAS: List[str] = STATIC_CAMERAS + WRIST_CAMERAS


@dataclasses.dataclass
class RobolatentObservation:
    """Robolatent 真机评估所需的最小观测。"""

    cam_head: np.ndarray
    cam_high: np.ndarray
    cam_left_wrist: np.ndarray
    cam_right_wrist: np.ndarray
    joint_state: np.ndarray
    timestamp: float


class RobolatentRealRobotInterface(ABC):
    """Robolatent 真机接口抽象层。

    该接口只暴露评估客户端需要的能力，具体实现可以是 ROS topic、gRPC
    或 Python SDK。当前文件提供 Piper ROS2 的实现。
    """

    @abstractmethod
    def reset(self) -> None:
        """将机器人移动到 episode 初始状态。"""

    @abstractmethod
    def get_observation(self) -> RobolatentObservation:
        """返回 4 路相机和 14 维双臂关节状态。"""

    @abstractmethod
    def send_action(self, action: np.ndarray) -> None:
        """发送 7 维左臂动作，右臂由具体实现决定是否保持当前位置。"""

    @abstractmethod
    def is_done(self) -> bool:
        """返回外部硬件接口是否判定 episode 结束。"""

    def shutdown(self) -> None:
        """释放硬件资源；无资源实现可以保持为空。"""


class PiperRobolatentRealRobotInterface(RobolatentRealRobotInterface):
    """Agilex Piper 双臂在 Robolatent 真机评估中的最小适配。

    订阅 RynnBrainOFT policy 需要的 4 个 RGB 视角：
    - cam_head: 默认映射到 /head_camera/color/image_raw
    - cam_high: 默认映射到 /agent_camera/color/image_raw
    - cam_left_wrist: 默认映射到 /left_wrist_camera/color/image_raw
    - cam_right_wrist: 默认映射到 /right_wrist_camera/color/image_raw

    关节状态读取 /puppet/joint_left 和 /puppet/joint_right，动作只覆盖左臂
    7 维，右臂保持发送动作前的当前状态。
    """

    def __init__(
        self,
        reset_position: Optional[np.ndarray] = None,
        image_timeout: float = 10.0,
        action_interp_num: int = 0,
        action_wait: bool = False,
        clip_gripper: bool = False,
        verify_reset: bool = False,
        reset_tolerance: float = 0.03,
        reset_settle_time: float = 0.5,
        reset_verify_timeout: float = 5.0,
        reset_verify_interval: float = 0.1,
        cam_head_name: str = "head_camera",
        cam_high_name: str = "agent_camera",
        cam_left_wrist_name: str = "left_wrist_camera",
        cam_right_wrist_name: str = "right_wrist_camera",
        left_wrist_brightness_offset: int = 0,
        right_wrist_brightness_offset: int = 0,
    ) -> None:
        self._reset_position = (
            np.asarray(reset_position, dtype=np.float32).reshape(14)
            if reset_position is not None
            else None
        )
        self._action_interp_num = int(action_interp_num)
        self._action_wait = bool(action_wait)
        self._clip_gripper = bool(clip_gripper)
        self._verify_reset = bool(verify_reset)
        self._reset_tolerance = float(reset_tolerance)
        self._reset_settle_time = float(reset_settle_time)
        self._reset_verify_timeout = float(reset_verify_timeout)
        self._reset_verify_interval = float(reset_verify_interval)
        self._camera_topics = {
            "cam_head": f"/{cam_head_name}/color/image_raw",
            "cam_high": f"/{cam_high_name}/color/image_raw",
            "cam_left_wrist": f"/{cam_left_wrist_name}/color/image_raw",
            "cam_right_wrist": f"/{cam_right_wrist_name}/color/image_raw",
        }
        self._brightness_offsets = {
            "cam_left_wrist": int(left_wrist_brightness_offset),
            "cam_right_wrist": int(right_wrist_brightness_offset),
        }
        if any(self._brightness_offsets.values()):
            logger.info("Robolatent wrist brightness offsets: %s", self._brightness_offsets)

        self._bridge = CvBridge()
        self._img_lock = threading.Lock()
        self._latest_images: Dict[str, Optional[np.ndarray]] = {
            logical_name: None for logical_name in self._camera_topics
        }
        self._done = False

        self._ros = RosOperatorForRos2()
        self._controller = AgilexController(self._ros)
        self._subscribe_robolatent_cameras()
        self._wait_for_ready(image_timeout)

    def _subscribe_robolatent_cameras(self) -> None:
        cam_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        for logical_name, topic in self._camera_topics.items():
            self._ros.ros_node.create_subscription(
                Image,
                topic,
                self._make_robolatent_image_cb(logical_name),
                cam_qos,
            )
            logger.info("Subscribed Robolatent camera %s: %s", logical_name, topic)

    def _make_robolatent_image_cb(self, logical_name: str):
        def cb(msg: Image) -> None:
            try:
                image = self._bridge.imgmsg_to_cv2(msg, "bgr8")
                # Keep the same channel order as collect_data.py; only apply optional brightness A/B tests.
                offset = self._brightness_offsets.get(logical_name, 0)
                if offset:
                    image = np.clip(image.astype(np.int16) + offset, 0, 255).astype(np.uint8)
                with self._img_lock:
                    self._latest_images[logical_name] = image
            except Exception as exc:
                logger.warning("Image decode failed for %s: %s", logical_name, exc)

        return cb

    def _wait_for_ready(self, timeout: float) -> None:
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self._img_lock:
                images_ready = all(v is not None for v in self._latest_images.values())
            joints_ready = (
                len(self._ros.puppet_arm_left_deque) > 0
                and len(self._ros.puppet_arm_right_deque) > 0
            )
            if images_ready and joints_ready:
                return
            time.sleep(0.05)

        with self._img_lock:
            missing = [k for k, v in self._latest_images.items() if v is None]
        raise RuntimeError(
            f"Robolatent robot interface is not ready within {timeout}s. "
            f"Missing images: {missing}"
        )

    def _get_joint_state(self) -> np.ndarray:
        if (
            len(self._ros.puppet_arm_left_deque) == 0
            or len(self._ros.puppet_arm_right_deque) == 0
        ):
            raise RuntimeError("No puppet arm JointState messages received yet")
        left, right = self._ros.get_puppet_arm_frame()
        left_pos = np.asarray(left.position, dtype=np.float32)
        right_pos = np.asarray(right.position, dtype=np.float32)
        if left_pos.shape[0] < 7 or right_pos.shape[0] < 7:
            raise RuntimeError(
                f"Expected 7 joints per arm, got left={left_pos.shape}, right={right_pos.shape}"
            )
        return np.concatenate([left_pos[:7], right_pos[:7]], axis=0).astype(np.float32)

    def _verify_reset_qpos(self) -> None:
        if self._reset_position is None:
            return

        time.sleep(max(0.0, self._reset_settle_time))
        deadline = time.time() + max(0.0, self._reset_verify_timeout)
        last_actual = self._get_joint_state()
        last_diff = np.abs(last_actual - self._reset_position)

        while True:
            last_actual = self._get_joint_state()
            last_diff = np.abs(last_actual - self._reset_position)
            max_diff = float(np.max(last_diff))
            left_diff = float(np.max(last_diff[:7]))
            right_diff = float(np.max(last_diff[7:]))
            if max_diff <= self._reset_tolerance:
                logger.info(
                    "Reset qpos verified: max_abs_diff=%.6f left=%.6f right=%.6f tolerance=%.6f",
                    max_diff,
                    left_diff,
                    right_diff,
                    self._reset_tolerance,
                )
                return

            if time.time() >= deadline:
                break
            time.sleep(max(0.01, self._reset_verify_interval))

        logger.error(
            "Reset qpos verification failed: max_abs_diff=%.6f tolerance=%.6f "
            "target=%s actual=%s",
            float(np.max(last_diff)),
            self._reset_tolerance,
            np.round(self._reset_position, 6),
            np.round(last_actual, 6),
        )
        raise RuntimeError(
            "Reset qpos verification failed: "
            f"max_abs_diff={float(np.max(last_diff)):.6f}, "
            f"left={float(np.max(last_diff[:7])):.6f}, "
            f"right={float(np.max(last_diff[7:])):.6f}, "
            f"tolerance={self._reset_tolerance:.6f}"
        )

    @override
    def reset(self) -> None:
        self._done = False
        if self._reset_position is None:
            logger.info("No Robolatent reset_position; skipping robot reset.")
            return
        logger.info("Moving Robolatent robot to reset position: %s", self._reset_position)
        self._controller.set_current_qpos(
            self._reset_position,
            interp_num=50,
            wait=True,
        )
        if self._verify_reset:
            self._verify_reset_qpos()

    @override
    def get_observation(self) -> RobolatentObservation:
        with self._img_lock:
            cam_head = self._latest_images["cam_head"]
            cam_high = self._latest_images["cam_high"]
            cam_left_wrist = self._latest_images["cam_left_wrist"]
            cam_right_wrist = self._latest_images["cam_right_wrist"]
            if cam_head is not None:
                cam_head = cam_head.copy()
            if cam_high is not None:
                cam_high = cam_high.copy()
            if cam_left_wrist is not None:
                cam_left_wrist = cam_left_wrist.copy()
            if cam_right_wrist is not None:
                cam_right_wrist = cam_right_wrist.copy()

        missing = [
            name
            for name, image in (
                ("cam_head", cam_head),
                ("cam_high", cam_high),
                ("cam_left_wrist", cam_left_wrist),
                ("cam_right_wrist", cam_right_wrist),
            )
            if image is None
        ]
        if missing:
            raise RuntimeError(f"Robolatent cameras have no frames yet: {missing}")

        return RobolatentObservation(
            cam_head=cam_head,
            cam_high=cam_high,
            cam_left_wrist=cam_left_wrist,
            cam_right_wrist=cam_right_wrist,
            joint_state=self._get_joint_state(),
            timestamp=time.time(),
        )

    @override
    def send_action(self, action: np.ndarray) -> None:
        left_action = np.asarray(action, dtype=np.float32).reshape(-1)
        if left_action.shape != (7,):
            raise ValueError(f"Expected 7-dim left-arm action, got shape {left_action.shape}")

        target = self._get_joint_state()
        target[:7] = left_action
        if self._clip_gripper:
            target[6] = np.clip(target[6], 0.0, 0.1)

        self._controller.set_current_qpos(
            target,
            interp_num=self._action_interp_num,
            wait=self._action_wait,
        )

    @override
    def is_done(self) -> bool:
        return self._done

    def mark_done(self) -> None:
        self._done = True

    @override
    def shutdown(self) -> None:
        shutdown = getattr(self._ros, "shutdown", None)
        if callable(shutdown):
            shutdown()


class PiperRealEnvironment(_environment.Environment):
    """openpi 推理时使用的本地硬件环境。"""

    def __init__(
        self,
        task: str,
        camera_poses_dir: str = "camera_poses",
        reset_position: Optional[np.ndarray] = None,
        render_height: int = 240,
        render_width: int = 320,
        intrinsics_timeout: float = 10.0,
        env_camera_order: Optional[List[str]] = None,
    ) -> None:
        if task not in TASK_INSTRUCTIONS:
            raise ValueError(f"Unknown task '{task}'. Available: {list(TASK_INSTRUCTIONS)}")
        self._prompt = TASK_INSTRUCTIONS[task]
        self._render_height = render_height
        self._render_width = render_width
        self._reset_position = (
            np.asarray(reset_position, dtype=float) if reset_position is not None else None
        )

        # env_image 的相机顺序(决定 stack 后 [V=N, 3, H, W] 的第 0 维语义)
        if env_camera_order is None:
            env_camera_order = ["front_camera", "agent_camera"]
        for cam in env_camera_order:
            if cam not in STATIC_CAMERAS:
                raise ValueError(
                    f"env_camera_order contains '{cam}' which is not a static camera. "
                    f"Allowed: {STATIC_CAMERAS}"
                )
        self._env_camera_order = list(env_camera_order)

        self._bridge = CvBridge()

        # ROS2 操作器(内部 rclpy.init + 节点 + 双臂订阅/发布)
        self._ros = RosOperatorForRos2()
        self._controller = AgilexController(self._ros)

        # 静态外参 + 手眼标定
        static_cam2world, wrist_T_cam_in_ee, wrist_T_base_in_aruco = load_camera_poses(camera_poses_dir)
        for cam in STATIC_CAMERAS:
            if cam not in static_cam2world:
                raise RuntimeError(f"Missing static cam2world for {cam} in {camera_poses_dir}/")
        for side in ("left", "right"):
            if side not in wrist_T_cam_in_ee:
                raise RuntimeError(f"Missing wrist hand-eye calibration ({side}) in {camera_poses_dir}/")
            if side not in wrist_T_base_in_aruco:
                raise RuntimeError(f"Missing arm base_in_aruco ({side}) in {camera_poses_dir}/")
        self._T_static = static_cam2world
        self._wrist_T_cam_in_ee = wrist_T_cam_in_ee
        self._wrist_T_base_in_aruco = wrist_T_base_in_aruco

        # 实时缓存
        self._img_lock = threading.Lock()
        self._latest_images: Dict[str, Optional[np.ndarray]] = {c: None for c in ALL_CAMERAS}
        self._K: Dict[str, Optional[np.ndarray]] = {c: None for c in ALL_CAMERAS}
        self._latest_end_left = None
        self._latest_end_right = None

        # 订阅
        self._subscribe_cameras()
        self._subscribe_end_pose()

        # 等待 camera_info 与第一帧 image 全部就绪
        self._wait_for_camera_data(intrinsics_timeout)

    # ------------------------- ROS 订阅 -------------------------

    def _subscribe_cameras(self) -> None:
        cam_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        info_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        node = self._ros.ros_node
        for cam in ALL_CAMERAS:
            node.create_subscription(
                Image,
                f"/{cam}/color/image_raw",
                self._make_image_cb(cam),
                cam_qos,
            )
            node.create_subscription(
                CameraInfo,
                f"/{cam}/color/camera_info",
                self._make_info_cb(cam),
                info_qos,
            )
        logger.info("Subscribed to %d cameras", len(ALL_CAMERAS))

    def _subscribe_end_pose(self) -> None:
        if PosCmd is None:
            raise RuntimeError("piper_msgs.msg.PosCmd not importable; cannot subscribe end-pose")
        ctrl_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        node = self._ros.ros_node
        node.create_subscription(PosCmd, "/puppet/end_left", self._end_left_cb, ctrl_qos)
        node.create_subscription(PosCmd, "/puppet/end_right", self._end_right_cb, ctrl_qos)

    def _make_image_cb(self, cam: str):
        def cb(msg: Image) -> None:
            try:
                rgb = self._bridge.imgmsg_to_cv2(msg, "rgb8")
                with self._img_lock:
                    self._latest_images[cam] = rgb
            except Exception as exc:
                logger.warning("Image decode failed for %s: %s", cam, exc)
        return cb

    def _make_info_cb(self, cam: str):
        def cb(msg: CameraInfo) -> None:
            if self._K[cam] is None:
                self._K[cam] = np.array(msg.k, dtype=float).reshape(3, 3)
                logger.info("Got intrinsics for %s", cam)
        return cb

    def _end_left_cb(self, msg) -> None:
        self._latest_end_left = msg

    def _end_right_cb(self, msg) -> None:
        self._latest_end_right = msg

    def _wait_for_camera_data(self, timeout: float) -> None:
        """阻塞至所有相机的 K 与第一帧 image 均就绪,任一超时即 raise。"""
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self._img_lock:
                imgs_ready = all(self._latest_images[c] is not None for c in ALL_CAMERAS)
            k_ready = all(self._K[c] is not None for c in ALL_CAMERAS)
            if imgs_ready and k_ready:
                return
            time.sleep(0.1)
        missing_k = [c for c in ALL_CAMERAS if self._K[c] is None]
        with self._img_lock:
            missing_img = [c for c in ALL_CAMERAS if self._latest_images[c] is None]
        raise RuntimeError(
            f"Camera data not ready within {timeout}s. "
            f"Missing camera_info: {missing_k}; missing image_raw: {missing_img}"
        )

    # ------------------------- 计算辅助 -------------------------

    def _compute_wrist_cam2world(self, side: str, end_msg) -> np.ndarray:
        T_base_in_aruco = self._wrist_T_base_in_aruco[side]
        T_cam_in_ee = self._wrist_T_cam_in_ee[side]
        T_ee_in_base = poscmd_to_matrix(
            end_msg.x, end_msg.y, end_msg.z,
            end_msg.roll, end_msg.pitch, end_msg.yaw,
        )
        return T_base_in_aruco @ T_ee_in_base @ T_cam_in_ee

    def _process_image(self, rgb_hwc: np.ndarray) -> np.ndarray:
        if rgb_hwc.dtype != np.uint8:
            rgb_hwc = image_tools.convert_to_uint8(rgb_hwc)
        return einops.rearrange(rgb_hwc, "h w c -> c h w")

    # ------------------------- Environment API -------------------------

    @override
    def reset(self) -> None:
        if self._reset_position is None:
            logger.info("No reset_position; skipping reset.")
            return
        logger.info("Moving to reset position: %s", self._reset_position)
        self._controller.set_current_qpos(self._reset_position, interp_num=50, wait=True)

    @override
    def is_episode_complete(self) -> bool:
        return False

    @override
    def get_observation(self) -> dict:
        with self._img_lock:
            imgs = {k: v for k, v in self._latest_images.items()}
        missing = [k for k, v in imgs.items() if v is None]
        if missing:
            raise RuntimeError(f"Cameras have no frames yet: {missing}")
        if self._latest_end_left is None or self._latest_end_right is None:
            raise RuntimeError("End-effector pose not received yet")

        left, right = self._ros.get_puppet_arm_frame()
        state = np.concatenate([
            np.asarray(left.position, dtype=np.float32),
            np.asarray(right.position, dtype=np.float32),
        ])

        proc = {k: self._process_image(v) for k, v in imgs.items()}

        T_left_wrist = self._compute_wrist_cam2world("left", self._latest_end_left).astype(np.float32)
        T_right_wrist = self._compute_wrist_cam2world("right", self._latest_end_right).astype(np.float32)

        env_order = self._env_camera_order
        return {
            "env_image":                np.stack([proc[c] for c in env_order], axis=0),
            "left_wrist_camera_image":  proc["left_wrist_camera"],
            "right_wrist_camera_image": proc["right_wrist_camera"],
            "anchor_image":             proc["head_camera"],
            "state":                    state,
            "actions":                  np.zeros((50, 14), dtype=np.float32),
            "prompt":                   self._prompt,
            "intrinsic_cv": {
                "env_image": np.stack(
                    [self._K[c] for c in env_order], axis=0
                ).astype(np.float32),
                "left_wrist_camera_image":  self._K["left_wrist_camera"].astype(np.float32),
                "right_wrist_camera_image": self._K["right_wrist_camera"].astype(np.float32),
                "anchor_image":             self._K["head_camera"].astype(np.float32),
            },
            "cam2world_cv": {
                "env_image": np.stack(
                    [self._T_static[c] for c in env_order], axis=0
                ).astype(np.float32),
                "left_wrist_camera_image":  T_left_wrist,
                "right_wrist_camera_image": T_right_wrist,
                "anchor_image":             self._T_static["head_camera"].astype(np.float32),
            },
            "mask_image": None,
        }

    @override
    def apply_action(self, action: dict) -> None:
        qpos = np.array(action["actions"], dtype=float).reshape(-1)
        if qpos.shape != (14,):
            raise ValueError(f"Expected 14-dim action, got shape {qpos.shape}")
        # qpos[[6, 13]] = np.maximum(qpos[[6, 13]], 0.0)
        self._controller.set_current_qpos(qpos, interp_num=0, wait=False)
