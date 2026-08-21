#!/usr/bin/env python3
"""Robolatent 真机 VLA websocket 评估客户端。

启动前通常需要：
  1. source ~/cobot_magic/piper_ros/install/setup.bash
  2. ros2 launch piper start_multi_piper.launch.py
  3. ros2 launch /home/agilex/cobot_magic/camera_ws_ros2/src/realsense-ros/realsense2_camera/launch/rs_launch_hby.py
  4. ssh -v -N -L 8000:127.0.0.1:8000 lab_server
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import functools
import hashlib
import inspect
import json
import logging
import os
import pickle
import select
import sys
import termios
import time
import tty
from pathlib import Path
from typing import Any, Deque, Optional

import cv2
import msgpack
import numpy as np
import websockets.sync.client

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

try:
    from openpi_client import msgpack_numpy
except ImportError:
    def _pack_array(obj):
        if (isinstance(obj, (np.ndarray, np.generic))) and obj.dtype.kind in ("V", "O", "c"):
            raise ValueError(f"Unsupported dtype: {obj.dtype}")
        if isinstance(obj, np.ndarray):
            return {
                b"__ndarray__": True,
                b"data": obj.tobytes(),
                b"dtype": obj.dtype.str,
                b"shape": obj.shape,
            }
        if isinstance(obj, np.generic):
            return {
                b"__npgeneric__": True,
                b"data": obj.item(),
                b"dtype": obj.dtype.str,
            }
        return obj

    def _unpack_array(obj):
        if b"__ndarray__" in obj:
            return np.ndarray(
                buffer=obj[b"data"],
                dtype=np.dtype(obj[b"dtype"]),
                shape=obj[b"shape"],
            )
        if b"__npgeneric__" in obj:
            return np.dtype(obj[b"dtype"]).type(obj[b"data"])
        return obj

    class _MsgpackNumpy:
        Packer = functools.partial(msgpack.Packer, default=_pack_array)
        packb = functools.partial(msgpack.packb, default=_pack_array)
        unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array)

    msgpack_numpy = _MsgpackNumpy()

from examples.new_deploy.piper_env import PiperRobolatentRealRobotInterface, RobolatentObservation


logger = logging.getLogger(__name__)


TASK_PROMPTS: dict[str, str] = {
    "PickXtimes": "pick up the blue block and place the blue block in the plate repeatedly",
    "PutBackBlock": "pick up the blue block, place it in the plate, push the button, then put the block back on the table",
    "UncoverBlock": "cover the blocks with lids then uncover them",
}

TASK_ALIASES: dict[str, str] = {
    "PickXTimes": "PickXtimes",
    # "pickxtimes": "PickXtimes",
    "putbackblock": "PutBackBlock",
    "uncoverblock": "UncoverBlock",
}

@dataclasses.dataclass
class PolicyObservation:
    views: list[np.ndarray]
    state: np.ndarray


class StarVLAWebsocketClientPolicy:
    """StarVLA websocket 客户端，发送 type/payload 协议并保留完整响应。"""

    def __init__(self, host: str, port: int) -> None:
        self._uri = f"ws://{host}:{port}"
        self._packer = msgpack_numpy.Packer()
        self._ws, self._server_metadata = self._wait_for_server()

    def _wait_for_server(self):
        logger.info("Waiting for StarVLA policy server at %s...", self._uri)
        while True:
            try:
                conn = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    open_timeout=60,
                    close_timeout=60,
                )
                metadata = msgpack_numpy.unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                logger.info("Still waiting for server...")
                time.sleep(5)

    def get_server_metadata(self) -> dict[str, Any]:
        return self._server_metadata

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._ws.send(self._packer.pack(payload))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    def reset(self) -> dict[str, Any]:
        return self._request({"type": "reset"})

    def add_buffer(self, buffer: dict[str, Any]) -> dict[str, Any]:
        return self._request(buffer)

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        return self._request({"type": "infer", "payload": obs})


def create_mmevla_client(host: str, port: int):
    """创建 StarVLA websocket 客户端。"""
    return StarVLAWebsocketClientPolicy(host=host, port=port)


class KeyboardPoller:
    """Linux 终端非阻塞按键读取。"""

    def __init__(self) -> None:
        self._enabled = sys.stdin.isatty()
        self._fd: Optional[int] = None
        self._old_settings: Any = None

    def start(self) -> None:
        if not self._enabled:
            logger.warning("stdin is not a TTY; stop keys are disabled.")
            return
        self._fd = sys.stdin.fileno()
        self._old_settings = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)

    def stop(self) -> None:
        if self._enabled and self._fd is not None and self._old_settings is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)

    def get_key(self) -> Optional[str]:
        if not self._enabled:
            return None
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None

class CamHighVideoRecorder:
    """流式保存 cam_high RGB 视频，避免整段缓存到内存。"""

    def __init__(self, path: Path, fps: int) -> None:
        self._path = path
        self._fps = int(fps)
        self._writer: Optional[cv2.VideoWriter] = None
        self._frame_count = 0
        path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    def add(self, rgb: np.ndarray) -> None:
        frame = ensure_uint8_rgb(rgb)
        h, w = frame.shape[:2]
        if self._writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(str(self._path), fourcc, self._fps, (w, h))
            if not self._writer.isOpened():
                raise RuntimeError(f"Failed to open video writer: {self._path}")
        self._writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        self._frame_count += 1

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None


def ensure_uint8_rgb(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected HWC RGB image, got shape {image.shape}")
    if image.dtype == np.uint8:
        return image
    return np.clip(image, 0, 255).astype(np.uint8)


def resize_image(image: np.ndarray, size: int = 224) -> np.ndarray:
    image = ensure_uint8_rgb(image)
    if image.shape[0] == size and image.shape[1] == size:
        return image
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)


RYNNBRAIN_VIEW_NAMES = ("cam_head", "cam_high", "cam_left_wrist", "cam_right_wrist")
HIGH_VIEW_INDEX = RYNNBRAIN_VIEW_NAMES.index("cam_high")
LEFT_WRIST_VIEW_INDEX = RYNNBRAIN_VIEW_NAMES.index("cam_left_wrist")


class DemoViewOverride:
    """Use one camera view from a demo HDF5 while keeping other observations real."""
    def __init__(
        self,
        path: Path,
        *,
        view_name: str,
        image_size: int,
        start: int,
        stride: int,
        hold_last: bool,
        option_prefix: str,
    ) -> None:
        import h5py

        self._path = path
        self._view_name = view_name
        self._view_index = RYNNBRAIN_VIEW_NAMES.index(view_name)
        self._image_size = int(image_size)
        self._start = int(start)
        self._stride = int(stride)
        self._hold_last = bool(hold_last)
        if self._stride <= 0:
            raise ValueError(f"{option_prefix}_stride must be positive")
        self._file = h5py.File(path, "r")
        self._dataset = self._file[f"observations/images/{view_name}"]
        if self._dataset.ndim != 4 or self._dataset.shape[-1] != 3:
            raise ValueError(
                f"Expected demo {view_name} shape (T,H,W,3), got {self._dataset.shape}"
            )
        if not 0 <= self._start < self.length:
            raise ValueError(
                f"{option_prefix}_start={self._start} out of range [0, {self.length})"
            )

    @property
    def length(self) -> int:
        return int(self._dataset.shape[0])

    def index_for_step(self, step: int) -> int:
        index = self._start + int(step) * self._stride
        if index < self.length:
            return index
        if self._hold_last:
            return self.length - 1
        raise IndexError(
            f"Demo {self._view_name} index {index} out of range for {self._path} length={self.length}"
        )

    def apply(self, obs: PolicyObservation, step: int) -> tuple[PolicyObservation, int]:
        index = self.index_for_step(step)
        views = list(obs.views)
        views[self._view_index] = resize_image(np.asarray(self._dataset[index]), self._image_size)
        return PolicyObservation(views=views, state=obs.state), index

    def close(self) -> None:
        self._file.close()


class DemoLeftWristOverride(DemoViewOverride):
    def __init__(
        self,
        path: Path,
        *,
        image_size: int,
        start: int,
        stride: int,
        hold_last: bool,
    ) -> None:
        super().__init__(
            path,
            view_name="cam_left_wrist",
            image_size=image_size,
            start=start,
            stride=stride,
            hold_last=hold_last,
            option_prefix="--demo_left_wrist",
        )


class DemoHighOverride(DemoViewOverride):
    def __init__(
        self,
        path: Path,
        *,
        image_size: int,
        start: int,
        stride: int,
        hold_last: bool,
    ) -> None:
        super().__init__(
            path,
            view_name="cam_high",
            image_size=image_size,
            start=start,
            stride=stride,
            hold_last=hold_last,
            option_prefix="--demo_high",
        )


def black_views(image_size: int) -> list[np.ndarray]:
    return [
        np.zeros((image_size, image_size, 3), dtype=np.uint8)
        for _ in RYNNBRAIN_VIEW_NAMES
    ]


def observation_views(
    obs: RobolatentObservation,
    image_size: int,
    missing_camera_strategy: str,
) -> list[np.ndarray]:
    raw_images = {
        name: getattr(obs, name, None)
        for name in RYNNBRAIN_VIEW_NAMES
    }
    fallback_for = {
        "cam_head": "cam_high",
        "cam_right_wrist": "cam_left_wrist",
    }
    views: list[np.ndarray] = []
    for name in RYNNBRAIN_VIEW_NAMES:
        image = raw_images.get(name)
        if image is None:
            if missing_camera_strategy == "strict":
                raise AttributeError(
                    f"Observation is missing required camera `{name}`. "
                    "This RynnBrainOFT checkpoint was trained with "
                    f"{list(RYNNBRAIN_VIEW_NAMES)}. For a debug-only run, pass "
                    "`--missing_camera_strategy copy` or `zero`."
                )
            if missing_camera_strategy == "copy":
                fallback = raw_images.get(fallback_for.get(name, "cam_high"))
                if fallback is None:
                    fallback = raw_images.get("cam_high")
                if fallback is None:
                    fallback = raw_images.get("cam_left_wrist")
                if fallback is None:
                    image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
                else:
                    image = fallback
            elif missing_camera_strategy == "zero":
                image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
            else:
                raise ValueError(f"Unknown missing camera strategy: {missing_camera_strategy}")
        views.append(resize_image(image, image_size))
    return views


def pack_left_state(joint_state: np.ndarray) -> np.ndarray:
    """从 14 维双臂 qpos 中取左臂 6 joints + 1 gripper。"""
    joint_state = np.asarray(joint_state, dtype=np.float32).reshape(-1)
    if joint_state.shape[0] < 7:
        raise ValueError(f"Expected at least 7-dim joint_state, got {joint_state.shape}")
    return np.concatenate([joint_state[:6], joint_state[6:7]], axis=0).astype(np.float32)


def convert_observation(
    obs: RobolatentObservation,
    image_size: int,
    missing_camera_strategy: str,
) -> PolicyObservation:
    return PolicyObservation(
        views=observation_views(obs, image_size, missing_camera_strategy),
        state=pack_left_state(obs.joint_state),
    )


def normalize_task(task: str) -> str:
    if task in TASK_PROMPTS:
        return task
    normalized = TASK_ALIASES.get(task) or TASK_ALIASES.get(task.lower())
    if normalized is not None:
        return normalized
    raise ValueError(f"Unknown task {task!r}; available tasks: {list(TASK_PROMPTS)}")


def load_action_stats(path: Path, unnorm_key: str | None) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        stats = json.load(f)
    if unnorm_key is None:
        if len(stats) != 1:
            raise ValueError(f"--unnorm_key is required; available keys: {list(stats)}")
        unnorm_key = next(iter(stats))
    if unnorm_key not in stats:
        raise ValueError(f"Unknown --unnorm_key {unnorm_key!r}; available keys: {list(stats)}")
    return stats[unnorm_key]["action"]


def unnormalize_actions(
    normalized_actions: np.ndarray,
    action_stats: dict[str, Any],
    *,
    gripper_binary_threshold: float,
    gripper_postprocess: str,
) -> np.ndarray:
    action_low = np.asarray(action_stats.get("min", action_stats.get("q01")), dtype=np.float32)
    action_high = np.asarray(action_stats.get("max", action_stats.get("q99")), dtype=np.float32)
    mask = np.asarray(action_stats.get("mask", np.ones_like(action_low, dtype=bool)), dtype=bool)
    normalized = np.asarray(normalized_actions, dtype=np.float32)
    if normalized.ndim != 2:
        raise ValueError(f"Expected normalized action shape (T, D), got {normalized.shape}")
    if normalized.shape[1] < 7:
        raise ValueError(f"Expected at least 7 action dims, got {normalized.shape}")

    clipped = np.clip(normalized[:, : len(action_low)], -1.0, 1.0)
    scaled = 0.5 * (clipped + 1.0) * (action_high - action_low) + action_low
    actions = np.where(mask, scaled, clipped)
    if gripper_postprocess == "binary":
        actions[:, 6] = np.where(
            clipped[:, 6] < gripper_binary_threshold,
            action_low[6],
            action_high[6],
        )
    elif gripper_postprocess != "raw":
        raise ValueError(f"Unsupported gripper_postprocess: {gripper_postprocess}")
    return actions.astype(np.float32)


def make_memory_views(
    observation_history: list[tuple[int, list[np.ndarray]]],
    *,
    current_step: int,
    image_size: int,
    max_memory_frames: int,
    memory_interval: int,
) -> list[list[np.ndarray]]:
    views_by_step = {step: views for step, views in observation_history}
    zero_views = black_views(image_size)
    memory: list[list[np.ndarray]] = []
    for i in range(1, max_memory_frames + 1):
        target_step = current_step - i * memory_interval
        views = views_by_step.get(target_step, zero_views)
        memory.append([view.astype(np.uint8) for view in views])
    return memory


def make_rynnbrain_payload(
    current_views: list[np.ndarray],
    observation_history: list[tuple[int, list[np.ndarray]]],
    *,
    prompt: str,
    current_step: int,
    image_size: int,
    max_memory_frames: int,
    memory_interval: int,
) -> dict[str, Any]:
    """Build the RynnBrainOFT payload used by the StarVLA policy server."""
    if len(current_views) != len(RYNNBRAIN_VIEW_NAMES):
        raise ValueError(
            f"Expected {len(RYNNBRAIN_VIEW_NAMES)} camera views, got {len(current_views)}"
        )
    lang = prompt
    return {
        "examples": [
            {
                "image": [view.astype(np.uint8) for view in current_views],
                "lang": lang,
                "memory": make_memory_views(
                    observation_history,
                    current_step=current_step,
                    image_size=image_size,
                    max_memory_frames=max_memory_frames,
                    memory_interval=memory_interval,
                ),
                "step": int(current_step),
            }
        ],
    }


def save_debug_image(path: Path, image: np.ndarray) -> None:
    frame = ensure_uint8_rgb(image)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), frame):
        raise RuntimeError(f"Failed to write image: {path}")


def image_metadata(path: Path, image: np.ndarray, root: Path) -> dict[str, Any]:
    arr = np.asarray(image)
    return {
        "file": str(path.relative_to(root)),
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
    }


def should_save_left_wrist_compare(args: argparse.Namespace, step: int) -> bool:
    if not args.save_left_wrist_compare:
        return False
    if step < args.left_wrist_compare_start:
        return False
    if args.left_wrist_compare_end >= 0 and step > args.left_wrist_compare_end:
        return False
    return (step - args.left_wrist_compare_start) % args.left_wrist_compare_stride == 0


def save_left_wrist_compare(
    save_dir: Path,
    task: str,
    episode_idx: int,
    step: int,
    real_policy_obs: PolicyObservation,
    demo_policy_obs: PolicyObservation,
    *,
    demo_left_wrist_index: int,
    timestamp: float,
) -> Path:
    dump_dir = (
        save_dir
        / "left_wrist_compare"
        / f"{task}_ep_{episode_idx:04d}"
        / f"step_{step:06d}"
    )
    dump_dir.mkdir(parents=True, exist_ok=True)

    real = ensure_uint8_rgb(real_policy_obs.views[LEFT_WRIST_VIEW_INDEX])
    demo = ensure_uint8_rgb(demo_policy_obs.views[LEFT_WRIST_VIEW_INDEX])
    if real.shape != demo.shape:
        raise ValueError(f"Cannot compare left wrist images with shapes {real.shape} and {demo.shape}")

    save_debug_image(dump_dir / "real_cam_left_wrist.png", real)
    save_debug_image(dump_dir / "demo_cam_left_wrist.png", demo)
    save_debug_image(dump_dir / "side_by_side_real_demo.png", np.concatenate([real, demo], axis=1))
    diff = np.abs(real.astype(np.int16) - demo.astype(np.int16)).astype(np.uint8)
    save_debug_image(dump_dir / "absdiff_cam_left_wrist.png", diff)

    metadata = {
        "step": step,
        "timestamp": timestamp,
        "demo_left_wrist_index": demo_left_wrist_index,
        "layout": "side_by_side_real_demo.png is [real, demo]",
        "real_file": "real_cam_left_wrist.png",
        "demo_file": "demo_cam_left_wrist.png",
        "side_by_side_file": "side_by_side_real_demo.png",
        "absdiff_file": "absdiff_cam_left_wrist.png",
        "mean_abs_diff": float(np.mean(diff)),
        "max_abs_diff": int(np.max(diff)),
        "real_state_left": real_policy_obs.state.astype(np.float32).tolist(),
        "demo_state_left": demo_policy_obs.state.astype(np.float32).tolist(),
    }
    save_json_atomic(dump_dir / "metadata.json", metadata)
    return dump_dir


def maybe_apply_demo_left_wrist(
    args: argparse.Namespace,
    demo_left_wrist: Optional[DemoLeftWristOverride],
    real_policy_obs: PolicyObservation,
    *,
    episode_idx: int,
    step: int,
    timestamp: float,
) -> tuple[PolicyObservation, Optional[int], Optional[Path]]:
    if demo_left_wrist is None:
        return real_policy_obs, None, None

    demo_policy_obs, demo_index = demo_left_wrist.apply(real_policy_obs, step)
    compare_dir = None
    if should_save_left_wrist_compare(args, step):
        compare_dir = save_left_wrist_compare(
            args.save_dir,
            args.task,
            episode_idx,
            step,
            real_policy_obs,
            demo_policy_obs,
            demo_left_wrist_index=demo_index,
            timestamp=timestamp,
        )
        logger.info("Saved left wrist compare: %s", compare_dir)
    return demo_policy_obs, demo_index, compare_dir


def save_server_input_dump(
    save_dir: Path,
    task: str,
    episode_idx: int,
    infer_idx: int,
    infer_payload: dict[str, Any],
    *,
    payload_bytes: int,
) -> Path:
    example = infer_payload["examples"][0]
    step = int(example.get("step", infer_idx))
    dump_dir = (
        save_dir
        / "server_inputs"
        / f"{task}_ep_{episode_idx:04d}"
        / f"infer_{infer_idx:04d}_step_{step:06d}"
    )
    dump_dir.mkdir(parents=True, exist_ok=True)

    current_images = []
    for view_idx, (view_name, image) in enumerate(zip(RYNNBRAIN_VIEW_NAMES, example["image"])):
        path = dump_dir / "current" / f"{view_idx}_{view_name}.png"
        save_debug_image(path, image)
        meta = {"view": view_name, "index": view_idx}
        meta.update(image_metadata(path, image, dump_dir))
        current_images.append(meta)

    memory_frames = []
    for memory_idx, memory_views in enumerate(example.get("memory", [])):
        frame_meta = []
        for view_idx, (view_name, image) in enumerate(zip(RYNNBRAIN_VIEW_NAMES, memory_views)):
            path = dump_dir / "memory" / f"memory_{memory_idx:02d}" / f"{view_idx}_{view_name}.png"
            save_debug_image(path, image)
            meta = {"view": view_name, "index": view_idx}
            meta.update(image_metadata(path, image, dump_dir))
            frame_meta.append(meta)
        memory_frames.append({"memory_index": memory_idx, "images": frame_meta})

    metadata = {
        "task": task,
        "episode_idx": episode_idx,
        "infer_idx": infer_idx,
        "step": step,
        "lang": example.get("lang"),
        "payload_bytes": payload_bytes,
        "payload_keys": list(infer_payload.keys()),
        "example_keys": list(example.keys()),
        "view_names": list(RYNNBRAIN_VIEW_NAMES),
        "current_images": current_images,
        "memory_frames": memory_frames,
        "payload_pickle": "payload.pkl",
    }
    save_json_atomic(dump_dir / "metadata.json", metadata)
    with open(dump_dir / "payload.pkl", "wb") as f:
        pickle.dump(infer_payload, f)
    return dump_dir


def wait_for_ack(response: dict[str, Any], key: str) -> None:
    if not response.get(key, False):
        raise RuntimeError(f"Policy server did not acknowledge {key}: {response}")


def config_fingerprint(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def save_json_atomic(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def load_json(path: Path) -> Optional[dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        logger.warning("Failed to load JSON: %s", path, exc_info=True)
        return None


def is_success(outcome: object) -> bool:
    text = str(outcome).strip().lower()
    return text == "success" or ("success" in text and "fail" not in text)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def parse_reset_qpos(reset_qpos: str) -> np.ndarray:
    values = [float(item.strip()) for item in reset_qpos.split(",") if item.strip()]
    if len(values) != 14:
        raise ValueError(f"--reset_qpos expects 14 comma-separated floats, got {len(values)}")
    return np.asarray(values, dtype=np.float32)


def load_dataset_initial_qpos(dataset_dir: Path, task: str) -> np.ndarray:
    import h5py

    path = dataset_dir / task / "episode_0.hdf5"
    if not path.exists():
        raise FileNotFoundError(f"Initial qpos dataset not found: {path}")
    with h5py.File(path, "r") as f:
        qpos = np.asarray(f["observations/qpos"][0], dtype=np.float32)
    if qpos.shape != (14,):
        raise ValueError(f"Expected initial qpos shape (14,), got {qpos.shape} from {path}")
    return qpos


def resolve_reset_position(args: argparse.Namespace) -> Optional[np.ndarray]:
    if args.reset_qpos is not None:
        reset_position = parse_reset_qpos(args.reset_qpos)
        logger.info("Using reset qpos from --reset_qpos: %s", np.round(reset_position, 4))
        return reset_position
    if args.reset_from_dataset:
        reset_position = load_dataset_initial_qpos(args.reset_dataset_dir, args.task)
        logger.info(
            "Using reset qpos from dataset %s/%s/episode_0.hdf5: %s",
            args.reset_dataset_dir,
            args.task,
            np.round(reset_position, 4),
        )
        return reset_position
    return None


def load_hdf5_left_action_slice(
    path: Path,
    *,
    start: int,
    end: Optional[int],
    steps: int,
) -> tuple[np.ndarray, int, int, int]:
    import h5py

    if not path.exists():
        raise FileNotFoundError(f"Hybrid replay HDF5 not found: {path}")
    with h5py.File(path, "r") as f:
        actions = np.asarray(f["action"], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] < 7:
        raise ValueError(f"Expected HDF5 action shape (T, >=7), got {actions.shape} from {path}")

    total = int(actions.shape[0])
    if start < 0 or start >= total:
        raise ValueError(f"Bad hybrid replay start={start}; HDF5 action length is {total}")
    if end is None:
        if steps <= 0:
            raise ValueError("--hybrid_replay_steps must be > 0 when --hybrid_replay_end is not set")
        end = start + steps
    end = min(int(end), total)
    if end <= start:
        raise ValueError(f"Bad hybrid replay range start={start}, end={end}, total={total}")
    return actions[start:end, :7].astype(np.float32), int(start), int(end), total


def save_episode_logs(
    save_dir: Path,
    task: str,
    episode_idx: int,
    metadata: dict[str, Any],
    step_records: list[dict[str, Any]],
) -> tuple[Path, Path]:
    log_dir = save_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{task}_ep_{episode_idx:04d}"
    json_path = log_dir / f"{stem}.json"
    pkl_path = log_dir / f"{stem}.pkl"

    payload = {
        "metadata": metadata,
        "steps": step_records,
    }
    save_json_atomic(json_path, payload)
    with open(pkl_path, "wb") as f:
        pickle.dump(payload, f)
    return json_path, pkl_path


def update_progress(
    save_dir: Path,
    *,
    task: str,
    episode_idx: int,
    outcome: str,
    video_path: Path,
    json_log_path: Path,
    pkl_log_path: Path,
    config: dict[str, Any],
    finished: bool,
) -> None:
    progress_path = save_dir / "progress.json"
    config_fp = config_fingerprint(config)
    progress = load_json(progress_path)
    if progress is None or progress.get("config_fingerprint") != config_fp:
        progress = {
            "team_id": "robolatent_real_robot",
            "config": config,
            "config_fingerprint": config_fp,
            "completed": {},
            "current": {"task_id": None, "episode_idx": None, "outcome": None},
            "finished": False,
            "metrics": None,
            "updated_at": time.time(),
        }

    completed_for_task = progress["completed"].setdefault(task, {})
    completed_for_task[str(episode_idx)] = {
        "outcome": outcome,
        "video_path": str(video_path),
        "json_log_path": str(json_log_path),
        "pkl_log_path": str(pkl_log_path),
    }
    progress["current"] = {
        "task_id": task,
        "episode_idx": episode_idx,
        "outcome": outcome,
    }

    success_count = sum(
        1 for record in completed_for_task.values() if is_success(record.get("outcome"))
    )
    num_episodes = len(completed_for_task)
    metrics = {
        "team_id": "robolatent_real_robot",
        "config": config,
        "per_task": {
            task: {
                "avg_success": success_count / max(1, num_episodes),
                "success_count": success_count,
                "num_episodes": num_episodes,
            }
        },
        "overall": {
            "avg_success": success_count / max(1, num_episodes),
            "total_success": success_count,
            "total_episodes": num_episodes,
        },
        "updated_at": time.time(),
    }
    progress["metrics"] = metrics
    progress["finished"] = bool(finished)
    progress["updated_at"] = time.time()
    save_json_atomic(progress_path, progress)
    save_json_atomic(save_dir / "metrics.json", metrics)


def process_control_key(
    key: Optional[str],
    step: int,
) -> Optional[str]:
    if key is None:
        return None
    if key in {"q", "\x03"}:
        return "failure"
    if key == "s":
        return "success"
    if key == "f":
        return "failure"
    return None


def run_episode(
    *,
    args: argparse.Namespace,
    client: Any,
    robot: PiperRobolatentRealRobotInterface,
    episode_idx: int,
    keyboard: KeyboardPoller,
) -> tuple[str, Path, Path, Path]:
    reset_resp = client.reset()
    wait_for_ack(reset_resp, "reset_finished")
    logger.info("Policy memory reset: %s", reset_resp)

    robot.reset()

    prompt = args.prompt or TASK_PROMPTS[args.task]
    action_plan: Deque[np.ndarray] = collections.deque()
    observation_history: list[tuple[int, list[np.ndarray]]] = []
    max_history_frames = args.max_memory_frames * args.memory_interval + args.action_horizon + 2
    step_records: list[dict[str, Any]] = []
    outcome: Optional[str] = None
    last_infer_info: dict[str, Any] = {}
    chunk_step_times: list[dict[str, float]] = []
    server_input_dump_count = 0
    demo_high: Optional[DemoHighOverride] = None
    if args.demo_high_hdf5 is not None:
        demo_high = DemoHighOverride(
            args.demo_high_hdf5,
            image_size=args.image_size,
            start=args.demo_high_start,
            stride=args.demo_high_stride,
            hold_last=args.demo_high_hold_last,
        )
        logger.info(
            "Demo high override enabled: path=%s start=%d stride=%d length=%d hold_last=%s",
            args.demo_high_hdf5,
            args.demo_high_start,
            args.demo_high_stride,
            demo_high.length,
            args.demo_high_hold_last,
        )
    demo_left_wrist: Optional[DemoLeftWristOverride] = None
    if args.demo_left_wrist_hdf5 is not None:
        demo_left_wrist = DemoLeftWristOverride(
            args.demo_left_wrist_hdf5,
            image_size=args.image_size,
            start=args.demo_left_wrist_start,
            stride=args.demo_left_wrist_stride,
            hold_last=args.demo_left_wrist_hold_last,
        )
        logger.info(
            "Demo left wrist override enabled: path=%s start=%d stride=%d length=%d hold_last=%s",
            args.demo_left_wrist_hdf5,
            args.demo_left_wrist_start,
            args.demo_left_wrist_stride,
            demo_left_wrist.length,
            args.demo_left_wrist_hold_last,
        )

    video_path = args.save_dir / "videos" / f"{args.task}_ep_{episode_idx:04d}_cam_high.mp4"
    recorder = CamHighVideoRecorder(video_path, fps=args.video_fps)

    raw_obs = robot.get_observation()
    real_policy_obs = convert_observation(raw_obs, args.image_size, args.missing_camera_strategy)
    initial_demo_high_index = None
    if demo_high is not None:
        real_policy_obs, initial_demo_high_index = demo_high.apply(real_policy_obs, 0)
        logger.info("Initial cam_high overridden from demo index %d", initial_demo_high_index)
    policy_obs, initial_demo_left_wrist_index, _initial_compare_dir = maybe_apply_demo_left_wrist(
        args,
        demo_left_wrist,
        real_policy_obs,
        episode_idx=episode_idx,
        step=0,
        timestamp=raw_obs.timestamp,
    )
    if initial_demo_left_wrist_index is not None:
        logger.info(
            "Initial cam_left_wrist overridden from demo index %d",
            initial_demo_left_wrist_index,
        )
    recorder.add(policy_obs.views[1])
    observation_history.append((0, policy_obs.views))

    logger.info(
        "Episode %d started: task=%s prompt=%r",
        episode_idx,
        args.task,
        prompt,
    )

    online_start_step = 0
    try:
        if args.hybrid_replay_hdf5 is not None:
            replay_actions, replay_start, replay_end, replay_total = load_hdf5_left_action_slice(
                args.hybrid_replay_hdf5,
                start=args.hybrid_replay_start,
                end=args.hybrid_replay_end,
                steps=args.hybrid_replay_steps,
            )
            replay_hz = float(args.hybrid_replay_control_hz or args.control_hz)
            logger.info(
                "Hybrid offline replay: path=%s range=[%d,%d) total=%d steps=%d hz=%.3f",
                args.hybrid_replay_hdf5,
                replay_start,
                replay_end,
                replay_total,
                len(replay_actions),
                replay_hz,
            )

            for replay_i, action in enumerate(replay_actions):
                step = online_start_step
                hdf5_action_index = replay_start + replay_i
                key_outcome = process_control_key(keyboard.get_key(), step)
                if key_outcome is not None:
                    outcome = key_outcome
                    break

                pre_action_state = policy_obs.state.copy()
                _t_step_start = time.perf_counter()
                robot.send_action(action)
                _t_send_done = time.perf_counter()
                if replay_hz > 0:
                    time.sleep(max(0.0, 1.0 / replay_hz - (time.perf_counter() - _t_step_start)))
                _t_sleep_done = time.perf_counter()
                raw_obs = robot.get_observation()
                _t_obs_done = time.perf_counter()

                real_policy_obs = convert_observation(
                    raw_obs,
                    args.image_size,
                    args.missing_camera_strategy,
                )
                demo_high_index = None
                if demo_high is not None:
                    real_policy_obs, demo_high_index = demo_high.apply(
                        real_policy_obs,
                        step + 1,
                    )
                policy_obs, demo_left_wrist_index, left_wrist_compare_dir = maybe_apply_demo_left_wrist(
                    args,
                    demo_left_wrist,
                    real_policy_obs,
                    episode_idx=episode_idx,
                    step=step + 1,
                    timestamp=raw_obs.timestamp,
                )
                recorder.add(policy_obs.views[1])
                observation_history.append((step + 1, policy_obs.views))
                if len(observation_history) > max_history_frames:
                    del observation_history[:-max_history_frames]
                _t_post_done = time.perf_counter()

                step_records.append(
                    {
                        "phase": "hybrid_offline_replay",
                        "step": step,
                        "hdf5_action_index": hdf5_action_index,
                        "demo_high_index": demo_high_index,
                        "demo_left_wrist_index": demo_left_wrist_index,
                        "left_wrist_compare_dir": (
                            str(left_wrist_compare_dir) if left_wrist_compare_dir is not None else None
                        ),
                        "timestamp": raw_obs.timestamp,
                        "action_left": action.tolist(),
                        "pre_action_state_left": pre_action_state.tolist(),
                        "action_minus_pre_state": (action - pre_action_state).tolist(),
                        "state_left": policy_obs.state.tolist(),
                        "joint_state_14d": raw_obs.joint_state.astype(np.float32).tolist(),
                        "timing": {
                            "send": _t_send_done - _t_step_start,
                            "sleep": _t_sleep_done - _t_send_done,
                            "obs": _t_obs_done - _t_sleep_done,
                            "post": _t_post_done - _t_obs_done,
                            "total": _t_post_done - _t_step_start,
                        },
                    }
                )
                if (
                    args.hybrid_replay_log_every_n_steps > 0
                    and (replay_i % args.hybrid_replay_log_every_n_steps == 0
                         or replay_i == len(replay_actions) - 1)
                ):
                    logger.info(
                        "hybrid replay step=%d hdf5_idx=%d action6=%.4f state6=%.4f "
                        "max|action-state|=%.4f",
                        step,
                        hdf5_action_index,
                        float(action[6]),
                        float(policy_obs.state[6]),
                        float(np.max(np.abs(action - pre_action_state))),
                    )

                online_start_step = step + 1

                if robot.is_done():
                    outcome = "success"
                    break

        if outcome is None and args.hybrid_replay_hdf5 is not None:
            logger.info(
                "Hybrid replay finished; online policy starts at step=%d state=%s",
                online_start_step,
                np.round(policy_obs.state, 4),
            )

        for online_step in range(args.max_steps):
            if outcome is not None:
                break
            step = online_start_step + online_step
            key_outcome = process_control_key(keyboard.get_key(), step)
            if key_outcome is not None:
                outcome = key_outcome
                break

            if not action_plan:
                if chunk_step_times:
                    n = len(chunk_step_times)
                    sum_send = sum(t["send"] for t in chunk_step_times)
                    sum_sleep = sum(t["sleep"] for t in chunk_step_times)
                    sum_obs = sum(t["obs"] for t in chunk_step_times)
                    sum_post = sum(t["post"] for t in chunk_step_times)
                    sum_total = sum(t["total"] for t in chunk_step_times)
                    logger.info(
                        "chunk timing: steps=%d total=%.3fs send=%.3fs sleep=%.3fs "
                        "get_obs=%.3fs post=%.3fs (per-step avg: send=%.4f obs=%.4f post=%.4f)",
                        n, sum_total, sum_send, sum_sleep, sum_obs, sum_post,
                        sum_send / n, sum_obs / n, sum_post / n,
                    )
                    chunk_step_times.clear()

                if step == 0:
                    for name, view in zip(RYNNBRAIN_VIEW_NAMES, policy_obs.views):
                        logger.info(
                            "PIXEL STAT %s: mean=%.1f std=%.1f",
                            name,
                            float(view.mean()),
                            float(view.std()),
                        )

                infer_payload = make_rynnbrain_payload(
                    policy_obs.views,
                    observation_history,
                    prompt=prompt,
                    current_step=step,
                    image_size=args.image_size,
                    max_memory_frames=args.max_memory_frames,
                    memory_interval=args.memory_interval,
                )

                _packer = getattr(client, "_packer", None)
                _payload_bytes = -1
                if _packer is not None:
                    try:
                        _payload_bytes = len(_packer.pack(infer_payload))
                    except Exception:
                        _payload_bytes = -1
                example = infer_payload["examples"][0]
                _n_frames = len(example["image"])
                _n_memory_frames = len(example["memory"])
                if args.save_server_inputs and (
                    args.server_input_dump_limit <= 0
                    or server_input_dump_count < args.server_input_dump_limit
                ):
                    dump_dir = save_server_input_dump(
                        args.save_dir,
                        args.task,
                        episode_idx,
                        server_input_dump_count,
                        infer_payload,
                        payload_bytes=_payload_bytes,
                    )
                    server_input_dump_count += 1
                    logger.info("Saved server input dump: %s", dump_dir)

                _t_infer_start = time.perf_counter()
                outputs = client.infer(infer_payload)
                _infer_rtt = time.perf_counter() - _t_infer_start
                logger.info(
                    "infer rtt: payload_bytes=%d views=%d memory_frames=%d rtt=%.3fs",
                    _payload_bytes, _n_frames, _n_memory_frames, _infer_rtt,
                )

                if not outputs.get("ok", True):
                    raise RuntimeError(f"Policy server error: {outputs}")
                data = outputs.get("data", outputs)
                if "normalized_actions" not in data:
                    raise RuntimeError(f"Policy response missing normalized_actions: {outputs}")
                normalized_actions = np.asarray(data["normalized_actions"], dtype=np.float32)
                if normalized_actions.ndim == 3:
                    normalized_actions = normalized_actions[0]
                action_low = np.asarray(
                    args.action_stats.get("min", args.action_stats.get("q01")),
                    dtype=np.float32,
                )
                action_high = np.asarray(
                    args.action_stats.get("max", args.action_stats.get("q99")),
                    dtype=np.float32,
                )
                action_mask = np.asarray(
                    args.action_stats.get("mask", np.ones_like(action_low, dtype=bool)),
                    dtype=bool,
                )
                clipped_actions = np.clip(normalized_actions[:, : len(action_low)], -1.0, 1.0)
                scaled_actions = (
                    0.5 * (clipped_actions + 1.0) * (action_high - action_low)
                    + action_low
                )
                before_postprocess_actions = np.where(action_mask, scaled_actions, clipped_actions)
                normalized_gripper = normalized_actions[:, 6].astype(np.float32)
                clipped_gripper = clipped_actions[:, 6].astype(np.float32)
                scaled_gripper = scaled_actions[:, 6].astype(np.float32)
                before_postprocess_gripper = before_postprocess_actions[:, 6].astype(np.float32)
                actions = unnormalize_actions(
                    normalized_actions,
                    args.action_stats,
                    gripper_binary_threshold=args.gripper_binary_threshold,
                    gripper_postprocess=args.gripper_postprocess,
                )
                if actions.ndim != 2 or actions.shape[1] < 7:
                    raise RuntimeError(f"Expected action chunk shape (T, >=7), got {actions.shape}")
                for action in actions[: args.action_horizon, :7]:
                    action_plan.append(action.astype(np.float32))
                if not action_plan:
                    raise RuntimeError(f"Policy returned no executable actions: {actions.shape}")
                last_infer_info = {
                    key: to_jsonable(value)
                    for key, value in data.items()
                    if key != "normalized_actions"
                }
                last_infer_info.update({
                    "normalized_gripper_min": float(np.min(normalized_gripper)),
                    "normalized_gripper_max": float(np.max(normalized_gripper)),
                    "normalized_gripper_chunk": normalized_gripper[: args.action_horizon].tolist(),
                    "gripper_raw_chunk": normalized_gripper[: args.action_horizon].tolist(),
                    "gripper_clipped_chunk": clipped_gripper[: args.action_horizon].tolist(),
                    "gripper_scaled_unnormalized_chunk": scaled_gripper[: args.action_horizon].tolist(),
                    "gripper_before_postprocess_chunk": before_postprocess_gripper[: args.action_horizon].tolist(),
                    "gripper_postprocess": args.gripper_postprocess,
                    "gripper_binary_threshold": float(args.gripper_binary_threshold),
                    "action_gripper_chunk": actions[: args.action_horizon, 6].tolist(),
                })
                logger.info(
                    "infer: normalized=%s actions=%s "
                    "gripper_raw[min=%.4f max=%.4f first=%s] "
                    "gripper_scaled=%s gripper_before_post=%s "
                    "gripper_action=%s postprocess=%s threshold=%.4f infer_info=%s",
                    normalized_actions.shape,
                    actions.shape,
                    float(np.min(normalized_gripper)),
                    float(np.max(normalized_gripper)),
                    np.round(normalized_gripper[: args.action_horizon], 4),
                    np.round(scaled_gripper[: args.action_horizon], 4),
                    np.round(before_postprocess_gripper[: args.action_horizon], 4),
                    np.round(actions[: args.action_horizon, 6], 4),
                    args.gripper_postprocess,
                    args.gripper_binary_threshold,
                    last_infer_info,
                )

            action = action_plan.popleft()
            pre_action_state = policy_obs.state.copy()

            _t_step_start = time.perf_counter()
            robot.send_action(action)
            _t_send_done = time.perf_counter()
            if args.control_hz > 0:
                time.sleep(1.0 / args.control_hz)
            _t_sleep_done = time.perf_counter()

            raw_obs = robot.get_observation()
            _t_obs_done = time.perf_counter()

            real_policy_obs = convert_observation(raw_obs, args.image_size, args.missing_camera_strategy)
            demo_high_index = None
            if demo_high is not None:
                real_policy_obs, demo_high_index = demo_high.apply(real_policy_obs, step + 1)
            policy_obs, demo_left_wrist_index, left_wrist_compare_dir = maybe_apply_demo_left_wrist(
                args,
                demo_left_wrist,
                real_policy_obs,
                episode_idx=episode_idx,
                step=step + 1,
                timestamp=raw_obs.timestamp,
            )
            recorder.add(policy_obs.views[1])
            observation_history.append((step + 1, policy_obs.views))
            if len(observation_history) > max_history_frames:
                del observation_history[:-max_history_frames]
            _t_post_done = time.perf_counter()

            chunk_step_times.append({
                "send": _t_send_done - _t_step_start,
                "sleep": _t_sleep_done - _t_send_done,
                "obs": _t_obs_done - _t_sleep_done,
                "post": _t_post_done - _t_obs_done,
                "total": _t_post_done - _t_step_start,
            })

            step_records.append(
                {
                    "step": step,
                    "timestamp": raw_obs.timestamp,
                    "demo_high_index": demo_high_index,
                    "demo_left_wrist_index": demo_left_wrist_index,
                    "left_wrist_compare_dir": (
                        str(left_wrist_compare_dir) if left_wrist_compare_dir is not None else None
                    ),
                    "action_left": action.tolist(),
                    "pre_action_state_left": pre_action_state.tolist(),
                    "action_minus_pre_state": (action - pre_action_state).tolist(),
                    "state_left": policy_obs.state.tolist(),
                    "joint_state_14d": raw_obs.joint_state.astype(np.float32).tolist(),
                    "infer_info": last_infer_info,
                }
            )
            if args.log_every_n_steps > 0 and step % args.log_every_n_steps == 0:
                logger.info(
                    "step=%d max|action-state|=%.4f action=%s state=%s",
                    step,
                    float(np.max(np.abs(action - pre_action_state))),
                    np.round(action, 4),
                    np.round(pre_action_state, 4),
                )

            key_outcome = process_control_key(keyboard.get_key(), step + 1)
            if key_outcome is not None:
                outcome = key_outcome
                break

            if robot.is_done():
                outcome = "success"
                break

        if outcome is None:
            outcome = "timeout"
    except KeyboardInterrupt:
        outcome = "interrupted"
        logger.info("Episode %d interrupted; saving logs before exit.", episode_idx)
    finally:
        recorder.close()
        if demo_high is not None:
            demo_high.close()
        if demo_left_wrist is not None:
            demo_left_wrist.close()

    metadata = {
        "task": args.task,
        "episode_idx": episode_idx,
        "outcome": outcome,
        "prompt": prompt,
        "max_steps": args.max_steps,
        "action_horizon": args.action_horizon,
        "control_hz": args.control_hz,
        "host": args.host,
        "port": args.port,
        "camera_views": list(RYNNBRAIN_VIEW_NAMES),
        "max_memory_frames": args.max_memory_frames,
        "memory_interval": args.memory_interval,
        "dataset_statistics": str(args.dataset_statistics),
        "gripper_postprocess": args.gripper_postprocess,
        "gripper_binary_threshold": args.gripper_binary_threshold,
        "hybrid_replay_hdf5": (
            str(args.hybrid_replay_hdf5) if args.hybrid_replay_hdf5 is not None else None
        ),
        "hybrid_replay_start": args.hybrid_replay_start,
        "hybrid_replay_end": args.hybrid_replay_end,
        "hybrid_replay_steps": args.hybrid_replay_steps,
        "hybrid_replay_control_hz": args.hybrid_replay_control_hz,
        "online_start_step": online_start_step,
        "demo_high_hdf5": (
            str(args.demo_high_hdf5) if args.demo_high_hdf5 is not None else None
        ),
        "demo_high_start": args.demo_high_start,
        "demo_high_stride": args.demo_high_stride,
        "demo_high_hold_last": args.demo_high_hold_last,
        "demo_left_wrist_hdf5": (
            str(args.demo_left_wrist_hdf5) if args.demo_left_wrist_hdf5 is not None else None
        ),
        "demo_left_wrist_start": args.demo_left_wrist_start,
        "demo_left_wrist_stride": args.demo_left_wrist_stride,
        "demo_left_wrist_hold_last": args.demo_left_wrist_hold_last,
        "save_left_wrist_compare": args.save_left_wrist_compare,
        "left_wrist_compare_start": args.left_wrist_compare_start,
        "left_wrist_compare_end": args.left_wrist_compare_end,
        "left_wrist_compare_stride": args.left_wrist_compare_stride,
        "left_wrist_brightness_offset": args.left_wrist_brightness_offset,
        "right_wrist_brightness_offset": args.right_wrist_brightness_offset,
        "video_path": str(video_path),
    }
    json_log_path, pkl_log_path = save_episode_logs(
        args.save_dir,
        args.task,
        episode_idx,
        metadata,
        step_records,
    )
    return outcome, video_path, json_log_path, pkl_log_path


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Robolatent VLA on the real Piper robot via websocket policy server."
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5694)
    parser.add_argument(
        "--task",
        type=str,
        default="UncoverBlock",
    )
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--action_horizon", type=int, default=8)
    parser.add_argument("--control_hz", type=float, default=15.0)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--max_memory_frames", type=int, default=5)
    parser.add_argument("--memory_interval", type=int, default=10)
    parser.add_argument(
        "--dataset_statistics",
        type=Path,
        default=Path(
            "/data/share-folder/linsixu_deploy/results/Checkpoints/"
            "robolatent_uncoverblock_left_RynnBrainOFT/dataset_statistics.json"
        ),
    )
    parser.add_argument("--unnorm_key", type=str, default="new_embodiment")
    parser.add_argument(
        "--gripper_postprocess",
        type=str,
        choices=("binary", "raw"),
        default="binary",
        help=(
            "binary maps gripper output to min/max using --gripper_binary_threshold; "
            "raw keeps the clipped mask=False gripper signal for debugging."
        ),
    )
    parser.add_argument("--gripper_binary_threshold", type=float, default=0.0)
    parser.add_argument("--save_dir", type=Path, default=Path("runs/robolatent_eval"))
    parser.add_argument("--video_fps", type=int, default=15)
    parser.add_argument("--image_timeout", type=float, default=10.0)
    parser.add_argument("--action_interp_num", type=int, default=0)
    parser.add_argument("--action_wait", action="store_true")
    parser.add_argument("--clip_gripper", action="store_true")
    parser.add_argument("--cam_head_name", type=str, default="head_camera")
    parser.add_argument("--cam_high_name", type=str, default="agent_camera")
    parser.add_argument("--cam_left_wrist_name", type=str, default="left_wrist_camera")
    parser.add_argument("--cam_right_wrist_name", type=str, default="right_wrist_camera")
    parser.add_argument(
        "--left_wrist_brightness_offset",
        type=int,
        default=0,
        help="Add this brightness offset to cam_left_wrist frames before policy input.",
    )
    parser.add_argument(
        "--right_wrist_brightness_offset",
        type=int,
        default=0,
        help="Add this brightness offset to cam_right_wrist frames before policy input.",
    )
    parser.add_argument(
        "--missing_camera_strategy",
        type=str,
        choices=("strict", "copy", "zero"),
        default="strict",
    )
    parser.add_argument("--reset_from_dataset", action="store_true")
    parser.add_argument(
        "--reset_dataset_dir",
        type=Path,
        default=Path("/home/agilex/YueBo/data/Robolatent"),
    )
    parser.add_argument("--reset_qpos", type=str, default=None)
    parser.add_argument("--verify_reset", action="store_true")
    parser.add_argument("--reset_tolerance", type=float, default=0.03)
    parser.add_argument("--reset_settle_time", type=float, default=0.5)
    parser.add_argument("--reset_verify_timeout", type=float, default=5.0)
    parser.add_argument("--reset_verify_interval", type=float, default=0.1)
    parser.add_argument("--log_every_n_steps", type=int, default=10)
    parser.add_argument("--save_server_inputs", action="store_true")
    parser.add_argument(
        "--server_input_dump_limit",
        type=int,
        default=0,
        help="Maximum number of infer payloads to dump when --save_server_inputs is set; 0 means unlimited.",
    )
    parser.add_argument(
        "--hybrid_replay_hdf5",
        type=Path,
        default=None,
        help="Optional HDF5 episode; replay its left-arm actions before online policy inference.",
    )
    parser.add_argument("--hybrid_replay_start", type=int, default=0)
    parser.add_argument("--hybrid_replay_end", type=int, default=None)
    parser.add_argument("--hybrid_replay_steps", type=int, default=0)
    parser.add_argument("--hybrid_replay_control_hz", type=float, default=None)
    parser.add_argument("--hybrid_replay_log_every_n_steps", type=int, default=10)
    parser.add_argument(
        "--demo_high_hdf5",
        type=Path,
        default=None,
        help="Optional HDF5 episode; replace cam_high images with demo frames during rollout.",
    )
    parser.add_argument("--demo_high_start", type=int, default=0)
    parser.add_argument("--demo_high_stride", type=int, default=1)
    parser.add_argument("--demo_high_hold_last", action="store_true")
    parser.add_argument(
        "--demo_left_wrist_hdf5",
        type=Path,
        default=None,
        help="Optional HDF5 episode; replace cam_left_wrist images with demo frames during rollout.",
    )
    parser.add_argument("--demo_left_wrist_start", type=int, default=0)
    parser.add_argument("--demo_left_wrist_stride", type=int, default=1)
    parser.add_argument("--demo_left_wrist_hold_last", action="store_true")
    parser.add_argument(
        "--save_left_wrist_compare",
        action="store_true",
        help="When demo left wrist override is enabled, save real/demo cam_left_wrist comparisons.",
    )
    parser.add_argument("--left_wrist_compare_start", type=int, default=100)
    parser.add_argument("--left_wrist_compare_end", type=int, default=130)
    parser.add_argument("--left_wrist_compare_stride", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    args.task = normalize_task(args.task)
    args.save_dir = args.save_dir.expanduser().resolve()
    args.save_dir.mkdir(parents=True, exist_ok=True)
    if args.hybrid_replay_hdf5 is not None:
        args.hybrid_replay_hdf5 = args.hybrid_replay_hdf5.expanduser().resolve()
    if args.demo_high_hdf5 is not None:
        args.demo_high_hdf5 = args.demo_high_hdf5.expanduser().resolve()
    if args.demo_left_wrist_hdf5 is not None:
        args.demo_left_wrist_hdf5 = args.demo_left_wrist_hdf5.expanduser().resolve()
    if args.save_left_wrist_compare and args.demo_left_wrist_hdf5 is None:
        raise ValueError("--save_left_wrist_compare requires --demo_left_wrist_hdf5")
    if args.left_wrist_compare_stride <= 0:
        raise ValueError("--left_wrist_compare_stride must be positive")
    if (
        args.left_wrist_compare_end >= 0
        and args.left_wrist_compare_end < args.left_wrist_compare_start
    ):
        raise ValueError("--left_wrist_compare_end must be >= --left_wrist_compare_start, or -1")

    config = {
        "host": args.host,
        "port": args.port,
        "task": args.task,
        "num_episodes": args.num_episodes,
        "max_steps": args.max_steps,
        "action_horizon": args.action_horizon,
        "control_hz": args.control_hz,
        "image_size": args.image_size,
        "max_memory_frames": args.max_memory_frames,
        "memory_interval": args.memory_interval,
        "camera_views": list(RYNNBRAIN_VIEW_NAMES),
        "dataset_statistics": str(args.dataset_statistics),
        "unnorm_key": args.unnorm_key,
        "gripper_postprocess": args.gripper_postprocess,
        "gripper_binary_threshold": args.gripper_binary_threshold,
        "missing_camera_strategy": args.missing_camera_strategy,
        "reset_from_dataset": args.reset_from_dataset,
        "reset_dataset_dir": str(args.reset_dataset_dir),
        "reset_qpos": args.reset_qpos,
        "verify_reset": args.verify_reset,
        "reset_tolerance": args.reset_tolerance,
        "reset_settle_time": args.reset_settle_time,
        "reset_verify_timeout": args.reset_verify_timeout,
        "reset_verify_interval": args.reset_verify_interval,
        "save_server_inputs": args.save_server_inputs,
        "server_input_dump_limit": args.server_input_dump_limit,
        "hybrid_replay_hdf5": (
            str(args.hybrid_replay_hdf5) if args.hybrid_replay_hdf5 is not None else None
        ),
        "hybrid_replay_start": args.hybrid_replay_start,
        "hybrid_replay_end": args.hybrid_replay_end,
        "hybrid_replay_steps": args.hybrid_replay_steps,
        "hybrid_replay_control_hz": args.hybrid_replay_control_hz,
        "demo_high_hdf5": (
            str(args.demo_high_hdf5) if args.demo_high_hdf5 is not None else None
        ),
        "demo_high_start": args.demo_high_start,
        "demo_high_stride": args.demo_high_stride,
        "demo_high_hold_last": args.demo_high_hold_last,
        "demo_left_wrist_hdf5": (
            str(args.demo_left_wrist_hdf5) if args.demo_left_wrist_hdf5 is not None else None
        ),
        "demo_left_wrist_start": args.demo_left_wrist_start,
        "demo_left_wrist_stride": args.demo_left_wrist_stride,
        "demo_left_wrist_hold_last": args.demo_left_wrist_hold_last,
        "save_left_wrist_compare": args.save_left_wrist_compare,
        "left_wrist_compare_start": args.left_wrist_compare_start,
        "left_wrist_compare_end": args.left_wrist_compare_end,
        "left_wrist_compare_stride": args.left_wrist_compare_stride,
        "left_wrist_brightness_offset": args.left_wrist_brightness_offset,
        "right_wrist_brightness_offset": args.right_wrist_brightness_offset,
    }
    reset_position = resolve_reset_position(args)
    args.action_stats = load_action_stats(args.dataset_statistics, args.unnorm_key)

    client = create_mmevla_client(host=args.host, port=args.port)
    logger.info("Connected policy server metadata: %s", client.get_server_metadata())

    robot_kwargs = {
        "reset_position": reset_position,
        "image_timeout": args.image_timeout,
        "action_interp_num": args.action_interp_num,
        "action_wait": args.action_wait,
        "clip_gripper": args.clip_gripper,
        "verify_reset": args.verify_reset,
        "reset_tolerance": args.reset_tolerance,
        "reset_settle_time": args.reset_settle_time,
        "reset_verify_timeout": args.reset_verify_timeout,
        "reset_verify_interval": args.reset_verify_interval,
        "cam_high_name": args.cam_high_name,
        "cam_left_wrist_name": args.cam_left_wrist_name,
        "left_wrist_brightness_offset": args.left_wrist_brightness_offset,
        "right_wrist_brightness_offset": args.right_wrist_brightness_offset,
    }
    try:
        robot_params = inspect.signature(PiperRobolatentRealRobotInterface).parameters
        if "cam_head_name" in robot_params:
            robot_kwargs["cam_head_name"] = args.cam_head_name
        if "cam_right_wrist_name" in robot_params:
            robot_kwargs["cam_right_wrist_name"] = args.cam_right_wrist_name
    except (TypeError, ValueError):
        logger.warning("Could not inspect robot interface signature; using legacy camera args.")

    robot = PiperRobolatentRealRobotInterface(**robot_kwargs)

    keyboard = KeyboardPoller()
    keyboard.start()
    try:
        for episode_idx in range(args.num_episodes):
            progress_path = args.save_dir / "progress.json"
            progress = load_json(progress_path) or {}
            completed = progress.get("completed", {}).get(args.task, {})
            if str(episode_idx) in completed:
                logger.info("Skip completed episode %d for task %s", episode_idx, args.task)
                continue

            outcome, video_path, json_log_path, pkl_log_path = run_episode(
                args=args,
                client=client,
                robot=robot,
                episode_idx=episode_idx,
                keyboard=keyboard,
            )
            update_progress(
                args.save_dir,
                task=args.task,
                episode_idx=episode_idx,
                outcome=outcome,
                video_path=video_path,
                json_log_path=json_log_path,
                pkl_log_path=pkl_log_path,
                config=config,
                finished=episode_idx == args.num_episodes - 1,
            )
            logger.info("Episode %d finished with outcome=%s", episode_idx, outcome)
    finally:
        keyboard.stop()
        robot.shutdown()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
