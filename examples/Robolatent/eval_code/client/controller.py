import logging
import time
from copy import deepcopy
from typing import List, Union

import numpy as np
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import JointState

from ros_bridge import RosOperator

logger = logging.getLogger(__name__)


# TODO: consider intepolation with max distance or trajectory generator using rlia, other than simple linear intepolation.
def mul_linear_expand(
    arr: np.ndarray, expand_times: Union[int, List[int]], is_interp: bool = True
):
    arr_len = arr.shape[0]
    dim = arr.shape[1]
    if isinstance(expand_times, int):
        interp_path = np.zeros(shape=(arr_len * expand_times, dim), dtype=float)
    else:
        assert len(expand_times) == arr_len - 1, "Invalid expand_times size."
        interp_path = np.zeros(shape=(sum(expand_times), dim), dtype=float)

    idx = 0
    for i in range(0, arr_len - 1):
        if isinstance(expand_times, int):
            sample_times = expand_times
        else:
            sample_times = expand_times[i]
        for k in range(sample_times):
            if is_interp:
                v = (sample_times - k) / sample_times * arr[i] + k / sample_times * arr[
                    i + 1
                ]
            else:
                v = arr[i]
            interp_path[idx] = v
            idx += 1
    interp_path = interp_path[:idx]
    return interp_path


class AgilexController:
    # TODO: add more feature to the controller

    def __init__(self, ros_operator: RosOperator, init_controller=False):
        self.ros_operator = ros_operator

        self.left_gripper_joint_limits = (
            0,
            0.1,
        )  # 0.1表示夹爪最大能够张开0.1m，且是全张开到底的。
        self.right_gripper_joint_limits = (0, 0.1)

        self.has_controller = False
        if init_controller:
            logger.warning(
                "init_controller=True is not supported without Embodychain dependencies. Ignoring."
            )

        self._previous_qpos = self.get_current_qpos()

    def get_current_qpos(self, name: str = None):
        left_arm_frame, right_arm_frame = self.ros_operator.get_puppet_arm_frame()
        left_arm_frame: JointState
        right_arm_frame: JointState
        qpos_left = np.array(left_arm_frame.position)
        qpos_right = np.array(right_arm_frame.position)
        # print("Left arm joint : {}.".format(np.round(qpos_left, 4)))
        # print("Right arm joint : {}.".format(np.round(qpos_right, 4)))
        if name == None:
            return np.concatenate([qpos_left, qpos_right])
        elif name == "left_arm":
            return qpos_left[:6]
        elif name == "right_arm":
            return qpos_right[:6]
        elif name == "left_eef":
            return qpos_left[6]
        elif name == "right_eef":
            return qpos_right[6]
        else:
            logger.error("Invalid name for get_current_qpos")

    def set_current_qpos(
        self, qpos: np.ndarray, name: str = None, interp_num=0, wait: bool = True
    ):
        qpos_target = deepcopy(qpos)
        qpos_left = qpos_target[:7]
        qpos_right = qpos_target[7:]

        if interp_num == 0:
            self.ros_operator.puppet_arm_publish(
                qpos_left.tolist(), qpos_right.tolist()
            )
            if wait:
                self._wait_qpos_for_motion_done(target_qpos=qpos)
            self._previous_qpos = np.concatenate((qpos_left, qpos_right))
        else:
            expand_times = [interp_num]
            expand_left = np.stack((self._previous_qpos[:7], qpos_left), axis=0)
            expand_right = np.stack((self._previous_qpos[7:], qpos_right), axis=0)
            new_qpos_left = mul_linear_expand(expand_left, expand_times)
            new_qpos_right = mul_linear_expand(expand_right, expand_times)
            for i in range(len(new_qpos_left)):
                # print(f"left arm joint : {np.round(new_qpos_left[i], 4)}")
                # print(f"right arm joint : {np.round(new_qpos_right[i], 4)}")
                self.ros_operator.puppet_arm_publish(
                    new_qpos_left[i].tolist(), new_qpos_right[i].tolist()
                )
            if wait:
                self._wait_qpos_for_motion_done(target_qpos=qpos)
            self._previous_qpos = np.concatenate(
                (new_qpos_left[-1], new_qpos_right[-1])
            )
        return True

    def set_current_xpos(self, name: str, xpos: np.ndarray):
        if self.has_controller is False:
            logger.warning("No controller is initialized")
            return False

        if name not in ["left_arm", "right_arm"]:
            logger.error("Invalid name for set_current_xpos")
        ret, qpos = self.sim_robot.get_ik(
            xpos=xpos,
            uid=name,
            qpos_seed=self.get_current_qpos(name=name),
            is_world_coordinates=False,
        )

        if ret is False:
            logger.warning("Failed to get IK solution")
            return False
        else:
            current_qpos = self.get_current_qpos()
            if name == "left_arm":
                current_qpos[:6] = qpos
            elif name == "right_arm":
                current_qpos[7:13] = qpos
            self.set_current_qpos(current_qpos)
            return True

    def set_current_xpos_L(
        self,
        name: str,
        xpos: np.ndarray,
        num_points: int = 10,
    ):
        """tcp末端笛卡尔直线运动"""
        if not isinstance(name, str) or name not in ["left_arm", "right_arm"]:
            logger.error("Invalid name for set_current_xpos")
            return

        if not isinstance(xpos, np.ndarray) or xpos.shape != (4, 4):
            logger.error("xpos must be a 4x4 homogeneous transformation matrix")
            return

        current_pose = self.get_current_xpos(name)

        start_pos = current_pose[:3, 3]
        end_pos = xpos[:3, 3]
        positions = np.linspace(start_pos, end_pos, num_points)

        start_rot = R.from_matrix(current_pose[:3, :3])
        end_rot = R.from_matrix(xpos[:3, :3])
        rotations = []

        for t in np.linspace(0, 1, num_points):
            rotations.append((start_rot * (end_rot * start_rot.inv()) ** t).as_matrix())

        interpolated_poses = []
        for pos, rot in zip(positions, rotations):
            pose = np.eye(4)
            pose[:3, :3] = rot
            pose[:3, 3] = pos
            interpolated_poses.append(pose)
        interpolated_qpos_list = []
        for interpolated_pose in interpolated_poses:
            res, qpos = self.sim_robot.get_ik(
                uid=name,
                qpos_seed=self.get_current_qpos(name),
                xpos=interpolated_pose,
                is_world_coordinates=False,
            )
            if not res:
                logger.log_warning("Failed to get IK solution on the line")
                return False
            else:
                interpolated_qpos_list.append(qpos)

        for qpos in interpolated_qpos_list:
            current_qpos = self.get_current_qpos()
            if name == "left_arm":
                current_qpos[:6] = qpos
            elif name == "right_arm":
                current_qpos[7:13] = qpos
            self.set_current_qpos(current_qpos)
        return True

    def get_current_xpos(self, name: str):
        """获取末端tcp的姿态,返回是齐次矩阵"""
        if self.has_controller is False:
            logger.log_error("No controller is initialized")

        if name not in ["left_arm", "right_arm"]:
            logger.log_error("Invalid name for get_current_xpos")

        xpos = self.sim_robot.get_fk(
            qpos=self.get_current_qpos(name=name), uid=name, is_world_coordinates=False
        )
        return xpos

    def get_current_xpos_origin(self, name: str):  # 一般不用
        """使用监听话题的方式获得末端法兰盘的xyz,roll,pitch,yaw(角度制)"""
        if name not in ["left_arm", "right_arm"]:
            logger.log_error("Invalid name for get_current_xpos")

        arm_xpos: PosCmd = self.ros_operator.get_puppet_arm_pos_frame(name)

        xpos = np.array(
            [
                arm_xpos.x,
                arm_xpos.y,
                arm_xpos.z,
                arm_xpos.roll,
                arm_xpos.pitch,
                arm_xpos.yaw,
            ]
        )
        return xpos

    def _wait_qpos_for_motion_done(
        self,
        timeout=10.0,
        poll_interval=0.1,
        distance_threshold=0.1,
        gripper_threshold=0.02,
        count_threshold=3,
        target_qpos=None,
    ):
        """阻塞直到指定机械臂关节运动完成"""
        start_time = time.time()
        ret = self.ros_operator.get_puppet_arm_frame()
        arm_val = np.concatenate([np.array(ret[0].position), np.array(ret[1].position)])

        count = 1
        while True:
            ret = self.ros_operator.get_puppet_arm_frame()
            new_arm_val = np.concatenate(
                [np.array(ret[0].position), np.array(ret[1].position)]
            )
            target_constraint = (
                np.linalg.norm(arm_val - new_arm_val) <= distance_threshold
            )
            # from IPython import embed
            # embed()
            if target_constraint:
                count += 1

            if count > count_threshold:
                return True
            arm_val = new_arm_val
            if time.time() - start_time > timeout:
                raise TimeoutError(f"qpos motion timeout after {timeout}s")
            time.sleep(poll_interval)

    def set_gripper(self, name: str, gripper_cmd: float):
        if name not in ["left_arm", "right_arm"]:
            logger.log_error("Invalid name for set_gripper")
        current_qpos = self.get_current_qpos()
        if name == None:
            current_qpos[6] = gripper_cmd
            current_qpos[13] = gripper_cmd
            self.set_current_qpos(current_qpos)
        elif name == "left_arm":
            current_qpos[6] = gripper_cmd
            self.set_current_qpos(current_qpos)
        elif name == "right_arm":
            current_qpos[13] = gripper_cmd
            self.set_current_qpos(current_qpos)
        else:
            logger.log_error("Invalid name for set_gripper")


if __name__ == "__main__":
    np.set_printoptions(5, suppress=True)
    ros_operator = RosOperator()
    controller = AgilexController(ros_operator)

    original_qpos = np.zeros(14)
    original_qpos[1] += 0.2
    original_qpos[8] += 0.2
    target_qpos = np.array(
        [0, 1.0, -1.2, 0, 1.22338, 0, 0.06986, 0, 1.0, -1.2, 0, 1.22338, 0, 0.0896]
    )

    target_qpos_2 = deepcopy(target_qpos)
    target_qpos_2[0] += 0.1
    target_qpos_2[3] += 0.1

    original_qpos = np.zeros(14)

    from IPython import embed

    embed()
