#!/usr/bin/env python3
"""
SimpleVLA socket client - Keep robot at default qpos only.

This script runs on the robot machine. It:
  * connects to ROS (ros_bridge + controller)
  * moves the robot to default position and keeps it there

Usage example:
    python examples/simplevla_client.py \
        --default_qpos "0,0,0,0,0,0,1,0,0,0,0,0,0,1"
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

from controller import AgilexController
import ros_bridge

RosOperator = ros_bridge.RosOperator
RosOperatorForRos2 = getattr(ros_bridge, "RosOperatorForRos2", None)

# ----------------------- Robot helpers -----------------------
def build_ros_operator(mode: str):
    ros1_available = getattr(ros_bridge, "rospy", None) is not None
    ros2_available = RosOperatorForRos2 is not None and getattr(ros_bridge, "rclpy", None) is not None

    def build_ros1():
        ros_distro = os.environ.get("ROS_DISTRO", "")
        print("Using ROS1 (ROS_DISTRO=%s)" % (ros_distro or "unknown"))
        return RosOperator()

    def build_ros2():
        print("Using ROS2.")
        return RosOperatorForRos2()

    if mode == "ros1":
        if not ros1_available:
            raise RuntimeError("ROS1 is unavailable on this machine.")
        return build_ros1()
    if mode == "ros2":
        if not ros2_available:
            raise RuntimeError("ROS2 is unavailable on this machine.")
        return build_ros2()
    # auto
    if ros1_available:
        return build_ros1()
    if ros2_available:
        return build_ros2()
    raise RuntimeError("Neither ROS1 nor ROS2 bindings are available.")


def get_qpos(con: AgilexController) -> np.ndarray:
    left_range = con.left_gripper_joint_limits
    right_range = con.right_gripper_joint_limits
    qpos = con.get_current_qpos()
    qpos = qpos.copy()
    qpos[6] = (qpos[6] - left_range[0]) / (left_range[1] - left_range[0] + 1e-8)
    qpos[13] = (qpos[13] - right_range[0]) / (right_range[1] - right_range[0] + 1e-8)
    return qpos


def set_qpos(con: AgilexController, qpos: np.ndarray, wait: bool = True):
    left_range = con.left_gripper_joint_limits
    right_range = con.right_gripper_joint_limits
    cmd = qpos.copy()
    cmd[6] = left_range[0] + (left_range[1] - left_range[0]) * cmd[6]
    cmd[13] = right_range[0] + (right_range[1] - right_range[0]) * cmd[13]
    con.set_current_qpos(cmd, interp_num=0, wait=wait)


def move_to_default(con: AgilexController, default_qpos: np.ndarray):
    print("Moving to default pose...")
    set_qpos(con, default_qpos, wait=True)

DEFAULT_QPOS = np.array(
    [
        -1.78963946e-03,
        4.98669991e-03,
        -4.61852954e-03,
        -2.55775952e-04,
        2.39215694e-03,
        -2.38695082e-03,
        9.98039226e-01,
        8.29102000e-01,
        2.19820343e+00,
        -1.55236671e+00,
        -5.48235317e-01,
        7.43960808e-01,
        1.15031849e+00,
        3.23529415e-01
    ],
    dtype=float,
)

DEFAULT_QPOS = np.array(
    [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1
    ],
    dtype=float,
)
# DEFAULT_QPOS = np.array(
#PickXTimes
#     [-0.020427, 0.006367, -0.000820, 0.031277, 0.340995, -0.095401, 0.000200,
#    -0.033144, 0.001989, -0.000645, 0.187453, 0.000000, -0.378064, 0.066700],
#UncoverBlock
#       [-2.04967000e-02,  5.44252805e-03, -2.09327997e-03,  1.03634804e-01,
#   3.49403322e-01, -7.61081725e-02,  9.99999975e-05, -3.13992016e-02,
#   1.98861607e-03, -6.45428023e-04,  1.87453225e-01,  0.00000000e+00,
#  -3.78063798e-01,  6.66999966e-02],
    # [0,0,0,0,0,0,0,
    #  0,0,0,0,0,0,0],
#     [-0.043034, 1.666164, -1.232017, -0.060007, 1.156781, 0.330354, 0.92900,
#    -0.033144, 0.001989, -0.000645, 0.187453, 0.000000, -0.378064, 0.066700],
#     [-0.021404, 0.734061, -0.592660, -0.052559, 0.249816, -0.034905, 0.000200,
#    -0.033144, 0.001989, -0.000645, 0.187453, 0.000000, -0.378064, 0.066700],
    # dtype=float,
# )


def parse_default_qpos(value: Optional[str]) -> np.ndarray:
    if not value:
        return DEFAULT_QPOS.copy()
    path = Path(value)
    if path.exists():
        if path.suffix in (".npy", ".npz"):
            arr = np.load(path)
            if isinstance(arr, np.lib.npyio.NpzFile):
                arr = arr[arr.files[0]]
            return np.asarray(arr).reshape(-1)
        if path.suffix in (".json", ".txt"):
            import json

            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return np.asarray(data, dtype=float).reshape(-1)
    return np.fromstring(value, sep=",")


def keep_at_default_position(controller: AgilexController, default_qpos: np.ndarray, hold_time: float = 3600.0):
    """
    Keep the robot at default position for specified time.
    
    Args:
        controller: Robot controller
        default_qpos: Default joint positions
        hold_time: How long to hold the position (seconds)
    """
    print(f"Keeping robot at default position for {hold_time} seconds")
    print("Press Ctrl+C to stop")
    
    start_time = time.time()
    try:
        while time.time() - start_time < hold_time:
            # Constantly set to default position
            set_qpos(controller, default_qpos, wait=False)
            time.sleep(0.01)  # Small sleep to prevent CPU overload
            
            # Print current qpos every 5 seconds for monitoring
            current_time = time.time() - start_time
            if current_time % 5 < 0.01:
                current_qpos = get_qpos(controller)
                print(f"Time: {current_time:.1f}s, Current qpos: {current_qpos}")
                
    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        # Make sure robot ends at default position
        set_qpos(controller, default_qpos, wait=True)
        print("Robot returned to default position")


# ----------------------- Main loop -----------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Keep robot at default qpos position")
    parser.add_argument("--default_qpos", type=str, default="", help="Default qpos position (comma-separated or file path)")
    parser.add_argument("--hold_time", type=float, default=3600.0, help="How long to hold position in seconds")
    parser.add_argument("--check_interval", type=float, default=5.0, help="How often to print status (seconds)")
    parser.add_argument(
        "--ros_version",
        choices=["auto", "ros1", "ros2"],
        default="auto",
        help="Select ROS stack (default: auto-detect).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Initialize ROS and controller
    ros_operator = build_ros_operator(args.ros_version)
    time.sleep(3)  # 休眠3秒让ros_operator中的订阅回调函数接收数据
    print("ROS operator constructed.")
    
    controller = AgilexController(ros_operator)
    print("Agilex controller initialized.")
    
    # Parse default qpos
    default_qpos = parse_default_qpos(args.default_qpos)
    print(f"Default qpos: {default_qpos}")
    
    # Move to and keep at default position
    try:
        move_to_default(controller, default_qpos)
        keep_at_default_position(controller, default_qpos, args.hold_time)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Cleanup
        if hasattr(ros_operator, "shutdown"):
            try:
                ros_operator.shutdown()
            except Exception:
                pass
        print("Cleanup completed")


if __name__ == "__main__":
    main()
