#!/usr/bin/env python3
"""Probe whether demo wrist frames trigger gripper decisions by themselves."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from examples.new_deploy.pi_infer import (
    LEFT_WRIST_VIEW_INDEX,
    TASK_PROMPTS,
    RYNNBRAIN_VIEW_NAMES,
    StarVLAWebsocketClientPolicy,
    black_views,
    convert_observation,
    load_action_stats,
    make_rynnbrain_payload,
    resize_image,
    save_json_atomic,
    save_server_input_dump,
    to_jsonable,
    unnormalize_actions,
)
from examples.new_deploy.piper_env import PiperRobolatentRealRobotInterface


logger = logging.getLogger("probe_demo_wrist_conditioning")


class DemoLeftWristFrames:
    def __init__(self, path: Path, image_size: int) -> None:
        import h5py

        self.path = path
        self.image_size = int(image_size)
        self.file = h5py.File(path, "r")
        self.qpos = np.asarray(self.file["observations/qpos"], dtype=np.float32)
        self.actions = np.asarray(self.file["action"], dtype=np.float32)
        self.left_wrist = self.file["observations/images/cam_left_wrist"]
        if self.qpos.ndim != 2 or self.qpos.shape[1] < 14:
            raise ValueError(f"Expected observations/qpos shape (T, >=14), got {self.qpos.shape}")
        if self.left_wrist.ndim != 4 or self.left_wrist.shape[-1] != 3:
            raise ValueError(
                f"Expected cam_left_wrist shape (T,H,W,3), got {self.left_wrist.shape}"
            )

    @property
    def length(self) -> int:
        return int(self.left_wrist.shape[0])

    def frame(self, index: int) -> np.ndarray:
        index = int(np.clip(index, 0, self.length - 1))
        return resize_image(np.asarray(self.left_wrist[index]), self.image_size)

    def target_qpos(self, index: int) -> np.ndarray:
        return self.qpos[int(index), :14].astype(np.float32)

    def close(self) -> None:
        self.file.close()


def replace_left_wrist(views: list[np.ndarray], wrist_image: np.ndarray) -> list[np.ndarray]:
    out = list(views)
    out[LEFT_WRIST_VIEW_INDEX] = wrist_image
    return out


def demo_index_for_query(args: argparse.Namespace, query_idx: int) -> int:
    if args.mode == "fixed":
        return args.demo_frame_index
    if args.mode == "sequence":
        return args.demo_frame_index + query_idx * args.demo_stride
    raise ValueError(f"Unsupported mode: {args.mode}")


def make_history(
    *,
    args: argparse.Namespace,
    demo: DemoLeftWristFrames,
    real_views: list[np.ndarray],
    current_step: int,
    current_demo_index: int,
) -> list[tuple[int, list[np.ndarray]]]:
    if args.memory_mode == "zero":
        return []

    history: list[tuple[int, list[np.ndarray]]] = []
    for i in range(1, args.max_memory_frames + 1):
        memory_step = current_step - i * args.memory_interval
        if memory_step < 0:
            continue
        if args.memory_mode == "fixed":
            memory_demo_index = current_demo_index
        elif args.memory_mode == "demo_past":
            memory_demo_index = memory_step
        else:
            raise ValueError(f"Unsupported memory_mode: {args.memory_mode}")
        history.append((memory_step, replace_left_wrist(real_views, demo.frame(memory_demo_index))))
    return history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reset to a demo frame and query the policy with fixed or sequential demo "
            "left-wrist frames. This distinguishes current-frame prediction from "
            "following the demo visual phase."
        )
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5678)
    parser.add_argument("--task", type=str, default="PickXtimes")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument(
        "--hdf5",
        type=Path,
        default=Path("/home/agilex/Sixu/data/origin_data/episode_0.hdf5"),
    )
    parser.add_argument(
        "--dataset_statistics",
        type=Path,
        default=Path("/home/agilex/Sixu/dataset_statistics.json"),
    )
    parser.add_argument("--unnorm_key", type=str, default="new_embodiment")
    parser.add_argument("--save_dir", type=Path, default=Path("runs/probe_demo_wrist_conditioning"))
    parser.add_argument("--demo_frame_index", type=int, default=108)
    parser.add_argument("--mode", choices=("fixed", "sequence"), default="fixed")
    parser.add_argument("--num_queries", type=int, default=1)
    parser.add_argument("--demo_stride", type=int, default=1)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--action_horizon", type=int, default=8)
    parser.add_argument("--max_memory_frames", type=int, default=5)
    parser.add_argument("--memory_interval", type=int, default=10)
    parser.add_argument("--memory_mode", choices=("demo_past", "fixed", "zero"), default="demo_past")
    parser.add_argument("--gripper_binary_threshold", type=float, default=0.5)
    parser.add_argument("--gripper_postprocess", choices=("binary", "raw"), default="binary")
    parser.add_argument("--execute_actions", action="store_true")
    parser.add_argument("--control_hz", type=float, default=15.0)
    parser.add_argument("--clip_gripper", action="store_true")
    parser.add_argument("--verify_reset", action="store_true")
    parser.add_argument("--reset_tolerance", type=float, default=0.03)
    parser.add_argument("--reset_settle_time", type=float, default=0.5)
    parser.add_argument("--reset_verify_timeout", type=float, default=8.0)
    parser.add_argument("--reset_verify_interval", type=float, default=0.1)
    parser.add_argument("--image_timeout", type=float, default=10.0)
    parser.add_argument("--action_interp_num", type=int, default=0)
    parser.add_argument("--save_server_inputs", action="store_true")
    parser.add_argument("--cam_head_name", type=str, default="head_camera")
    parser.add_argument("--cam_high_name", type=str, default="agent_camera")
    parser.add_argument("--cam_left_wrist_name", type=str, default="left_wrist_camera")
    parser.add_argument("--cam_right_wrist_name", type=str, default="right_wrist_camera")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.num_queries <= 0:
        raise ValueError("--num_queries must be positive")
    if args.demo_stride <= 0:
        raise ValueError("--demo_stride must be positive")

    args.hdf5 = args.hdf5.expanduser().resolve()
    args.dataset_statistics = args.dataset_statistics.expanduser().resolve()
    args.save_dir = args.save_dir.expanduser().resolve()
    args.save_dir.mkdir(parents=True, exist_ok=True)

    demo = DemoLeftWristFrames(args.hdf5, args.image_size)
    if not 0 <= args.demo_frame_index < demo.length:
        raise ValueError(
            f"--demo_frame_index={args.demo_frame_index} out of range [0, {demo.length})"
        )

    action_stats = load_action_stats(args.dataset_statistics, args.unnorm_key)
    prompt = args.prompt or TASK_PROMPTS.get(args.task, args.task)
    target_qpos = demo.target_qpos(args.demo_frame_index)

    config = vars(args).copy()
    config.update({
        "hdf5": str(args.hdf5),
        "dataset_statistics": str(args.dataset_statistics),
        "save_dir": str(args.save_dir),
        "camera_views": list(RYNNBRAIN_VIEW_NAMES),
        "target_qpos": target_qpos.tolist(),
    })
    save_json_atomic(args.save_dir / "config.json", config)

    logger.info(
        "Probe setup: mode=%s frame=%d num_queries=%d memory_mode=%s target_qpos6=%.4f",
        args.mode,
        args.demo_frame_index,
        args.num_queries,
        args.memory_mode,
        float(target_qpos[6]),
    )

    client = StarVLAWebsocketClientPolicy(args.host, args.port)
    reset_resp = client.reset()
    logger.info("Policy memory reset: %s", reset_resp)

    robot = PiperRobolatentRealRobotInterface(
        reset_position=target_qpos,
        image_timeout=args.image_timeout,
        action_interp_num=args.action_interp_num,
        action_wait=True,
        clip_gripper=args.clip_gripper,
        verify_reset=args.verify_reset,
        reset_tolerance=args.reset_tolerance,
        reset_settle_time=args.reset_settle_time,
        reset_verify_timeout=args.reset_verify_timeout,
        reset_verify_interval=args.reset_verify_interval,
        cam_head_name=args.cam_head_name,
        cam_high_name=args.cam_high_name,
        cam_left_wrist_name=args.cam_left_wrist_name,
        cam_right_wrist_name=args.cam_right_wrist_name,
    )

    records: list[dict[str, Any]] = []
    try:
        robot.reset()
        for query_idx in range(args.num_queries):
            raw_obs = robot.get_observation()
            real_obs = convert_observation(raw_obs, args.image_size, "strict")
            demo_index = int(np.clip(demo_index_for_query(args, query_idx), 0, demo.length - 1))
            current_step = args.demo_frame_index + query_idx * args.demo_stride
            current_views = replace_left_wrist(real_obs.views, demo.frame(demo_index))
            history = make_history(
                args=args,
                demo=demo,
                real_views=real_obs.views,
                current_step=current_step,
                current_demo_index=demo_index,
            )
            payload = make_rynnbrain_payload(
                current_views,
                history,
                prompt=prompt,
                current_step=current_step,
                image_size=args.image_size,
                max_memory_frames=args.max_memory_frames,
                memory_interval=args.memory_interval,
            )

            payload_bytes = -1
            packer = getattr(client, "_packer", None)
            if packer is not None:
                try:
                    payload_bytes = len(packer.pack(payload))
                except Exception:
                    payload_bytes = -1
            if args.save_server_inputs:
                save_server_input_dump(
                    args.save_dir,
                    args.task,
                    0,
                    query_idx,
                    payload,
                    payload_bytes=payload_bytes,
                )

            outputs = client.infer(payload)
            data = outputs.get("data", outputs)
            if "normalized_actions" not in data:
                raise RuntimeError(f"Policy response missing normalized_actions: {outputs}")
            normalized_actions = np.asarray(data["normalized_actions"], dtype=np.float32)
            if normalized_actions.ndim == 3:
                normalized_actions = normalized_actions[0]
            actions = unnormalize_actions(
                normalized_actions,
                action_stats,
                gripper_binary_threshold=args.gripper_binary_threshold,
                gripper_postprocess=args.gripper_postprocess,
            )
            raw_gripper = normalized_actions[:, 6].astype(np.float32)
            action_gripper = actions[:, 6].astype(np.float32)

            record = {
                "query_idx": query_idx,
                "mode": args.mode,
                "current_step": current_step,
                "demo_left_wrist_index": demo_index,
                "real_state_left": real_obs.state.astype(np.float32).tolist(),
                "normalized_gripper_chunk": raw_gripper[: args.action_horizon].tolist(),
                "action_gripper_chunk": action_gripper[: args.action_horizon].tolist(),
                "normalized_gripper_min": float(np.min(raw_gripper)),
                "normalized_gripper_max": float(np.max(raw_gripper)),
                "action_gripper_min": float(np.min(action_gripper)),
                "action_gripper_max": float(np.max(action_gripper)),
                "infer_info": {
                    key: to_jsonable(value)
                    for key, value in data.items()
                    if key != "normalized_actions"
                },
            }
            records.append(record)
            logger.info(
                "query=%d step=%d demo_idx=%d raw[min=%.4f max=%.4f first=%s] action6=%s",
                query_idx,
                current_step,
                demo_index,
                float(np.min(raw_gripper)),
                float(np.max(raw_gripper)),
                np.round(raw_gripper[: args.action_horizon], 4),
                np.round(action_gripper[: args.action_horizon], 4),
            )

            if args.execute_actions:
                for action in actions[: args.action_horizon, :7]:
                    robot.send_action(action)
                    if args.control_hz > 0:
                        time.sleep(1.0 / args.control_hz)
    finally:
        robot.shutdown()
        demo.close()

    save_json_atomic(
        args.save_dir / "probe_results.json",
        {
            "config": config,
            "records": records,
        },
    )
    logger.info("Saved probe results: %s", args.save_dir / "probe_results.json")


if __name__ == "__main__":
    main()
