#!/usr/bin/env python3
"""Replay a fixed offline action sequence on the Piper Robolatent client."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


logger = logging.getLogger("replay_offline_actions")


def _add_import_paths() -> None:
    here = Path(__file__).resolve().parent
    candidates = [
        here,
        here.parent,
        Path("/home/agilex/Sixu/examples/new_deploy"),
        Path("/home/agilex/Sixu"),
    ]
    for path in candidates:
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


def load_actions(path: Path, sequence: str) -> tuple[np.ndarray, dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "sequences" not in data:
        raise ValueError(f"{path} does not look like an offline replay action file")
    if sequence not in data["sequences"]:
        raise ValueError(f"Unknown sequence {sequence!r}; available: {list(data['sequences'])}")
    actions = np.asarray(data["sequences"][sequence]["actions"], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(f"Expected action sequence shape (N, 7), got {actions.shape}")
    return actions, data


def parse_qpos(text: str) -> np.ndarray:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if len(values) != 14:
        raise ValueError(f"Expected 14 comma-separated qpos values, got {len(values)}")
    return np.asarray(values, dtype=np.float32)


def make_video_writer(path: Path, frame: np.ndarray, fps: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {path}")
    return writer


def save_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay offline absolute 7D left-arm actions without calling the policy server."
    )
    parser.add_argument(
        "--actions_json",
        type=Path,
        default=Path("/data/share-folder/linsixu_deploy/analysis_outputs/hdf5_offline_replay_actions.json"),
    )
    parser.add_argument(
        "--sequence",
        choices=("policy_chunk_stride8", "policy_first_step", "hdf5_gt"),
        default="policy_chunk_stride8",
    )
    parser.add_argument("--save_dir", type=Path, default=Path("runs/offline_action_replay"))
    parser.add_argument("--control_hz", type=float, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--execute", action="store_true", help="Actually send actions to the robot.")
    parser.add_argument("--no_reset", action="store_true")
    parser.add_argument("--reset_qpos", type=str, default=None)
    parser.add_argument("--verify_reset_tolerance", type=float, default=0.05)
    parser.add_argument("--allow_reset_mismatch", action="store_true")
    parser.add_argument("--post_reset_wait", type=float, default=1.0)
    parser.add_argument("--image_timeout", type=float, default=10.0)
    parser.add_argument("--action_interp_num", type=int, default=0)
    parser.add_argument("--action_wait", action="store_true")
    parser.add_argument("--clip_gripper", action="store_true")
    parser.add_argument("--video_fps", type=float, default=15.0)
    parser.add_argument("--log_every_n_steps", type=int, default=10)
    parser.add_argument("--cam_head_name", type=str, default="head_camera")
    parser.add_argument("--cam_high_name", type=str, default="agent_camera")
    parser.add_argument("--cam_left_wrist_name", type=str, default="left_wrist_camera")
    parser.add_argument("--cam_right_wrist_name", type=str, default="right_wrist_camera")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    actions, metadata = load_actions(args.actions_json, args.sequence)
    control_hz = float(args.control_hz or metadata.get("control_hz", 15.0))
    end = len(actions) if args.num_steps is None else min(len(actions), args.start + args.num_steps)
    if not (0 <= args.start < end <= len(actions)):
        raise ValueError(f"Bad replay range start={args.start}, end={end}, len={len(actions)}")
    actions = actions[args.start:end]

    reset_qpos = None
    if args.reset_qpos is not None:
        reset_qpos = parse_qpos(args.reset_qpos)
    elif metadata.get("reset_qpos") is not None:
        reset_qpos = np.asarray(metadata["reset_qpos"], dtype=np.float32)

    args.save_dir = args.save_dir.expanduser().resolve()
    args.save_dir.mkdir(parents=True, exist_ok=True)

    run_metadata = {
        "actions_json": str(args.actions_json),
        "sequence": args.sequence,
        "start": args.start,
        "end": end,
        "num_steps": int(len(actions)),
        "execute": bool(args.execute),
        "control_hz": control_hz,
        "reset_qpos": to_jsonable(reset_qpos),
        "clip_gripper": bool(args.clip_gripper),
        "action_interp_num": args.action_interp_num,
        "action_wait": bool(args.action_wait),
        "source_summary": metadata.get("sequences", {}).get(args.sequence, {}).get("summary", {}),
    }
    save_json(args.save_dir / "metrics.json", run_metadata)

    logger.info("Loaded %d actions from %s sequence=%s", len(actions), args.actions_json, args.sequence)
    logger.info(
        "Action stats: min=%s median=%s max=%s",
        np.round(np.min(actions, axis=0), 4),
        np.round(np.median(actions, axis=0), 4),
        np.round(np.max(actions, axis=0), 4),
    )
    logger.info("Gripper open count (>0.05): %d/%d", int(np.sum(actions[:, 6] > 0.05)), len(actions))

    if not args.execute:
        logger.info("Dry run only. Add --execute to send actions to the robot.")
        return

    _add_import_paths()
    from piper_env import PiperRobolatentRealRobotInterface

    robot = PiperRobolatentRealRobotInterface(
        reset_position=None if args.no_reset else reset_qpos,
        image_timeout=args.image_timeout,
        action_interp_num=args.action_interp_num,
        action_wait=args.action_wait,
        clip_gripper=args.clip_gripper,
        cam_head_name=args.cam_head_name,
        cam_high_name=args.cam_high_name,
        cam_left_wrist_name=args.cam_left_wrist_name,
        cam_right_wrist_name=args.cam_right_wrist_name,
    )

    records: list[dict[str, Any]] = []
    writer = None
    video_path = args.save_dir / "videos" / f"{args.sequence}_cam_high.mp4"

    try:
        if not args.no_reset:
            robot.reset()
            if args.post_reset_wait > 0:
                time.sleep(args.post_reset_wait)
            obs = robot.get_observation()
            reset_err = None
            if reset_qpos is not None:
                reset_err = np.asarray(obs.joint_state, dtype=np.float32) - reset_qpos
                reset_absmax = float(np.max(np.abs(reset_err)))
                logger.info("Reset verify absmax=%.6f state=%s", reset_absmax, np.round(obs.joint_state, 4))
                if reset_absmax > args.verify_reset_tolerance and not args.allow_reset_mismatch:
                    raise RuntimeError(
                        f"Reset mismatch absmax={reset_absmax:.6f} > "
                        f"{args.verify_reset_tolerance}; use --allow_reset_mismatch to continue."
                    )
        else:
            obs = robot.get_observation()

        writer = make_video_writer(video_path, obs.cam_high, args.video_fps)
        writer.write(obs.cam_high)

        for i, action in enumerate(actions):
            step = args.start + i
            t0 = time.perf_counter()
            pre_obs = robot.get_observation()
            robot.send_action(action)
            send_done = time.perf_counter()
            if control_hz > 0:
                time.sleep(max(0.0, 1.0 / control_hz - (time.perf_counter() - t0)))
            post_obs = robot.get_observation()
            if writer is not None:
                writer.write(post_obs.cam_high)

            record = {
                "step": int(step),
                "timestamp": time.time(),
                "action_left": action.tolist(),
                "pre_action_state_left": pre_obs.joint_state[:7].tolist(),
                "state_left": post_obs.joint_state[:7].tolist(),
                "joint_state_14d": post_obs.joint_state.tolist(),
                "action_minus_pre_state": (action - pre_obs.joint_state[:7]).tolist(),
                "timing": {
                    "send_sec": send_done - t0,
                    "total_sec": time.perf_counter() - t0,
                },
            }
            records.append(record)

            if args.log_every_n_steps > 0 and (i % args.log_every_n_steps == 0 or i == len(actions) - 1):
                logger.info(
                    "step=%d/%d action6=%.4f pre6=%.4f state6=%.4f max_delta=%.4f",
                    i,
                    len(actions),
                    float(action[6]),
                    float(pre_obs.joint_state[6]),
                    float(post_obs.joint_state[6]),
                    float(np.max(np.abs(action - pre_obs.joint_state[:7]))),
                )
    finally:
        if writer is not None:
            writer.release()
        robot.shutdown()
        save_json(
            args.save_dir / "logs" / f"{args.sequence}.json",
            {
                "metadata": {
                    **run_metadata,
                    "video_path": str(video_path),
                    "completed_steps": len(records),
                },
                "steps": records,
            },
        )

    logger.info("Replay complete. Logs: %s", args.save_dir)


if __name__ == "__main__":
    main()
