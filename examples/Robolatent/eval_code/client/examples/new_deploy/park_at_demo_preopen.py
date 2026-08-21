#!/usr/bin/env python3
"""Move Piper to a demo frame just before the left gripper opens."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np


logger = logging.getLogger("park_at_demo_preopen")

CAMERA_NAMES = ("cam_head", "cam_high", "cam_left_wrist", "cam_right_wrist")
HDF5_IMAGE_PATHS = {
    "cam_head": "observations/images/cam_head",
    "cam_high": "observations/images/cam_high",
    "cam_left_wrist": "observations/images/cam_left_wrist",
    "cam_right_wrist": "observations/images/cam_right_wrist",
}


def add_import_paths() -> None:
    here = Path(__file__).resolve().parent
    repo_root = here.parents[1]
    for path in (repo_root, here):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def save_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = np.asarray(image)
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    if not cv2.imwrite(str(path), frame):
        raise RuntimeError(f"Failed to write image: {path}")


def resize_like(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if image.shape[:2] == reference.shape[:2]:
        return image
    h, w = reference.shape[:2]
    return cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)


def image_stats(image: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(image)
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": int(np.min(arr)),
        "max": int(np.max(arr)),
    }


def load_demo(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    import h5py

    with h5py.File(path, "r") as f:
        qpos = np.asarray(f["observations/qpos"], dtype=np.float32)
        action = np.asarray(f["action"], dtype=np.float32)
        images = {
            name: np.asarray(f[hdf5_path], dtype=np.uint8)
            for name, hdf5_path in HDF5_IMAGE_PATHS.items()
            if hdf5_path in f
        }
    if qpos.ndim != 2 or qpos.shape[1] < 14:
        raise ValueError(f"Expected observations/qpos shape (T, >=14), got {qpos.shape}")
    if action.ndim != 2 or action.shape[1] < 14:
        raise ValueError(f"Expected action shape (T, >=14), got {action.shape}")
    return qpos[:, :14], action[:, :14], images


def find_preopen_index(
    qpos: np.ndarray,
    action: np.ndarray,
    *,
    source: str,
    threshold: float,
    before_open_offset: int,
    target_index: Optional[int],
) -> tuple[int, Optional[int], float]:
    if target_index is not None:
        if target_index < 0 or target_index >= len(qpos):
            raise ValueError(f"--target_index out of range: {target_index}, len={len(qpos)}")
        return int(target_index), None, float(qpos[target_index, 6])

    signal = qpos[:, 6] if source == "qpos" else action[:, 6]
    open_indices = np.flatnonzero(signal > threshold)
    if open_indices.size == 0:
        raise ValueError(
            f"No gripper open frame found from {source} with threshold={threshold}. "
            f"Signal min={float(np.min(signal)):.6f}, max={float(np.max(signal)):.6f}"
        )
    first_open_index = int(open_indices[0])
    index = max(0, first_open_index - int(before_open_offset))
    return index, first_open_index, float(signal[index])


def save_snapshot(
    save_dir: Path,
    *,
    snapshot_name: str,
    demo_images: dict[str, np.ndarray],
    target_index: int,
    obs: Any,
) -> Path:
    snapshot_dir = save_dir / "snapshots" / snapshot_name
    real_images = {
        "cam_head": obs.cam_head,
        "cam_high": obs.cam_high,
        "cam_left_wrist": obs.cam_left_wrist,
        "cam_right_wrist": obs.cam_right_wrist,
    }

    metadata: dict[str, Any] = {
        "target_index": target_index,
        "timestamp": time.time(),
        "joint_state_14d": np.asarray(obs.joint_state, dtype=np.float32).tolist(),
        "views": {},
    }
    for name in CAMERA_NAMES:
        real = np.asarray(real_images[name])
        demo = np.asarray(demo_images[name][target_index])
        demo = resize_like(demo, real)
        side_by_side = np.concatenate([real, demo], axis=1)
        diff = np.abs(real.astype(np.int16) - demo.astype(np.int16)).astype(np.uint8)

        view_dir = snapshot_dir / name
        save_image(view_dir / "real.png", real)
        save_image(view_dir / "demo.png", demo)
        save_image(view_dir / "side_by_side_real_demo.png", side_by_side)
        save_image(view_dir / "absdiff.png", diff)

        metadata["views"][name] = {
            "real": image_stats(real),
            "demo": image_stats(demo),
            "mean_abs_diff": float(np.mean(diff)),
            "max_abs_diff": int(np.max(diff)),
        }

    save_json(snapshot_dir / "metadata.json", metadata)
    return snapshot_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Move the real robot to the demo qpos just before the left gripper opens, "
            "then save real/demo camera comparisons for object alignment."
        )
    )
    parser.add_argument(
        "--hdf5",
        type=Path,
        default=Path("/home/agilex/Sixu/data/origin_data/episode_0.hdf5"),
    )
    parser.add_argument("--save_dir", type=Path, default=Path("runs/park_demo_preopen"))
    parser.add_argument("--target_index", type=int, default=None)
    parser.add_argument("--source", choices=("qpos", "action"), default="action")
    parser.add_argument("--gripper_open_threshold", type=float, default=0.005)
    parser.add_argument("--before_open_offset", type=int, default=1)
    parser.add_argument("--execute", action="store_true", help="Actually move the robot.")
    parser.add_argument("--image_timeout", type=float, default=10.0)
    parser.add_argument("--verify_target", action="store_true")
    parser.add_argument("--target_tolerance", type=float, default=0.03)
    parser.add_argument("--target_verify_timeout", type=float, default=8.0)
    parser.add_argument("--target_settle_time", type=float, default=0.5)
    parser.add_argument("--action_interp_num", type=int, default=50)
    parser.add_argument("--hold", action="store_true", help="Keep saving snapshots until Ctrl+C.")
    parser.add_argument("--snapshot_interval", type=float, default=2.0)
    parser.add_argument("--cam_head_name", type=str, default="head_camera")
    parser.add_argument("--cam_high_name", type=str, default="agent_camera")
    parser.add_argument("--cam_left_wrist_name", type=str, default="left_wrist_camera")
    parser.add_argument("--cam_right_wrist_name", type=str, default="right_wrist_camera")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    args.hdf5 = args.hdf5.expanduser().resolve()
    args.save_dir = args.save_dir.expanduser().resolve()
    args.save_dir.mkdir(parents=True, exist_ok=True)

    if args.before_open_offset < 0:
        raise ValueError("--before_open_offset must be non-negative")
    if args.snapshot_interval <= 0:
        raise ValueError("--snapshot_interval must be positive")

    qpos, action, demo_images = load_demo(args.hdf5)
    missing_images = [name for name in CAMERA_NAMES if name not in demo_images]
    if missing_images:
        raise RuntimeError(f"HDF5 is missing demo images: {missing_images}")

    target_index, first_open_index, target_gripper_signal = find_preopen_index(
        qpos,
        action,
        source=args.source,
        threshold=args.gripper_open_threshold,
        before_open_offset=args.before_open_offset,
        target_index=args.target_index,
    )
    target_qpos = qpos[target_index].astype(np.float32)

    metadata = {
        "hdf5": str(args.hdf5),
        "target_index": target_index,
        "first_open_index": first_open_index,
        "source": args.source,
        "gripper_open_threshold": args.gripper_open_threshold,
        "before_open_offset": args.before_open_offset,
        "target_gripper_signal": target_gripper_signal,
        "target_qpos": target_qpos.tolist(),
        "target_action": action[target_index].astype(np.float32).tolist(),
        "execute": bool(args.execute),
    }
    save_json(args.save_dir / "metadata.json", metadata)

    logger.info(
        "Selected target_index=%d first_open_index=%s source=%s threshold=%.4f target_qpos6=%.4f target_action6=%.4f",
        target_index,
        first_open_index,
        args.source,
        args.gripper_open_threshold,
        float(target_qpos[6]),
        float(action[target_index, 6]),
    )
    logger.info("Target qpos: %s", np.round(target_qpos, 6))

    if not args.execute:
        logger.info("Dry run only. Add --execute to move the robot.")
        return

    add_import_paths()
    from examples.new_deploy.piper_env import PiperRobolatentRealRobotInterface

    robot = PiperRobolatentRealRobotInterface(
        reset_position=target_qpos,
        image_timeout=args.image_timeout,
        action_interp_num=args.action_interp_num,
        action_wait=True,
        clip_gripper=False,
        verify_reset=args.verify_target,
        reset_tolerance=args.target_tolerance,
        reset_settle_time=args.target_settle_time,
        reset_verify_timeout=args.target_verify_timeout,
        cam_head_name=args.cam_head_name,
        cam_high_name=args.cam_high_name,
        cam_left_wrist_name=args.cam_left_wrist_name,
        cam_right_wrist_name=args.cam_right_wrist_name,
    )

    try:
        robot.reset()
        obs = robot.get_observation()
        actual = np.asarray(obs.joint_state, dtype=np.float32)
        diff = actual - target_qpos
        logger.info(
            "Parked at target. max_abs_diff=%.6f left=%.6f right=%.6f actual=%s",
            float(np.max(np.abs(diff))),
            float(np.max(np.abs(diff[:7]))),
            float(np.max(np.abs(diff[7:]))),
            np.round(actual, 6),
        )
        snapshot_dir = save_snapshot(
            args.save_dir,
            snapshot_name=f"target_{target_index:06d}",
            demo_images=demo_images,
            target_index=target_index,
            obs=obs,
        )
        logger.info("Saved alignment snapshot: %s", snapshot_dir)

        if args.hold:
            logger.info("Holding. Adjust the object, then inspect refreshed snapshots. Press Ctrl+C to stop.")
            snapshot_i = 0
            while True:
                time.sleep(args.snapshot_interval)
                obs = robot.get_observation()
                snapshot_i += 1
                snapshot_dir = save_snapshot(
                    args.save_dir,
                    snapshot_name=f"hold_{snapshot_i:04d}_target_{target_index:06d}",
                    demo_images=demo_images,
                    target_index=target_index,
                    obs=obs,
                )
                logger.info("Saved alignment snapshot: %s", snapshot_dir)
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        robot.shutdown()


if __name__ == "__main__":
    main()
