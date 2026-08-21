"""
Convert a CALVIN zip dataset directly into a StarVLA-friendly LeRobot dataset.

Compared with RoboTron-Mani's original `convert_to_lerobot_zip.py`, this script:
1. Auto-detects the dataset root inside the zip (`task_ABC_D`, `task_D_D`, etc.).
2. Defaults to a StarVLA-compatible 8D state:
   [x, y, z, roll, pitch, yaw, gripper_width, gripper_action]
3. Stores camera streams as LeRobot videos, which matches StarVLA's video loader.
4. Copies `examples/calvin/train_files/modality.json` into `meta/modality.json`.

Example:
python examples/calvin/convert_to_lerobot_starvla.py \
  --zip-path /mnt/data/jiangnan/calvin/task_ABC_D.zip \
  --output-root /mnt/data/jiangnan/lerobot \
  --repo-id task_ABC_D_lerobot \
  --splits training
"""

from __future__ import annotations

import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

import numpy as np
import tyro

try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset


DEFAULT_MODALITY_JSON = Path(__file__).resolve().parent / "train_files" / "modality.json"


@dataclass(frozen=True)
class Args:
    zip_path: str
    output_root: str = "/mnt/data/jiangnan/lerobot"
    repo_id: str | None = None
    fps: int = 10
    splits: Literal["training", "validation", "both"] = "training"
    action_key: Literal["rel_actions", "actions"] = "rel_actions"
    state_format: Literal["starvla8", "raw15"] = "starvla8"
    robot_type: str = "panda"
    max_episodes: int | None = None
    overwrite: bool = False
    copy_modality_json: bool = True
    modality_json_path: str = str(DEFAULT_MODALITY_JSON)


def _load_npy_from_zip(zf: zipfile.ZipFile, member: str):
    with zf.open(member, "r") as f:
        return np.load(f, allow_pickle=True)


def _infer_dataset_root(zf: zipfile.ZipFile) -> str:
    for member in zf.namelist():
        parts = PurePosixPath(member).parts
        if len(parts) >= 3 and parts[1] in {"training", "validation"}:
            return parts[0]
    raise ValueError("Failed to infer dataset root from zip contents.")


def _load_language_annotations(
    zf: zipfile.ZipFile, dataset_root: str, split: Literal["training", "validation"]
) -> tuple[np.ndarray, np.ndarray]:
    member = f"{dataset_root}/{split}/lang_annotations/auto_lang_ann.npy"
    lang_data = _load_npy_from_zip(zf, member).item()
    return lang_data["info"]["indx"], lang_data["language"]["ann"]


def _load_step_npz(
    zf: zipfile.ZipFile,
    dataset_root: str,
    split: Literal["training", "validation"],
    step_id: int,
) -> dict[str, np.ndarray]:
    member = f"{dataset_root}/{split}/episode_{step_id:07d}.npz"
    with zf.open(member, "r") as f:
        npz = np.load(f, allow_pickle=True)
        try:
            return {k: npz[k] for k in npz.files}
        finally:
            npz.close()


def _build_state(robot_obs: np.ndarray, state_format: str) -> np.ndarray:
    robot_obs = robot_obs.astype(np.float32)
    if state_format == "raw15":
        return robot_obs

    if robot_obs.shape[0] < 15:
        raise ValueError(f"Expected robot_obs with >=15 dims, got shape {robot_obs.shape}")

    # CALVIN 15D proprioception:
    # [eef_xyz(3), eef_rpy(3), gripper_width(1), joint_pos(7), gripper_action(1)]
    return np.concatenate(
        [
            robot_obs[:6],
            robot_obs[6:7],    # mapped to StarVLA modality's `state.pad`
            robot_obs[14:15],  # mapped to `state.gripper`
        ],
        axis=0,
    ).astype(np.float32)


def _get_features(state_format: str) -> dict:
    state_dim = 8 if state_format == "starvla8" else 15
    return {
        "image": {
            "dtype": "video",
            "shape": (200, 200, 3),
            "names": ["height", "width", "channel"],
        },
        "wrist_image": {
            "dtype": "video",
            "shape": (84, 84, 3),
            "names": ["height", "width", "channel"],
        },
        "state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": ["state"],
        },
        "actions": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["actions"],
        },
    }


def _normalize_task(task) -> str:
    return str(task).strip().split("\n")[0]


def main(args: Args) -> None:
    zip_path = Path(args.zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file not found: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        dataset_root = _infer_dataset_root(zf)

        repo_id = args.repo_id or f"{dataset_root}_lerobot"
        output_root = Path(args.output_root)
        dataset_path = output_root / repo_id

        if dataset_path.exists():
            if args.overwrite:
                shutil.rmtree(dataset_path)
            else:
                raise FileExistsError(
                    f"Output dataset already exists: {dataset_path}. "
                    "Use --overwrite true to replace it."
                )

        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            root=dataset_path,
            robot_type=args.robot_type,
            fps=args.fps,
            features=_get_features(args.state_format),
        )

        if args.splits == "both":
            splits: list[Literal["training", "validation"]] = ["training", "validation"]
        else:
            splits = [args.splits]

        total_saved_episodes = 0
        for split in splits:
            episode_ranges, tasks = _load_language_annotations(zf, dataset_root, split)
            for episode_idx, (start_idx, end_idx) in enumerate(episode_ranges):
                task = _normalize_task(tasks[episode_idx])

                for step_id in range(int(start_idx), int(end_idx) + 1):
                    step = _load_step_npz(zf, dataset_root, split, step_id)
                    frame = {
                        "image": step["rgb_static"],
                        "wrist_image": step["rgb_gripper"],
                        "state": _build_state(step["robot_obs"], args.state_format),
                        "actions": step[args.action_key].astype(np.float32),
                    }
                    dataset.add_frame(frame, task=task)

                dataset.save_episode()
                total_saved_episodes += 1

                if args.max_episodes is not None and total_saved_episodes >= args.max_episodes:
                    break

            if args.max_episodes is not None and total_saved_episodes >= args.max_episodes:
                break

    if args.copy_modality_json:
        modality_src = Path(args.modality_json_path)
        if not modality_src.exists():
            raise FileNotFoundError(f"modality.json not found: {modality_src}")
        modality_dst = dataset_path / "meta" / "modality.json"
        modality_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(modality_src, modality_dst)

    print(f"Conversion finished.")
    print(f"Input zip: {zip_path}")
    print(f"Detected dataset root: {dataset_root}")
    print(f"Output dataset: {dataset_path}")
    print(f"Suggested StarVLA data_mix entry:")
    print(f'    ("{repo_id}", 1.0, "libero_franka"),')


if __name__ == "__main__":
    main(tyro.cli(Args))
