"""Build task/quality curriculum splits for a CALVIN LeRobot dataset.

This script is intentionally offline: it analyzes the converted LeRobot dataset
and writes JSON/Markdown files that can be inspected before wiring a sampler
into training.

Example:
python examples/calvin/train_files/build_calvin_curriculum.py \
  --dataset-root /mnt/data/jiangnan/lerobot/task_ABC_D_lerobot \
  --output-dir examples/calvin/train_files/curriculum_calvin_abcd
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import pandas as pd


HARD_KEYWORDS = (
    "drawer",
    "sliding",
    "slider",
    "door",
    "place",
    "put",
    "pick",
    "grasp",
    "lift",
    "rotate",
    "turn it",
)

EASY_KEYWORDS = (
    "switch",
    "button",
    "light",
    "lamp",
    "led",
)


@dataclass
class EpisodeRecord:
    episode_index: int
    task: str
    task_index: int | None
    length: int
    action_l2_mean: float | None = None
    action_delta_l2_mean: float | None = None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize_task(task: str) -> str:
    return str(task).strip().split("\n")[0]


def percentile_rank(values: dict[int, float]) -> dict[int, float]:
    if not values:
        return {}

    sorted_items = sorted(values.items(), key=lambda item: item[1])
    denom = max(len(sorted_items) - 1, 1)
    ranks: dict[int, float] = {}
    for rank, (key, _) in enumerate(sorted_items):
        ranks[key] = rank / denom
    return ranks


def find_episode_parquet(dataset_root: Path, episode_index: int) -> Path | None:
    episode_name = f"episode_{episode_index:06d}.parquet"
    matches = list((dataset_root / "data").glob(f"chunk-*/{episode_name}"))
    if matches:
        return matches[0]
    return None


def compute_action_metrics(parquet_path: Path) -> tuple[float, float]:
    df = pd.read_parquet(parquet_path, columns=["actions"])
    actions = np.stack(df["actions"].to_numpy()).astype(np.float32)
    action_l2 = np.linalg.norm(actions[:, :6], axis=1)
    if len(actions) > 1:
        action_delta_l2 = np.linalg.norm(np.diff(actions[:, :6], axis=0), axis=1)
        delta_mean = float(np.mean(action_delta_l2))
    else:
        delta_mean = 0.0
    return float(np.mean(action_l2)), delta_mean


def load_records(
    dataset_root: Path,
    scan_actions: bool,
    max_episodes: int | None,
) -> list[EpisodeRecord]:
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    tasks_path = dataset_root / "meta" / "tasks.jsonl"
    if not episodes_path.exists():
        raise FileNotFoundError(f"Missing episodes metadata: {episodes_path}")

    task_to_index: dict[str, int] = {}
    if tasks_path.exists():
        for row in read_jsonl(tasks_path):
            task_to_index[normalize_task(row["task"])] = int(row["task_index"])

    records: list[EpisodeRecord] = []
    for row in read_jsonl(episodes_path):
        task = normalize_task(row["tasks"][0] if row.get("tasks") else "")
        record = EpisodeRecord(
            episode_index=int(row["episode_index"]),
            task=task,
            task_index=task_to_index.get(task),
            length=int(row["length"]),
        )

        if scan_actions:
            parquet_path = find_episode_parquet(dataset_root, record.episode_index)
            if parquet_path is not None:
                record.action_l2_mean, record.action_delta_l2_mean = compute_action_metrics(parquet_path)

        records.append(record)
        if max_episodes is not None and len(records) >= max_episodes:
            break

    return records


def keyword_adjustment(task: str) -> float:
    text = task.lower()
    hard_hits = sum(1 for kw in HARD_KEYWORDS if kw in text)
    easy_hits = sum(1 for kw in EASY_KEYWORDS if kw in text)
    return min(0.20, 0.07 * hard_hits) - min(0.12, 0.04 * easy_hits)


def build_task_scores(records: list[EpisodeRecord]) -> dict[str, dict[str, Any]]:
    by_task: dict[str, list[EpisodeRecord]] = defaultdict(list)
    for record in records:
        by_task[record.task].append(record)

    length_by_task = {task: mean(r.length for r in rows) for task, rows in by_task.items()}
    length_rank = percentile_rank(length_by_task)

    action_values = {
        task: mean(r.action_l2_mean for r in rows if r.action_l2_mean is not None)
        for task, rows in by_task.items()
        if any(r.action_l2_mean is not None for r in rows)
    }
    delta_values = {
        task: mean(r.action_delta_l2_mean for r in rows if r.action_delta_l2_mean is not None)
        for task, rows in by_task.items()
        if any(r.action_delta_l2_mean is not None for r in rows)
    }
    action_rank = percentile_rank(action_values)
    delta_rank = percentile_rank(delta_values)

    scores: dict[str, dict[str, Any]] = {}
    for task, rows in by_task.items():
        length_component = length_rank.get(task, 0.5)
        action_component = action_rank.get(task, 0.5)
        delta_component = delta_rank.get(task, 0.5)
        score = (
            0.55 * length_component
            + 0.25 * action_component
            + 0.20 * delta_component
            + keyword_adjustment(task)
        )
        score = float(np.clip(score, 0.0, 1.0))
        scores[task] = {
            "task": task,
            "task_index": rows[0].task_index,
            "num_episodes": len(rows),
            "num_steps": int(sum(r.length for r in rows)),
            "mean_length": float(mean(r.length for r in rows)),
            "mean_action_l2": (
                float(mean(r.action_l2_mean for r in rows if r.action_l2_mean is not None))
                if any(r.action_l2_mean is not None for r in rows)
                else None
            ),
            "mean_action_delta_l2": (
                float(mean(r.action_delta_l2_mean for r in rows if r.action_delta_l2_mean is not None))
                if any(r.action_delta_l2_mean is not None for r in rows)
                else None
            ),
            "difficulty_score": score,
            "episode_indices": [r.episode_index for r in rows],
        }

    return scores


def assign_groups(task_scores: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    tasks = sorted(task_scores, key=lambda task: task_scores[task]["difficulty_score"])
    if not tasks:
        return {"easy": [], "medium": [], "hard": []}

    groups = {"easy": [], "medium": [], "hard": []}
    n = len(tasks)
    easy_cut = math.ceil(n / 3)
    hard_cut = math.ceil(2 * n / 3)
    groups["easy"] = tasks[:easy_cut]
    groups["medium"] = tasks[easy_cut:hard_cut]
    groups["hard"] = tasks[hard_cut:]
    return groups


def build_curriculum(
    records: list[EpisodeRecord],
    task_scores: dict[str, dict[str, Any]],
    groups: dict[str, list[str]],
    max_train_steps: int,
) -> dict[str, Any]:
    group_to_episodes = {
        group: sorted(
            episode
            for task in tasks
            for episode in task_scores[task]["episode_indices"]
        )
        for group, tasks in groups.items()
    }

    episode_to_group: dict[int, str] = {}
    for group, episodes in group_to_episodes.items():
        for episode in episodes:
            episode_to_group[episode] = group

    return {
        "metadata": {
            "num_episodes": len(records),
            "num_tasks": len(task_scores),
            "max_train_steps": max_train_steps,
            "difficulty_formula": (
                "0.55*length_percentile + 0.25*action_magnitude_percentile "
                "+ 0.20*action_delta_percentile + keyword_adjustment"
            ),
        },
        "stages": [
            {
                "name": "warmup_easy",
                "until_step": int(max_train_steps * 0.20),
                "group_weights": {"easy": 0.70, "medium": 0.30, "hard": 0.00},
            },
            {
                "name": "mixed",
                "until_step": int(max_train_steps * 0.60),
                "group_weights": {"easy": 0.30, "medium": 0.50, "hard": 0.20},
            },
            {
                "name": "hard_focus",
                "until_step": max_train_steps,
                "group_weights": {"easy": 0.20, "medium": 0.35, "hard": 0.45},
            },
        ],
        "groups": {
            group: {
                "tasks": tasks,
                "episode_indices": group_to_episodes[group],
                "num_tasks": len(tasks),
                "num_episodes": len(group_to_episodes[group]),
            }
            for group, tasks in groups.items()
        },
        "episode_to_group": {str(k): v for k, v in sorted(episode_to_group.items())},
        "tasks": task_scores,
    }


def write_summary(path: Path, curriculum: dict[str, Any]) -> None:
    lines = [
        "# CALVIN Curriculum Summary",
        "",
        f"- Episodes: {curriculum['metadata']['num_episodes']}",
        f"- Tasks: {curriculum['metadata']['num_tasks']}",
        "",
        "## Stages",
        "",
    ]
    for stage in curriculum["stages"]:
        lines.append(f"- `{stage['name']}` until step `{stage['until_step']}`: {stage['group_weights']}")

    lines.extend(["", "## Groups", ""])
    for group, info in curriculum["groups"].items():
        lines.append(f"### {group}")
        lines.append(f"- tasks: {info['num_tasks']}")
        lines.append(f"- episodes: {info['num_episodes']}")
        for task in info["tasks"][:20]:
            task_info = curriculum["tasks"][task]
            lines.append(
                f"- score={task_info['difficulty_score']:.3f}, "
                f"eps={task_info['num_episodes']}, len={task_info['mean_length']:.1f}: {task}"
            )
        if len(info["tasks"]) > 20:
            lines.append(f"- ... {len(info['tasks']) - 20} more tasks")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-train-steps", type=int, default=100000)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument(
        "--no-scan-actions",
        action="store_true",
        help="Use only episode length and task text. Faster, but less quality-aware.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_records(
        dataset_root=args.dataset_root,
        scan_actions=not args.no_scan_actions,
        max_episodes=args.max_episodes,
    )
    task_scores = build_task_scores(records)
    groups = assign_groups(task_scores)
    curriculum = build_curriculum(records, task_scores, groups, args.max_train_steps)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    curriculum_path = args.output_dir / "calvin_curriculum.json"
    summary_path = args.output_dir / "calvin_curriculum_summary.md"
    curriculum_path.write_text(json.dumps(curriculum, indent=2, ensure_ascii=False), encoding="utf-8")
    write_summary(summary_path, curriculum)

    print(f"Wrote curriculum: {curriculum_path}")
    print(f"Wrote summary: {summary_path}")
    for group, info in curriculum["groups"].items():
        print(f"{group}: {info['num_tasks']} tasks, {info['num_episodes']} episodes")


if __name__ == "__main__":
    main()
