#!/usr/bin/env python3
"""Probe policy sensitivity to perturbed demo cam_left_wrist images.

This script does not touch ROS or the real robot. It sends HDF5 demo observations
directly to the StarVLA websocket server and reports the predicted gripper values.
"""

from __future__ import annotations

import argparse
import functools
import json
import logging
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import msgpack
import numpy as np
import websockets.sync.client


logger = logging.getLogger("probe_demo_wrist_perturb")

RYNNBRAIN_VIEW_NAMES = ("cam_head", "cam_high", "cam_left_wrist", "cam_right_wrist")
LEFT_WRIST_VIEW_INDEX = RYNNBRAIN_VIEW_NAMES.index("cam_left_wrist")

TASK_PROMPTS: dict[str, str] = {
    "PickXtimes": "pick up the blue block and place the blue block in the plate repeatedly",
    "PutBackBlock": "pick up the blue block, place it in the plate, push the button, then put the block back on the table",
    "UncoverBlock": "cover the blocks with lids then uncover them",
}


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


class StarVLAWebsocketClientPolicy:
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

    def reset(self) -> dict[str, Any]:
        return self._request({"type": "reset"})

    def infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request({"type": "infer", "payload": payload})

    def get_server_metadata(self) -> dict[str, Any]:
        return self._server_metadata

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._ws.send(self._packer.pack(payload))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    def payload_bytes(self, payload: dict[str, Any]) -> int:
        return len(self._packer.pack(payload))


def ensure_uint8_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected HWC image, got shape {image.shape}")
    if image.dtype == np.uint8:
        return image
    return np.clip(image, 0, 255).astype(np.uint8)


def resize_image(image: np.ndarray, size: int) -> np.ndarray:
    image = ensure_uint8_image(image)
    if image.shape[0] == size and image.shape[1] == size:
        return image
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)


def black_views(image_size: int) -> list[np.ndarray]:
    return [
        np.zeros((image_size, image_size, 3), dtype=np.uint8)
        for _ in RYNNBRAIN_VIEW_NAMES
    ]


class DemoEpisode:
    def __init__(self, path: Path, image_size: int) -> None:
        import h5py

        self.path = path
        self.image_size = int(image_size)
        self.file = h5py.File(path, "r")
        self.images = {
            name: self.file[f"observations/images/{name}"]
            for name in RYNNBRAIN_VIEW_NAMES
        }
        self.length = int(self.images[RYNNBRAIN_VIEW_NAMES[0]].shape[0])

    def close(self) -> None:
        self.file.close()

    def views(self, step: int) -> list[np.ndarray]:
        if step < 0:
            return black_views(self.image_size)
        if step >= self.length:
            raise IndexError(f"step={step} out of range for {self.path} length={self.length}")
        return [
            resize_image(np.asarray(self.images[name][step]), self.image_size)
            for name in RYNNBRAIN_VIEW_NAMES
        ]


def parse_steps(text: str) -> list[int]:
    steps: list[int] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            parts = [int(x) for x in item.split(":")]
            if len(parts) not in (2, 3):
                raise argparse.ArgumentTypeError(f"Bad step range: {item}")
            start, end = parts[:2]
            stride = parts[2] if len(parts) == 3 else 1
            if stride <= 0:
                raise argparse.ArgumentTypeError(f"Bad step stride: {item}")
            steps.extend(range(start, end + 1, stride))
        else:
            steps.append(int(item))
    if not steps:
        raise argparse.ArgumentTypeError("--steps must not be empty")
    return sorted(dict.fromkeys(steps))


def apply_wrist_perturbation(
    image: np.ndarray,
    *,
    step: int,
    noise_std: float,
    brightness: float,
    contrast: float,
    shift_x: float,
    shift_y: float,
    scale: float,
    rotate_deg: float,
    seed: int,
) -> np.ndarray:
    out = ensure_uint8_image(image).astype(np.float32)

    if scale != 1.0 or rotate_deg != 0.0 or shift_x != 0.0 or shift_y != 0.0:
        h, w = out.shape[:2]
        center = ((w - 1) / 2.0, (h - 1) / 2.0)
        matrix = cv2.getRotationMatrix2D(center, rotate_deg, scale)
        matrix[0, 2] += shift_x
        matrix[1, 2] += shift_y
        out = cv2.warpAffine(
            out,
            matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

    if contrast != 1.0 or brightness != 0.0:
        out = out * float(contrast) + float(brightness)

    if noise_std > 0:
        rng = np.random.default_rng(seed + int(step) * 1009)
        out = out + rng.normal(0.0, noise_std, size=out.shape).astype(np.float32)

    return np.clip(out, 0, 255).astype(np.uint8)


def make_payload(
    current_views: list[np.ndarray],
    memory_views: list[list[np.ndarray]],
    *,
    prompt: str,
    step: int,
) -> dict[str, Any]:
    return {
        "examples": [
            {
                "image": [view.astype(np.uint8) for view in current_views],
                "lang": prompt,
                "memory": [
                    [view.astype(np.uint8) for view in views]
                    for views in memory_views
                ],
                "step": int(step),
            }
        ]
    }


def make_memory_views(
    demo: DemoEpisode,
    *,
    current_step: int,
    max_memory_frames: int,
    memory_interval: int,
    perturb_memory: bool,
    args: argparse.Namespace,
) -> list[list[np.ndarray]]:
    memory: list[list[np.ndarray]] = []
    for i in range(1, max_memory_frames + 1):
        target_step = current_step - i * memory_interval
        views = demo.views(target_step)
        if perturb_memory and target_step >= 0:
            views[LEFT_WRIST_VIEW_INDEX] = apply_wrist_perturbation(
                views[LEFT_WRIST_VIEW_INDEX],
                step=target_step,
                noise_std=args.noise_std,
                brightness=args.brightness,
                contrast=args.contrast,
                shift_x=args.shift_x,
                shift_y=args.shift_y,
                scale=args.scale,
                rotate_deg=args.rotate_deg,
                seed=args.seed,
            )
        memory.append(views)
    return memory


def save_debug_images(
    save_dir: Path,
    *,
    step: int,
    clean_views: list[np.ndarray],
    perturbed_views: list[np.ndarray],
) -> None:
    step_dir = save_dir / "images" / f"step_{step:06d}"
    step_dir.mkdir(parents=True, exist_ok=True)
    for idx, name in enumerate(RYNNBRAIN_VIEW_NAMES):
        cv2.imwrite(str(step_dir / f"{idx}_{name}.png"), perturbed_views[idx])
    clean = clean_views[LEFT_WRIST_VIEW_INDEX]
    perturbed = perturbed_views[LEFT_WRIST_VIEW_INDEX]
    cv2.imwrite(str(step_dir / "clean_cam_left_wrist.png"), clean)
    cv2.imwrite(str(step_dir / "perturbed_cam_left_wrist.png"), perturbed)
    cv2.imwrite(
        str(step_dir / "left_wrist_clean_vs_perturbed.png"),
        np.concatenate([clean, perturbed], axis=1),
    )
    diff = np.abs(clean.astype(np.int16) - perturbed.astype(np.int16)).astype(np.uint8)
    cv2.imwrite(str(step_dir / "left_wrist_absdiff.png"), diff)


def save_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


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


def run_probe(args: argparse.Namespace) -> None:
    prompt = args.prompt or TASK_PROMPTS[args.task]
    demo = DemoEpisode(args.hdf5, args.image_size)
    client = StarVLAWebsocketClientPolicy(args.host, args.port)
    logger.info("Connected policy server metadata: %s", client.get_server_metadata())
    if not args.no_server_reset:
        reset_resp = client.reset()
        logger.info("Policy memory reset: %s", reset_resp)

    args.save_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    try:
        for step in args.steps:
            clean_views = demo.views(step)
            perturbed_views = [view.copy() for view in clean_views]
            perturbed_views[LEFT_WRIST_VIEW_INDEX] = apply_wrist_perturbation(
                perturbed_views[LEFT_WRIST_VIEW_INDEX],
                step=step,
                noise_std=args.noise_std,
                brightness=args.brightness,
                contrast=args.contrast,
                shift_x=args.shift_x,
                shift_y=args.shift_y,
                scale=args.scale,
                rotate_deg=args.rotate_deg,
                seed=args.seed,
            )
            memory = make_memory_views(
                demo,
                current_step=step,
                max_memory_frames=args.max_memory_frames,
                memory_interval=args.memory_interval,
                perturb_memory=args.perturb_memory,
                args=args,
            )
            payload = make_payload(perturbed_views, memory, prompt=prompt, step=step)
            payload_bytes = client.payload_bytes(payload)

            if args.save_images:
                save_debug_images(
                    args.save_dir,
                    step=step,
                    clean_views=clean_views,
                    perturbed_views=perturbed_views,
                )
            if args.save_payloads:
                payload_dir = args.save_dir / "payloads" / f"step_{step:06d}"
                payload_dir.mkdir(parents=True, exist_ok=True)
                with open(payload_dir / "payload.pkl", "wb") as f:
                    pickle.dump(payload, f)

            t0 = time.perf_counter()
            outputs = client.infer(payload)
            rtt = time.perf_counter() - t0
            if not outputs.get("ok", True):
                raise RuntimeError(f"Policy server error: {outputs}")
            data = outputs.get("data", outputs)
            if "normalized_actions" not in data:
                raise RuntimeError(f"Policy response missing normalized_actions: {outputs}")
            actions = np.asarray(data["normalized_actions"], dtype=np.float32)
            if actions.ndim == 3:
                actions = actions[0]
            gripper = actions[:, 6]
            open_count = int(np.sum(gripper[: args.action_horizon] >= args.open_threshold))
            record = {
                "step": step,
                "payload_bytes": payload_bytes,
                "rtt_sec": rtt,
                "normalized_actions_shape": list(actions.shape),
                "gripper_min": float(np.min(gripper)),
                "gripper_max": float(np.max(gripper)),
                "gripper_chunk": gripper[: args.action_horizon].tolist(),
                "open_threshold": args.open_threshold,
                "open_count_in_horizon": open_count,
                "outputs_extra": {
                    key: to_jsonable(value)
                    for key, value in data.items()
                    if key != "normalized_actions"
                },
            }
            records.append(record)
            logger.info(
                "step=%d gripper[min=%.4f max=%.4f first=%s] open_count=%d/%d rtt=%.3fs",
                step,
                record["gripper_min"],
                record["gripper_max"],
                np.round(gripper[: args.action_horizon], 4),
                open_count,
                args.action_horizon,
                rtt,
            )
    finally:
        demo.close()

    metadata = {
        "hdf5": str(args.hdf5),
        "host": args.host,
        "port": args.port,
        "task": args.task,
        "prompt": prompt,
        "steps": args.steps,
        "image_size": args.image_size,
        "max_memory_frames": args.max_memory_frames,
        "memory_interval": args.memory_interval,
        "perturb_memory": args.perturb_memory,
        "noise_std": args.noise_std,
        "brightness": args.brightness,
        "contrast": args.contrast,
        "shift_x": args.shift_x,
        "shift_y": args.shift_y,
        "scale": args.scale,
        "rotate_deg": args.rotate_deg,
        "seed": args.seed,
        "open_threshold": args.open_threshold,
    }
    save_json(args.save_dir / "results.json", {"metadata": metadata, "records": records})
    logger.info("Saved results: %s", args.save_dir / "results.json")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send perturbed demo cam_left_wrist images to the policy server without moving the robot."
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5678)
    parser.add_argument("--hdf5", type=Path, default=Path("/home/agilex/Sixu/data/origin_data/episode_0.hdf5"))
    parser.add_argument("--task", type=str, default="PickXtimes", choices=tuple(TASK_PROMPTS))
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--steps", type=parse_steps, default=parse_steps("112,120,168,176,248"))
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--max_memory_frames", type=int, default=5)
    parser.add_argument("--memory_interval", type=int, default=10)
    parser.add_argument("--action_horizon", type=int, default=8)
    parser.add_argument("--open_threshold", type=float, default=0.5)
    parser.add_argument("--no_server_reset", action="store_true")

    parser.add_argument("--noise_std", type=float, default=0.0)
    parser.add_argument("--brightness", type=float, default=0.0)
    parser.add_argument("--contrast", type=float, default=1.0)
    parser.add_argument("--shift_x", type=float, default=0.0)
    parser.add_argument("--shift_y", type=float, default=0.0)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--rotate_deg", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    memory_group = parser.add_mutually_exclusive_group()
    memory_group.add_argument("--perturb_memory", dest="perturb_memory", action="store_true")
    memory_group.add_argument("--no_perturb_memory", dest="perturb_memory", action="store_false")
    parser.set_defaults(perturb_memory=True)

    parser.add_argument("--save_dir", type=Path, default=Path("runs/probe_demo_wrist_perturb"))
    image_group = parser.add_mutually_exclusive_group()
    image_group.add_argument("--save_images", dest="save_images", action="store_true")
    image_group.add_argument("--no_save_images", dest="save_images", action="store_false")
    parser.set_defaults(save_images=True)
    parser.add_argument("--save_payloads", action="store_true")
    args = parser.parse_args(argv)
    args.hdf5 = args.hdf5.expanduser().resolve()
    args.save_dir = args.save_dir.expanduser().resolve()
    if args.noise_std < 0:
        raise ValueError("--noise_std must be non-negative")
    if args.scale <= 0:
        raise ValueError("--scale must be positive")
    if args.max_memory_frames < 0:
        raise ValueError("--max_memory_frames must be non-negative")
    if args.memory_interval <= 0:
        raise ValueError("--memory_interval must be positive")
    return args


def main(argv: Optional[list[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    run_probe(parse_args(argv))


if __name__ == "__main__":
    main()
