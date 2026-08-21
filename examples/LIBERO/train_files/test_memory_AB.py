"""
Step A + B probe for memory dataflow (CPU-only, no model weights loaded).

A) Verify dataloader outputs `memory` / `step` with expected shapes.
B) Verify build_rynnbrain_inputs_with_memorys-like concatenation produces
   the expected `pixel_values_mem.shape`, `memorys_length`, `steps`.

Run:
  python examples/LIBERO/train_files/test_memory_AB.py \
      --config_yaml examples/LIBERO/train_files/starvla_cotrain_libero.yaml
"""

import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from transformers import AutoProcessor

from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn


def describe_sample(sample: dict, idx: int):
    print(f"\n--- sample[{idx}] keys: {list(sample.keys())} ---")
    img = sample["image"]
    print(f"  image: list(len={len(img)}), first={type(img[0]).__name__} size={getattr(img[0], 'size', None)}")
    print(f"  lang:  {sample['lang'][:80]!r}")
    print(f"  action shape: {sample['action'].shape}, dtype={sample['action'].dtype}")
    if "memory" in sample:
        mem = sample["memory"]
        print(f"  memory: outer_len(max_step)={len(mem)}, inner_len(num_views)={len(mem[0])}")
        print(f"          frame[0][0] type={type(mem[0][0]).__name__} size={getattr(mem[0][0], 'size', None)}")
        print(f"          frame[-1][0] type={type(mem[-1][0]).__name__} size={getattr(mem[-1][0], 'size', None)}")
        print(f"  step: {sample['step']}")
    else:
        print("  [!!] sample['memory'] NOT FOUND — dataloader memory path did not trigger.")


def step_B_concat(memorys, processor):
    """Mirror build_rynnbrain_inputs_with_memorys exactly (without the chat template / model side)."""
    all_memory_images = []
    for mem in memorys:
        for frame in mem:
            all_memory_images.extend(frame)
    processed = processor.image_processor(images=all_memory_images, return_tensors="pt")
    pixel_values_mem = processed["pixel_values"]
    return {
        "pixel_values_mem_shape": tuple(pixel_values_mem.shape),
        "memorys_length": len(memorys[0]),
        "total_memory_images_flat": len(all_memory_images),
        "extra_keys": {k: tuple(v.shape) if torch.is_tensor(v) else type(v).__name__
                       for k, v in processed.items() if k != "pixel_values"},
        "processed": processed,
    }


def step_B_grid_check(batch, processor, mem_result):
    """
    Extra consistency checks:
      1. Main image processing — does it return the same per-image grid as memory?
         (If yes, the downstream model can safely assume a fixed patch count per image.)
      2. Print memory image_grid_thw[:3] so we can confirm every memory frame is
         really (t=1, h=16, w=16) → 256 patches.
      3. Cross-check: pixel_values_mem.shape[0] == sum of patches implied by image_grid_thw.
    """
    print("\n  --- grid consistency checks ---")

    # (2) Memory image grid
    mem_proc = mem_result["processed"]
    mem_grid = mem_proc.get("image_grid_thw", None)
    if mem_grid is None:
        print("  [!!] processor did not return image_grid_thw for memory — unexpected for Qwen3-VL")
        return
    print(f"  memory image_grid_thw[:3]   = {mem_grid[:3].tolist()}")
    unique_mem_grids = torch.unique(mem_grid, dim=0).tolist()
    print(f"  memory unique grids         = {unique_mem_grids}")

    # (3) Cross-check total patches
    patches_per_img = (mem_grid[:, 0] * mem_grid[:, 1] * mem_grid[:, 2]).tolist()
    total_patches = sum(patches_per_img)
    print(f"  memory implied total patches = {total_patches} (pixel_values_mem.shape[0]={mem_result['pixel_values_mem_shape'][0]})")
    if total_patches != mem_result["pixel_values_mem_shape"][0]:
        print("  [!!] mismatch — grid_thw does NOT account for all patches")
    else:
        print("  OK: grid_thw accounts for all patches")

    # (1) Main image grid — run processor on batch's main images
    main_imgs = []
    for ex in batch:
        main_imgs.extend(ex["image"])
    main_proc = processor.image_processor(images=main_imgs, return_tensors="pt")
    main_grid = main_proc.get("image_grid_thw", None)
    if main_grid is None:
        print("  [!!] main image has no image_grid_thw either")
        return
    print(f"  main   image_grid_thw[:3]   = {main_grid[:3].tolist()}")
    unique_main_grids = torch.unique(main_grid, dim=0).tolist()
    print(f"  main   unique grids         = {unique_main_grids}")

    if unique_main_grids == unique_mem_grids:
        print("  OK: main and memory images share identical grid → downstream can assume fixed patch count")
    else:
        print("  [!!] main and memory grids DIFFER → model must use image_grid_thw to split memory tokens,"
              " but build_rynnbrain_inputs_with_memorys DROPS it. Likely bug.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str,
                        default="./examples/LIBERO/train_files/starvla_cotrain_libero.yaml")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_batches", type=int, default=1)
    parser.add_argument("--skip_B", action="store_true", help="Only run dataloader check (Step A).")
    parser.add_argument("--base_vlm", type=str, default="playground/Pretrained_models/RynnBrain-CoP-8B",
                        help="Local path to VLM (overrides yaml's framework.qwenvl.base_vlm for Step B).")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config_yaml)
    vla_cfg = cfg.datasets.vla_data
    print("========== CONFIG (memory-related) ==========")
    print(f"  vla_data.memory   = {vla_cfg.get('memory', False)}")
    print(f"  vla_data.max_step = {vla_cfg.get('max_step', None)}")
    print(f"  vla_data.interval = {vla_cfg.get('interval', None)}")
    print(f"  framework.qwenvl.memory          = {cfg.framework.qwenvl.get('memory', False)}")
    print(f"  framework.qwenvl.max_memory_step = {cfg.framework.qwenvl.get('max_memory_step', None)}")
    print(f"  framework.qwenvl.base_vlm        = {cfg.framework.qwenvl.base_vlm}")

    # ------------ Step A: dataloader ------------
    print("\n========== STEP A: dataloader output ==========")
    dataset = get_vla_dataset(data_cfg=vla_cfg)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0, collate_fn=collate_fn)

    batches = []
    for bi, batch in enumerate(loader):
        print(f"\n[batch {bi}] size={len(batch)}")
        for i, s in enumerate(batch):
            describe_sample(s, i)
        batches.append(batch)
        if bi + 1 >= args.num_batches:
            break

    if args.skip_B:
        return

    # ------------ Step B: processor concat ------------
    print("\n========== STEP B: build_rynnbrain_inputs_with_memorys concat ==========")
    model_id = args.base_vlm if args.base_vlm else cfg.framework.qwenvl.base_vlm
    print(f"Loading processor from: {model_id}")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    for bi, batch in enumerate(batches):
        if "memory" not in batch[0]:
            print(f"[batch {bi}] no 'memory' key — skip Step B")
            continue
        memorys = [ex["memory"] for ex in batch]
        steps = [ex["step"] for ex in batch]
        result = step_B_concat(memorys, processor)
        print(f"\n[batch {bi}] Step B result:")
        print(f"  pixel_values_mem.shape      = {result['pixel_values_mem_shape']}")
        print(f"  memorys_length              = {result['memorys_length']}")
        print(f"  total_memory_images (flat)  = {result['total_memory_images_flat']}")
        print(f"  steps                       = {steps}")
        print(f"  processor other outputs     = {result['extra_keys']}")

        # sanity: per-sample memory lengths all equal?
        lens = [len(m) for m in memorys]
        if len(set(lens)) != 1:
            print(f"  [!!] memory lengths differ across batch: {lens} — flatten logic will break.")
        else:
            print(f"  OK: all samples have memory length = {lens[0]}")

        step_B_grid_check(batch, processor, result)


if __name__ == "__main__":
    main()
