import json
import os
from accelerate.logging import get_logger
import numpy as np
from torch.utils.data import DataLoader
import torch.distributed as dist
from pathlib import Path
from starVLA.dataloader.vlm_datasets import make_vlm_dataloader

logger = get_logger(__name__)

def save_dataset_statistics(dataset_statistics, run_dir):
    """Saves a `dataset_statistics.json` file."""
    out_path = run_dir / "dataset_statistics.json"
    with open(out_path, "w") as f_json:
        for _, stats in dataset_statistics.items():
            for k in stats["action"].keys():
                if isinstance(stats["action"][k], np.ndarray):
                    stats["action"][k] = stats["action"][k].tolist()
            if "proprio" in stats:
                for k in stats["proprio"].keys():
                    if isinstance(stats["proprio"][k], np.ndarray):
                        stats["proprio"][k] = stats["proprio"][k].tolist()
            if "num_trajectories" in stats:
                if isinstance(stats["num_trajectories"], np.ndarray):
                    stats["num_trajectories"] = stats["num_trajectories"].item()
            if "num_transitions" in stats:
                if isinstance(stats["num_transitions"], np.ndarray):
                    stats["num_transitions"] = stats["num_transitions"].item()
        json.dump(dataset_statistics, f_json, indent=2)
    logger.info(f"Saved dataset statistics file at path {out_path}")



def build_dataloader(cfg, dataset_py="lerobot_datasets_oxe"): # TODO now here only is get dataset, we need mv dataloader to here

    if dataset_py == "lerobot_datasets":
        from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn
        vla_dataset_cfg = cfg.datasets.vla_data
        # 修改这里防止硬编码，可以直接从yaml文件中读取参数
        num_workers = int(vla_dataset_cfg.get("num_workers", 0))
        pin_memory = bool(vla_dataset_cfg.get("pin_memory", False))
        persistent_workers = bool(vla_dataset_cfg.get("persistent_workers", False)) if num_workers > 0 else False
        prefetch_factor = vla_dataset_cfg.get("prefetch_factor", None) if num_workers > 0 else None

        vla_dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)

        dataloader_kwargs = {
            "batch_size": cfg.datasets.vla_data.per_device_batch_size,
            "collate_fn": collate_fn,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "persistent_workers": persistent_workers,
        }
        if prefetch_factor is not None:
            dataloader_kwargs["prefetch_factor"] = prefetch_factor

        logger.info(
            "Building VLA DataLoader with num_workers=%s, pin_memory=%s, persistent_workers=%s, "
            "prefetch_factor=%s",
            num_workers,
            pin_memory,
            persistent_workers,
            prefetch_factor,
        )

        vla_train_dataloader = DataLoader(
            vla_dataset,
            **dataloader_kwargs,
            # shuffle=True
        )
        if not dist.is_initialized() or dist.get_rank() == 0:
            output_dir = Path(cfg.output_dir)
            vla_dataset.save_dataset_statistics(output_dir / "dataset_statistics.json")
        return vla_train_dataloader
    elif dataset_py == "vlm_datasets":
        vlm_data_module = make_vlm_dataloader(cfg)
        vlm_train_dataloader = vlm_data_module["train_dataloader"]
        
        return vlm_train_dataloader
