#!/usr/bin/env python3
"""Load a converted YAM conveyor dataset through Long-WAM's real training data pipeline (CPU).

  LONGWAM_DATA_ROOT=... LONGWAM_CHECKPOINT_ROOT=... python check_dataset.py [--dummy_text] [--out sample.png]

--dummy_text writes a zero text embedding for each prompt so the loader runs before the real UMT5 cache
exists (only for this check; training needs scripts/precompute_text_embeds.py task=yam_conveyor).
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from longwam.utils.config_resolvers import register_default_resolvers
from longwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT, get_text_cache_path

ap = argparse.ArgumentParser()
ap.add_argument("--dummy_text", action="store_true")
ap.add_argument("--out", default="/tmp/longwam_sample.png")
ap.add_argument("--repo", default=os.path.expanduser("~/longwam/LongLive/Long-WAM"))
args = ap.parse_args()

register_default_resolvers()
with initialize_config_dir(config_dir=os.path.join(args.repo, "configs"), version_base=None):
    cfg = compose(config_name="train", overrides=["task=yam_conveyor"])
dcfg = cfg.data.train
root = Path(dcfg.dataset_dirs[0])
tasks = [json.loads(l)["task"] for l in (root / "meta/tasks.jsonl").read_text().splitlines() if l.strip()]
print("tasks:", tasks)
if args.dummy_text:
    for t in tasks:
        p = Path(get_text_cache_path(dcfg.text_embedding_cache_dir, DEFAULT_PROMPT.format(task=t), dcfg.context_len))
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"context": torch.zeros(dcfg.context_len, 4096, dtype=torch.bfloat16),
                    "mask": torch.ones(dcfg.context_len, dtype=torch.bool)}, p)
        print("dummy text cache:", p)

ds = instantiate(dcfg)
print("dataset length (samples):", len(ds))
s = ds[len(ds) // 2]
for k, v in s.items():
    if isinstance(v, torch.Tensor):
        print(f"  {k:14s} {tuple(v.shape)} {v.dtype} min {v.float().min():.3f} max {v.float().max():.3f}")
    else:
        print(f"  {k:14s} {type(v).__name__}: {str(v)[:90]}")
video = s["video"]  # [3, T, H, W] in [-1, 1]
frames = ((video.float().clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 3, 0).numpy()
import cv2
row = np.concatenate([frames[i] for i in np.linspace(0, len(frames) - 1, 4).astype(int)], axis=1)
cv2.imwrite(args.out, cv2.cvtColor(row, cv2.COLOR_RGB2BGR))
print("wrote", args.out)
