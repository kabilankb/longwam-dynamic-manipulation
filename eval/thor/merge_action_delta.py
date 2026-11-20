#!/usr/bin/env python3
"""Build a servable fine-tuned checkpoint directory from the base Long-WAM-YAM bundle + an action delta.

  python merge_action_delta.py BASE_DIR action_step_000250.pt OUT_DIR

BASE_DIR holds the released bundle (model.pt, config.yaml, dataset_stats.json). OUT_DIR gets a merged
model.pt plus copies of config.yaml / dataset_stats.json, ready for `CKPT_DIR=OUT_DIR serve_yam.sh`.
"""
import argparse
import shutil
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base_dir", type=Path)
    ap.add_argument("delta", type=Path)
    ap.add_argument("out_dir", type=Path)
    args = ap.parse_args()

    base = torch.load(args.base_dir / "model.pt", map_location="cpu", weights_only=False)
    delta = torch.load(args.delta, map_location="cpu", weights_only=False)
    mot = base["mot"]
    missing = [k for k in delta["mot_action"] if k not in mot]
    if missing:
        raise SystemExit(f"delta keys not in base: {missing[:3]}")
    for k, v in delta["mot_action"].items():
        if mot[k].shape != v.shape:
            raise SystemExit(f"shape mismatch for {k}: {tuple(mot[k].shape)} vs {tuple(v.shape)}")
        mot[k] = v.to(mot[k].dtype)
    if "proprio_encoder" in delta and "proprio_encoder" in base:
        base["proprio_encoder"] = delta["proprio_encoder"]
    if delta.get("step") is not None:
        base["step"] = delta["step"]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tmp = args.out_dir / "model.pt.tmp"
    torch.save(base, tmp)
    tmp.replace(args.out_dir / "model.pt")
    for name in ("config.yaml", "dataset_stats.json"):
        shutil.copy2(args.base_dir / name, args.out_dir / name)
    print(f"merged {len(delta['mot_action'])} action tensors (step {delta.get('step')}) -> {args.out_dir}")


if __name__ == "__main__":
    main()
