#!/usr/bin/env python3
"""Cut the trained part out of a Long-WAM weights checkpoint (train_modules=action).

With the video expert frozen, only `mot.mixtures.action.*` and the proprio encoder change. Saving just those
turns a 12 GB `step_XXXXXX.pt` into a ~2 GB file that is cheap to move; `eval/thor/merge_action_delta.py`
puts it back into the base Long-WAM-YAM `model.pt`.

  python extract_action_delta.py checkpoints/weights/step_000250.pt action_step_000250.pt [--base base/model.pt]

--base (optional) verifies that every other tensor is identical to the base checkpoint, i.e. that the
run really left the video expert untouched.
"""
import argparse
import json
from pathlib import Path

import torch

ACTION_PREFIX = "mixtures.action."


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--base", type=Path, default=None)
    args = ap.parse_args()

    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    mot = payload["mot"]
    action = {k: v for k, v in mot.items() if k.startswith(ACTION_PREFIX)}
    if not action:
        raise SystemExit(f"no '{ACTION_PREFIX}*' tensors in {args.ckpt}")
    delta = {"mot_action": action, "step": payload.get("step")}
    if payload.get("proprio_encoder") is not None:
        delta["proprio_encoder"] = payload["proprio_encoder"]

    if args.base is not None:
        base = torch.load(args.base, map_location="cpu", weights_only=False)["mot"]
        changed = [k for k, v in mot.items() if not k.startswith(ACTION_PREFIX) and not torch.equal(v, base[k])]
        if changed:
            raise SystemExit(f"{len(changed)} non-action tensors differ from the base, e.g. {changed[:3]}")
        print("verified: all non-action tensors equal the base checkpoint")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    torch.save(delta, tmp)
    tmp.replace(args.out)
    n = sum(v.numel() for v in action.values())
    meta = {"source": args.ckpt.name, "step": delta["step"], "action_params": n,
            "proprio_encoder": "proprio_encoder" in delta, "dtype": str(next(iter(action.values())).dtype)}
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta), f"-> {args.out} ({args.out.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
