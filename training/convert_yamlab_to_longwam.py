#!/usr/bin/env python3
"""Convert a YAMLab LeRobot dataset (e.g. ConveyorCubePick) into the layout Long-WAM-YAM trains on.

Changes (everything else is kept, videos are hard-linked, not re-encoded):
  * cameras  observation.images.{top_rgb,left_rgb,right_rgb} -> observation.images.{top,left_wrist,right_wrist}
    (the camera names the Long-WAM YAM adapter and checkpoint use)
  * gripper  state/action columns 6 and 13: yamlab finger position in metres (-0.0475 open .. 0 closed)
             -> ABC-130K convention 0..1, 1 = open (same mapping as the yamlab6 eval client;
             pass --gripper_one_is_closed to flip both)
  * meta     info.json features renamed; stats.json renamed and state/action stats recomputed

Usage:
  python convert_yamlab_to_longwam.py SRC_DATASET DST_DATASET [--gripper_one_is_closed]
"""
import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

CAMERA_RENAME = {
    "observation.images.top_rgb": "observation.images.top",
    "observation.images.left_rgb": "observation.images.left_wrist",
    "observation.images.right_rgb": "observation.images.right_wrist",
}
GRIPPER_COLUMNS = (6, 13)
FINGER_OPEN_M = -0.0475


def finger_to_unit(x, one_is_closed):
    g = np.clip(np.asarray(x, np.float64) / FINGER_OPEN_M, 0.0, 1.0)
    return 1.0 - g if one_is_closed else g


def convert_vec_column(series, one_is_closed):
    arr = np.stack(series.to_numpy()).astype(np.float64)
    for c in GRIPPER_COLUMNS:
        arr[:, c] = finger_to_unit(arr[:, c], one_is_closed)
    return arr


def link_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--gripper_one_is_closed", action="store_true")
    args = ap.parse_args()
    src, dst = args.src.resolve(), args.dst.resolve()
    info = json.loads((src / "meta/info.json").read_text())
    missing = [k for k in CAMERA_RENAME if k not in info["features"]]
    if missing:
        raise SystemExit(f"{src} lacks {missing}; is it a consolidated YAMLab dataset?")
    if dst.exists():
        raise SystemExit(f"{dst} exists; choose a new destination")

    # data: convert gripper columns, keep everything else
    states, actions = [], []
    for pq in sorted((src / "data").rglob("*.parquet")):
        df = pd.read_parquet(pq)
        s = convert_vec_column(df["observation.state"], args.gripper_one_is_closed)
        a = convert_vec_column(df["action"], args.gripper_one_is_closed)
        df["observation.state"] = list(s.astype(np.float32))
        df["action"] = list(a.astype(np.float32))
        out = dst / pq.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out)
        states.append(s)
        actions.append(a)

    # videos: hard-link under the renamed camera directories
    n_videos = 0
    for mp4 in sorted((src / "videos").rglob("*.mp4")):
        rel = mp4.relative_to(src)
        parts = [CAMERA_RENAME.get(p, p) for p in rel.parts]
        link_or_copy(mp4, dst.joinpath(*parts))
        n_videos += 1

    # meta
    (dst / "meta").mkdir(parents=True, exist_ok=True)
    for f in (src / "meta").iterdir():
        if f.name not in ("info.json", "stats.json") and f.is_file():
            shutil.copy2(f, dst / "meta" / f.name)
    info["features"] = {CAMERA_RENAME.get(k, k): v for k, v in info["features"].items()}
    for key in ("observation.state", "action"):
        names = info["features"][key].get("names")
        if isinstance(names, list) and len(names) == 14:
            for c in GRIPPER_COLUMNS:
                names[c] = names[c].replace("finger.pos", "gripper")
    info["longwam_conversion"] = {
        "source": str(src), "camera_rename": CAMERA_RENAME,
        "gripper": "0..1, 1 = closed" if args.gripper_one_is_closed else "0..1, 1 = open",
    }
    (dst / "meta/info.json").write_text(json.dumps(info, indent=4))
    stats_path = src / "meta/stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        stats = {CAMERA_RENAME.get(k, k): v for k, v in stats.items()}
        for key, arr in (("observation.state", np.concatenate(states)), ("action", np.concatenate(actions))):
            stats[key] = {"mean": arr.mean(0).tolist(), "std": arr.std(0).tolist(),
                          "min": arr.min(0).tolist(), "max": arr.max(0).tolist(), "count": [int(len(arr))]}
        (dst / "meta/stats.json").write_text(json.dumps(stats, indent=4))

    allf = np.concatenate(states)
    print(f"converted {info.get('total_episodes')} episodes, {len(allf)} frames, {n_videos} videos -> {dst}")
    print("gripper state range (left, right):", np.round([allf[:, 6].min(), allf[:, 6].max(),
                                                           allf[:, 13].min(), allf[:, 13].max()], 3))


if __name__ == "__main__":
    main()
