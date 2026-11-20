"""
Evaluate Long-WAM-YAM (served remotely, e.g. on a Jetson Thor) closed-loop in yamlab6.

The sim runs here; every control step's three RGB views + 14-D joint state are streamed (JPEG)
to ``longwam_yam_server.py`` and a 32-step action chunk comes back every ``--execute_steps``
steps. The sim waits for each chunk (synchronous), so inference latency does not affect physics.

Unit conversion (yamlab <-> ABC-130K checkpoint):
  arm joints: radians in both (same YAM joint conventions)
  gripper:    yamlab finger pos  -0.0475 (open) .. 0 (closed)  <->  checkpoint 0..1
              default 1 = open; pass --gripper_one_is_closed to flip.

Usage (GUI; use --headless instead of --viz kit to run without a window):
  D=yamlab_datasets/tasks_data/PutPotOnCooktop/objects
  python scripts/eval/eval_longwam_remote.py --server $LONGWAM_SERVER \
    --task PutPotOnCooktop-v0 \
    --asset pot=$D/Pot/pot_000 --asset cooktop=$D/Cooktop/cooktop_000 \
    --num_episodes 5 --timeout_steps 600 --enable_cameras --viz kit
"""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate a remote Long-WAM-YAM policy on a YAM bimanual task")
parser.add_argument("--server", type=str, default=None,
                    help="host:port of longwam_yam_server.py (default: $LONGWAM_SERVER)")
parser.add_argument("--task", type=str, default="PutPotOnCooktop-v0")
parser.add_argument("--instruction", type=str, default=None, help="Language instruction (default: per task)")
parser.add_argument("--asset", action="append", default=[], metavar="NAME=PATH")
parser.add_argument("--num_episodes", type=int, default=5)
parser.add_argument("--timeout_steps", type=int, default=600)
parser.add_argument("--execute_steps", type=int, default=24,
                    help="Actions executed from each 32-step chunk before replanning")
parser.add_argument("--camera_width", type=int, default=320)
parser.add_argument("--camera_height", type=int, default=240)
parser.add_argument("--jpeg_quality", type=int, default=90)
parser.add_argument("--gripper_one_is_closed", action="store_true",
                    help="Checkpoint gripper 1.0 means closed (default: 1.0 means open)")
parser.add_argument("--enable_gripper_clamp", action="store_true", default=False)
parser.add_argument("--init_joint_pos_randomization", type=float, default=0.0)
parser.add_argument("--enable_domain_randomization", action="store_true", default=False)
parser.add_argument("--hdris_path", type=str, default=None,
                    help="Lighting randomization from this HDRI folder (e.g. assets/hdris); implies DR")
parser.add_argument("--output_dir", type=str, default="outputs/eval/longwam_yam")
parser.add_argument("--no_video", action="store_true", help="Do not save per-episode mosaic videos")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(device=None)
args = parser.parse_args()
import os as _os
args.server = args.server or _os.environ.get("LONGWAM_SERVER")
if not args.server:
    parser.error("pass --server HOST:PORT or set LONGWAM_SERVER")

from yamlab.configs import (resolve_device, resolve_assets, parse_asset_args,
                            get_objects_randomization, get_task_kwargs)

args.device = resolve_device(args.task, "evaluation", args.device)
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import json
import os
import socket
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from longwam_wire import recv_msg, send_msg  # noqa: E402

from yamlab.robot.spec import get_robot  # noqa: E402
from yamlab.utils.task_creation import create_task_environment  # noqa: E402

ROBOT = get_robot("yam")
CAMERAS = ("top", "left_wrist", "right_wrist")
FINGER_OPEN = ROBOT.raw["gripper"]["left_finger_open_pos"]  # -0.0475
DEFAULT_INSTRUCTIONS = {
    "PutPotOnCooktop": "put the pot on the cooktop",
    "HangMug": "hang the mug on the mug tree",
    "HangMugOnTree": "hang the mug on the mug tree",
    "PutAppleInBowl": "put the apple in the bowl",
    # Must match the dataset's task text exactly (the fine-tune's text embedding).
    "ConveyorCubePick": "pick the red cube from the conveyor and place it in the blue bin",
}
PRIMITIVE_TASKS = {"ConveyorCubePick-v0"}  # scene objects are primitives: no --asset folders


def finger_to_ckpt(finger):
    g = np.clip(np.asarray(finger, np.float32) / FINGER_OPEN, 0.0, 1.0)  # open -> 1
    return 1.0 - g if args.gripper_one_is_closed else g


def ckpt_to_finger(g):
    g = np.clip(np.asarray(g, np.float32), 0.0, 1.0)
    if args.gripper_one_is_closed:
        g = 1.0 - g
    return g * FINGER_OPEN


class RemotePolicy:
    def __init__(self, address):
        host, port = address.rsplit(":", 1)
        self.sock = socket.create_connection((host, int(port)), timeout=900)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.info = self.call({"op": "info"})["info"]

    def call(self, header, blobs=()):
        send_msg(self.sock, header, blobs)
        reply, _ = recv_msg(self.sock)
        if not reply.get("ok"):
            raise RuntimeError(f"server error: {reply.get('error')}")
        return reply

    def reset(self, instruction):
        return self.call({"op": "reset", "instruction": instruction})["episode"]

    def step(self, episode, pending, infer):
        obs, blobs = [], []
        for step, state, images in pending:
            idx = {}
            for cam in CAMERAS:
                ok, buf = cv2.imencode(".jpg", cv2.cvtColor(images[cam], cv2.COLOR_RGB2BGR),
                                       [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
                assert ok
                idx[cam] = len(blobs)
                blobs.append(buf.tobytes())
            obs.append({"step": step, "state": state.tolist(), "images": idx})
        return self.call({"op": "step", "episode": episode, "obs": obs, "infer": infer}, blobs)


def read_obs(obs_dict):
    p = obs_dict.get("policy", obs_dict)
    left_g = finger_to_ckpt(p["left_gripper_state"].reshape(-1)[0].item())
    right_g = finger_to_ckpt(p["right_gripper_state"].reshape(-1)[0].item())
    state = np.concatenate([
        p["left_arm_joint_pos"][0].cpu().numpy(), [left_g],
        p["right_arm_joint_pos"][0].cpu().numpy(), [right_g],
    ]).astype(np.float32)
    images = {}
    for cam in CAMERAS:
        img = p[f"{cam}_camera_rgb"][0]
        if img.dtype != torch.uint8:
            img = (img.float() * (255 if img.max() <= 1.0 else 1)).clamp(0, 255).to(torch.uint8)
        images[cam] = np.ascontiguousarray(img[..., :3].cpu().numpy())
    return state, images


def to_env_action(a):
    a = np.asarray(a, np.float32).copy()
    a[6] = ckpt_to_finger(a[6])
    a[13] = ckpt_to_finger(a[13])
    return torch.as_tensor(a, dtype=torch.float32).unsqueeze(0)


def mosaic(images):
    top = cv2.resize(images["top"], (320, 256))
    wrists = np.concatenate([cv2.resize(images["left_wrist"], (160, 128)),
                             cv2.resize(images["right_wrist"], (160, 128))], axis=1)
    return np.concatenate([top, wrists], axis=0)


def main():
    task_short = args.task.split("-")[0]
    instruction = args.instruction or DEFAULT_INSTRUCTIONS.get(task_short, task_short)
    out_dir = Path(args.output_dir) / time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    policy = RemotePolicy(args.server)
    print(f"[LWAM] connected to {args.server}: {policy.info}")
    horizon = int(policy.info["action_horizon"])
    if not 0 < args.execute_steps <= horizon:
        raise SystemExit(f"--execute_steps must be in 1..{horizon}")

    assets = ({} if args.task in PRIMITIVE_TASKS
              else resolve_assets(args.task, cli_assets=parse_asset_args(args.asset)))
    if not assets and args.task not in PRIMITIVE_TASKS:
        raise SystemExit("No object assets specified (--asset name=/path).")
    task_kwargs = get_task_kwargs(args.task, "evaluation")
    task_kwargs.update(timeout_steps=args.timeout_steps, enable_gripper_grasp_clamp=args.enable_gripper_clamp)
    env = create_task_environment(
        task_name=args.task, assets_instance_paths=assets,
        objects_randomization=get_objects_randomization(args.task),
        init_joint_pos_randomization=args.init_joint_pos_randomization,
        mode="evaluation", observation_modalities=["rgb", "proprioception"], num_envs=1,
        device=args.device, camera_width=args.camera_width, camera_height=args.camera_height,
        enable_domain_randomization=args.enable_domain_randomization or args.hdris_path is not None,
        hdris_path=os.path.abspath(args.hdris_path) if args.hdris_path else None, **task_kwargs,
    )

    obs_dict, _ = env.reset()
    results = []
    while simulation_app.is_running() and len(results) < args.num_episodes:
        ep_id = policy.reset(instruction)
        step, chunk, chunk_pos, pending = 0, None, 0, []
        infer_times, picked, success, frames, states, actions = [], False, False, [], [], []
        t_ep = time.time()
        while True:
            state, images = read_obs(obs_dict)
            pending.append((step, state, images))
            if not args.no_video:
                frames.append(mosaic(images))
            need = chunk is None or chunk_pos >= args.execute_steps
            if need or len(pending) >= 8:  # flush history in small batches between chunks
                reply = policy.step(ep_id, pending, infer=need)
                pending = []
                if need:
                    chunk, chunk_pos = np.asarray(reply["actions"], np.float32), 0
                    infer_times.append(reply["infer_s"])
                    print(f"[LWAM] ep {len(results) + 1} step {step}: chunk {chunk.shape} "
                          f"infer {reply['infer_s']:.2f}s")
            a = chunk[chunk_pos]
            chunk_pos += 1
            states.append(state.tolist())
            actions.append(a.tolist())
            obs_dict, _, terminated, truncated, _ = env.step(to_env_action(a).to(env.device))
            step += 1
            if hasattr(env, "stage1_success"):
                picked |= bool(env.stage1_success[0])
            if bool(terminated[0]) or bool(truncated[0]):
                success = bool(terminated[0]) and not bool(truncated[0])
                break
        r = {"episode": len(results) + 1, "success": success, "pick": picked or success, "steps": step,
             "wall_s": round(time.time() - t_ep, 1), "chunks": len(infer_times),
             "infer_mean_s": round(float(np.mean(infer_times)), 3) if infer_times else None,
             "instruction": instruction}
        results.append(r)
        print(f"[LWAM] episode {r['episode']}/{args.num_episodes}: {'SUCCESS' if success else 'FAIL'} "
              f"(pick {'yes' if r['pick'] else 'no'}, {step} steps, {r['chunks']} chunks, "
              f"infer {r['infer_mean_s']}s/chunk, wall {r['wall_s']}s)")
        np.savez_compressed(out_dir / f"ep{r['episode']:03d}_traj.npz",
                            state=np.asarray(states), action=np.asarray(actions))
        if frames:
            h, w = frames[0].shape[:2]
            vw = cv2.VideoWriter(str(out_dir / f"ep{r['episode']:03d}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
            for f in frames:
                vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            vw.release()

    n = len(results)
    summary = {"task": args.task, "instruction": instruction, "server": args.server, "server_info": policy.info,
               "execute_steps": args.execute_steps, "gripper_one_is_closed": args.gripper_one_is_closed,
               "episodes": results, "success": sum(r["success"] for r in results), "pick": sum(r["pick"] for r in results), "n": n}
    (out_dir / "results.json").write_text(json.dumps(summary, indent=2))
    if n:
        print("=" * 60)
        print(f"[LWAM] {args.task}: {summary['success']}/{n} succeeded, {summary['pick']}/{n} reached pick. "
              f"Results in {out_dir}")
        print("=" * 60)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
