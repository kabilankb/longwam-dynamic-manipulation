"""Probe the ConveyorCubePick scene: renders the top camera, tracks the cube on the belt, prints arm link poses."""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=90)
parser.add_argument("--out", type=str, default="outputs/conveyor/probe")
parser.add_argument("--camera_width", type=int, default=320)
parser.add_argument("--camera_height", type=int, default=240)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import os
import cv2
import numpy as np
import torch

from yamlab.utils.task_creation import create_task_environment

os.makedirs(args.out, exist_ok=True)
env = create_task_environment(task_name="ConveyorCubePick-v0", assets_instance_paths={}, mode="evaluation",
                              observation_modalities=["rgb", "proprioception"], num_envs=1, device="cuda:0",
                              camera_width=args.camera_width, camera_height=args.camera_height,
                              timeout_steps=2000)
obs, _ = env.reset()
p = obs["policy"]
print("obs keys:", {k: tuple(v.shape) for k, v in p.items()})
arm = env.scene["right_arm"]
print("right_arm joints:", arm.joint_names)
print("right_arm bodies:", arm.body_names)
q0 = arm.data.joint_pos.torch[0].cpu().numpy()
print("right_arm q0:", np.round(q0, 3))
origin = env.scene.env_origins[0]
for name in ("link_6", "left_finger", "right_finger"):
    i = arm.body_names.index(name)
    pose = arm.data.body_link_pose_w.torch[0, i].cpu().numpy()
    print(f"{name}: pos {np.round(pose[:3] - origin.cpu().numpy(), 4)} quat(xyzw) {np.round(pose[3:], 4)}")

act = torch.zeros(1, 14, device=env.device)
act[0, :6] = env.scene["left_arm"].data.joint_pos.torch[0, :6]
act[0, 7:13] = arm.data.joint_pos.torch[0, :6]
act[0, 6] = act[0, 13] = -0.0475
for t in range(args.steps):
    obs, _, term, trunc, info = env.step(act)
    if t % 15 == 0 or t == args.steps - 1:
        c = (env.scene["cube"].data.root_pos_w.torch[0] - origin).cpu().numpy()
        print(f"step {t}: cube {np.round(c, 4)} on_belt {bool(env.cube_on_belt()[0])}")
img = obs["policy"]["top_camera_rgb"][0].cpu().numpy()
cv2.imwrite(f"{args.out}/top.png", cv2.cvtColor(img[..., :3].astype(np.uint8), cv2.COLOR_RGB2BGR))
for cam in ("left_wrist", "right_wrist"):
    img = obs["policy"][f"{cam}_camera_rgb"][0].cpu().numpy()
    cv2.imwrite(f"{args.out}/{cam}.png", cv2.cvtColor(img[..., :3].astype(np.uint8), cv2.COLOR_RGB2BGR))
env.close()
app.close()
