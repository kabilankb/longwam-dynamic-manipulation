"""
Automatic demonstration generation for ConveyorCubePick-v0 (YAM bimanual + conveyor belt).

A scripted expert drives the right arm with damped-least-squares IK on privileged cube state:
hover over the belt -> track the moving cube while descending -> close -> lift -> carry to the bin ->
lower -> release -> retreat. The left arm holds its home pose. Every episode randomizes the belt speed,
the pick point along the belt, hover height and timing (the cube's spawn pose is randomized by the task's
reset event). Only successful episodes (cube in the bin, released) are written to the LeRobot dataset; failed
attempts are discarded.

The dataset has the same layout as the other YAMLab datasets (30 Hz, 14-D state/action, three 320x240
cameras: top_rgb, left_rgb, right_rgb), so the ACT trainer and the Long-WAM converter can read it.

Usage:
  python scripts/conveyor/generate_conveyor_data.py --num_episodes 200 \
      --output datasets/yam_conveyor_cube --headless --enable_cameras
  # watch it: replace --headless with --viz kit
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Generate ConveyorCubePick demos with a scripted IK expert")
parser.add_argument("--num_episodes", type=int, default=10, help="Successful episodes to keep")
parser.add_argument("--max_attempts", type=int, default=None, help="Stop after this many attempts (default 3x)")
parser.add_argument("--output", type=str, default="datasets/yam_conveyor_cube")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--speed_range", type=float, nargs=2, default=[0.04, 0.07], help="Belt speed range (m/s)")
parser.add_argument("--max_steps", type=int, default=720, help="Control steps per attempt (30 Hz)")
parser.add_argument("--camera_width", type=int, default=320)
parser.add_argument("--camera_height", type=int, default=240)
parser.add_argument("--task_description", type=str,
                    default="pick the red cube from the conveyor and place it in the blue bin")
parser.add_argument("--no_record", action="store_true", help="Run the expert without writing a dataset")
parser.add_argument("--hdris_path", type=str, default=None,
                    help="Enable lighting randomization (dome intensity/colour + HDRI) from this folder "
                         "(train/ and eval/ subfolders allowed), e.g. assets/hdris")
parser.add_argument("--debug_dir", type=str, default=None, help="Save a top-camera mp4 + phase log per attempt")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import json
import math
import os
import time

import cv2
import numpy as np
import torch

from yamlab.robot.yam import ROBOT
from yamlab.utils.task_creation import create_task_environment
from yamlab.envs.tasks.conveyor_cube_pick_manager_cfg import BIN, CONVEYOR, CUBE_SIZE, TABLE_Z

OPEN = ROBOT.finger_open("left")      # -0.0475
CLOSED = ROBOT.finger_closed("left")  # 0.0
LINK6 = "link_6"


def skew(v):
    z = torch.zeros_like(v[..., 0])
    return torch.stack([torch.stack([z, -v[..., 2], v[..., 1]], -1),
                        torch.stack([v[..., 2], z, -v[..., 0]], -1),
                        torch.stack([-v[..., 1], v[..., 0], z], -1)], -2)


def quat_xyzw_to_mat(q):
    x, y, z, w = q.unbind(-1)
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


def rot_error(R_des, R):
    """Axis-angle vector (world frame) rotating R onto R_des."""
    Re = R_des @ R.T
    angle = torch.arccos(torch.clamp((torch.trace(Re) - 1) / 2, -1.0, 1.0))
    if angle < 1e-6:
        return torch.zeros(3, device=R.device)
    axis = torch.stack([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]])
    s = torch.sin(angle)
    if s.abs() < 1e-4:  # ~180 deg: fall back to a principal axis
        d = torch.diagonal(Re)
        axis = torch.sqrt(torch.clamp((d + 1) / 2, min=0.0))
        return axis / axis.norm() * angle
    return axis / (2 * s) * angle


def topdown_rotation(yaw, device):
    """link_6 orientation with the gripper pointing straight down and fingers closing along world (-sin, cos)."""
    y = torch.tensor([-math.sin(yaw), math.cos(yaw), 0.0], device=device)
    z = torch.tensor([0.0, 0.0, -1.0], device=device)
    x = torch.linalg.cross(y, z)
    return torch.stack([x, y, z], dim=1)  # columns are the local axes in world


class ArmIK:
    """Damped-least-squares IK on a fingertip-centred tool point (TCP) of one YAM arm."""

    def __init__(self, env, arm_name):
        self.env, self.arm = env, env.scene[arm_name]
        self.b6 = self.arm.body_names.index(LINK6)
        self.jac_idx = self.b6 - 1 if self.arm.is_fixed_base else self.b6
        self.bl = self.arm.body_names.index("left_finger")
        self.br = self.arm.body_names.index("right_finger")
        kp = ROBOT.raw["fingers"]
        dev = env.device
        self.tip_l = torch.tensor(kp["lf"]["keypoints"][0], device=dev)
        self.tip_r = torch.tensor(kp["rf"]["keypoints"][0], device=dev)
        self.origin = env.scene.env_origins[0]
        self.limits = self.arm.data.joint_pos_limits.torch[0, :6]

    def state(self):
        d = self.arm.data
        pose = d.body_link_pose_w.torch[0]
        p6, R6 = pose[self.b6, :3] - self.origin, quat_xyzw_to_mat(pose[self.b6, 3:7])
        tl = pose[self.bl, :3] - self.origin + quat_xyzw_to_mat(pose[self.bl, 3:7]) @ self.tip_l
        tr = pose[self.br, :3] - self.origin + quat_xyzw_to_mat(pose[self.br, 3:7]) @ self.tip_r
        approach = R6[:, 2]
        tcp = 0.5 * (tl + tr) - approach * 0.015  # finger pad centre, 1.5 cm up from the tips
        return p6, R6, tcp

    def solve(self, tcp_target, R_target, max_lin=0.025, damping=0.04):
        d = self.arm.data
        q = d.joint_pos.torch[0, :6]
        _, R6, tcp = self.state()
        J = d.body_com_jacobian_w.torch[0, self.jac_idx, :, :6]
        com = d.body_com_pos_w.torch[0, self.b6] - self.origin
        J_lin = J[:3] - skew(tcp - com) @ J[3:]
        Jt = torch.cat([J_lin, J[3:]], 0)
        e_pos = tcp_target - tcp
        n = e_pos.norm()
        if n > max_lin:
            e_pos = e_pos * (max_lin / n)
        e_rot = rot_error(R_target, R6)
        e_rot = torch.clamp(e_rot, -0.15, 0.15)
        e = torch.cat([e_pos, 0.8 * e_rot])
        dq = Jt.T @ torch.linalg.solve(Jt @ Jt.T + damping ** 2 * torch.eye(6, device=q.device), e)
        dq = torch.clamp(dq, -0.08, 0.08)
        return torch.clamp(q + dq, self.limits[:, 0], self.limits[:, 1])


class ConveyorExpert:
    def __init__(self, env, rng, q_ready=None, q_bin=None, verbose=False):
        self.env, self.rng, self.q_ready, self.q_bin, self.verbose = env, rng, q_ready, q_bin, verbose
        self.ik = ArmIK(env, "right_arm")
        self.left_home = env.scene["left_arm"].data.joint_pos.torch[0, :6].clone()
        self.right_home = env.scene["right_arm"].data.joint_pos.torch[0, :6].clone()
        self.reset()

    def reset(self):
        r = self.rng
        self.phase, self.t = ("ready" if self.q_ready is not None else "hover"), 0
        self.ready_steps = int(r.integers(40, 55))
        self.lost = 0
        self.grasp_heading = None
        self.R_hold = None
        self.pick_y = float(r.uniform(-0.06, 0.04))
        self.hover_dz = float(r.uniform(0.10, 0.15))
        self.descend_steps = int(r.integers(20, 30))
        self.close_steps = int(r.integers(10, 15))
        self.lift_dz = float(r.uniform(0.12, 0.17))
        self.carry_steps = int(r.integers(35, 50))
        self.drop_dz = float(r.uniform(0.10, 0.13))
        b = self.env.bin_center()[0]  # the bin moves per episode
        self.bin_xy = (float(b[0]) + float(r.uniform(-0.015, 0.015)), float(b[1]) + float(r.uniform(-0.015, 0.015)))
        self.yaw = None
        self.grip = OPEN
        self.from_pt = None

    def _cube(self):
        c = self.env.scene["cube"]
        pos = c.data.root_pos_w.torch[0] - self.ik.origin
        q = c.data.root_quat_w.torch[0]
        yaw = math.atan2(2 * (q[3] * q[2] + q[0] * q[1]).item(), 1 - 2 * (q[1] ** 2 + q[2] ** 2).item())
        return pos, yaw

    def _heading(self, p):
        base = ROBOT.arm_position("right")
        return math.atan2(float(p[1]) - base[1], float(p[0]) - base[0])

    def _goto(self, phase):
        if self.verbose:
            cube, _ = self._cube()
            g = [bool(x[0]) for x in self.env.robot.is_grasping("cube")]
            print(f"    -> {phase:8s} cube {np.round(cube.cpu().numpy(), 3)} grasp(L,R) {g}", flush=True)
        self.phase, self.t = phase, 0
        self.from_pt = self.ik.state()[2].clone()

    def act(self):
        env, dev = self.env, self.env.device
        cube, cube_yaw = self._cube()
        v = torch.tensor([0.0, -env.conveyor_speed, 0.0], device=dev)
        if self.yaw is None:  # grasp yaw: cube faces modulo 90 deg, closest to the default
            self.yaw = (cube_yaw + math.pi / 4) % (math.pi / 2) - math.pi / 4
        R = topdown_rotation(self.yaw, dev)
        if self.phase in ("lower", "release", "retreat") and self.R_hold is not None:
            R = self.R_hold  # after the carry: keep the orientation the gripper arrived with
        hover = torch.tensor([CONVEYOR["x"], self.pick_y, CONVEYOR["top_z"] + CUBE_SIZE / 2 + self.hover_dz], device=dev)
        self.t += 1
        p = self.phase

        if p == "ready":  # smooth joint-space move home -> ready pose above the belt
            a = 0.5 - 0.5 * math.cos(math.pi * min(1.0, self.t / self.ready_steps))
            q_r = self.right_home + a * (self.q_ready - self.right_home)
            if self.t >= self.ready_steps:
                self._goto("hover")
            return self._action(q_r)
        if p == "hover":
            target = hover
            lead = (cube[1] - self.pick_y).item() / max(env.conveyor_speed, 1e-3)  # s until the cube arrives
            if self.t > 25 and lead < self.descend_steps / 30.0 + 0.3:
                self._goto("track")
        elif p == "track":  # follow the cube while descending onto it
            a = min(1.0, self.t / self.descend_steps)
            z = hover[2] + a * (cube[2] - hover[2])
            target = cube + v * 0.12
            target = torch.stack([target[0], target[1], z])
            if self.t >= self.descend_steps + 4:
                self._goto("close")
        elif p == "close":
            self.grip = CLOSED
            target = torch.stack([cube[0], (cube + v * 0.06)[1], cube[2]])
            if self.t >= self.close_steps:
                self._goto("lift")
        elif p == "lift":
            a = min(1.0, self.t / 18)
            target = self.from_pt + torch.tensor([0.0, 0.0, a * self.lift_dz], device=dev)
            if self.t >= 22:
                if not bool(env.robot.is_grasping("cube")[1][0]):
                    return None  # missed: abort the attempt
                self.grasp_heading = self._heading(self.ik.state()[2])
                self._goto("carry")
        elif p == "carry":  # joint-space move from the lift pose to the calibrated pose above the bin
            self.lost = 0 if bool(env.robot.is_grasping("cube")[1][0]) else self.lost + 1
            if self.lost > 4:
                return None  # dropped during the carry
            if self.t == 1:
                self.q_from = env.scene["right_arm"].data.joint_pos.torch[0, :6].clone()
            a = 0.5 - 0.5 * math.cos(math.pi * min(1.0, self.t / self.carry_steps))
            if self.t >= self.carry_steps + 6:
                self._goto("lower")
                self.R_hold = self.ik.state()[1].clone()
            return self._action(self.q_from + a * (self.q_bin - self.q_from))
        elif p == "lower":
            a = 0.5 - 0.5 * math.cos(math.pi * min(1.0, self.t / 22))
            goal = torch.tensor([self.bin_xy[0], self.bin_xy[1], TABLE_Z + self.drop_dz], device=dev)
            target = self.from_pt + a * (goal - self.from_pt)
            if self.t >= 26:
                self._goto("release")
        elif p == "release":
            self.grip = OPEN
            target = self.from_pt
            if self.t >= 15:
                self._goto("retreat")
        else:  # retreat: straight up, then back toward home joints
            target = self.from_pt + torch.tensor([0.0, 0.0, min(0.10, self.t * 0.006)], device=dev)

        if p == "retreat" and self.t > 20:
            q_r = self.env.scene["right_arm"].data.joint_pos.torch[0, :6]
            q_r = q_r + torch.clamp(self.right_home - q_r, -0.05, 0.05)
        else:
            q_r = self.ik.solve(target, R)
        return self._action(q_r)

    def _action(self, q_r):
        a = torch.zeros(1, 14, device=self.env.device)
        a[0, :6] = self.left_home
        a[0, 6] = OPEN
        a[0, 7:13] = q_r
        a[0, 13] = self.grip
        return a


def main():
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    env = create_task_environment(
        task_name="ConveyorCubePick-v0", assets_instance_paths={}, mode="evaluation",
        observation_modalities=["rgb", "proprioception"], num_envs=1, device="cuda:0",
        camera_width=args.camera_width, camera_height=args.camera_height,
        timeout_steps=args.max_steps + 100,
        enable_domain_randomization=args.hdris_path is not None,
        hdris_path=os.path.abspath(args.hdris_path) if args.hdris_path else None,
    )
    out = os.path.abspath(args.output)
    if not args.no_record:
        if os.path.exists(out):
            raise SystemExit(f"{out} already exists; choose a new --output")
        env.configure_for_lerobot_recording(output_dir=out, repo_id=f"local/{os.path.basename(out)}", fps=30,
                                            task_name=args.task_description, image_downsample_factor=1)
    # One unrecorded calibration run: let the hover IK converge from home, keep that joint pose as the
    # "ready" pose, so every recorded episode starts with a smooth home -> ready joint move.
    env.conveyor_speed = 0.0
    env.reset()
    calib = ConveyorExpert(env, np.random.default_rng(12345))
    calib.pick_y, calib.hover_dz = -0.01, 0.13
    for _ in range(150):
        env.step(calib.act())
    q_ready = env.scene["right_arm"].data.joint_pos.torch[0, :6].clone()
    print(f"[CONVEYOR] ready pose {np.round(q_ready.cpu().numpy(), 3)}, TCP {np.round(calib.ik.state()[2].cpu().numpy(), 3)}")
    bin_hover = torch.tensor([BIN["x"], BIN["y"], TABLE_Z + 0.20], device=env.device)
    R_down = topdown_rotation(0.0, env.device)
    for _ in range(200):
        env.step(calib._action(calib.ik.solve(bin_hover, R_down)))
    q_bin = env.scene["right_arm"].data.joint_pos.torch[0, :6].clone()
    err = (calib.ik.state()[2] - bin_hover).norm().item()
    print(f"[CONVEYOR] bin pose {np.round(q_bin.cpu().numpy(), 3)}, TCP error {err * 100:.1f} cm")
    if err > 0.02:
        raise SystemExit("calibration could not reach the bin hover pose")

    max_attempts = args.max_attempts or 3 * args.num_episodes
    kept, attempts, log = 0, 0, []
    t0 = time.time()
    while kept < args.num_episodes and attempts < max_attempts and app.is_running():
        attempts += 1
        env.conveyor_speed = float(rng.uniform(*args.speed_range))
        env.reset()
        expert = ConveyorExpert(env, rng, q_ready=q_ready, q_bin=q_bin, verbose=bool(args.debug_dir))
        if not args.no_record:
            env.lerobot_recorder.start_recording()
        result, steps, frames = "timeout", 0, []
        for steps in range(1, args.max_steps + 1):
            action = expert.act()
            if action is None:
                result = "dropped" if expert.phase == "carry" else "missed_grasp"
                break
            obs, _, terminated, truncated, info = env.step(action)
            if args.debug_dir:
                img = obs["policy"]["top_camera_rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
                cv2.putText(img, f"{expert.phase} {steps}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
                frames.append(img)
            if bool(env.get_task_success()[0]):
                result = "success"
                break
            if bool(terminated[0]) or bool(truncated[0]):
                result = "terminated"
                break
        ok = result == "success"
        if not args.no_record:
            if ok:
                env.lerobot_recorder.save_episode(0)
            else:
                env.lerobot_recorder.discard_episode(0)
        kept += ok
        if args.debug_dir and frames:
            os.makedirs(args.debug_dir, exist_ok=True)
            vw = cv2.VideoWriter(os.path.join(args.debug_dir, f"attempt{attempts:03d}_{result}.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), 30, frames[0].shape[1::-1])
            for f in frames:
                vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            vw.release()
        log.append({"attempt": attempts, "result": result, "steps": steps, "speed": round(env.conveyor_speed, 4),
                    "bin_xy": [round(float(x), 3) for x in env.bin_center()[0, :2]],
                    "pick_y": round(expert.pick_y, 3)})
        print(f"[CONVEYOR] attempt {attempts}: {result} ({steps} steps, belt {env.conveyor_speed:.3f} m/s, "
              f"bin {np.round(env.bin_center()[0, :2].cpu().numpy(), 3)}) "
              f"-> kept {kept}/{args.num_episodes}, {time.time() - t0:.0f}s", flush=True)

    if not args.no_record and kept:
        env.consolidate_lerobot_dataset(run_compute_stats=True)
        env.close_lerobot_recorder()
        with open(os.path.join(out, "meta", "generation_log.json"), "w") as f:
            json.dump({"task": "ConveyorCubePick-v0", "instruction": args.task_description, "seed": args.seed,
                       "lighting_randomization": args.hdris_path, "bin_randomization": True,
                       "speed_range": args.speed_range, "kept": kept, "attempts": attempts, "log": log}, f, indent=1)
    print(f"[CONVEYOR] done: {kept} successful / {attempts} attempts "
          f"({100 * kept / max(attempts, 1):.0f}%), {time.time() - t0:.0f}s. Dataset: {out if kept and not args.no_record else 'none'}")
    env.close()
    app.close()


if __name__ == "__main__":
    main()
