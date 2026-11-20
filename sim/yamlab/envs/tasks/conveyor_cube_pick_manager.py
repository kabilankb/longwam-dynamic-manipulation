"""ConveyorCubePick task environment: pick a cube off a moving conveyor belt and drop it into a bin.

Conveyor model: the belt is a static, low-friction collider. Before every control step the env sets the cube's
linear velocity to the belt velocity while the cube rests on the belt and no gripper holds it, so the cube rides
the belt toward the end stop at a constant speed. Once a gripper grasps it (or it is lifted off the belt), the env
stops driving it.

Stage 1 (pick): the cube is lifted above the belt top and held by a gripper, for N consecutive steps.
Stage 2 (place): the cube lies inside the bin's inner footprint, near the bin floor, held by no gripper.
"""

import gymnasium as gym
import torch
from typing import Any, Dict

from isaaclab.managers import SceneEntityCfg

from yamlab.configs import get_task_config
from yamlab.utils.task_logic import check_pick_success

from .conveyor_cube_pick_manager_cfg import BIN, CONVEYOR, CUBE_SIZE, TABLE_Z, ConveyorCubePickManagerEnvCfg
from .yam_bimanual_env import YamBimanualEnv, make_task_env

_SUCCESS = get_task_config("ConveyorCubePick-v0")["success"]

__all__ = ["ConveyorCubePickManager", "make_conveyor_cube_pick_env"]


def make_conveyor_cube_pick_env(**kwargs):
    kwargs.setdefault("pick_by_two_hands", False)
    return make_task_env(ConveyorCubePickManagerEnvCfg, ConveyorCubePickManager, **kwargs)


class ConveyorCubePickManager(YamBimanualEnv):
    """Pick a cube from a moving conveyor and place it into a bin."""

    def __init__(self, cfg: ConveyorCubePickManagerEnvCfg, render_mode: str | None = None, **kwargs):
        self.pick_by_two_hands = kwargs.pop("pick_by_two_hands", False)
        kwargs.pop("check_gripper_release_for_ontop", None)
        self.is_evaluation_mode = kwargs.pop("is_evaluation_mode", False)
        #: Belt speed (m/s along -Y); the data generator may change it per episode.
        self.conveyor_speed = float(kwargs.pop("conveyor_speed", CONVEYOR["speed"]))
        self.conveyor_enabled = True

        super().__init__(cfg, render_mode, **kwargs)

        self.cube_init_z = CONVEYOR["top_z"] + CUBE_SIZE / 2
        n, dev = self.num_envs, self.device
        self.stage1_success = torch.zeros(n, dtype=torch.bool, device=dev)
        self.stage2_success = torch.zeros(n, dtype=torch.bool, device=dev)
        self._consecutive_pick_steps = torch.zeros(n, dtype=torch.int32, device=dev)
        self._consecutive_place_steps = torch.zeros(n, dtype=torch.int32, device=dev)
        self._pick_required = int(_SUCCESS["pick_required_consecutive_steps"])
        self._place_required = int(_SUCCESS["place_required_consecutive_steps_eval" if self.is_evaluation_mode
                                            else "place_required_consecutive_steps_teleop"])
        print(f"[INFO] ConveyorCubePick initialized: belt speed {self.conveyor_speed} m/s, "
              f"bin at ({BIN['x']}, {BIN['y']})")

    # ------------------------------------------------------------------ helpers
    def get_object_names_for_physics_material(self) -> list[str]:
        return []  # the cube's friction is set on its own spawn material; the belt must stay low-friction

    def cube_grasped(self) -> torch.Tensor:
        """(num_envs,) bool: either gripper holds the cube."""
        left, right = self.robot.is_grasping("cube")
        return left | right

    def cube_on_belt(self) -> torch.Tensor:
        pos = self.scene["cube"].data.root_pos_w.torch - self.scene.env_origins
        c = CONVEYOR
        return ((pos[:, 0] - c["x"]).abs() < c["width"] / 2 + 0.01) & \
               (pos[:, 1] < c["y_start"] + 0.02) & (pos[:, 1] > c["y_end"] - 0.02) & \
               (pos[:, 2] < c["top_z"] + CUBE_SIZE / 2 + 0.012)

    def _drive_conveyor(self) -> None:
        if not self.conveyor_enabled or self.conveyor_speed == 0.0:
            return
        cube = self.scene["cube"]
        driven = self.cube_on_belt() & ~self.cube_grasped()
        if not driven.any():
            return
        ids = driven.nonzero(as_tuple=False).squeeze(-1)
        pos = cube.data.root_pos_w.torch[ids] - self.scene.env_origins[ids]
        vel = cube.data.root_vel_w.torch[ids].clone()
        at_stop = pos[:, 1] <= CONVEYOR["y_end"] + CUBE_SIZE / 2 + 0.004
        vel[:, 0] = 0.0
        vel[:, 1] = torch.where(at_stop, torch.zeros_like(vel[:, 1]), torch.full_like(vel[:, 1], -self.conveyor_speed))
        vel[:, 2] = torch.clamp(vel[:, 2], max=0.0)
        vel[:, 3:] = 0.0
        cube.write_root_velocity_to_sim_index(root_velocity=vel, env_ids=ids)

    def bin_center(self) -> torch.Tensor:
        """(num_envs, 3) bin floor centre in env-local coordinates (the bin moves per episode)."""
        return self.scene["bin_floor"].data.root_pos_w.torch - self.scene.env_origins

    def cube_in_bin(self) -> torch.Tensor:
        pos = self.scene["cube"].data.root_pos_w.torch - self.scene.env_origins
        b = self.bin_center()
        half = BIN["inner"] / 2 - CUBE_SIZE * 0.35
        return ((pos[:, 0] - b[:, 0]).abs() < half) & ((pos[:, 1] - b[:, 1]).abs() < half) & \
               (pos[:, 2] < TABLE_Z + BIN["floor"] + CUBE_SIZE + 0.015)

    # ------------------------------------------------------------------ success
    def reset_success_check(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self.stage1_success[env_ids] = False
        self.stage2_success[env_ids] = False
        self._consecutive_pick_steps[env_ids] = 0
        self._consecutive_place_steps[env_ids] = 0
        if hasattr(self, "_success_reward_given"):
            self._success_reward_given[env_ids] = False

    def update_stage_success(self):
        pick = check_pick_success(self, SceneEntityCfg("cube"), init_obj_height=self.cube_init_z,
                                  pick_threshold=_SUCCESS["pick_lift_threshold_m"], check_contact=True,
                                  pick_by_two_hands=self.pick_by_two_hands, grasp=self.robot)
        inc = pick & ~self.stage1_success
        self._consecutive_pick_steps[inc] += 1
        self._consecutive_pick_steps[~pick & ~self.stage1_success] = 0
        self.stage1_success |= self._consecutive_pick_steps >= self._pick_required

        place = self.cube_in_bin() & ~self.cube_grasped()
        check = self.stage1_success & ~self.stage2_success
        self._consecutive_place_steps[place & check] += 1
        self._consecutive_place_steps[~place & check] = 0
        self.stage2_success |= self._consecutive_place_steps >= self._place_required

    def get_task_success(self):
        return self.stage1_success & self.stage2_success

    def get_current_task_info(self) -> Dict[str, Any]:
        cube = self.scene["cube"]
        return {
            "cube_position": cube.data.root_pos_w.torch[0].cpu().numpy().tolist(),
            "cube_orientation": cube.data.root_quat_w.torch[0].cpu().numpy().tolist(),
            "conveyor_speed": self.conveyor_speed,
            "bin_position": self.bin_center()[0].cpu().numpy().tolist(),
            "total_objs": 1,
            "episode_count": self._episode_count,
            "intermediate_success": {"pick": bool(self.stage1_success[0]), "place": bool(self.stage2_success[0]),
                                     "task_success": bool(self.stage1_success[0] and self.stage2_success[0])},
        }

    # ------------------------------------------------------------------ loop
    def step(self, actions):
        self._drive_conveyor()
        pre_step_success = self.get_task_success().clone()
        obs, reward, terminated, truncated, info = super().step(actions)
        if hasattr(self, "reset_buf") and torch.any(self.reset_buf):
            self.reset_success_check(self.reset_buf.nonzero(as_tuple=False).squeeze(-1))
        self.update_stage_success()
        current = self.get_task_success()
        info["stage1_success"] = self.stage1_success
        info["stage2_success"] = self.stage2_success
        info["success"] = torch.where(terminated, pre_step_success, current)
        info["q_score"] = torch.where(terminated | truncated, pre_step_success.float(), current.float())
        return obs, reward, terminated, truncated, info

    def reset(self, warm_up=True, seed=None, env_ids=None, options=None):
        self.reset_success_check(env_ids)
        return super().reset(warm_up=warm_up, seed=seed, env_ids=env_ids, options=options)


gym.register(
    id="ConveyorCubePick-v0",
    entry_point="yamlab.envs.tasks.conveyor_cube_pick_manager:make_conveyor_cube_pick_env",
)
