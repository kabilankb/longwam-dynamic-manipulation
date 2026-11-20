"""Configuration for the ConveyorCubePick task (YAM bimanual + conveyor belt) using ManagerBasedRLEnv.

A 4 cm cube rides a conveyor belt that runs along the table's -Y direction, in front of both arms. The right arm
picks it off the moving belt and drops it into a bin on its right. Everything except the robot is a primitive:
the belt, its side rails and end stop and the bin walls are static colliders (no rigid body), and the cube is a
dynamic cuboid. The belt motion is applied by the task env (it sets the cube's velocity while the cube rests on
the belt), see conveyor_cube_pick_manager.py.
"""

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg, RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils import configclass

from yamlab.configs import pose_range_for
from yamlab.envs.yam_bimanual_scene import YamBimanualSceneCfg
from yamlab.robot.yam import ROBOT
from yamlab.utils.task_logic import check_obj_below_table, compute_sparse_success_reward

from .yam_bimanual_env_cfg import BaseEventsCfg, YamBimanualEnvCfg

TASK_NAME = "ConveyorCubePick-v0"

__all__ = ["ConveyorCubePickSceneCfg", "ConveyorCubePickEventsCfg", "ConveyorCubePickManagerEnvCfg",
           "CONVEYOR", "BIN", "BIN_PARTS", "CUBE_SIZE"]

TABLE_Z = ROBOT.table_position[2]  # 0.7517

#: Conveyor geometry (world frame, metres). The belt runs along Y; the cube travels toward -Y (to the right arm).
CONVEYOR = {
    "x": 0.47,              # belt centreline
    "y_start": 0.45,        # upstream end (left side of the table)
    "y_end": -0.20,         # downstream end, closed by an end stop
    "width": 0.16,          # belt width (X)
    "height": 0.04,         # belt top is TABLE_Z + height
    "rail_height": 0.015,   # side rails above the belt top
    "rail_thickness": 0.01,
    "speed": 0.06,          # m/s along -Y
    "friction": 0.05,       # low, so the cube slides at the commanded speed between control steps
}
CONVEYOR["top_z"] = TABLE_Z + CONVEYOR["height"]
CONVEYOR["length"] = CONVEYOR["y_start"] - CONVEYOR["y_end"]
CONVEYOR["y_center"] = 0.5 * (CONVEYOR["y_start"] + CONVEYOR["y_end"])

#: Bin on the table to the right arm's front-right (inner footprint, wall height).
BIN = {"x": 0.52, "y": -0.36, "inner": 0.14, "wall_height": 0.05, "wall_thickness": 0.01, "floor": 0.006}

CUBE_SIZE = 0.04
CUBE_SPAWN = (CONVEYOR["x"], CONVEYOR["y_start"] - 0.06, CONVEYOR["top_z"] + CUBE_SIZE / 2 + 0.002)


def _static_box(name: str, size, pos, color, friction: float = 0.8) -> AssetBaseCfg:
    """A static collider box (no rigid body): conveyor parts and bin walls."""
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.CuboidCfg(
            size=tuple(size),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=friction, dynamic_friction=friction, friction_combine_mode="min"),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.6),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(pos)),
    )


BIN_PARTS = ("bin_floor", "bin_wall_near", "bin_wall_far", "bin_wall_left", "bin_wall_right")


def _bin_part(name: str, size, local_pos) -> RigidObjectCfg:
    """One bin box as a kinematic rigid body, positioned relative to the nominal bin centre on the table."""
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.CuboidCfg(
            size=tuple(size),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.7),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.35, 0.75), roughness=0.6),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(BIN["x"] + local_pos[0], BIN["y"] + local_pos[1], TABLE_Z + local_pos[2]), rot=(0.0, 0.0, 0.0, 1.0)),
    )


def reset_bin_pose(env, env_ids, pose_range: dict, asset_names=BIN_PARTS):
    """Move all bin parts by one shared random XY offset (rigidly, no rotation)."""
    n = len(env_ids)
    rx, ry = pose_range.get("x", (0.0, 0.0)), pose_range.get("y", (0.0, 0.0))
    off = torch.zeros(n, 3, device=env.device)
    off[:, 0] = torch.empty(n, device=env.device).uniform_(rx[0], rx[1])
    off[:, 1] = torch.empty(n, device=env.device).uniform_(ry[0], ry[1])
    for name in asset_names:
        part = env.scene[name]
        pose = part.data.default_root_pose.torch[env_ids].clone()
        pose[:, :3] += env.scene.env_origins[env_ids] + off
        part.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        part.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=env.device), env_ids=env_ids)


C = CONVEYOR
_RAIL_Z = C["top_z"] + C["rail_height"] / 2 - 0.0001
_RAIL_DX = C["width"] / 2 + C["rail_thickness"] / 2
_B = BIN
_WALL_D = _B["inner"] / 2 + _B["wall_thickness"] / 2
_WALL_LEN = _B["inner"] + 2 * _B["wall_thickness"]


@configclass
class ConveyorCubePickSceneCfg(YamBimanualSceneCfg):
    """Bimanual workstation scene with a conveyor belt, a cube on it and a bin."""

    conveyor_frame = _static_box(
        "ConveyorFrame", (C["width"] + 0.04, C["length"], C["height"] - 0.004),
        (C["x"], C["y_center"], TABLE_Z + (C["height"] - 0.004) / 2), (0.25, 0.25, 0.27))
    conveyor_belt = _static_box(
        "ConveyorBelt", (C["width"], C["length"], 0.004),
        (C["x"], C["y_center"], C["top_z"] - 0.002), (0.05, 0.05, 0.05), friction=C["friction"])
    conveyor_rail_near = _static_box(
        "ConveyorRailNear", (C["rail_thickness"], C["length"], C["rail_height"]),
        (C["x"] - _RAIL_DX, C["y_center"], _RAIL_Z), (0.75, 0.75, 0.78), friction=0.1)
    conveyor_rail_far = _static_box(
        "ConveyorRailFar", (C["rail_thickness"], C["length"], C["rail_height"]),
        (C["x"] + _RAIL_DX, C["y_center"], _RAIL_Z), (0.75, 0.75, 0.78), friction=0.1)
    conveyor_end_stop = _static_box(
        "ConveyorEndStop", (C["width"] + 2 * C["rail_thickness"], 0.012, 0.03),
        (C["x"], C["y_end"] - 0.006, C["top_z"] + 0.015), (0.9, 0.75, 0.1))

    # The bin is five kinematic boxes that the reset event moves together (one random offset per episode).
    bin_floor = _bin_part("BinFloor", (_WALL_LEN, _WALL_LEN, _B["floor"]), (0.0, 0.0, _B["floor"] / 2))
    bin_wall_near = _bin_part("BinWallNear", (_B["wall_thickness"], _WALL_LEN, _B["wall_height"]),
                              (-_WALL_D, 0.0, _B["wall_height"] / 2))
    bin_wall_far = _bin_part("BinWallFar", (_B["wall_thickness"], _WALL_LEN, _B["wall_height"]),
                             (_WALL_D, 0.0, _B["wall_height"] / 2))
    bin_wall_left = _bin_part("BinWallLeft", (_WALL_LEN, _B["wall_thickness"], _B["wall_height"]),
                              (0.0, _WALL_D, _B["wall_height"] / 2))
    bin_wall_right = _bin_part("BinWallRight", (_WALL_LEN, _B["wall_thickness"], _B["wall_height"]),
                               (0.0, -_WALL_D, _B["wall_height"] / 2))

    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cube",
        spawn=sim_utils.CuboidCfg(
            size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=1.0,
                                                         solver_position_iteration_count=16,
                                                         solver_velocity_iteration_count=1),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=0.9),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.08, 0.06), roughness=0.4),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=CUBE_SPAWN, rot=(0.0, 0.0, 0.0, 1.0)),
    )


@configclass
class ConveyorCubePickEventsCfg(BaseEventsCfg):
    """Reset event: the cube appears at the belt's upstream end with a small pose jitter (ranges in the YAML)."""

    reset_bin_position = EventTermCfg(
        func=reset_bin_pose,
        mode="reset",
        params={"pose_range": pose_range_for(TASK_NAME, "bin")},
    )

    reset_cube_position = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": pose_range_for(TASK_NAME, "cube"), "velocity_range": {},
                "asset_cfg": SceneEntityCfg("cube", body_names=".*")},
    )


@configclass
class ConveyorCubePickRewardsCfg:
    sparse_success_reward = RewardTermCfg(func=compute_sparse_success_reward, params={}, weight=1.0)


def check_task_success(env):
    return torch.logical_and(env.stage1_success, env.stage2_success)


@configclass
class ConveyorCubePickTerminationsCfg:
    """Timeout, task success, or the cube falling off the table."""

    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    task_success = TerminationTermCfg(func=check_task_success, params={})
    cube_below_table = TerminationTermCfg(
        func=check_obj_below_table,
        params={"asset_cfg": SceneEntityCfg("cube"), "table_z": TABLE_Z, "margin": 0.01},
    )


@configclass
class ConveyorCubePickManagerEnvCfg(YamBimanualEnvCfg):
    """Full configuration for the ConveyorCubePick ManagerBasedRLEnv (num_envs defaults to 1)."""

    scene: ConveyorCubePickSceneCfg = ConveyorCubePickSceneCfg(num_envs=1, env_spacing=3.0)
    events: ConveyorCubePickEventsCfg = ConveyorCubePickEventsCfg()
    rewards: ConveyorCubePickRewardsCfg = ConveyorCubePickRewardsCfg()
    terminations: ConveyorCubePickTerminationsCfg = ConveyorCubePickTerminationsCfg()

    obj_name_to_event_name: dict[str, str] = {"cube": "reset_cube_position", "bin": "reset_bin_position"}

    def configure_assets_instance_paths(self, assets_instance_paths: dict[str, str]):
        """The cube is a primitive (no USD instance folder): only register its contact-filter prim."""
        self._contact_body_links = {"cube": ""}
        if assets_instance_paths:
            print(f"[WARNING] ConveyorCubePick uses primitives; ignoring --asset {sorted(assets_instance_paths)}")


ConveyorCubePickManagerEnvCfg.CONTACT_OBJECT_NAMES = ["cube"]
