# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path
import math
from math import ceil, sqrt

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.sim import SimulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors.ray_caster import patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
import isaaclab.terrains as terrain_gen
from isaaclab.sensors.frame_transformer import FrameTransformerCfg
from isaaclab.sensors import ContactSensorCfg



BITTLE_ASSET_DIR = Path(__file__).resolve().parent


@configclass
class BittleEnvCfg(DirectRLEnvCfg):
    # ====== ENV / TIMING ======
    decimation = 2
    episode_length_s = 10
    action_space = 8
    observation_space = 53
    state_space = 38
    action_scale = 3

    # ====== SIMULATION ======
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        device="cuda",
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,
        env_spacing=10.0,
        replicate_physics=True,
        filter_collisions=True,
    )

    # ====== ROBOT ======
    bittle = ArticulationCfg(
        prim_path="/World/envs/env_.*/bittle",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{BITTLE_ASSET_DIR}/assets/Bittle_URDF/bittle/bittle.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
            joint_pos={
                "left_back_shoulder_joint" : 0.0, 
                "left_front_shoulder_joint" : 0.0, 
                "right_back_shoulder_joint" : 0.0, 
                "right_front_shoulder_joint" : 0.0, 
                "left_back_knee_joint" : 0.0, 
                "left_front_knee_joint" : 0.0, 
                "right_back_knee_joint" : 0.0, 
                "right_front_knee_joint" : 0.0
            },
            joint_vel={".*": 0.0},
        ),
        actuators={
            "body_pd": ImplicitActuatorCfg(
                joint_names_expr=["left_.*", "right_.*"],
                effort_limit_sim=120.0,
                velocity_limit_sim=20.0,
                stiffness=60.0,
                damping=3.0,
            ),
        },
    )
    robot_cfg: ArticulationCfg = bittle

    # ====== FRAME TRANSFORMER (for feet) ======
    # lf_rf_transformer: FrameTransformerCfg = FrameTransformerCfg(
    #     prim_path="/World/envs/env_.*/bittle/pelvis",
    #     target_frames=[
    #         FrameTransformerCfg.FrameCfg(
    #             prim_path="/World/envs/env_.*/bittle/leftFoot",
    #             name="leftFoot"
    #         ),
    #         FrameTransformerCfg.FrameCfg(
    #             prim_path="/World/envs/env_.*/bittle/rightFoot",
    #             name="rightFoot"
    #         ),
    #     ],
    #     update_period=0.0,
    #     history_length=1,
    #     debug_vis=False,
    # )

    # ====== REWARD WEIGHTS (from GymWrapper) ======
    rew_upright_gain = 5.0          # upright_bonus
    rew_smooth_gain = 4.0           # smooth_bonus
    rew_posture_pen = 2.0           # posture_penalty
    rew_jerk_pen = 0.5              # jerk_penalty
    rew_velocity_pen = 0.2          # velocity_penalty
    rew_z_pen = 3.0                 # z_penalty
    rew_distance_pen = 8.0          # distance-to-goal penalty
    rew_goal_align_bonus = 2.0      # alignment bonus
    rew_goal_arrival_bonus = 20.0   # at-goal reward (constant)
    rew_tipping_pen = 5.0           # tipping penalty (constant)
    rew_recovery_bonus = 2.0        # recovering bonus (constant)
    
    min_base_height = 0.4
    start_z_height = 1
    spawn_yaw_range = 3

    # ====== RAY CASTER (pelvis → ground) ======
    # ray_caster: RayCasterCfg = RayCasterCfg(
    #     prim_path="/World/envs/env_.*/bittle/torso",
    #     update_period=1 / 60,
    #     offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, -0.05)),
    #     mesh_prim_paths=["/World/Ground"],
    #     max_distance=2.0,
    #     pattern_cfg=patterns.GridPatternCfg(
    #         size=(0.0, 0.0),
    #         resolution=1,
    #         direction=(0, 0.0, -1.0),
    #     ),
    #     debug_vis=True,
    # )

    # ====== TERRAIN (generator) ======
    def __post_init__(self):
        super().__post_init__()
        N = int(self.scene.num_envs)
        S = float(self.scene.env_spacing)
        rows = int(ceil(sqrt(N)))
        cols = int(ceil(N / rows))

        generator = TerrainGeneratorCfg(
            size=(S, S),
            border_width=0.1,
            border_height=-1.0,
            num_rows=rows,
            num_cols=cols,
            color_scheme="height",
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            use_cache=False,
            sub_terrains={
                "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.1),
                "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
                    proportion=0.2, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
                ),
                "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
                    proportion=0.2, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
                ),
                "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
                    proportion=0.05, step_height_range=(0.0, 0.1), step_width=0.3,
                    platform_width=3.0, border_width=1.0, holes=False
                ),
                "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
                    proportion=0.05, step_height_range=(0.0, 0.1), step_width=0.3,
                    platform_width=3.0, border_width=1.0, holes=False
                ),
                "wave_terrain": terrain_gen.HfWaveTerrainCfg(
                    proportion=0.3, amplitude_range=(0.0, 0.2), num_waves=4, border_width=0.25
                ),
                "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
                    proportion=0.3, noise_range=(0.0, 0.06), noise_step=0.02, border_width=0.25
                ),
            },
        )

        # self.scene.terrain = TerrainImporterCfg(
        #     prim_path="/World/Ground",
        #     terrain_type="generator",
        #     terrain_generator=generator,
        #     max_init_terrain_level=0,
        #     collision_group=-1,
        #     physics_material=sim_utils.RigidBodyMaterialCfg(
        #         friction_combine_mode="multiply",
        #         restitution_combine_mode="multiply",
        #         static_friction=1.0,
        #         dynamic_friction=1.0,
        #     ),
        #     debug_vis=False,
        # )
