# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform
from pxr import UsdGeom, UsdPhysics, PhysxSchema, UsdShade, UsdLux, Gf, Sdf

from .bittle_adil_env_cfg import BittleEnvCfg


class BittleEnv(DirectRLEnv):
    cfg: BittleEnvCfg

    def __init__(self, cfg: BittleEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # robot state handles
        self.joint_ids = list(range(len(self.robot.data.joint_pos)))
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        self._actuated_ids: torch.Tensor | None = None

        # per-env goal points (xyz)
        self.goal_points = torch.zeros((self.scene.num_envs, 3), device=self.device)
        self.sample_goals(first_time=True)

        # reward bookkeeping
        self.prev_actions = torch.zeros(
            (self.scene.num_envs, len(self.joint_pos)), device=self.device
        )
        self.prev_distance = torch.zeros(self.scene.num_envs, device=self.device)
        self.was_tipped_last = torch.zeros(self.scene.num_envs, dtype=torch.bool, device=self.device)

        # spawn state tensors (persistent across resets)
        self.spawn_root_states = self.robot.data.default_root_state.clone()
        self.spawn_joint_pos = self.robot.data.default_joint_pos.clone()
        self.spawn_joint_vel = self.robot.data.default_joint_vel.clone()

        self.first_reset = True

    def set_log_dir(self, log_dir: str):
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir=log_dir)
        self.global_step = 0

    # === Scene setup ===
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)

        # ground
        spawn_ground_plane(prim_path="/World/Ground", cfg=GroundPlaneCfg())

        # multi-env replication
        self.scene.clone_environments(copy_from_source=False)

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        self.scene.articulations["robot"] = self.robot  # standardized name

        # lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.8, 0.8, 0.8))
        light_cfg.func("/World/Light", light_cfg)

    # === Point Sampler ===
    def sample_points(self, env_ids: torch.Tensor, z_offset: float = 2) -> torch.Tensor:
        origins = self.scene.env_origins[env_ids]
        half_size = self.cfg.scene.env_spacing / 2.0
        margin = 0.5

        x_min = origins[:, 0] - half_size + margin
        x_max = origins[:, 0] + half_size - margin
        y_min = origins[:, 1] - half_size + margin
        y_max = origins[:, 1] + half_size - margin

        x = sample_uniform(x_min, x_max, (len(env_ids),), device=self.device)
        y = sample_uniform(y_min, y_max, (len(env_ids),), device=self.device)
        z = origins[:, 2] + z_offset

        return torch.stack([x, y, z], dim=-1)

    # === Goal Sampling ===
    def sample_goals(self, env_ids: torch.Tensor | None = None, first_time=False):
        if env_ids is None:
            env_ids = torch.arange(self.scene.num_envs, device=self.device)

        self.goal_points[env_ids] = self.sample_points(env_ids, z_offset=0.5)
        # self._spawn_goal_markers(env_ids)

    def _spawn_goal_markers(self, env_ids: torch.Tensor):
        for i in env_ids.tolist():
            prim_path = f"/World/goals/goal_{i}"
            pos = self.goal_points[i].cpu().numpy().tolist()

            # Create sphere marker config
            sphere_cfg = sim_utils.SphereCfg(
                radius=0.1,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=[0.0, 1.0, 0.0]),
                rigid_props=None,
            )

            goal_cfg = RigidObjectCfg(prim_path=prim_path, spawn=sphere_cfg)

            if prim_path in self.scene.rigid_objects:
                self.scene.rigid_objects[prim_path].set_world_pose(pos)
            else:
                goal = RigidObject(cfg=goal_cfg)
                self.scene.rigid_objects.add(goal)
                goal.set_world_pose(pos)

    # === RL loop hooks ===
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = torch.clamp(actions.to(self.device), -1.0, 1.0)

    def _apply_action(self) -> None:
        targets = self.actions * float(self.cfg.action_scale)
        self.robot.set_joint_position_target(targets, joint_ids=self._actuated_ids)

    def _get_observations(self) -> dict:

        pos = self.robot.data.root_link_pos_w
        quat = self.robot.data.root_link_quat_w
        roll, pitch = self._extract_roll_pitch()
        yaw = torch.atan2(
            2 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]),
            1 - 2 * (quat[:, 2] ** 2 + quat[:, 3] ** 2),
        )
        obs = torch.cat(
            (
                pos,
                roll.unsqueeze(-1),
                pitch.unsqueeze(-1),
                yaw.unsqueeze(-1),
                self.joint_pos,
                self.joint_vel,
            ),
            dim=-1,
        )

        # print(f"pos: {pos[0]}")

        return {"policy": obs, "rnd_state": obs}

    def _get_rewards(self) -> torch.Tensor:
        
        pos = self.robot.data.root_link_pos_w
        quat = self.robot.data.root_link_quat_w
        roll, pitch = self._extract_roll_pitch()
        yaw = torch.atan2(
            2 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]),
            1 - 2 * (quat[:, 2] ** 2 + quat[:, 3] ** 2),
        )

        delta = torch.abs(self.actions - self.prev_actions)
        self.prev_actions = self.actions.clone()

        dist_to_goal = torch.norm(self.goal_points[:, :2] - pos[:, :2], dim=-1)
        self.prev_distance = dist_to_goal

        upright_bonus = torch.clamp(1.5 - (torch.abs(roll) + torch.abs(pitch)), min=0.0, max=1.5)
        smooth_bonus = torch.exp(-torch.norm(delta, dim=-1))
        posture_penalty = (torch.clamp(torch.abs(roll) - 0.2, min=0.0) ** 2 +
                           torch.clamp(torch.abs(pitch) - 0.2, min=0.0) ** 2)
        jerk_penalty = torch.norm(delta, dim=-1)
        velocity_penalty = torch.sum(torch.tanh(torch.abs(self.joint_vel) / 100), dim=-1)
        z_penalty = torch.clamp(-0.2 - pos[:, 2], min=0.0)

        goal_vec = self.goal_points[:, :2] - pos[:, :2]
        robot_forward = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)
        goal_alignment_bonus = torch.relu(
            torch.sum(goal_vec * robot_forward, dim=-1) /
            (torch.norm(goal_vec, dim=-1) + 1e-6)
        )

        at_goal = (dist_to_goal < 0.1) & (torch.abs(roll) < 0.3) & (torch.abs(pitch) < 0.3)
        goal_arrival_bonus = torch.where(
            at_goal, torch.tensor(20.0, device=self.device), torch.tensor(0.0, device=self.device)
        )

        is_tipped = (torch.abs(roll) > 0.8) | (torch.abs(pitch) > 0.8)
        tipping_penalty = torch.where(
            is_tipped, torch.tensor(5.0, device=self.device), torch.tensor(0.0, device=self.device)
        )

        recovering_bonus = torch.where(
            self.was_tipped_last & ~is_tipped,
            torch.tensor(2.0, device=self.device),
            torch.tensor(0.0, device=self.device),
        )
        self.was_tipped_last = is_tipped

        reward = (
            self.cfg.rew_upright_gain * upright_bonus +
            self.cfg.rew_smooth_gain * smooth_bonus -
            self.cfg.rew_posture_pen * posture_penalty -
            self.cfg.rew_jerk_pen * jerk_penalty -
            self.cfg.rew_velocity_pen * velocity_penalty -
            self.cfg.rew_z_pen * z_penalty -
            self.cfg.rew_distance_pen * dist_to_goal +
            self.cfg.rew_goal_align_bonus * goal_alignment_bonus +
            goal_arrival_bonus -
            tipping_penalty +
            recovering_bonus
        )
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        pos_z = self.robot.data.root_link_pos_w[:, 2]
        fallen = pos_z < self.cfg.min_base_height
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        pos = self.robot.data.root_link_pos_w
        roll, pitch = self._extract_roll_pitch()
        dist = torch.norm(self.goal_points[:, :2] - pos[:, :2], dim=-1)
        success = (dist < 0.1) & (torch.abs(roll) < 0.3) & (torch.abs(pitch) < 0.3)

        if success.any():
            self.sample_goals(env_ids=torch.nonzero(success).squeeze(-1))

        return fallen | success, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):

        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # Check which envs succeeded
        pos = self.robot.data.root_link_pos_w[env_ids]
        roll, pitch = self._extract_roll_pitch()
        dist = torch.norm(self.goal_points[env_ids, :2] - pos[:, :2], dim=-1)
        success = (dist < 0.1) & (torch.abs(roll[env_ids]) < 0.3) & (torch.abs(pitch[env_ids]) < 0.3)
        succ_ids = env_ids[success]

        if self.first_reset or len(succ_ids) > 0:
            
            if self.first_reset:
                succ_ids = env_ids
                self.first_reset = False
            # Pull defaults for successful envs
            joint_pos = self.robot.data.default_joint_pos[succ_ids]
            joint_vel = self.robot.data.default_joint_vel[succ_ids]
            root_state = self.robot.data.default_root_state[succ_ids].clone()

            # Sample XY from training ground and enforce configured Z height
            spawn_points = self.sample_points(succ_ids, z_offset=0.0)
            spawn_points[:, 2] = self.cfg.start_z_height
            root_state[:, 0:3] = spawn_points

            # Randomize yaw
            yaw = sample_uniform(
                -self.cfg.spawn_yaw_range, self.cfg.spawn_yaw_range,
                (len(succ_ids), 1), device=root_state.device
            )
            root_state[:, 3:7] = torch.cat(
                [
                    torch.zeros((len(succ_ids), 2), device=self.device),
                    torch.sin(yaw / 2), torch.cos(yaw / 2)
                ],
                dim=-1
            )

            # Save updated spawn state for next reset
            self.spawn_root_states[succ_ids] = root_state
            self.spawn_joint_pos[succ_ids] = joint_pos
            self.spawn_joint_vel[succ_ids] = joint_vel

        # Reapply cached spawn state (works for both success & fail cases)
        root_state = self.spawn_root_states[env_ids]
        joint_pos = self.spawn_joint_pos[env_ids]
        joint_vel = self.spawn_joint_vel[env_ids]

        self.robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


    def _extract_roll_pitch(self):
        quat = self.robot.data.root_link_quat_w
        roll = torch.atan2(
            2 * (quat[:, 0] * quat[:, 1] + quat[:, 2] * quat[:, 3]),
            1 - 2 * (quat[:, 1] ** 2 + quat[:, 2] ** 2),
        )
        pitch = torch.asin(2 * (quat[:, 0] * quat[:, 2] - quat[:, 3] * quat[:, 1]))
        return roll, pitch
