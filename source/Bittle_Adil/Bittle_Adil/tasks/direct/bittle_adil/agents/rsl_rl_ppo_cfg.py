# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
    RslRlPpoActorCriticCfg,
    RslRlRndCfg,
)

@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 16
    max_iterations = 150
    save_interval = 50
    experiment_name = "BittleAdil"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,           # initial Gaussian exploration noise
        noise_std_type="scalar",      # scalar or log std
        actor_hidden_dims=[256, 256], # MLP for actor
        critic_hidden_dims=[256, 256],# MLP for critic
        activation="elu",             # activation (elu, relu, tanh)
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        num_learning_epochs=10,
        num_mini_batches=8,
        learning_rate=1e-4,
        schedule="adaptive",             # could also be "linear" for LR decay
        gamma=0.98,                   # discount factor
        lam=0.95,                     # GAE lambda
        clip_param=0.2,               # PPO clipping
        value_loss_coef=1.0,
        entropy_coef=0.01,            # 🔥 increase entropy for more exploration
        desired_kl=0.02,              # optional KL target for stable updates
        max_grad_norm=1.0,
        use_clipped_value_loss=True,

        # ===============================
        # Intrinsic Curiosity via RND
        # ===============================
        rnd_cfg=RslRlRndCfg(
            weight=0.1,                          # scale of intrinsic reward
            weight_schedule=RslRlRndCfg.LinearWeightScheduleCfg(
                final_value=0.0,                 # fade out curiosity to 0
                initial_step=0,
                final_step=1_000_000,            # after 1M steps, curiosity = 0
            ),
            reward_normalization=True,           # normalize intrinsic reward
            state_normalization=True,            # normalize input states
            learning_rate=1e-4,                  # RND module LR
            predictor_hidden_dims=[128, 128],    # predictor net size
            target_hidden_dims=[128, 128],       # target net size
        )
    )

    runner_cfg = RslRlOnPolicyRunnerCfg(
        seed=42,
        device="cuda",                # or "cpu"
        num_steps_per_env=16,         # steps per env per PPO update
        max_iterations=20000,         # total training iterations
        empirical_normalization=True, # normalize observations
        clip_actions=1.0,             # clip action outputs
        save_interval=200,            # save model every 500 iters
        experiment_name="ucr_humanoid",
        logger="tensorboard",         # or "wandb", "neptune"
        policy=policy,
        algorithm=algorithm,
    )



