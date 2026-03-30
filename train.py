"""
COLREGs RL training with pacSTL monitoring and MaskablePPO.

Usage:
    python train.py --timesteps 3000000 --dt 0.5
"""

import argparse
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from pacstl.core.factory import create as create_spec
from pacstl.domains.colregs.utils import USV_DEFAULT

from colregs_gym import ColregsGym
from reachable_sets import preload_reachable_sets, select_reachable_set


def make_env(dt: float = 0.5) -> ColregsGym:
    env = ColregsGym(dt=dt, vessel_model=USV_DEFAULT, grid_width=25, grid_height=30)
    env.set_encounter(
        start_position=(1.5, 7.5, 0.0),
        wave_conditions=(0.05, 1.5, 0),
        encounter_type="crossing",
        separation=10,
        target_speed=0.06,
        goal_ahead_distance=25.0,
        collision_radius=1.0,
        simtime=150.0,
    )

    spec = create_spec("colregs", "crossing_detection")
    ellipsoids = preload_reachable_sets()
    tube = select_reachable_set(ellipsoids, env.encounter_speed)
    env.configure_monitoring(spec, tube, sampling_rate=1)

    return env


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train COLREGs MaskablePPO agent")
    parser.add_argument("--timesteps", type=int, default=3_000_000)
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument("--checkpoint-dir", default="checkpoints/ppo_mask")
    args = parser.parse_args()

    env = make_env(dt=args.dt)

    model = MaskablePPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./logs/colregs_ppo/",
        policy_kwargs=dict(net_arch=[128, 128]),
        n_steps=2048,
        batch_size=64,
        learning_rate=3e-4,
    )

    model.learn(total_timesteps=args.timesteps)
    model.save(f"{args.checkpoint_dir}/colregs_maskable_ppo")

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20)
    print(f"Evaluation: mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")