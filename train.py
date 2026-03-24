"""
Minimal COLREGs RL training loop with pacSTL monitoring.

Usage:
    python train.py --episodes 10 --dt 0.08
"""

import argparse
from pathlib import Path

import numpy as np
import interval

from mchorcrux.numpy_core.controllers.adaptive_seakeeping import (
    MRACShipController,
    heading_to_goal,
)

from colregs_gym import ColregsGym
from ddpg.agent import Agent
from pacstl.core.factory import create as create_spec
import pacstl.domains
from pacstl.domains.colregs.utils import USV_DEFAULT, VesselModel
from reachable_sets import preload_reachable_sets, select_reachable_set
from plotting import plot_robustness 



# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def decode_action(action: np.ndarray, obs: np.ndarray):
    """Map RL action in [-1,1]^2 to (heading_ref, speed_ref)."""
    a_heading, a_speed = action
    n, e, psi = obs[0], obs[1], obs[2]
    gn, ge = obs[6], obs[7]
    psi_goal = heading_to_goal(n, e, gn, ge)
    psi_d = psi_goal + a_heading * np.deg2rad(45.0)
    u_d = 0.5 * (a_speed + 1.0) * 1.0
    return psi_d, u_d


def shape_terminal_reward(reward: float, reason: str) -> float:
    """Apply sparse terminal bonuses/penalties."""
    bonuses = {"collision": -50.0, "goal_reached": 50.0, "time_limit": -10.0}
    return reward + bonuses.get(reason, 0.0)


# ------------------------------------------------------------------
# Environment factory
# ------------------------------------------------------------------

def make_env(dt: float = 0.08) -> ColregsGym:
    env = ColregsGym(dt=dt, vessel_model=USV_DEFAULT, grid_width=25, grid_height=30)
    env.set_encounter(
        start_position=(1.5, 7.5, 0.0),
        wave_conditions=(0.05, 1.5, 0),
        encounter_type="crossing",
        separation=30,
        target_speed=0.3,
        goal_ahead_distance=25.0,
        collision_radius=1.0,
        simtime=150.0,
    )
    return env


# ------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------

def train(num_episodes: int, dt: float, checkpoint_dir: str):
    # pacSTL setup
    spec = create_spec("colregs", "crossing_rule")
    ellipsoids = preload_reachable_sets()

    env = make_env(dt=dt)
    tube = select_reachable_set(ellipsoids, env.encounter_speed)
    env.configure_monitoring(spec, tube, sampling_rate=1)

    obs = env.reset()

    agent = Agent(
        alpha=0.000025,
        beta=0.00025,
        input_dims=obs.shape[0],
        tau=0.01,
        env=env,
        n_actions=2,
        chkpt_dir=checkpoint_dir,
    )

    ckpt_path = Path(checkpoint_dir)
    if any(ckpt_path.iterdir()):
        print("Loading existing models...")
        agent.load_models()

    scores = []
    robustness_log = []

    for ep in range(num_episodes):
        obs = env.reset()
        controller = MRACShipController(dt=dt)

        done = False
        score = 0.0
        step = 0
        ep_robustness = []
        current_robustness = interval.interval(-np.inf, -np.inf)
        while not done:
            if current_robustness.l > 0:
                print("crossing situation detected")
                action_mask = env.get_action_mask("crossing")
                action = agent.choose_action(action_mask, obs)
            else:
                action = agent.choose_action(obs)
            # action = agent.choose_action(obs)
            psi_d, u_d = decode_action(action, obs)
            tau = controller.compute_action_minimal(env.get_state(), psi_d, u_d)

            next_obs, reward, done, info = env.step(tau)

            # Collect robustness when available
            if "robustness" in info:
                ep_robustness.append(info["robustness"])
                current_robustness = info["robustness"]

            if done:
                reward = shape_terminal_reward(reward, info.get("reason", ""))

            agent.remember(obs, action, reward, next_obs, int(done))
            if step % 10 == 0:
                agent.learn()

            obs = next_obs
            score += reward
            step += 1

        scores.append(score)
        robustness_log.append(ep_robustness)
        avg = np.mean(scores[max(0, len(scores) - 10):])
        print(f"Episode {ep + 1:4d}/{num_episodes}  score={score:+8.2f}  avg10={avg:+8.2f}")
        agent.save_models()

    return scores, robustness_log


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train COLREGs DDPG agent")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--dt", type=float, default=0.08)
    parser.add_argument("--checkpoint-dir", default="checkpoints/ddpg")
    args = parser.parse_args()

    scores, robustness = train(
        num_episodes=args.episodes,
        dt=args.dt,
        checkpoint_dir=args.checkpoint_dir,
    )

    print(robustness)
    #plot_robustness(robustness)

