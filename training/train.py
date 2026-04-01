"""
Train COLREGs MaskablePPO agent with pacSTL monitoring.

The monitoring radius gates pacSTL evaluation and action masking so that
the optimizer is only invoked when the encounter vessel is close enough
for the robustness calculation to be meaningful and well-conditioned.
"""

import argparse
import os
from sb3_contrib import MaskablePPO
from pacstl.core.factory import create as create_spec
from pacstl.domains.colregs.utils import USV_DEFAULT

from colregs_gym import ColregsGym
from utils.reachable_sets import preload_reachable_sets, select_reachable_set
from training.callbacks import ColregsMonitorCallback


def make_env(dt: float = 0.5, monitoring_radius: float = None,
             monitoring_radius_safety_factor: float = 2.0,
             robustness_margin: float = 2.0) -> ColregsGym:
    """Create a COLREGs encounter environment with pacSTL monitoring.

    Parameters
    ----------
    dt : float
        Simulation time step.
    monitoring_radius : float or None
        Explicit monitoring radius in meters. If None (recommended), the
        radius is auto-computed from vessel speeds and maneuver horizon:
            radius = safety_factor * (v_max + v_target) * (T_maneuver + T_pred)
    monitoring_radius_safety_factor : float
        Multiplier for the auto-computed radius (default 2.0).
        Increase if you see encounters being detected too late.
        Decrease if the optimizer is still struggling (unlikely with
        reasonable encounter geometries).
    robustness_margin : float
        Actions with robustness upper bound above -margin are masked out.
        Higher values = more conservative (earlier intervention, more actions
        masked). Lower values = agent has more freedom but risks running
        out of compliant actions. Recommended range: 1.0 - 3.0.
    """
    env = ColregsGym(
        dt=dt,
        vessel_model=USV_DEFAULT,
        grid_width=25,
        grid_height=30,
        monitoring_radius=monitoring_radius,
        monitoring_radius_safety_factor=monitoring_radius_safety_factor,
        robustness_margin=robustness_margin,
    )

    env.set_encounter(
        start_position=(1.5, 7.5, 0.0),
        wave_conditions=(0.05, 1.5, 0),
        encounter_type="crossing",
        separation=6,
        target_speed=0.2,
        goal_ahead_distance=25.0,
        collision_radius=1.0,
        simtime=250.0,
    )

    spec = create_spec("colregs", "crossing_detection")
    ellipsoids = preload_reachable_sets()
    tube = select_reachable_set(ellipsoids, env.encounter_speed)

    # configure_monitoring will refine the auto-computed radius using the
    # actual prediction horizon from the ellipsoid time keys
    env.configure_monitoring(spec, tube, sampling_rate=5)

    return env


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train COLREGs MaskablePPO agent")
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument("--checkpoint-dir", default="checkpoints/ppo_mask")
    parser.add_argument("--plot-dir", default="plots")
    parser.add_argument("--plot-every", type=int, default=1)
    parser.add_argument("--monitoring-radius", type=float, default=None,
                        help="Override monitoring radius (meters). "
                             "If unset, auto-computed from speeds and maneuver horizon.")
    parser.add_argument("--monitoring-safety-factor", type=float, default=2.0,
                        help="Safety factor for auto-computed monitoring radius.")
    parser.add_argument("--robustness-margin", type=float, default=2.0,
                        help="Robustness margin for action masking. Actions with "
                             "upper bound > -margin are masked. Range: 1.0-3.0.")

    args = parser.parse_args()

    model_path = f"{args.checkpoint_dir}/colregs_maskable_ppo.zip"

    env = make_env(
        dt=args.dt,
        monitoring_radius=args.monitoring_radius,
        monitoring_radius_safety_factor=args.monitoring_safety_factor,
        robustness_margin=args.robustness_margin,
    )
    eval_env = make_env(
        dt=args.dt,
        monitoring_radius=args.monitoring_radius,
        monitoring_radius_safety_factor=args.monitoring_safety_factor,
        robustness_margin=args.robustness_margin,
    )

    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        model = MaskablePPO.load(model_path, env=env)
    else:
        model = MaskablePPO(
            "MlpPolicy",
            env,
            verbose=0,  
            tensorboard_log="./logs/colregs_ppo/",
            policy_kwargs=dict(net_arch=[128, 128]),
            n_steps=128,
            batch_size=32,
            learning_rate=3e-4,
        )

    callback = ColregsMonitorCallback(
        eval_env=eval_env,
        plot_dir=args.plot_dir,
        plot_every_episodes=args.plot_every,
    )

    model.learn(total_timesteps=args.timesteps, callback=callback)
    model.save(f"{args.checkpoint_dir}/colregs_maskable_ppo")