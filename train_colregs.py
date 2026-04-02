import argparse
import os
from pathlib import Path

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from gym.colregs_gym import COLREGsGym

try:
    from pacstl.core.factory import create as create_spec
    from pacstl.domains.colregs.utils import USV_DEFAULT
except ImportError as exc:
    raise ImportError(
        "This training script requires pacstl to be installed and available."
    ) from exc


def configure_monitoring(env, sampling_rate):
    try:
        from utils.reachable_sets import preload_reachable_sets, select_reachable_set
    except ImportError:
        return False

    spec = create_spec("colregs", "crossing_detection")
    ellipsoids = preload_reachable_sets()
    tube = select_reachable_set(ellipsoids, env.encounter_scenario.target_speed)
    env.configure_monitoring(spec, tube, sampling_rate=sampling_rate)
    return True


def make_env(
    dt: float,
    grid_width: float,
    grid_height: float,
    monitoring_radius: float | None,
    monitoring_radius_safety_factor: float,
    robustness_margin: float,
    sampling_rate: int,
    enable_monitoring: bool,
):
    env = COLREGsGym(
        dt=dt,
        vessel_model=USV_DEFAULT,
        grid_width=grid_width,
        grid_height=grid_height,
        monitoring_radius=monitoring_radius,
        monitoring_radius_safety_factor=monitoring_radius_safety_factor,
        robustness_margin=robustness_margin,
    )

    env.set_encounter(
        start_position=(1.5, 7.5, 0.0),
        wave_conditions=(0.05, 1.5, 0.0),
        encounter_type="crossing",
        separation=6.0,
        target_speed=0.2,
        goal_ahead_distance=25.0,
        collision_radius=1.0,
        simtime=250.0,
    )

    if enable_monitoring:
        monitoring_enabled = configure_monitoring(env, sampling_rate)
        if not monitoring_enabled:
            print(
                "Monitoring helpers were not available, training will continue "
                "without pacSTL masking."
            )

    return Monitor(env)


def build_model(env, checkpoint_path: Path, tensorboard_log: str | None):
    if checkpoint_path.exists():
        print(f"Loading model from {checkpoint_path}")
        return MaskablePPO.load(str(checkpoint_path), env=env)

    return MaskablePPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=tensorboard_log,
        policy_kwargs=dict(net_arch=[128, 128]),
        n_steps=128,
        batch_size=32,
        learning_rate=3e-4,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train COLREGs MaskablePPO agent")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument("--grid-width", type=float, default=25.0)
    parser.add_argument("--grid-height", type=float, default=30.0)
    parser.add_argument("--checkpoint-dir", default="checkpoints/ppo_mask")
    parser.add_argument("--checkpoint-freq", type=int, default=10_000)
    parser.add_argument("--tensorboard-log", default="logs/colregs_ppo")
    parser.add_argument("--monitoring-radius", type=float, default=None)
    parser.add_argument("--monitoring-safety-factor", type=float, default=2.0)
    parser.add_argument("--robustness-margin", type=float, default=2.0)
    parser.add_argument("--sampling-rate", type=int, default=5)
    parser.add_argument(
        "--disable-monitoring",
        action="store_true",
        help="Skip pacSTL monitoring setup and train with unit masks.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    Path(args.tensorboard_log).mkdir(parents=True, exist_ok=True)

    model_path = checkpoint_dir / "colregs_maskable_ppo.zip"

    env = make_env(
        dt=args.dt,
        grid_width=args.grid_width,
        grid_height=args.grid_height,
        monitoring_radius=args.monitoring_radius,
        monitoring_radius_safety_factor=args.monitoring_safety_factor,
        robustness_margin=args.robustness_margin,
        sampling_rate=args.sampling_rate,
        enable_monitoring=not args.disable_monitoring,
    )

    model = build_model(env, model_path, args.tensorboard_log)

    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=str(checkpoint_dir),
        name_prefix="colregs_maskable_ppo",
    )

    model.learn(
        total_timesteps=args.timesteps,
        callback=checkpoint_callback,
        reset_num_timesteps=False,
    )

    model.save(str(model_path))
    env.close()


if __name__ == "__main__":
    main()
