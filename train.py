import argparse
import shutil
from pathlib import Path

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from gym.colregs_gym import COLREGsGym
from gym.callback import ColregsMonitorCallback
from gym.utils.config import load_config

try:
    from pacstl.core.factory import create as create_spec
    from pacstl.domains.colregs.utils import USV_DEFAULT
except ImportError as exc:
    raise ImportError(
        "This training script requires pacstl to be installed and available."
    ) from exc


def configure_monitoring(env, monitoring_cfg):
    try:
        from gym.utils.reachable_sets import preload_reachable_sets, select_reachable_set
    except ImportError:
        return False

    spec = create_spec("colregs", "crossing_detection")
    ellipsoids = preload_reachable_sets()
    tube = select_reachable_set(ellipsoids, env.encounter_scenario.target_speed)
    env.configure_monitoring(
        spec,
        tube,
        sampling_rate=monitoring_cfg.get("robustness_sampling_rate", 5),
    )
    return True


def make_env(config, enable_monitoring):
    env_cfg = config.get("environment_configuration", {})
    encounter_cfg = dict(env_cfg.get("encounter_configuration", {}))
    monitoring_cfg = env_cfg.get("monitoring_configuration", {})

    env = COLREGsGym(
        vessel_model=USV_DEFAULT,
        config=config,
    )

    encounter_cfg.pop("maneuver_horizon", None)
    env.set_encounter(**encounter_cfg)

    if enable_monitoring:
        monitoring_enabled = configure_monitoring(env, monitoring_cfg)
        if not monitoring_enabled:
            print(
                "Monitoring helpers were not available, training will continue "
                "without pacSTL masking."
            )

    return env


def build_model(env, checkpoint_path: Path, train_cfg):
    if checkpoint_path.exists():
        print(f"Loading model from {checkpoint_path}")
        return MaskablePPO.load(str(checkpoint_path), env=env)

    if train_cfg.get("algorithm", "PPO") != "PPO":
        raise NotImplementedError("Only PPO is currently supported.")

    policy_net_arch = train_cfg.get("policy_net_arch", [128, 128])
    return MaskablePPO(
        train_cfg.get("policy", "MlpPolicy"),
        env,
        verbose=1,
        tensorboard_log=train_cfg.get("tensorboard_log"),
        policy_kwargs=dict(net_arch=policy_net_arch),
        n_steps=train_cfg.get("n_steps", 128),
        batch_size=train_cfg.get("batch_size", 32),
        learning_rate=train_cfg.get("learning_rate", 3e-4),
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train COLREGs MaskablePPO agent")
    parser.add_argument("--config", default="configuration/config.yaml")
    parser.add_argument(
        "--disable-monitoring",
        action="store_true",
        help="Skip pacSTL monitoring setup and train with unit masks.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Override training timesteps from config.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    train_cfg = config.get("training_configuration", {})

    checkpoint_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints/ppo_mask"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_log = train_cfg.get("tensorboard_log", "logs/colregs_ppo")
    Path(tensorboard_log).mkdir(parents=True, exist_ok=True)

    model_path = checkpoint_dir / "colregs_maskable_ppo_100000_steps.zip"

    if train_cfg.get("remove_existing_logging", False):
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
        shutil.rmtree(tensorboard_log, ignore_errors=True)
        shutil.rmtree(train_cfg.get("plot_dir", "plots"), ignore_errors=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        Path(tensorboard_log).mkdir(parents=True, exist_ok=True)

    env = make_env(config, enable_monitoring=not args.disable_monitoring)
    env = Monitor(env)

    eval_env = make_env(config, enable_monitoring=not args.disable_monitoring)

    train_cfg = dict(train_cfg)
    train_cfg["tensorboard_log"] = tensorboard_log
    model = build_model(env, model_path, train_cfg)

    checkpoint_callback = CheckpointCallback(
        save_freq=train_cfg.get("checkpoint_freq", 10000),
        save_path=str(checkpoint_dir),
        name_prefix="colregs_maskable_ppo",
    )
    monitor_callback = ColregsMonitorCallback(
        eval_env=eval_env,
        plot_dir=train_cfg.get("plot_dir", "plots"),
        plot_every_episodes=train_cfg.get("plot_every_episodes", 1),
    )
    callback = CallbackList([checkpoint_callback, monitor_callback])

    model.learn(
        total_timesteps=args.timesteps or train_cfg.get("training_timesteps", 100000),
        callback=callback,
        reset_num_timesteps=False,
    )

    model.save(str(model_path))
    env.close()


if __name__ == "__main__":
    main()
