import argparse
import os
import shutil
from pathlib import Path

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

import wandb

from gym.colregs_gym import COLREGsGym
from gym.callback import ColregsMonitorCallback
from gym.utils.config import load_config

try:
    from pacstl.core.factory import create as create_spec
    from pacstl.domains.colregs.utils import EGO_VESSEL_DEFAULT, USV_DEFAULT
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
    
    ego_vessel = EGO_VESSEL_DEFAULT
    env.maneuver_spec_factory = lambda T_end, T_start=None, _ev=ego_vessel: create_spec(
        "colregs", "maneuver_verified", T_end=T_end,
        **({"T_start": T_start} if T_start is not None else {}),
        ego_vessel=_ev,
    )
    env.configure_maneuver_monitoring(env.maneuver_spec_factory)
    return True


def make_env(config, enable_monitoring, enable_episode_logging=True):
    env_cfg = config.get("environment_configuration", {})
    encounter_cfg = dict(env_cfg.get("encounter_configuration", {}))
    monitoring_cfg = env_cfg.get("monitoring_configuration", {})

    env = COLREGsGym(
        vessel_model=USV_DEFAULT,
        ego_vessel_model=EGO_VESSEL_DEFAULT,
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

    if not enable_episode_logging:
        env.callback = None

    return env


def _optional_positive_int(value, name):
    if value is None:
        return None
    parsed = int(value)
    if parsed < 1:
        raise ValueError(f"{name} must be >= 1, got {parsed}")
    return parsed


def resolve_num_envs(train_cfg):
    configured = _optional_positive_int(
        train_cfg.get("num_envs", train_cfg.get("num_workers")),
        "training_configuration.num_envs",
    )
    slurm_cpus = _optional_positive_int(
        os.environ.get("SLURM_CPUS_PER_TASK"),
        "SLURM_CPUS_PER_TASK",
    )

    num_envs = configured or slurm_cpus or 1
    if slurm_cpus is not None:
        num_envs = min(num_envs, slurm_cpus)
    return num_envs, slurm_cpus


def configure_cpu_threading(num_envs):
    if num_envs <= 1:
        return

    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(name, "1")

    try:
        import torch
    except ImportError:
        return
    torch.set_num_threads(1)


def make_train_env_fn(config, enable_monitoring, enable_episode_logging):
    def _init():
        return make_env(
            config,
            enable_monitoring=enable_monitoring,
            enable_episode_logging=enable_episode_logging,
        )

    return _init


def build_train_env(config, num_envs, train_cfg, enable_monitoring):
    enable_episode_logging = bool(
        train_cfg.get("print_episode_summary", num_envs == 1)
    )
    env_fns = [
        make_train_env_fn(
            config,
            enable_monitoring=enable_monitoring,
            enable_episode_logging=enable_episode_logging,
        )
        for _ in range(num_envs)
    ]

    if num_envs == 1:
        return VecMonitor(DummyVecEnv(env_fns))

    start_method = train_cfg.get("vec_env_start_method", "forkserver")
    return VecMonitor(SubprocVecEnv(env_fns, start_method=start_method))


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


def build_wandb_config(config, args):
    return {
        **config,
        "run_metadata": {
            "config_path": str(Path(args.config).resolve()),
            "disable_monitoring": args.disable_monitoring,
            "timesteps_override": args.timesteps,
        },
    }


def save_wandb_config_file(run, config_path):
    source_path = Path(config_path).resolve()
    saved_path = Path(run.dir) / "config.yaml"
    shutil.copy2(source_path, saved_path)
    wandb.save(str(saved_path), base_path=run.dir, policy="now")


def main():
    args = parse_args()
    config = load_config(args.config)
    train_cfg = config.get("training_configuration", {})
    num_envs, slurm_cpus = resolve_num_envs(train_cfg)
    configure_cpu_threading(num_envs)

    if slurm_cpus is None:
        print(f"Using {num_envs} training environment(s).")
    else:
        print(
            f"Using {num_envs} training environment(s) "
            f"from SLURM_CPUS_PER_TASK={slurm_cpus}."
        )

    checkpoint_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints/ppo_mask"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_log = train_cfg.get("tensorboard_log", "logs/colregs_ppo")
    Path(tensorboard_log).mkdir(parents=True, exist_ok=True)

    wandb_cfg = config.get("wandb_configuration", {})
    run = None
    if wandb_cfg.get("enabled", False):
        run = wandb.init(
            project=wandb_cfg.get("project", "colregs-ppo"),
            name=wandb_cfg.get("run_name"),
            group=wandb_cfg.get("group"),
            config=build_wandb_config(config, args),
        )
        save_wandb_config_file(run, args.config)

    model_path = checkpoint_dir / "colregs_maskable_ppo.zip"

    if train_cfg.get("remove_existing_logging", False):
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
        shutil.rmtree(tensorboard_log, ignore_errors=True)
        shutil.rmtree(train_cfg.get("plot_dir", "plots"), ignore_errors=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        Path(tensorboard_log).mkdir(parents=True, exist_ok=True)

    env = build_train_env(
        config,
        num_envs=num_envs,
        train_cfg=train_cfg,
        enable_monitoring=not args.disable_monitoring,
    )

    eval_env = make_env(
        config,
        enable_monitoring=not args.disable_monitoring,
        enable_episode_logging=False,
    )

    train_cfg = dict(train_cfg)
    train_cfg["tensorboard_log"] = tensorboard_log
    model = build_model(env, model_path, train_cfg)

    checkpoint_callback = CheckpointCallback(
        save_freq=max(int(train_cfg.get("checkpoint_freq", 10000)) // num_envs, 1),
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
    eval_env.close()

    if run:
        run.finish()


if __name__ == "__main__":
    main()
