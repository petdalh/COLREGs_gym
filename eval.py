import argparse
import csv
import json
import math
import numbers
import os
import random
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from types import SimpleNamespace
from pathlib import Path

import yaml


REWARD_KEYS = [
    "reward_acceleration",
    "reward_reverse_driving",
    "reward_termination",
    "reward_velocity",
    "reward_goal_distance",
    "reward_lateral_deviation",
    "reward_wrong_side_crossing",
    "reward_safe_distance",
    "reward_reference_tracking",
    "reward_fallback",
]

MASK_DIAGNOSTIC_KEYS = [
    "masking/effective_depth",
    "masking/candidate_actions",
    "masking/search_safe_actions",
    "masking/certified_actions",
    "masking/best_robustness",
    "masking/search_time_ms",
    "masking/nodes_evaluated",
    "masking/nodes_pruned",
    "masking/certified_before_full_depth",
    "masking/allowed_certified_depth_mean",
    "masking/fallback_action",
    "masking/fallback_action_yaw_deg_s",
    "masking/fallback_allowed_all",
    "masking/min_distance_interval",
    "masking/maneuver_verified_lower_min",
    "masking/maneuver_verified_lower_max",
    "masking/maneuver_verified_upper_min",
    "masking/maneuver_verified_upper_max",
    "masking/occupancy_clear_lower_min",
    "masking/occupancy_clear_lower_max",
    "masking/occupancy_clear_upper_min",
    "masking/occupancy_clear_upper_max",
    "masking/collision_possible_lower_min",
    "masking/collision_possible_lower_max",
    "masking/collision_possible_upper_min",
    "masking/collision_possible_upper_max",
]

CONTROL_KEYS = [
    "control/heading_deg",
    "control/heading_cmd_deg",
    "control/heading_error_deg",
    "control/surge_velocity",
    "control/surge_cmd",
    "control/speed_error",
    "control/yaw_rate_cmd_deg_s",
    "control/surge_accel_cmd",
    "control/sway_velocity",
    "control/yaw_rate_deg_s",
    "control/tau_surge",
    "control/tau_yaw",
]

SUMMARY_METRICS = [
    "collision_rate",
    "goal_rate",
    "timeout_rate",
    "wrong_side_crossing_rate",
    "fallback_rate",
    "avg_episode_reward",
]


@dataclass(frozen=True)
class EvalTarget:
    agent: str
    seed: int
    config_path: Path
    model_path: Path


@dataclass
class RunningMean:
    total: float = 0.0
    count: int = 0

    def add(self, value):
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return
        if not math.isfinite(numeric_value):
            return
        self.total += numeric_value
        self.count += 1

    def mean(self):
        if self.count == 0:
            return None
        return self.total / self.count


@dataclass
class EpisodeBuilder:
    reward: float = 0.0
    length: int = 0
    wrong_side_crossing: bool = False
    fallback_steps: int = 0
    encounter_steps: int = 0
    reward_sums: dict = field(
        default_factory=lambda: {key: 0.0 for key in REWARD_KEYS}
    )
    reward_counts: dict = field(
        default_factory=lambda: {key: 0 for key in REWARD_KEYS}
    )

    def step(self, reward, info):
        self.reward += float(reward)
        self.length += 1

        if info.get("encounter_active"):
            self.encounter_steps += 1
            if info.get("mask_fallback"):
                self.fallback_steps += 1

        for key in REWARD_KEYS:
            value = info.get(key)
            if value is None:
                continue
            numeric_value = float(value)
            if not math.isfinite(numeric_value):
                continue
            self.reward_sums[key] += numeric_value
            self.reward_counts[key] += 1
            if key == "reward_wrong_side_crossing" and not math.isclose(
                numeric_value, 0.0
            ):
                self.wrong_side_crossing = True

    def finish(self, reason, episode_index):
        if reason is None:
            reason = "goal_reached"
        return {
            "episode_index": int(episode_index),
            "reason": str(reason),
            "reward": float(self.reward),
            "length": int(self.length),
            "wrong_side_crossing": bool(self.wrong_side_crossing),
            "fallback_steps": int(self.fallback_steps),
            "encounter_steps": int(self.encounter_steps),
            "reward_sums": dict(self.reward_sums),
            "reward_counts": dict(self.reward_counts),
        }


@dataclass
class EvalAccumulator:
    target: EvalTarget
    eval_steps_target: int
    total_steps: int = 0
    episodes: list = field(default_factory=list)
    mask_allowed: RunningMean = field(default_factory=RunningMean)
    mask_allowed_fraction: RunningMean = field(default_factory=RunningMean)
    fallback_steps: int = 0
    encounter_steps: int = 0
    mask_diagnostics: dict = field(
        default_factory=lambda: {key: RunningMean() for key in MASK_DIAGNOSTIC_KEYS}
    )
    control_values: dict = field(
        default_factory=lambda: {key: RunningMean() for key in CONTROL_KEYS}
    )
    total_actions: int | None = None

    def observe_step(self, reward, info):
        self.total_steps += 1

        for key in CONTROL_KEYS:
            value = info.get(key)
            if value is None:
                continue
            if "error" in key:
                value = abs(float(value))
            self.control_values[key].add(value)

        if not info.get("encounter_active"):
            return

        self.encounter_steps += 1
        if info.get("mask_fallback"):
            self.fallback_steps += 1

        allowed_count = info.get("mask_allowed_count")
        if allowed_count is not None:
            self.mask_allowed.add(allowed_count)
            if self.total_actions:
                self.mask_allowed_fraction.add(float(allowed_count) / self.total_actions)

        for key in MASK_DIAGNOSTIC_KEYS:
            self.mask_diagnostics[key].add(info.get(key))

    def add_episode(self, episode):
        self.episodes.append(episode)

    def checkpoint_metrics(self):
        episode_count = len(self.episodes)
        reasons = [episode["reason"] for episode in self.episodes]

        metrics = {
            "agent": self.target.agent,
            "seed": self.target.seed,
            "config_path": str(self.target.config_path),
            "model_path": str(self.target.model_path),
            "eval_steps_target": int(self.eval_steps_target),
            "eval_steps_completed": int(self.total_steps),
            "episode_count": int(episode_count),
            "collision_count": sum(reason == "collision" for reason in reasons),
            "goal_count": sum(reason == "goal_reached" for reason in reasons),
            "timeout_count": sum(reason == "time_limit" for reason in reasons),
            "wrong_side_crossing_count": sum(
                episode["wrong_side_crossing"] for episode in self.episodes
            ),
            "fallback_steps": int(self.fallback_steps),
            "encounter_steps": int(self.encounter_steps),
        }

        metrics.update(
            {
                "collision_rate": _rate(metrics["collision_count"], episode_count),
                "goal_rate": _rate(metrics["goal_count"], episode_count),
                "timeout_rate": _rate(metrics["timeout_count"], episode_count),
                "wrong_side_crossing_rate": _rate(
                    metrics["wrong_side_crossing_count"], episode_count
                ),
                "fallback_rate": _rate(self.fallback_steps, self.encounter_steps),
                "avg_episode_reward": _mean(
                    episode["reward"] for episode in self.episodes
                ),
                "avg_episode_length": _mean(
                    episode["length"] for episode in self.episodes
                ),
                "masking_allowed_actions": self.mask_allowed.mean(),
                "masking_allowed_action_fraction": self.mask_allowed_fraction.mean(),
            }
        )

        for key in REWARD_KEYS:
            sums = [episode["reward_sums"][key] for episode in self.episodes]
            means = []
            for episode in self.episodes:
                count = episode["reward_counts"][key]
                if count:
                    means.append(episode["reward_sums"][key] / count)
            metrics[f"reward_component_sums_{key}"] = _mean(sums)
            metrics[f"reward_components_{key}"] = _mean(means)

        for key, running_mean in self.mask_diagnostics.items():
            metrics[_column_name(key)] = running_mean.mean()
        for key, running_mean in self.control_values.items():
            metrics[_column_name(key)] = running_mean.mean()

        return metrics


def _rate(numerator, denominator):
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _mean(values):
    values = [float(value) for value in values if value is not None]
    if not values:
        return None
    return float(statistics.fmean(values))


def _column_name(key):
    return key.replace("/", "_")


def _format_csv_value(value):
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def load_manifest(path):
    manifest_path = Path(path).expanduser()
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Eval manifest does not exist: {manifest_path}. "
            "Create one from configuration/eval_manifest.example.yaml."
        )

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = yaml.safe_load(f) or {}

    entries = manifest.get("checkpoints", manifest)
    if not isinstance(entries, list):
        raise ValueError(
            "Eval manifest must contain a list under 'checkpoints' or be a list."
        )

    base_dir = manifest_path.parent
    targets = []
    seen = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"Manifest entry {index} must be a mapping.")

        missing = [
            key
            for key in ("agent", "seed", "config_path", "model_path")
            if key not in entry
        ]
        if missing:
            raise ValueError(
                f"Manifest entry {index} is missing required field(s): "
                f"{', '.join(missing)}"
            )

        agent = str(entry["agent"])
        seed = int(entry["seed"])
        identity = (agent, seed)
        if identity in seen:
            raise ValueError(
                f"Duplicate manifest entry for agent={agent!r}, seed={seed}."
            )
        seen.add(identity)

        config_path = _resolve_manifest_path(base_dir, entry["config_path"])
        model_path = _resolve_manifest_path(base_dir, entry["model_path"])
        if not config_path.exists():
            raise FileNotFoundError(
                f"Config path for agent={agent}, seed={seed} does not exist: "
                f"{config_path}"
            )
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model path for agent={agent}, seed={seed} does not exist: "
                f"{model_path}"
            )

        targets.append(
            EvalTarget(
                agent=agent,
                seed=seed,
                config_path=config_path,
                model_path=model_path,
            )
        )

    if not targets:
        raise ValueError("Eval manifest contains no checkpoints.")

    return targets


def load_eval_config(path):
    config_path = Path(path).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"Eval config does not exist: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    if not isinstance(config, dict):
        raise ValueError("Eval config must be a YAML mapping.")
    return config


def resolve_eval_args(args):
    config = load_eval_config(args.config) if args.config else {}

    manifest = _arg_or_config(args.manifest, config, "manifest")
    if manifest is None:
        manifest = args.config if args.config and "checkpoints" in config else None
    if manifest is None:
        manifest = "configuration/eval_manifest.yaml"

    eval_steps = _arg_or_config(args.eval_steps, config, "eval_steps")
    if eval_steps is None:
        raise ValueError(
            "eval_steps must be set in the eval config or with --eval-steps."
        )
    eval_steps = int(eval_steps)

    eval_seed_base = _arg_or_config(args.eval_seed_base, config, "eval_seed_base")
    if eval_seed_base is None:
        eval_seed_base = 1_000_000

    return SimpleNamespace(
        config=args.config,
        manifest=manifest,
        eval_steps=eval_steps,
        output_dir=_arg_or_config(args.output_dir, config, "output_dir"),
        eval_seed_base=int(eval_seed_base),
        progress_log_interval=int(
            _arg_or_config(args.progress_log_interval, config, "progress_log_interval", 500)
        ),
        disable_monitoring=bool(
            _arg_or_config(args.disable_monitoring, config, "disable_monitoring", False)
        ),
        wandb=bool(_arg_or_config(args.wandb, config, "wandb", False)),
        wandb_project=_arg_or_config(
            args.wandb_project, config, "wandb_project", "colregs-ppo"
        ),
        wandb_run_name=_arg_or_config(args.wandb_run_name, config, "wandb_run_name"),
        wandb_group=_arg_or_config(args.wandb_group, config, "wandb_group"),
    )


def _arg_or_config(arg_value, config, key, default=None):
    if arg_value is not None:
        return arg_value
    return config.get(key, default)


def _resolve_manifest_path(base_dir, value):
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    repo_relative = Path.cwd() / path
    if repo_relative.exists():
        return repo_relative.resolve()
    return (base_dir / path).resolve()


def set_eval_seed(seed):
    random.seed(seed)
    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None:
        np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_target(
    target,
    eval_steps,
    eval_seed_base,
    disable_monitoring=False,
    progress_log_interval=500,
    progress_callback=None,
):
    from sb3_contrib import MaskablePPO

    from gym.utils.config import load_config
    from train import make_env

    config = load_config(str(target.config_path))
    set_eval_seed(eval_seed_base + target.seed)

    env = make_env(
        config,
        enable_monitoring=not disable_monitoring,
        enable_episode_logging=False,
        seed=eval_seed_base + target.seed,
    )
    try:
        model = MaskablePPO.load(str(target.model_path), env=env)
        if hasattr(model, "policy") and hasattr(model.policy, "set_training_mode"):
            model.policy.set_training_mode(False)

        accumulator = EvalAccumulator(
            target=target,
            eval_steps_target=eval_steps,
            total_actions=getattr(env.action_space, "n", None),
        )
        _run_rollout(
            model,
            env,
            accumulator,
            eval_steps,
            eval_seed_base,
            progress_log_interval=progress_log_interval,
            progress_callback=progress_callback,
        )
        return accumulator
    finally:
        env.close()


def _run_rollout(
    model,
    env,
    accumulator,
    eval_steps,
    eval_seed_base,
    progress_log_interval=500,
    progress_callback=None,
):
    try:
        import torch
    except ImportError:
        torch = None

    episode_index = 0
    obs, _ = env.reset(seed=_episode_seed(eval_seed_base, accumulator.target.seed, 0))
    episode = EpisodeBuilder()
    start_time = time.monotonic()
    next_progress_step = max(int(progress_log_interval), 1)

    while accumulator.total_steps < eval_steps or episode.length > 0:
        masks = env.action_masks() if hasattr(env, "action_masks") else None
        if torch is None:
            action, _ = model.predict(obs, action_masks=masks, deterministic=True)
        else:
            with torch.no_grad():
                action, _ = model.predict(obs, action_masks=masks, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(_scalar_action(action))
        info = info or {}
        episode.step(reward, info)
        accumulator.observe_step(reward, info)
        if (
            progress_callback is not None
            and accumulator.total_steps >= next_progress_step
        ):
            progress_callback(
                accumulator=accumulator,
                elapsed_s=time.monotonic() - start_time,
            )
            while next_progress_step <= accumulator.total_steps:
                next_progress_step += max(int(progress_log_interval), 1)

        if not (terminated or truncated):
            continue

        accumulator.add_episode(
            episode.finish(info.get("reason"), episode_index=episode_index)
        )
        episode_index += 1
        if accumulator.total_steps >= eval_steps:
            break
        obs, _ = env.reset(
            seed=_episode_seed(eval_seed_base, accumulator.target.seed, episode_index)
        )
        episode = EpisodeBuilder()


def _episode_seed(eval_seed_base, training_seed, episode_index):
    return int(eval_seed_base) + int(training_seed) * 100000 + int(episode_index)


def _scalar_action(action):
    if isinstance(action, numbers.Integral):
        return int(action)
    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None:
        action_array = np.asarray(action)
        if action_array.size == 1:
            return int(action_array.item())
    if hasattr(action, "item"):
        return int(action.item())
    if isinstance(action, (list, tuple)) and len(action) == 1:
        return int(action[0])
    return action


def summarize_by_agent(checkpoint_rows):
    numeric_keys = sorted(
        {
            key
            for row in checkpoint_rows
            for key, value in row.items()
            if isinstance(value, numbers.Real)
            and math.isfinite(float(value))
            and key != "seed"
        }
    )
    rows = []
    grouped = defaultdict(list)
    for row in checkpoint_rows:
        grouped[row["agent"]].append(row)

    for agent in sorted(grouped):
        agent_rows = grouped[agent]
        for key in numeric_keys:
            values = [
                float(row[key])
                for row in agent_rows
                if row.get(key) is not None and math.isfinite(float(row[key]))
            ]
            if not values:
                continue
            rows.append(
                {
                    "agent": agent,
                    "metric": key,
                    "mean": float(statistics.fmean(values)),
                    "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
                    "n": len(values),
                }
            )
    return rows


def write_outputs(output_dir, checkpoint_rows, summary_rows, episode_rows):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    checkpoint_csv = output_path / "per_checkpoint.csv"
    summary_csv = output_path / "per_agent_summary.csv"
    episodes_jsonl = output_path / "raw_episodes.jsonl"
    summary_md = output_path / "summary.md"

    _write_csv(checkpoint_csv, checkpoint_rows)
    _write_csv(summary_csv, summary_rows)
    with open(episodes_jsonl, "w", encoding="utf-8") as f:
        for row in episode_rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    _write_summary_markdown(summary_md, summary_rows)

    return {
        "per_checkpoint": checkpoint_csv,
        "per_agent_summary": summary_csv,
        "raw_episodes": episodes_jsonl,
        "summary": summary_md,
    }


def _write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    preferred = [
        "agent",
        "seed",
        "metric",
        "mean",
        "std",
        "n",
        "config_path",
        "model_path",
    ]
    fieldnames = [key for key in preferred if key in fieldnames] + [
        key for key in fieldnames if key not in preferred
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format_csv_value(row.get(key)) for key in fieldnames})


def _write_summary_markdown(path, summary_rows):
    by_agent_metric = {
        (row["agent"], row["metric"]): row for row in summary_rows
    }
    agents = sorted({row["agent"] for row in summary_rows})
    lines = [
        "# COLREGs Evaluation Summary",
        "",
        "| Agent | Collision | Goal | Timeout | Wrong-side | Fallback | Avg reward |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for agent in agents:
        values = [
            _format_mean_std(by_agent_metric.get((agent, metric)))
            for metric in SUMMARY_METRICS
        ]
        lines.append(f"| {agent} | " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_mean_std(row):
    if row is None:
        return ""
    return f"{row['mean']:.4g} +/- {row['std']:.4g}"


def start_wandb_run(args, targets):
    if not args.wandb:
        return None, None
    import wandb

    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        group=args.wandb_group,
        config={
            "manifest": str(Path(args.manifest).resolve()),
            "eval_steps": args.eval_steps,
            "eval_seed_base": args.eval_seed_base,
            "progress_log_interval": args.progress_log_interval,
            "disable_monitoring": args.disable_monitoring,
            "checkpoint_count": len(targets),
            "checkpoints": [
                {
                    "agent": target.agent,
                    "seed": target.seed,
                    "config_path": str(target.config_path),
                    "model_path": str(target.model_path),
                }
                for target in targets
            ],
        },
        job_type="eval",
    )
    wandb.log({"eval/started": 1, "eval/completed_checkpoints": 0})
    return wandb, run


def log_wandb_checkpoint(wandb_module, checkpoint_row, checkpoint_index):
    if wandb_module is None:
        return
    wandb_module.log(
        {
            "eval/completed_checkpoints": checkpoint_index + 1,
            **_wandb_checkpoint_scalars(checkpoint_row),
        },
        step=checkpoint_index + 1,
    )


def log_progress(wandb_module, target, checkpoint_index, total_checkpoints):
    def _log(accumulator, elapsed_s):
        completed_steps = min(accumulator.total_steps, accumulator.eval_steps_target)
        progress = _rate(completed_steps, accumulator.eval_steps_target)
        eta_s = _eta_seconds(elapsed_s, progress)
        print(
            "Progress "
            f"agent={target.agent} seed={target.seed} "
            f"checkpoint={checkpoint_index + 1}/{total_checkpoints} "
            f"steps={accumulator.total_steps}/{accumulator.eval_steps_target} "
            f"({progress * 100:.1f}%) "
            f"elapsed={_format_duration(elapsed_s)} "
            f"eta={_format_duration(eta_s)}",
            flush=True,
        )
        if wandb_module is not None:
            wandb_module.log(
                {
                    "eval/current_checkpoint_index": checkpoint_index + 1,
                    "eval/current_checkpoint_steps": accumulator.total_steps,
                    "eval/current_checkpoint_progress": progress,
                    "eval/current_checkpoint_elapsed_s": elapsed_s,
                    "eval/current_checkpoint_eta_s": eta_s,
                }
            )

    return _log


def finish_wandb_run(wandb_module, run, output_files, checkpoint_rows, summary_rows):
    if wandb_module is None or run is None:
        return
    try:
        wandb_module.log(
            {
                "eval/completed": 1,
                "per_checkpoint": _wandb_table(wandb_module, checkpoint_rows),
                "per_agent_summary": _wandb_table(wandb_module, summary_rows),
            }
        )
        artifact = wandb_module.Artifact(
            name=f"{run.id}_eval_outputs",
            type="eval-results",
            metadata={"eval_steps": run.config.get("eval_steps")},
        )
        for path in output_files.values():
            artifact.add_file(str(path), name=path.name)
        run.log_artifact(artifact)
    finally:
        run.finish()


def abort_wandb_run(wandb_module, run):
    if wandb_module is None or run is None:
        return
    try:
        wandb_module.log({"eval/failed": 1})
    finally:
        run.finish(exit_code=1)


def _wandb_checkpoint_scalars(row):
    prefix = (
        f"checkpoint/{_sanitize_wandb_key_part(row['agent'])}"
        f"_seed_{row['seed']}"
    )
    return {
        f"{prefix}/{key}": float(value)
        for key, value in row.items()
        if key != "seed"
        and isinstance(value, numbers.Real)
        and math.isfinite(float(value))
    }


def _sanitize_wandb_key_part(value):
    text = str(value)
    sanitized = "".join(
        char if char.isalnum() or char in ("-", "_", ".") else "_"
        for char in text
    )
    return sanitized.strip("_") or "value"


def _eta_seconds(elapsed_s, progress):
    if progress <= 0.0:
        return None
    return max(elapsed_s * (1.0 - progress) / progress, 0.0)


def _format_duration(seconds):
    if seconds is None or not math.isfinite(float(seconds)):
        return "unknown"
    seconds = int(round(float(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def _wandb_table(wandb_module, rows):
    if not rows:
        return wandb_module.Table(columns=[])
    columns = sorted({key for row in rows for key in row})
    return wandb_module.Table(
        columns=columns,
        data=[
            [_format_csv_value(row.get(column)) for column in columns]
            for row in rows
        ],
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate frozen COLREGs MaskablePPO checkpoints."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="YAML eval config containing manifest, eval_steps, W&B, and output settings.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="YAML manifest mapping agent/seed to config and checkpoint paths.",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=None,
        help="Minimum number of policy decision steps per checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for CSV/JSONL/Markdown outputs.",
    )
    parser.add_argument(
        "--eval-seed-base",
        type=int,
        default=None,
        help="Base seed used for deterministic evaluation scenario resets.",
    )
    parser.add_argument(
        "--progress-log-interval",
        type=int,
        default=None,
        help="Print and log W&B progress every N eval steps per checkpoint.",
    )
    parser.add_argument(
        "--disable-monitoring",
        action="store_true",
        default=None,
        help="Skip pacSTL monitoring setup, matching train.py's debug option.",
    )
    parser.add_argument(
        "--enable-monitoring",
        dest="disable_monitoring",
        action="store_false",
        help="Enable pacSTL monitoring even if the eval config disables it.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        default=None,
        help="Log evaluation outputs to a separate W&B eval run.",
    )
    parser.add_argument(
        "--no-wandb",
        dest="wandb",
        action="store_false",
        help="Disable W&B logging even if the eval config enables it.",
    )
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-group", default=None)
    return parser.parse_args()


def main():
    args = resolve_eval_args(parse_args())
    if args.eval_steps < 1:
        raise ValueError("--eval-steps must be >= 1")

    targets = load_manifest(args.manifest)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or os.path.join("eval_runs", timestamp)
    wandb_module, wandb_run = start_wandb_run(args, targets)

    checkpoint_rows = []
    episode_rows = []
    try:
        for checkpoint_index, target in enumerate(targets):
            print(
                f"Evaluating agent={target.agent} seed={target.seed} "
                f"model={target.model_path}"
            )
            accumulator = run_target(
                target,
                eval_steps=args.eval_steps,
                eval_seed_base=args.eval_seed_base,
                disable_monitoring=args.disable_monitoring,
                progress_log_interval=args.progress_log_interval,
                progress_callback=log_progress(
                    wandb_module,
                    target,
                    checkpoint_index,
                    len(targets),
                ),
            )
            checkpoint_row = accumulator.checkpoint_metrics()
            checkpoint_rows.append(checkpoint_row)
            log_wandb_checkpoint(wandb_module, checkpoint_row, checkpoint_index)
            for episode in accumulator.episodes:
                episode_rows.append(
                    {
                        "agent": target.agent,
                        "seed": target.seed,
                        "config_path": str(target.config_path),
                        "model_path": str(target.model_path),
                        **episode,
                    }
                )

        summary_rows = summarize_by_agent(checkpoint_rows)
        output_files = write_outputs(
            output_dir,
            checkpoint_rows,
            summary_rows,
            episode_rows,
        )
        finish_wandb_run(
            wandb_module,
            wandb_run,
            output_files,
            checkpoint_rows,
            summary_rows,
        )
    except Exception:
        abort_wandb_run(wandb_module, wandb_run)
        raise

    print(f"Wrote evaluation outputs to {Path(output_dir).resolve()}")
    for name, path in output_files.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
