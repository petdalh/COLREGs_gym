import collections
import csv
import json
import os

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

import wandb

from gym.callback.plotting import (
    DEFAULT_CONTROL_SURGE_EMA_ALPHA,
    plot_control_timeseries,
    plot_episode_trajectory,
    plot_robustness,
    plot_surge_command_fraction,
)


class ColregsMonitorCallback(BaseCallback):
    _DEFAULT_PLOT_SMOOTHING = {
        "trajectory_window": 1,
        "speed_profile_window": 1,
        "robustness_window": 1,
        "control_heading_window": 1,
        "control_heading_error_window": 1,
        "control_surge_ema_alpha": DEFAULT_CONTROL_SURGE_EMA_ALPHA,
        "control_surge_command_window": 1,
        "control_actuator_window": 1,
        "training_step_reward_window": 200,
        "training_episode_reward_window": 20,
        "training_episode_length_window": 20,
        "training_event_rate_window": 20,
    }

    def __init__(
        self,
        eval_env,
        plot_dir="plots",
        plot_every_episodes=50,
        metrics_log_path=None,
        plot_smoothing=None,
    ):
        super().__init__()
        self.eval_env = getattr(eval_env, "unwrapped", eval_env)
        self.plot_dir = plot_dir
        self.metrics_log_path = metrics_log_path
        self._metrics_file = None
        self._metrics_writer = None
        self.plot_every = int(plot_every_episodes)
        if self.plot_every < 0:
            raise ValueError("plot_every_episodes must be >= 0")
        self.next_plot_episode = self.plot_every if self.plot_every > 0 else None
        self.plot_smoothing = {
            **self._DEFAULT_PLOT_SMOOTHING,
            **(plot_smoothing or {}),
        }

        self.step_rewards = []
        self.step_timesteps = []
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_reasons = []
        self.episode_count = 0

        self._REWARD_KEYS = [
            "reward_acceleration",
            "reward_reverse_driving",
            "reward_termination",
            "reward_velocity",
            "reward_goal_distance",
            "reward_lateral_deviation",
            "reward_wrong_side_crossing",
            "reward_safe_distance",
            "reward_reference_tracking", 
            "reward_fallback"
        ]
        self._rolling_reward_means = {
            k: collections.deque(maxlen=10) for k in self._REWARD_KEYS
        }

        self._rolling_mask_allowed = collections.deque(maxlen=10)
        self._rolling_mask_allowed_fraction = collections.deque(maxlen=10)
        self._rolling_fallback_rate = collections.deque(maxlen=10)
        self._MASK_DIAGNOSTIC_KEYS = [
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
        self._rolling_mask_diagnostic_means = {
            k: collections.deque(maxlen=10) for k in self._MASK_DIAGNOSTIC_KEYS
        }

        self._CONTROL_KEYS = [
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
        self._rolling_control_means = {
            k: collections.deque(maxlen=10) for k in self._CONTROL_KEYS
        }

        self._n_envs = 0
        self._ep_reward_sums = []
        self._ep_reward_steps = []
        self._ep_mask_allowed = []
        self._ep_mask_allowed_fraction = []
        self._ep_fallback_flags = []
        self._ep_mask_diagnostic_values = []
        self._ep_control_sums = []
        self._ep_control_steps = []
        self._ep_wrong_side_crossing = []
        self._total_actions = self._resolve_action_count(self.eval_env)
        self._cumulative_collisions = 0
        self._cumulative_goals = 0
        self._cumulative_timeouts = 0
        self._cumulative_wrong_side_crossings = 0

        self._metrics_fieldnames = [
            "timesteps",
            "episode_count",
            "avg_episode_reward",
            "avg_episode_length",
            "goal_rate",
            "collision_rate",
            "timeout_rate",
            *[f"reward_components_{k}" for k in self._REWARD_KEYS],
            "masking_allowed_actions",
            "masking_allowed_action_fraction",
            "masking_fallback_rate",
            *[self._metric_column_name(k) for k in self._MASK_DIAGNOSTIC_KEYS],
            "events_collisions_cumulative",
            "events_goals_cumulative",
            "events_timeouts_cumulative",
            "events_wrong_side_crossings_cumulative",
            *[self._metric_column_name(k) for k in self._CONTROL_KEYS],
        ]

        os.makedirs(plot_dir, exist_ok=True)

    @staticmethod
    def _metric_column_name(key):
        return key.replace("/", "_")

    @staticmethod
    def _format_metric_value(value):
        if value is None:
            return ""
        if isinstance(value, float) and not np.isfinite(value):
            return ""
        return value

    def _smoothing_window(self, key):
        return max(
            int(self.plot_smoothing.get(key, self._DEFAULT_PLOT_SMOOTHING[key])),
            1,
        )

    def _smoothing_float(self, key):
        return float(self.plot_smoothing.get(key, self._DEFAULT_PLOT_SMOOTHING[key]))

    def _write_metrics_row(self, metrics):
        if not self.metrics_log_path:
            return

        if self._metrics_writer is None:
            os.makedirs(os.path.dirname(self.metrics_log_path), exist_ok=True)
            self._metrics_file = open(self.metrics_log_path, "w", newline="")
            self._metrics_writer = csv.DictWriter(
                self._metrics_file,
                fieldnames=self._metrics_fieldnames,
                delimiter="\t",
                extrasaction="ignore",
            )
            self._metrics_writer.writeheader()

        row = {
            self._metric_column_name(key): self._format_metric_value(value)
            for key, value in metrics.items()
        }
        row["timesteps"] = int(self.num_timesteps)
        self._metrics_writer.writerow(row)
        self._metrics_file.flush()

    @staticmethod
    def _json_safe(value):
        if isinstance(value, dict):
            return {
                str(k): ColregsMonitorCallback._json_safe(v)
                for k, v in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [ColregsMonitorCallback._json_safe(v) for v in value]
        if isinstance(value, np.ndarray):
            return ColregsMonitorCallback._json_safe(value.tolist())
        if isinstance(value, np.generic):
            return ColregsMonitorCallback._json_safe(value.item())
        if isinstance(value, float):
            return value if np.isfinite(value) else None
        return value

    @staticmethod
    def _wandb_artifact_name(value):
        return "".join(
            char if char.isalnum() or char in ("-", "_", ".") else "_"
            for char in str(value)
        ).strip("_")

    def _build_action_plot_data(self, actions):
        vessel_action = self.eval_env.vessel_action
        n_accel = len(vessel_action.surge_accel_commands)
        action_indices = [int(action) for action in actions]
        return {
            "action_indices": action_indices,
            "surge_accel_cmd": [
                float(vessel_action.surge_accel_commands[action % n_accel])
                for action in action_indices
            ],
            "yaw_rate_cmd_deg_s": [
                float(vessel_action.yaw_rate_commands_deg_s[action // n_accel])
                for action in action_indices
            ],
        }

    def _build_eval_episode_plot_data(self, actions, info, episode_reward):
        if hasattr(self.eval_env, "_build_episode_plot_data"):
            plot_data = self.eval_env._build_episode_plot_data()
        else:
            plot_data = {
                "ego_traj": [list(p) for p in self.eval_env.history_ego],
                "enc_traj": [list(p) for p in self.eval_env.history_enc],
                "goal": list(self.eval_env.goal[:2]),
                "dt": float(self.eval_env.dt),
                "robustness_dt": float(
                    self.eval_env.dt * getattr(self.eval_env, "decision_interval", 1)
                ),
                "history_surge_command_fraction": [
                    float(v) for v in self.eval_env.history_surge_command_fraction
                ],
            }

        plot_data["actions"] = self._build_action_plot_data(actions)
        plot_data["episode_reward"] = float(episode_reward)
        plot_data["episode_length"] = int(len(actions))
        plot_data["termination_reason"] = info.get("reason")
        return plot_data

    def _write_trajectory_json(self, tag, source, episode_num, plot_data, env_idx=None):
        trajectory_dir = os.path.join(self.plot_dir, "trajectory_data")
        os.makedirs(trajectory_dir, exist_ok=True)
        json_path = os.path.join(trajectory_dir, f"{tag}.json")

        metadata = {
            "tag": tag,
            "source": source,
            "episode_num": int(episode_num),
            "num_timesteps": int(self.num_timesteps),
            "plot_every_episodes": int(self.plot_every),
            "plot_smoothing": self._json_safe(self.plot_smoothing),
        }
        if env_idx is not None:
            metadata["env_idx"] = int(env_idx)

        payload = {
            "metadata": metadata,
            "trajectory": plot_data,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                self._json_safe(payload),
                f,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
        return json_path

    def _log_trajectory_artifact(self, json_path, tag, source, episode_num, env_idx=None):
        if not wandb.run:
            return

        metadata = {
            "tag": tag,
            "source": source,
            "episode_num": int(episode_num),
            "num_timesteps": int(self.num_timesteps),
        }
        if env_idx is not None:
            metadata["env_idx"] = int(env_idx)

        run_id = getattr(wandb.run, "id", "run")
        artifact = wandb.Artifact(
            name=self._wandb_artifact_name(f"{run_id}_{tag}_trajectory_data"),
            type="trajectory-data",
            metadata=metadata,
        )
        artifact.add_file(json_path, name=os.path.basename(json_path))
        wandb.run.log_artifact(artifact)

    def _close_metrics_log(self):
        if self._metrics_file is not None:
            self._metrics_file.close()
            self._metrics_file = None
            self._metrics_writer = None

    @staticmethod
    def _resolve_action_count(env):
        action_space = getattr(env, "action_space", None)
        action_count = getattr(action_space, "n", None)
        if action_count is not None:
            return int(action_count)

        unwrapped = getattr(env, "unwrapped", None)
        if unwrapped is not None and unwrapped is not env:
            return ColregsMonitorCallback._resolve_action_count(unwrapped)

        envs = getattr(env, "envs", None)
        if envs:
            return ColregsMonitorCallback._resolve_action_count(envs[0])

        return None

    def _ensure_env_accumulators(self, n_envs):
        if n_envs == self._n_envs:
            return

        self._n_envs = n_envs
        self._ep_reward_sums = [
            {k: 0.0 for k in self._REWARD_KEYS} for _ in range(n_envs)
        ]
        self._ep_reward_steps = [0 for _ in range(n_envs)]
        self._ep_mask_allowed = [[] for _ in range(n_envs)]
        self._ep_mask_allowed_fraction = [[] for _ in range(n_envs)]
        self._ep_fallback_flags = [[] for _ in range(n_envs)]
        self._ep_mask_diagnostic_values = [
            {k: [] for k in self._MASK_DIAGNOSTIC_KEYS} for _ in range(n_envs)
        ]
        self._ep_control_sums = [
            {k: 0.0 for k in self._CONTROL_KEYS} for _ in range(n_envs)
        ]
        self._ep_control_steps = [0 for _ in range(n_envs)]
        self._ep_wrong_side_crossing = [False for _ in range(n_envs)]

    def _on_training_start(self):
        if self._total_actions is None:
            self._total_actions = self._resolve_action_count(self.training_env)

    def _on_step(self):
        rewards = self.locals.get("rewards")
        if rewards is not None:
            self.step_rewards.append(float(np.mean(rewards)))
            self.step_timesteps.append(int(self.num_timesteps))

        infos = self.locals.get("infos", [])
        if isinstance(infos, dict):
            infos = [infos]
        self._ensure_env_accumulators(len(infos))

        for env_idx, info in enumerate(infos):
            for key in self._REWARD_KEYS:
                val = info.get(key)
                if val is not None:
                    val = float(val)
                    self._ep_reward_sums[env_idx][key] += val
                    if (
                        key == "reward_wrong_side_crossing"
                        and np.isfinite(val)
                        and not np.isclose(val, 0.0)
                    ):
                        self._ep_wrong_side_crossing[env_idx] = True
            self._ep_reward_steps[env_idx] += 1

            for key in self._CONTROL_KEYS:
                val = info.get(key)
                if val is not None:
                    # use abs for error terms so rolling means show magnitude trends
                    self._ep_control_sums[env_idx][key] += (
                        abs(float(val)) if "error" in key else float(val)
                    )
            self._ep_control_steps[env_idx] += 1

            if info.get("encounter_active"):
                allowed_count = int(info["mask_allowed_count"])
                self._ep_mask_allowed[env_idx].append(allowed_count)
                if self._total_actions:
                    self._ep_mask_allowed_fraction[env_idx].append(
                        allowed_count / self._total_actions
                    )
                self._ep_fallback_flags[env_idx].append(int(info["mask_fallback"]))
                for key in self._MASK_DIAGNOSTIC_KEYS:
                    val = info.get(key)
                    if val is not None:
                        val = float(val)
                        if np.isfinite(val):
                            self._ep_mask_diagnostic_values[env_idx][key].append(val)

            if "episode" in info:
                self.episode_count += 1
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])

                reason = info.get("reason", "unknown")
                self.episode_reasons.append(reason)
                if reason == "collision":
                    self._cumulative_collisions += 1
                    self._plot_collision_episode(info, self.episode_count, env_idx)
                elif reason == "goal_reached":
                    self._cumulative_goals += 1
                elif reason == "time_limit":
                    self._cumulative_timeouts += 1

                if self._ep_wrong_side_crossing[env_idx]:
                    self._cumulative_wrong_side_crossings += 1
                self._ep_wrong_side_crossing[env_idx] = False

                if self._ep_reward_steps[env_idx] > 0:
                    for k in self._REWARD_KEYS:
                        self._rolling_reward_means[k].append(
                            self._ep_reward_sums[env_idx][k]
                            / self._ep_reward_steps[env_idx]
                        )
                self._ep_reward_sums[env_idx] = {k: 0.0 for k in self._REWARD_KEYS}
                self._ep_reward_steps[env_idx] = 0

                if self._ep_control_steps[env_idx] > 0:
                    for k in self._CONTROL_KEYS:
                        self._rolling_control_means[k].append(
                            self._ep_control_sums[env_idx][k]
                            / self._ep_control_steps[env_idx]
                        )
                self._ep_control_sums[env_idx] = {k: 0.0 for k in self._CONTROL_KEYS}
                self._ep_control_steps[env_idx] = 0

                if self._ep_mask_allowed[env_idx]:
                    self._rolling_mask_allowed.append(
                        float(np.mean(self._ep_mask_allowed[env_idx]))
                    )
                if self._ep_mask_allowed_fraction[env_idx]:
                    self._rolling_mask_allowed_fraction.append(
                        float(np.mean(self._ep_mask_allowed_fraction[env_idx]))
                    )
                if self._ep_fallback_flags[env_idx]:
                    self._rolling_fallback_rate.append(
                        float(np.mean(self._ep_fallback_flags[env_idx]))
                    )
                for key in self._MASK_DIAGNOSTIC_KEYS:
                    values = self._ep_mask_diagnostic_values[env_idx][key]
                    if values:
                        self._rolling_mask_diagnostic_means[key].append(
                            float(np.mean(values))
                        )
                self._ep_mask_allowed[env_idx] = []
                self._ep_mask_allowed_fraction[env_idx] = []
                self._ep_fallback_flags[env_idx] = []
                self._ep_mask_diagnostic_values[env_idx] = {
                    k: [] for k in self._MASK_DIAGNOSTIC_KEYS
                }

                if self.episode_count % 1 == 0:
                    avg_r = np.mean(self.episode_rewards[-10:])
                    avg_l = np.mean(self.episode_lengths[-10:])
                    recent_reasons = self.episode_reasons[-10:]
                    collisions = sum(1 for r in recent_reasons if r == "collision")
                    goals = sum(1 for r in recent_reasons if r == "goal_reached")
                    timeouts = sum(1 for r in recent_reasons if r == "time_limit")
                    rew_log = {
                        f"reward_components/{k}": float(np.mean(v))
                        for k, v in self._rolling_reward_means.items()
                        if v
                    }
                    masking_log = {}
                    if self._rolling_mask_allowed:
                        masking_log["masking/allowed_actions"] = float(
                            np.mean(self._rolling_mask_allowed)
                        )
                    if self._rolling_mask_allowed_fraction:
                        masking_log["masking/allowed_action_fraction"] = float(
                            np.mean(self._rolling_mask_allowed_fraction)
                        )
                    if self._rolling_fallback_rate:
                        masking_log["masking/fallback_rate"] = float(
                            np.mean(self._rolling_fallback_rate)
                        )
                    masking_log.update(
                        {
                            key: float(np.mean(values))
                            for key, values in (
                                self._rolling_mask_diagnostic_means.items()
                            )
                            if values
                        }
                    )
                    event_log = {
                        "events/collisions_cumulative": self._cumulative_collisions,
                        "events/goals_cumulative": self._cumulative_goals,
                        "events/timeouts_cumulative": self._cumulative_timeouts,
                        "events/wrong_side_crossings_cumulative": (
                            self._cumulative_wrong_side_crossings
                        ),
                    }
                    control_log = {
                        k: float(np.mean(v))
                        for k, v in self._rolling_control_means.items()
                        if v
                    }
                    metrics_log = {
                        "episode_count": self.episode_count,
                        "avg_episode_reward": avg_r,
                        "avg_episode_length": avg_l,
                        "goal_rate": goals / 10,
                        "collision_rate": collisions / 10,
                        "timeout_rate": timeouts / 10,
                        **rew_log,
                        **masking_log,
                        **event_log,
                        **control_log,
                    }
                    self._write_metrics_row(metrics_log)

                    if wandb.run:
                        wandb.log(metrics_log, step=self.num_timesteps)

                while (
                    self.next_plot_episode is not None
                    and self.episode_count >= self.next_plot_episode
                ):
                    self._run_eval_episode(episode_num=self.next_plot_episode)
                    self.next_plot_episode += self.plot_every

        return True

    def _run_eval_episode(self, episode_num=None):
        if episode_num is None:
            episode_num = self.episode_count

        obs, _ = self.eval_env.reset()
        done = False
        ep_actions = []
        ep_reward = 0.0
        info = {}

        while not done:
            masks = self.eval_env.action_masks()
            action, _ = self.model.predict(obs, action_masks=masks, deterministic=True)
            ep_actions.append(int(action))
            obs, reward, terminated, truncated, info = self.eval_env.step(action)
            ep_reward += float(reward)
            done = terminated or truncated

        tag = f"ep{episode_num:05d}"
        step_dt = self.eval_env.dt * getattr(self.eval_env, "decision_interval", 5)

        plot_episode_trajectory(
            ego_traj=self.eval_env.history_ego,
            ego_speed_traj=self.eval_env.history_surge_command_fraction,
            enc_traj=self.eval_env.history_enc,
            goal=self.eval_env.goal[:2],
            dt=self.eval_env.dt,
            episode_num=episode_num,
            collision_radius=self.eval_env.encounter_scenario.collision_radius,
            obstacle_speed_mps=self.eval_env.state.encounter_speed,
            smoothing_window=self._smoothing_window("trajectory_window"),
            save_path=os.path.join(self.plot_dir, f"traj_{tag}.png"),
        )

        plot_surge_command_fraction(
            history_surge_command_fraction=self.eval_env.history_surge_command_fraction,
            dt=self.eval_env.dt,
            smoothing_window=self._smoothing_window("speed_profile_window"),
            save_path=os.path.join(self.plot_dir, f"speed_{tag}.png"),
        )

        plot_robustness(
            ep_robustness=self.eval_env.history_rob,
            dt=step_dt,
            save_path=os.path.join(self.plot_dir, f"rob_{tag}.png"),
            title="Crossing Detection",
            smoothing_window=self._smoothing_window("robustness_window"),
        )

        plot_robustness(
            ep_robustness=self.eval_env.history_maneuver_rob,
            dt=step_dt,
            save_path=os.path.join(self.plot_dir, f"maneuver_rob_{tag}.png"),
            title="Maneuver Spec",
            smoothing_window=self._smoothing_window("robustness_window"),
        )

        plot_control_timeseries(
            history_heading_deg=self.eval_env.history_heading_deg,
            history_heading_cmd_deg=self.eval_env.history_heading_cmd_deg,
            history_heading_error_deg=self.eval_env.history_heading_error_deg,
            history_surge=self.eval_env.history_surge,
            history_surge_cmd=self.eval_env.history_surge_cmd,
            history_tau_surge=self.eval_env.history_tau_surge,
            history_tau_yaw=self.eval_env.history_tau_yaw,
            dt=self.eval_env.dt,
            surge_ema_alpha=self._smoothing_float("control_surge_ema_alpha"),
            heading_smoothing_window=self._smoothing_window("control_heading_window"),
            heading_error_smoothing_window=self._smoothing_window(
                "control_heading_error_window"
            ),
            surge_cmd_smoothing_window=self._smoothing_window(
                "control_surge_command_window"
            ),
            actuator_smoothing_window=self._smoothing_window("control_actuator_window"),
            save_path=os.path.join(self.plot_dir, f"control_{tag}.png"),
        )

        trajectory_data = self._build_eval_episode_plot_data(
            ep_actions,
            info,
            ep_reward,
        )
        trajectory_json_path = self._write_trajectory_json(
            tag=tag,
            source="eval",
            episode_num=episode_num,
            plot_data=trajectory_data,
        )

        if wandb.run:
            self._log_trajectory_artifact(
                trajectory_json_path,
                tag=tag,
                source="eval",
                episode_num=episode_num,
            )
            log_dict = {
                "trajectory": wandb.Image(os.path.join(self.plot_dir, f"traj_{tag}.png")),
                "speed_profile": wandb.Image(os.path.join(self.plot_dir, f"speed_{tag}.png")),
                "robustness/crossing": wandb.Image(os.path.join(self.plot_dir, f"rob_{tag}.png")),
            }
            maneuver_rob_path = os.path.join(self.plot_dir, f"maneuver_rob_{tag}.png")
            if os.path.exists(maneuver_rob_path):
                log_dict["robustness/maneuver"] = wandb.Image(maneuver_rob_path)

            control_plot_path = os.path.join(self.plot_dir, f"control_{tag}.png")
            if os.path.exists(control_plot_path):
                log_dict["control/timeseries"] = wandb.Image(control_plot_path)

            if ep_actions:
                vessel_action = self.eval_env.vessel_action
                n_accel = len(vessel_action.surge_accel_commands)
                accel_choices = [
                    float(vessel_action.surge_accel_commands[a % n_accel])
                    for a in ep_actions
                ]
                yaw_rate_choices = [
                    float(vessel_action.yaw_rate_commands_deg_s[a // n_accel])
                    for a in ep_actions
                ]
                log_dict["actions/surge_accel_cmd"] = wandb.Histogram(accel_choices)
                log_dict["actions/yaw_rate_cmd_deg_s"] = wandb.Histogram(yaw_rate_choices)

            wandb.log(log_dict)

    def _plot_collision_episode(self, info, episode_num, env_idx):
        plot_data = info.get("collision_plot_data")
        if not plot_data:
            return

        ego_traj = plot_data.get("ego_traj", [])
        goal = plot_data.get("goal")
        if not ego_traj or goal is None:
            return

        tag = f"collision_ep{episode_num:05d}_env{env_idx:02d}"
        traj_path = os.path.join(self.plot_dir, f"traj_{tag}.png")
        control_path = os.path.join(self.plot_dir, f"control_{tag}.png")
        rob_path = os.path.join(self.plot_dir, f"rob_{tag}.png")
        maneuver_rob_path = os.path.join(self.plot_dir, f"maneuver_rob_{tag}.png")

        plot_episode_trajectory(
            ego_traj=ego_traj,
            ego_speed_traj=plot_data.get("history_surge_command_fraction"),
            enc_traj=plot_data.get("enc_traj", []),
            goal=goal,
            dt=plot_data.get("dt", 0.5),
            episode_num=episode_num,
            collision_radius=plot_data.get("collision_radius", 0.0),
            obstacle_speed_mps=plot_data.get("obstacle_speed_mps"),
            smoothing_window=self._smoothing_window("trajectory_window"),
            save_path=traj_path,
        )

        plot_control_timeseries(
            history_heading_deg=plot_data.get("history_heading_deg", []),
            history_heading_cmd_deg=plot_data.get("history_heading_cmd_deg", []),
            history_heading_error_deg=plot_data.get("history_heading_error_deg", []),
            history_surge=plot_data.get("history_surge", []),
            history_surge_cmd=plot_data.get("history_surge_cmd", []),
            history_tau_surge=plot_data.get("history_tau_surge", []),
            history_tau_yaw=plot_data.get("history_tau_yaw", []),
            dt=plot_data.get("dt", 0.5),
            surge_ema_alpha=self._smoothing_float("control_surge_ema_alpha"),
            heading_smoothing_window=self._smoothing_window("control_heading_window"),
            heading_error_smoothing_window=self._smoothing_window(
                "control_heading_error_window"
            ),
            surge_cmd_smoothing_window=self._smoothing_window(
                "control_surge_command_window"
            ),
            actuator_smoothing_window=self._smoothing_window("control_actuator_window"),
            save_path=control_path,
        )

        robustness_dt = plot_data.get("robustness_dt", plot_data.get("dt", 0.5))
        history_rob = plot_data.get("history_rob", [])
        if any(rob is not None for rob in history_rob):
            plot_robustness(
                ep_robustness=history_rob,
                dt=robustness_dt,
                save_path=rob_path,
                title="Crossing Detection",
                smoothing_window=self._smoothing_window("robustness_window"),
            )

        history_maneuver_rob = plot_data.get("history_maneuver_rob", [])
        if any(rob is not None for rob in history_maneuver_rob):
            plot_robustness(
                ep_robustness=history_maneuver_rob,
                dt=robustness_dt,
                save_path=maneuver_rob_path,
                title="Maneuver Spec",
                smoothing_window=self._smoothing_window("robustness_window"),
            )

        trajectory_json_path = self._write_trajectory_json(
            tag=tag,
            source="collision",
            episode_num=episode_num,
            plot_data=plot_data,
            env_idx=env_idx,
        )

        if wandb.run:
            self._log_trajectory_artifact(
                trajectory_json_path,
                tag=tag,
                source="collision",
                episode_num=episode_num,
                env_idx=env_idx,
            )
            log_dict = {"collisions/trajectory": wandb.Image(traj_path)}
            if os.path.exists(control_path):
                log_dict["collisions/control"] = wandb.Image(control_path)
            if os.path.exists(rob_path):
                log_dict["collisions/robustness_crossing"] = wandb.Image(rob_path)
            if os.path.exists(maneuver_rob_path):
                log_dict["collisions/robustness_maneuver"] = wandb.Image(
                    maneuver_rob_path
                )
            wandb.log(log_dict, step=self.num_timesteps)

    def _on_training_end(self):
        try:
            self._save_training_curves()
            self._run_eval_episode()
        finally:
            self._close_metrics_log()

    def _save_training_curves(self):
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(6, 1, figsize=(10, 19))

        axes[0].plot(self.step_timesteps, self.step_rewards, alpha=0.15, color="purple")
        step_window = self._smoothing_window("training_step_reward_window")
        if len(self.step_rewards) >= step_window:
            rolling = np.convolve(
                self.step_rewards,
                np.ones(step_window) / step_window,
                mode="valid",
            )
            axes[0].plot(self.step_timesteps[step_window - 1 :], rolling, color="purple")
        axes[0].set_ylabel("Step Reward")
        axes[0].set_xlabel("Training Timesteps")
        axes[0].set_title("Training Progress")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(self.episode_rewards, alpha=0.3, color="blue")
        reward_window = self._smoothing_window("training_episode_reward_window")
        if len(self.episode_rewards) >= reward_window:
            rolling = np.convolve(
                self.episode_rewards,
                np.ones(reward_window) / reward_window,
                mode="valid",
            )
            axes[1].plot(
                range(reward_window - 1, reward_window - 1 + len(rolling)),
                rolling,
                color="blue",
            )
        axes[1].set_ylabel("Episode Reward")
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(self.episode_lengths, alpha=0.3, color="green")
        length_window = self._smoothing_window("training_episode_length_window")
        if len(self.episode_lengths) >= length_window:
            rolling = np.convolve(
                self.episode_lengths,
                np.ones(length_window) / length_window,
                mode="valid",
            )
            axes[2].plot(
                range(length_window - 1, length_window - 1 + len(rolling)),
                rolling,
                color="green",
            )
        axes[2].set_ylabel("Episode Length")
        axes[2].grid(True, alpha=0.3)

        window = self._smoothing_window("training_event_rate_window")
        if len(self.episode_reasons) >= window:
            goals = []
            collisions = []
            timeouts = []
            for i in range(window, len(self.episode_reasons) + 1):
                chunk = self.episode_reasons[i - window : i]
                goals.append(sum(1 for r in chunk if r == "goal_reached") / window)
                collisions.append(sum(1 for r in chunk if r == "collision") / window)
                timeouts.append(sum(1 for r in chunk if r == "time_limit") / window)
            x = range(window, len(self.episode_reasons) + 1)

            axes[3].plot(x, goals, color="#2ecc71")
            axes[3].set_ylabel("Goal Rate")
            axes[3].set_ylim(0, 1)
            axes[3].grid(True, alpha=0.3)

            axes[4].plot(x, collisions, color="#e74c3c")
            axes[4].set_ylabel("Collision Rate")
            axes[4].set_ylim(0, 1)
            axes[4].grid(True, alpha=0.3)

            axes[5].plot(x, timeouts, color="#f39c12")
            axes[5].set_ylabel("Timeout Rate")
            axes[5].set_ylim(0, 1)
            axes[5].grid(True, alpha=0.3)
        else:
            for ax in axes[3:]:
                ax.set_ylabel(f"Rate (last {window})")
                ax.set_ylim(0, 1)
                ax.grid(True, alpha=0.3)

        for ax in axes[1:]:
            ax.set_xlabel("Episode")

        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, "training_curves.png"), dpi=150)
        plt.close()

        if wandb.run:
            wandb.log({"training_curves": wandb.Image(os.path.join(self.plot_dir, "training_curves.png"))})
            
