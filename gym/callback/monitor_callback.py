import collections
import os

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

import wandb

from gym.callback.plotting import (
    plot_control_timeseries,
    plot_episode_trajectory,
    plot_robustness,
    plot_surge_command_fraction,
)


class ColregsMonitorCallback(BaseCallback):
    def __init__(self, eval_env, plot_dir="plots", plot_every_episodes=50):
        super().__init__()
        self.eval_env = getattr(eval_env, "unwrapped", eval_env)
        self.plot_dir = plot_dir
        self.plot_every = int(plot_every_episodes)
        if self.plot_every < 0:
            raise ValueError("plot_every_episodes must be >= 0")
        self.next_plot_episode = self.plot_every if self.plot_every > 0 else None

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
        ]
        self._rolling_reward_means = {
            k: collections.deque(maxlen=10) for k in self._REWARD_KEYS
        }

        self._rolling_mask_allowed = collections.deque(maxlen=10)
        self._rolling_mask_allowed_fraction = collections.deque(maxlen=10)
        self._rolling_fallback_rate = collections.deque(maxlen=10)

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
        self._ep_control_sums = []
        self._ep_control_steps = []
        self._ep_wrong_side_crossing = []
        self._total_actions = self._resolve_action_count(self.eval_env)
        self._cumulative_collisions = 0
        self._cumulative_goals = 0
        self._cumulative_timeouts = 0
        self._cumulative_wrong_side_crossings = 0

        os.makedirs(plot_dir, exist_ok=True)

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

            if "episode" in info:
                self.episode_count += 1
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])

                reason = info.get("reason", "unknown")
                self.episode_reasons.append(reason)
                if reason == "collision":
                    self._cumulative_collisions += 1
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
                self._ep_mask_allowed[env_idx] = []
                self._ep_mask_allowed_fraction[env_idx] = []
                self._ep_fallback_flags[env_idx] = []

                if self.episode_count % 1 == 0:
                    avg_r = np.mean(self.episode_rewards[-10:])
                    avg_l = np.mean(self.episode_lengths[-10:])
                    recent_reasons = self.episode_reasons[-10:]
                    collisions = sum(1 for r in recent_reasons if r == "collision")
                    goals = sum(1 for r in recent_reasons if r == "goal_reached")
                    timeouts = sum(1 for r in recent_reasons if r == "time_limit")
                    if wandb.run:
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
                        wandb.log(
                            {
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
                            },
                            step=self.num_timesteps,
                        )

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

        while not done:
            masks = self.eval_env.action_masks()
            action, _ = self.model.predict(obs, action_masks=masks, deterministic=True)
            ep_actions.append(int(action))
            obs, reward, terminated, truncated, info = self.eval_env.step(action)
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
            save_path=os.path.join(self.plot_dir, f"traj_{tag}.png"),
        )

        plot_surge_command_fraction(
            history_surge_command_fraction=self.eval_env.history_surge_command_fraction,
            dt=self.eval_env.dt,
            save_path=os.path.join(self.plot_dir, f"speed_{tag}.png"),
        )

        plot_robustness(
            ep_robustness=self.eval_env.history_rob,
            dt=step_dt,
            save_path=os.path.join(self.plot_dir, f"rob_{tag}.png"),
            title="Crossing Detection",
        )

        plot_robustness(
            ep_robustness=self.eval_env.history_maneuver_rob,
            dt=step_dt,
            save_path=os.path.join(self.plot_dir, f"maneuver_rob_{tag}.png"),
            title="Maneuver Spec",
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
            save_path=os.path.join(self.plot_dir, f"control_{tag}.png"),
        )

        if wandb.run:
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

    def _on_training_end(self):
        self._save_training_curves()
        self._run_eval_episode()

    def _save_training_curves(self):
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(6, 1, figsize=(10, 19))

        axes[0].plot(self.step_timesteps, self.step_rewards, alpha=0.15, color="purple")
        if len(self.step_rewards) >= 200:
            rolling = np.convolve(self.step_rewards, np.ones(200) / 200, mode="valid")
            axes[0].plot(self.step_timesteps[199:], rolling, color="purple")
        axes[0].set_ylabel("Step Reward")
        axes[0].set_xlabel("Training Timesteps")
        axes[0].set_title("Training Progress")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(self.episode_rewards, alpha=0.3, color="blue")
        if len(self.episode_rewards) >= 20:
            rolling = np.convolve(self.episode_rewards, np.ones(20) / 20, mode="valid")
            axes[1].plot(range(19, 19 + len(rolling)), rolling, color="blue")
        axes[1].set_ylabel("Episode Reward")
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(self.episode_lengths, alpha=0.3, color="green")
        if len(self.episode_lengths) >= 20:
            rolling = np.convolve(self.episode_lengths, np.ones(20) / 20, mode="valid")
            axes[2].plot(range(19, 19 + len(rolling)), rolling, color="green")
        axes[2].set_ylabel("Episode Length")
        axes[2].grid(True, alpha=0.3)

        window = 20
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
                ax.set_ylabel("Rate (last 20)")
                ax.set_ylim(0, 1)
                ax.grid(True, alpha=0.3)

        for ax in axes[1:]:
            ax.set_xlabel("Episode")

        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, "training_curves.png"), dpi=150)
        plt.close()

        if wandb.run:
            wandb.log({"training_curves": wandb.Image(os.path.join(self.plot_dir, "training_curves.png"))})
            
