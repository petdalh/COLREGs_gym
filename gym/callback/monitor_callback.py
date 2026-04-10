import os

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from gym.callback.plotting import plot_episode_trajectory, plot_robustness


class ColregsMonitorCallback(BaseCallback):
    def __init__(self, eval_env, plot_dir="plots", plot_every_episodes=50):
        super().__init__()
        self.eval_env = getattr(eval_env, "unwrapped", eval_env)
        self.plot_dir = plot_dir
        self.plot_every = plot_every_episodes

        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_reasons = []
        self.episode_count = 0

        os.makedirs(plot_dir, exist_ok=True)

    def _on_step(self):
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_count += 1
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])

                reason = info.get("reason", "unknown")
                self.episode_reasons.append(reason)

                if self.episode_count % 10 == 0:
                    avg_r = np.mean(self.episode_rewards[-10:])
                    avg_l = np.mean(self.episode_lengths[-10:])
                    recent_reasons = self.episode_reasons[-10:]
                    collisions = sum(1 for r in recent_reasons if r == "collision")
                    goals = sum(1 for r in recent_reasons if r == "goal_reached")
                    timeouts = sum(1 for r in recent_reasons if r == "time_limit")
                    print(
                        f"Ep {self.episode_count:5d} | "
                        f"avg_r={avg_r:+8.2f} | avg_len={avg_l:5.1f} | "
                        f"last10: {goals}G {collisions}C {timeouts}T"
                    )

                if self.episode_count % self.plot_every == 0:
                    self._run_eval_episode()

        return True

    def _run_eval_episode(self):
        obs, _ = self.eval_env.reset()
        done = False

        while not done:
            masks = self.eval_env.action_masks()
            action, _ = self.model.predict(obs, action_masks=masks, deterministic=True)
            obs, reward, terminated, truncated, info = self.eval_env.step(action)
            done = terminated or truncated

        tag = f"ep{self.episode_count:05d}"

        plot_episode_trajectory(
            ego_traj=self.eval_env.history_ego,
            ego_speed_traj=self.eval_env.history_speed_multiplier,
            enc_traj=self.eval_env.history_enc,
            goal=self.eval_env.goal[:2],
            dt=self.eval_env.dt,
            episode_num=self.episode_count,
            save_path=os.path.join(self.plot_dir, f"traj_{tag}.png"),
        )

        plot_robustness(
            ep_robustness=self.eval_env.history_rob,
            dt=self.eval_env.dt * getattr(self.eval_env, "decision_interval", 5),
            save_path=os.path.join(self.plot_dir, f"rob_{tag}.png"),
        )

    def _on_training_end(self):
        self._save_training_curves()
        self._run_eval_episode()

    def _save_training_curves(self):
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(5, 1, figsize=(10, 16), sharex=True)

        axes[0].plot(self.episode_rewards, alpha=0.3, color="blue")
        if len(self.episode_rewards) >= 20:
            rolling = np.convolve(self.episode_rewards, np.ones(20) / 20, mode="valid")
            axes[0].plot(range(19, 19 + len(rolling)), rolling, color="blue")
        axes[0].set_ylabel("Episode Reward")
        axes[0].set_title("Training Progress")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(self.episode_lengths, alpha=0.3, color="green")
        if len(self.episode_lengths) >= 20:
            rolling = np.convolve(self.episode_lengths, np.ones(20) / 20, mode="valid")
            axes[1].plot(range(19, 19 + len(rolling)), rolling, color="green")
        axes[1].set_ylabel("Episode Length")
        axes[1].grid(True, alpha=0.3)

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

            axes[2].plot(x, goals, color="#2ecc71")
            axes[2].set_ylabel("Goal Rate")
            axes[2].set_ylim(0, 1)
            axes[2].grid(True, alpha=0.3)

            axes[3].plot(x, collisions, color="#e74c3c")
            axes[3].set_ylabel("Collision Rate")
            axes[3].set_ylim(0, 1)
            axes[3].grid(True, alpha=0.3)

            axes[4].plot(x, timeouts, color="#f39c12")
            axes[4].set_ylabel("Timeout Rate")
            axes[4].set_ylim(0, 1)
            axes[4].grid(True, alpha=0.3)
        else:
            for ax in axes[2:]:
                ax.set_ylabel("Rate (last 20)")
                ax.set_ylim(0, 1)
                ax.grid(True, alpha=0.3)

        axes[4].set_xlabel("Episode")

        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, "training_curves.png"), dpi=150)
        plt.close()