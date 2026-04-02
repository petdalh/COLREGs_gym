class EpisodeLogger:
    def on_episode_end(self, info, terminated, reward):
        """Log episode outcome. Called when terminated or truncated."""
        reason = info.get("reason", "goal_reached")
        status = "Terminated" if terminated else "Truncated"
        print(
            f"--- Episode {status} | Reason: {reason} "
            f"| Reward: {reward.episode_total:.2f} ---"
        )
