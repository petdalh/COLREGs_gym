import numpy as np

from gym.utils.geometry import cross_track_error, within_monitoring_radius


class Reward:
    def __init__(self, config):
        self.w_cte = config.get("w_cte", 1.0)
        self.cte_clip = config.get("cte_clip", 10.0)
        self.prev_dist_to_goal = None
        self.episode_total = 0.0

    def get_reward(self, state):
        """
        Compute reward from state.

        Uses state.position, state.goal, state.encounter_vessel_eta,
        state.monitoring_radius, state.sim_state, state.nominal_path_start.
        """
        pos_xy = state.position
        goal = state.goal
        if goal is None and isinstance(state.sim_state, dict):
            goal = state.sim_state.get("goal")
        if goal is None:
            goal = (0.0, 0.0)
        gn, ge = goal[:2]
        dist = np.hypot(gn - pos_xy[0], ge - pos_xy[1])

        progress = (
            self.prev_dist_to_goal - dist
            if self.prev_dist_to_goal is not None
            else 0.0
        )
        self.prev_dist_to_goal = dist

        in_radius = within_monitoring_radius(
            state.encounter_vessel_eta, state.monitoring_radius, state.sim_state
        )
        if not in_radius:
            cte = cross_track_error(pos_xy, state.nominal_path_start, goal[:2])
            reward = progress - self.w_cte * min(abs(cte), self.cte_clip)
        else:
            reward = progress

        self.episode_total += reward
        return reward

    def reset(self):
        self.prev_dist_to_goal = None
        self.episode_total = 0.0
