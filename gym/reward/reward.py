import numpy as np

from gym.utils.geometry import within_monitoring_radius


class Reward:
    def __init__(self, config: dict):
        self.reward_handlers: dict[str, callable] = {}
        self.reward_reset_handlers: dict[str, callable] = {}
        self.episode_total = 0.0

    def get_reward(self, state, in_radius=None) -> tuple[float, dict]:
        if in_radius is None:
            in_radius = state.in_monitoring_radius
        if in_radius is None:
            in_radius = within_monitoring_radius(
                state.encounter_vessel_eta, state.monitoring_radius, state.sim_state
            )

        info = {}
        total = 0.0
        for key, handler in self.reward_handlers.items():
            r = handler(state, in_radius)
            info[key] = r
            total += r

        self.episode_total += total
        return total, info

    def reset(self):
        self.episode_total = 0.0
        for fn in self.reward_reset_handlers.values():
            fn()
