from functools import partial

import numpy as np

from gym.utils.geometry import cross_track_error, within_monitoring_radius


class Reward:
    def __init__(self, config: dict):
        self.reward_handlers: dict[str, callable] = {}
        self.reward_reset_handlers: dict[str, callable] = {}
        self.episode_total = 0.0

        handler_map = {
            "reward_progress": self._build_progress,
            "reward_cte":      self._build_cte,
            "reward_speed":    self._build_speed,
            "reward_heading":  self._build_heading,
            "reward_fallback": self._build_fallback,
        }

        for key, builder in handler_map.items():
            cfg = config.get(key, {})
            if cfg.get("is_enabled", False):
                builder(cfg)

    # --- builders ---

    def _build_progress(self, cfg):
        self._prev_dist = None
        self.reward_handlers["reward_progress"] = partial(
            self._reward_progress, cfg["coefficient"]
        )
        self.reward_reset_handlers["reward_progress"] = lambda: setattr(
            self, "_prev_dist", None
        )

    def _build_cte(self, cfg):
        clip = cfg.get("clip", 5.0)
        self.reward_handlers["reward_cte"] = partial(
            self._reward_cte, cfg["coefficient"], clip
        )

    def _build_speed(self, cfg):
        self.reward_handlers["reward_speed"] = partial(
            self._reward_speed, cfg["coefficient"]
        )

    def _build_heading(self, cfg):
        self.reward_handlers["reward_heading"] = partial(
            self._reward_heading, cfg["coefficient"]
        )

    def _build_fallback(self, cfg):
        self.reward_handlers["reward_fallback"] = partial(
            self._reward_fallback, cfg["coefficient"]
        )

    # --- handlers ---

    def _reward_progress(self, coeff, state, in_radius):
        goal = state.goal
        if goal is None:
            goal = state.sim_state.get("goal") or (0.0, 0.0)
        pos = state.position
        dist = np.hypot(goal[0] - pos[0], goal[1] - pos[1])
        progress = (self._prev_dist - dist) if self._prev_dist is not None else 0.0
        self._prev_dist = dist
        return coeff * progress

    def _reward_cte(self, coeff, clip, state, in_radius):
        if in_radius:
            return 0.0
        goal = state.goal
        if goal is None:
            goal = state.sim_state.get("goal") or (0.0, 0.0)
        cte = cross_track_error(state.position, state.nominal_path_start, goal[:2])
        return coeff * min(abs(cte), clip)

    def _reward_speed(self, coeff, state, in_radius):
        if not in_radius:
            return 0.0
        return coeff * (1.0 - state._current_speed_multiplier)

    def _reward_heading(self, coeff, state, in_radius):
        if not in_radius:
            return 0.0
        return coeff * abs(state._current_heading_offset)

    def _reward_fallback(self, coeff, state, in_radius):
        if not in_radius or not state.fallback_used:
            return 0.0
        return coeff

    # --- core ---

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
