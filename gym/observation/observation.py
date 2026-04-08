import numpy as np
from gym.utils.geometry import relative_polar


class Observation:
    def __init__(self, config):
        self.config = config
        self._prev_d_g = None
        self._prev_d_t = None

    def reset(self):
        self._prev_d_g = None
        self._prev_d_t = None

    def get(self, state) -> np.ndarray:
        eta = state.sim_state["eta"]
        nu = state.sim_state["nu"]
        ego_pos = eta[:2]        # [n, e]
        psi = float(eta[-1])     # heading (rad)
        u, v, r = nu[0], nu[1], nu[2]

        # --- goal ---
        goal = state.goal
        if goal is None:
            goal = state.sim_state.get("goal")
        goal_pos = np.asarray(goal[:2], dtype=float) if goal is not None else ego_pos

        d_g, beta_g = relative_polar(ego_pos, psi, goal_pos)
        if self._prev_d_g is None:
            self._prev_d_g = d_g
        d_dot_g = d_g - self._prev_d_g
        self._prev_d_g = d_g

        # --- encounter vessel ---
        if state.encounter_vessel_eta is not None:
            enc_pos = state.encounter_vessel_eta[:2]
            d_t, beta_t = relative_polar(ego_pos, psi, enc_pos)
        else:
            d_t = state.monitoring_radius if state.monitoring_radius is not None else 0.0
            beta_t = 0.0

        if self._prev_d_t is None:
            self._prev_d_t = d_t
        d_dot_t = d_t - self._prev_d_t
        self._prev_d_t = d_t

        obs = np.array(
            [u, v, r, d_g, d_dot_g, beta_g, d_t, d_dot_t, beta_t],
            dtype=np.float32,
        )

        if not np.all(np.isfinite(obs)):
            obs = np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)

        return obs
