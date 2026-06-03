import numpy as np
from gym.utils.geometry import bearing_to_goal, relative_polar, wrap_angle


class Observation:
    BASE_DIM = 11
    TRACKING_ERROR_DIM = 2

    def __init__(self, config):
        self.config = config or {}
        self.include_reference_tracking_errors = bool(
            self.config.get("include_reference_tracking_errors", False)
        )
        self._prev_d_g = None
        self._prev_d_t = None

    @property
    def dim(self):
        if self.include_reference_tracking_errors:
            return self.BASE_DIM + self.TRACKING_ERROR_DIM
        return self.BASE_DIM

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
        goal_bearing = bearing_to_goal(ego_pos, goal_pos)

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

        psi_d = state._current_heading_cmd
        u_d = state._current_surge_cmd
        v_max = state.ego_vessel_model.v_max
        if psi_d is not None and u_d is not None and v_max and v_max > 0:
            psi_d_rel_goal = float(wrap_angle(psi_d - goal_bearing))
            u_d_fraction = float(np.clip(u_d / v_max, 0.0, 1.0))
        else:
            psi_d_rel_goal = 0.0
            u_d_fraction = float(state.initial_surge_command_fraction)

        obs_values = [
            u,
            v,
            r,
            d_g,
            d_dot_g,
            beta_g,
            d_t,
            d_dot_t,
            beta_t,
            psi_d_rel_goal,
            u_d_fraction,
        ]

        if self.include_reference_tracking_errors:
            if psi_d is not None:
                heading_tracking_error = float(wrap_angle(psi_d - psi) / np.pi)
            else:
                heading_tracking_error = 0.0

            if u_d is not None and v_max and v_max > 0:
                speed_tracking_error = float((u_d - u) / v_max)
            else:
                speed_tracking_error = 0.0

            obs_values.extend([heading_tracking_error, speed_tracking_error])

        obs = np.array(obs_values, dtype=np.float32)

        if not np.all(np.isfinite(obs)):
            obs = np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)

        return obs
