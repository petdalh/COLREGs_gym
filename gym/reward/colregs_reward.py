"""Crossing-specific COLREGS reward — faithful port of ColregsReward."""
from functools import partial

import numpy as np

from gym.reward.reward import Reward
from gym.utils.geometry import bearing_to_goal, cross_track_error, wrap_angle


class ColregsReward(Reward):
    """
    Extends the base Reward class with reward terms ported from ColregsReward
    (E. Meyer et al., COLREG-Compliant Collision Avoidance for USV Using DRL).

    All parameters are read exclusively from the config dict — no values are
    inferred from the action space or environment configuration.
    """

    def __init__(self, config: dict):
        super().__init__(config)  # registers all base handlers

        extra_handlers = {
            "reward_fallback":          self._build_fallback,
            "reward_acceleration":      self._build_acceleration,
            "reward_reverse_driving":   self._build_reverse_driving,
            "reward_termination":       self._build_termination,
            "reward_velocity":          self._build_velocity,
            "reward_goal_distance":     self._build_goal_distance,
            "reward_lateral_deviation": self._build_lateral_deviation,
            "reward_wrong_side_crossing": self._build_wrong_side_crossing,
            "reward_safe_distance":     self._build_safe_distance,
        }
        for key, builder in extra_handlers.items():
            cfg = config.get(key, {})
            if cfg.get("is_enabled", False):
                builder(cfg)

    # ------------------------------------------------------------------ #
    # reward_fallback                                                     #
    # ------------------------------------------------------------------ #

    def _build_fallback(self, cfg):
        self.reward_handlers["reward_fallback"] = partial(
            self._reward_fallback, cfg["coefficient"]
        )

    @staticmethod
    def _reward_fallback(coeff, state, in_radius):
        if not in_radius or not state.fallback_used:
            return 0.0
        return coeff

    # ------------------------------------------------------------------ #
    # reward_acceleration                                                 #
    # ------------------------------------------------------------------ #

    def _build_acceleration(self, cfg):
        self._prev_yaw_rate_cmd_deg_s = None
        self._prev_surge_accel_cmd = None
        self.reward_handlers["reward_acceleration"] = partial(
            self._reward_acceleration,
            cfg.get("coefficient", 1.0),
            cfg.get("yaw_rate_delta_coefficient", 1.0),
            cfg.get("surge_accel_delta_coefficient", 1.0),
        )
        self.reward_reset_handlers["reward_acceleration"] = self._reset_acceleration

    def _reset_acceleration(self):
        self._prev_yaw_rate_cmd_deg_s = None
        self._prev_surge_accel_cmd = None

    def _reward_acceleration(
        self,
        coeff,
        yaw_rate_delta_coeff,
        surge_accel_delta_coeff,
        state,
        in_radius,
    ):
        # These are the held agent commands (r_i, a_j), not the autopilot
        # torques. Penalise command changes separately so deg/s and m/s^2
        # are not silently added with equal units.
        yaw_rate_cmd_deg_s = float(np.degrees(state._current_yaw_rate_cmd))
        surge_accel_cmd = float(state._current_surge_accel_cmd)

        if self._prev_yaw_rate_cmd_deg_s is None:
            self._prev_yaw_rate_cmd_deg_s = yaw_rate_cmd_deg_s
            self._prev_surge_accel_cmd = surge_accel_cmd
            return 0.0

        yaw_rate_delta = abs(yaw_rate_cmd_deg_s - self._prev_yaw_rate_cmd_deg_s)
        surge_accel_delta = abs(
            surge_accel_cmd - self._prev_surge_accel_cmd
        )
        self._prev_yaw_rate_cmd_deg_s = yaw_rate_cmd_deg_s
        self._prev_surge_accel_cmd = surge_accel_cmd
        return coeff * (
            yaw_rate_delta_coeff * yaw_rate_delta
            + surge_accel_delta_coeff * surge_accel_delta
        )

    # ------------------------------------------------------------------ #
    # reward_reverse_driving                                              #
    # ------------------------------------------------------------------ #

    def _build_reverse_driving(self, cfg):
        self.reward_handlers["reward_reverse_driving"] = partial(
            self._reward_reverse_driving, cfg["coefficient"]
        )

    @staticmethod
    def _reward_reverse_driving(coeff, state, in_radius):
        return coeff if _surge_speed(state) < 0.0 else 0.0

    # ------------------------------------------------------------------ #
    # reward_termination                                                   #
    # ------------------------------------------------------------------ #

    def _build_termination(self, cfg):
        self.reward_handlers["reward_termination"] = partial(
            self._reward_termination,
            cfg["reward_goal_reached"],
            cfg["reward_collision"],
            cfg["reward_time_out"],
            cfg["reward_out_of_bounds"],
        )

    @staticmethod
    def _reward_termination(r_goal, r_collision, r_timeout, r_oob, state, in_radius):
        reason = getattr(state, "terminal_reason", None)
        if reason == "goal_reached":
            return r_goal
        if reason == "collision":
            return r_collision
        if reason == "time_limit":
            return r_timeout
        if reason in ("out_of_bounds", "diverged", "state_blowup"):
            return r_oob
        return 0.0

    # ------------------------------------------------------------------ #
    # reward_velocity  (port of velocity_penalty)                         #
    # ------------------------------------------------------------------ #

    def _build_velocity(self, cfg):
        self.reward_handlers["reward_velocity"] = partial(
            self._reward_velocity,
            cfg["low_speed_mps"],
            cfg["high_speed_mps"],
            cfg["coefficient"],
        )

    @staticmethod
    def _reward_velocity(low_thr, high_thr, coeff, state, in_radius):
        # Use realised surge speed u, matching the reference reward's v_ego
        # term. The integrated command fraction is only a controller setpoint.
        v = _surge_speed(state)
        if np.isnan(v):
            return 0.0
        penalty = 0.0
        if v <= low_thr:
            penalty = v - low_thr      # negative
        elif v >= high_thr:
            penalty = high_thr - v     # negative
        return penalty * coeff

    # ------------------------------------------------------------------ #
    # reward_goal_distance  (port of goal_distance_reward)                #
    # ------------------------------------------------------------------ #

    def _build_goal_distance(self, cfg):
        self._prev_eucl_dist = None
        self._prev_orient_dist = None
        self.reward_handlers["reward_goal_distance"] = partial(
            self._reward_goal_distance, cfg
        )
        self.reward_reset_handlers["reward_goal_distance"] = self._reset_goal_distance

    def _reset_goal_distance(self):
        self._prev_eucl_dist = None
        self._prev_orient_dist = None

    def _reward_goal_distance(self, cfg, state, in_radius):
        """
        Three-phase goal progress reward (far approach / final approach /
        linearised final approach).

        Coefficients map to the reference as:
          c1 = final_approach_start_dist
          c2 = euclidean_advance_coeff
          c3 = orientation_advance_coeff
          c5 = final_approach_euclidean_dist_coeff  (negative to enable)
          c6 = final_approach_distance_coeff
          c7 = final_approach_linearization_coeff   (positive to enable)
        """
        goal = state.goal
        if goal is None:
            return 0.0
        pos = state.position
        ego_psi = float(state.sim_state["eta"][2])

        eucl_dist = np.hypot(goal[0] - pos[0], goal[1] - pos[1])
        goal_bearing = bearing_to_goal(pos, goal)
        orient_dist = abs(wrap_angle(ego_psi - goal_bearing))

        # Initialise on first call (mirrors ColregsReward.reset() behaviour)
        if self._prev_eucl_dist is None:
            self._prev_eucl_dist = eucl_dist
            self._prev_orient_dist = orient_dist
            return 0.0

        eucl_advance = self._prev_eucl_dist - eucl_dist
        orient_advance_signed = self._prev_orient_dist - orient_dist  # positive = rotating toward goal
        orient_advance = abs(orient_advance_signed)
        self._prev_eucl_dist = eucl_dist
        self._prev_orient_dist = orient_dist

        c1 = cfg["final_approach_start_dist"]
        c2 = cfg["euclidean_advance_coeff"]
        c3 = cfg["orientation_advance_coeff"]
        c5 = cfg["final_approach_euclidean_dist_coeff"]   # negative to enable
        c6 = cfg["final_approach_distance_coeff"]
        c7 = cfg["final_approach_linearization_coeff"]    # positive to enable

        if c7 > 0 and eucl_dist < c1:
            orient = orient_dist if orient_dist != 0.0 else 1.0
            if eucl_advance < 0:
                rew = c7 / ((eucl_dist + 1e-8) * abs(orient))
            else:
                rew = c7 / (eucl_dist + 1e-8)
        elif eucl_dist < c1:
            rew = c6 * (orient_advance * c3 + eucl_advance * c2)
        else:
            rew = c2 * eucl_advance + c3 * orient_advance_signed

        # Final-approach euclidean distance penalty (disabled when c5 >= 0)
        if c5 < 0 and eucl_dist < c1:
            rew += c5 * eucl_dist

        return rew / cfg["timestep_normalization"]

    # ------------------------------------------------------------------ #
    # reward_lateral_deviation  (port of lat_dist_rew)                    #
    # ------------------------------------------------------------------ #

    def _build_lateral_deviation(self, cfg):
        self.reward_handlers["reward_lateral_deviation"] = partial(
            self._reward_lateral_deviation, cfg
        )

    @staticmethod
    def _reward_lateral_deviation(cfg, state, in_radius):
        """
        Penalise cross-track error when NOT in the final approach zone.

        Differs from the base reward_cte in that the condition is
        euclidean_distance > final_approach_start_dist (matching the
        reference) rather than not in_radius.
        """
        goal = state.goal
        if goal is None:
            return 0.0
        pos = state.position
        eucl_dist = np.hypot(goal[0] - pos[0], goal[1] - pos[1])
        if eucl_dist <= cfg["final_approach_start_dist"]:
            return 0.0
        lat_dist = abs(cross_track_error(pos, state.nominal_path_start, goal[:2]))
        c4 = cfg["coefficient"]
        c8 = cfg["clip"]
        if c8 != 0:
            return float(np.clip(c4 * lat_dist, -c8, c8))
        return c4 * lat_dist

    # ------------------------------------------------------------------ #
    # reward_wrong_side_crossing                                          #
    # ------------------------------------------------------------------ #

    def _build_wrong_side_crossing(self, cfg):
        self._wrong_side_crossing_triggered = False
        self._prev_wrong_side_rel_n = None
        self.reward_handlers["reward_wrong_side_crossing"] = partial(
            self._reward_wrong_side_crossing,
            cfg["coefficient"],
            cfg.get("east_margin_m", 0.0),
        )
        self.reward_reset_handlers["reward_wrong_side_crossing"] = (
            self._reset_wrong_side_crossing
        )

    def _reset_wrong_side_crossing(self):
        self._wrong_side_crossing_triggered = False
        self._prev_wrong_side_rel_n = None

    def _reward_wrong_side_crossing(self, coeff, east_margin_m, state, in_radius):
        if self._wrong_side_crossing_triggered:
            return 0.0
        if state.encounter_vessel_eta is None or state.sim_state is None:
            return 0.0

        ego_eta = state.sim_state["eta"]
        ego_n = float(ego_eta[0])
        ego_e = float(ego_eta[1])
        enc_n = float(state.encounter_vessel_eta[0])
        enc_e = float(state.encounter_vessel_eta[1])

        rel_n = ego_n - enc_n
        rel_e = ego_e - enc_e
        prev_rel_n = self._prev_wrong_side_rel_n
        self._prev_wrong_side_rel_n = rel_n

        if prev_rel_n is None:
            return 0.0

        crossed_obstacle_north_line = prev_rel_n < 0.0 <= rel_n
        passed_left_of_obstacle = rel_e < -float(east_margin_m)
        if crossed_obstacle_north_line and passed_left_of_obstacle:
            self._wrong_side_crossing_triggered = True
            return coeff
        return 0.0

    # ------------------------------------------------------------------ #
    # reward_safe_distance  (port of safe_distance_penalty)               #
    # ------------------------------------------------------------------ #

    def _build_safe_distance(self, cfg):
        params = {k: v for k, v in cfg.items() if k != "is_enabled"}
        self.reward_handlers["reward_safe_distance"] = partial(
            self._reward_safe_distance, params
        )

    def _reward_safe_distance(self, params, state, in_radius):
        """
        Directional proximity penalty, applied only when in_radius=True.

        Inspired by E. Meyer et al., "COLREG-Compliant Collision Avoidance
        for Unmanned Surface Vehicle Using Deep Reinforcement Learning".

        Adapted to NED coordinates (eta = [n, e, psi]).
        The rotation matrix structure and velocity_y sign convention are
        kept identical to the reference for mathematical fidelity.
        """
        if not in_radius or state.encounter_vessel_eta is None:
            return 0.0

        eta = state.sim_state["eta"]
        ego_n, ego_e, ego_psi = float(eta[0]), float(eta[1]), float(eta[2])
        enc_n = float(state.encounter_vessel_eta[0])
        enc_e = float(state.encounter_vessel_eta[1])
        enc_psi = float(state.encounter_vessel_eta[2])
        enc_speed = float(state.encounter_speed or 0.0)

        distance = np.hypot(enc_n - ego_n, enc_e - ego_e)
        if distance < 1e-6:
            return 0.0

        # Relative bearing in ego body frame, normalised to [-pi, pi]
        bearing = np.arctan2(enc_e - ego_e, enc_n - ego_n)
        obs_rel_rad = wrap_angle(bearing - ego_psi)
        obs_rel_deg = np.degrees(obs_rel_rad)

        # Static sector weight (which COLREGS sector the obstacle is in)
        sector_weight = self._sector_static(obs_rel_deg, params)

        # Encounter vessel velocity expressed in ego frame.
        R = np.array([
            [np.cos(ego_psi), -np.sin(ego_psi)],
            [np.sin(ego_psi),  np.cos(ego_psi)],
        ])
        v_enc = enc_speed * np.array([np.cos(enc_psi), np.sin(enc_psi)])
        v_body = R @ v_enc
        velocity_y = -1.0 * v_body[1]

        # Dynamic sector weight (depends on velocity_y sign)
        sector_weight_dyn = self._sector_dynamic(obs_rel_deg, velocity_y, params)

        weighting_term = -1.0 / (1.0 + np.exp(np.abs(obs_rel_deg)))

        raw = params["magnitude"] * np.exp(
            (sector_weight_dyn * velocity_y - sector_weight) * distance
        )
        return float(np.clip(weighting_term * raw, params["clip"], 0.0))

    @staticmethod
    def _sector_static(deg, p):
        if 0.0 <= deg < 112.5:
            return p["starboard_static_coeff"]
        elif -112.5 <= deg < 0.0:
            return p["port_static_coeff"]
        else:
            return p["stern_static_coeff"]

    @staticmethod
    def _sector_dynamic(deg, vy, p):
        if 0.0 <= deg < 112.5:
            return p["starboard_dyn_plus_coeff"] if vy >= 0.0 else p["starboard_dyn_minus_coeff"]
        elif -112.5 <= deg < 0.0:
            return p["port_dyn_plus_coeff"] if vy >= 0.0 else p["port_dyn_minus_coeff"]
        else:
            return p["stern_dyn_plus_coeff"] if vy >= 0.0 else p["stern_dyn_minus_coeff"]


def _surge_speed(state) -> float:
    if state.sim_state is None or "nu" not in state.sim_state:
        return np.nan
    return float(state.sim_state["nu"][0])
