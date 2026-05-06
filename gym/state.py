import numpy as np
from gym.utils.geometry import propagate_vessel, wrap_angle


class State:
    def __init__(
        self,
        monitoring_radius,
        n_actions,
        ego_vessel_model,
        initial_surge_command_fraction=0.8,
        min_surge_command_mps=0.0,
    ):
        self.monitoring_radius = monitoring_radius
        self.ego_vessel_model = ego_vessel_model
        self.initial_surge_command_fraction = float(initial_surge_command_fraction)
        self.min_surge_command_mps = float(min_surge_command_mps)

        self.encounter_vessel_eta = None
        self._encounter_speed = None

        self._encounter_active = False
        self._active_maneuver_spec = None
        self._cached_mask = np.ones(n_actions, dtype=bool)
        self._n_actions = n_actions

        self._current_heading_cmd: float | None = None
        self._current_surge_cmd: float | None = None
        self._current_yaw_rate_cmd: float = 0.0
        self._current_surge_accel_cmd: float = 0.0
        self._current_psi_d_dot: float = 0.0
        self._current_psi_d_ddot: float = 0.0
        self._current_u_d_dot: float = 0.0
        self._current_goal_bearing_dot: float = 0.0
        self._psi_d_offset: float = 0.0
        self.fallback_used: bool = False

        self.sim_state = None
        self.in_monitoring_radius = None
        self.terminal_reason: str | None = None

        self.encounter_scenario = None
        self._goal = None
        self._nominal_path_start = None

        self.history_ego = []
        self.history_enc = []
        self.history_rob = []
        self.history_maneuver_rob = []
        self.history_in_radius = []
        self.history_surge_command_fraction = []
        self._current_surge_command_fraction = np.nan

        self.history_heading_deg = []
        self.history_heading_cmd_deg = []
        self.history_heading_error_deg = []
        self.history_surge = []
        self.history_surge_cmd = []
        self.history_yaw_rate_cmd_deg_s = []
        self.history_surge_accel_cmd = []
        self.history_tau_surge = []
        self.history_tau_yaw = []
        self._current_tau = np.zeros(3)

    def update_sim(self, sim_state):
        """Store the latest sim state snapshot from get_state()."""
        self.sim_state = sim_state
        self.in_monitoring_radius = None

    @property
    def position(self):
        """Current vessel position [x, y]."""
        return self.sim_state["eta"][:2]

    @property
    def encounter_type(self):
        if self.encounter_scenario is not None:
            return self.encounter_scenario.encounter_type
        return None

    @property
    def encounter_speed(self):
        if self.encounter_scenario is not None and self.encounter_scenario.target_speed is not None:
            return self.encounter_scenario.target_speed
        return self._encounter_speed

    @encounter_speed.setter
    def encounter_speed(self, value):
        self._encounter_speed = value

    @property
    def encounter_radius(self):
        if self.encounter_scenario is not None:
            return self.encounter_scenario.collision_radius
        return None

    @property
    def encounter_max_time(self):
        if self.encounter_scenario is not None:
            return self.encounter_scenario.simtime
        return None

    @property
    def goal(self):
        if self.encounter_scenario is not None and self.encounter_scenario.goal is not None:
            return self.encounter_scenario.goal
        return self._goal

    @goal.setter
    def goal(self, value):
        self._goal = value

    @property
    def nominal_path_start(self):
        if self.encounter_scenario is not None and self.encounter_scenario.nominal_path_start is not None:
            return self.encounter_scenario.nominal_path_start
        return self._nominal_path_start

    @nominal_path_start.setter
    def nominal_path_start(self, value):
        self._nominal_path_start = value

    def propagate_encounter(self, dt):
        """Advance the encounter vessel by one timestep."""
        if self.encounter_vessel_eta is not None:
            self.encounter_vessel_eta = propagate_vessel(
                vessel_eta=self.encounter_vessel_eta,
                encounter_speed=self.encounter_speed,
                dt=dt,
            )

    def is_diverged(self):
        """Check if the sim state has diverged (non-finite or too large)."""
        return (
            not np.all(np.isfinite(self.sim_state["eta"]))
            or not np.all(np.isfinite(self.sim_state["nu"]))
            or np.any(np.abs(self.sim_state["eta"][:2]) > 1e6)
            or np.any(np.abs(self.sim_state["nu"]) > 2)
        )

    def set_current_tau(self, tau: np.ndarray):
        self._current_tau = np.asarray(tau, dtype=float)

    def initialize_command_references(self, sim_state):
        """Reset carried MRAC references at the start of an episode."""
        eta = sim_state["eta"]
        goal = sim_state.get("goal")
        if goal is not None:
            self._current_heading_cmd = float(
                np.arctan2(float(goal[1]) - float(eta[1]), float(goal[0]) - float(eta[0]))
            )
        else:
            self._current_heading_cmd = float(eta[-1])
        self._current_surge_cmd = self._clip_surge_cmd(
            self.initial_surge_command_fraction * self.ego_vessel_model.v_max
        )
        self._current_surge_command_fraction = (
            self._current_surge_cmd / self.ego_vessel_model.v_max
            if self.ego_vessel_model.v_max
            else np.nan
        )
        _, goal_bearing_dot = self._goal_bearing_and_rate(sim_state)
        self._current_yaw_rate_cmd = 0.0
        self._current_surge_accel_cmd = 0.0
        self._current_psi_d_dot = goal_bearing_dot
        self._current_psi_d_ddot = 0.0
        self._current_u_d_dot = 0.0
        self._current_goal_bearing_dot = goal_bearing_dot
        self._psi_d_offset = 0.0

    def _goal_bearing_and_rate(self, sim_state=None):
        sim_state = self.sim_state if sim_state is None else sim_state
        eta = np.asarray(sim_state["eta"], dtype=float)
        nu = np.asarray(sim_state["nu"][:3], dtype=float)
        psi = float(eta[-1])
        u, v, r = nu

        goal = sim_state.get("goal")
        if goal is None:
            return psi, float(r)

        goal = np.asarray(goal[:2], dtype=float)
        delta = goal - eta[:2]
        range_sq = float(np.dot(delta, delta))
        bearing = float(np.arctan2(delta[1], delta[0]))
        if range_sq <= 1e-12:
            return bearing, 0.0

        x_dot = float(u * np.cos(psi) - v * np.sin(psi))
        y_dot = float(u * np.sin(psi) + v * np.cos(psi))
        bearing_dot = float((delta[1] * x_dot - delta[0] * y_dot) / range_sq)
        return bearing, bearing_dot

    def apply_rate_command(self, yaw_rate_cmd: float, surge_accel_cmd: float, dt: float):
        """Integrate goal-relative rate commands into carried MRAC references.

        The yaw rate accumulates a heading offset from the current goal bearing,
        so yaw_rate=0 always commands heading toward the goal (offset=0).
        """
        if self._current_heading_cmd is None or self._current_surge_cmd is None:
            self.initialize_command_references(self.sim_state)

        self._current_yaw_rate_cmd = float(yaw_rate_cmd)
        self._current_surge_accel_cmd = float(surge_accel_cmd)

        self._psi_d_offset += yaw_rate_cmd * dt

        goal_bearing, goal_bearing_dot = self._goal_bearing_and_rate()
        self._current_heading_cmd = goal_bearing + self._psi_d_offset
        psi_d_dot = goal_bearing_dot + float(yaw_rate_cmd)
        psi_d_ddot = (
            (goal_bearing_dot - self._current_goal_bearing_dot) / dt
            if dt > 0.0
            else 0.0
        )

        previous_surge_cmd = float(self._current_surge_cmd)
        self._current_surge_cmd = self._clip_surge_cmd(
            self._current_surge_cmd + surge_accel_cmd * dt
        )
        u_d_dot = (
            (self._current_surge_cmd - previous_surge_cmd) / dt
            if dt > 0.0
            else 0.0
        )
        self._current_surge_command_fraction = (
            self._current_surge_cmd / self.ego_vessel_model.v_max
            if self.ego_vessel_model.v_max
            else np.nan
        )
        self._current_psi_d_dot = psi_d_dot
        self._current_psi_d_ddot = psi_d_ddot
        self._current_u_d_dot = u_d_dot
        self._current_goal_bearing_dot = goal_bearing_dot
        return (
            self._current_heading_cmd,
            self._current_surge_cmd,
            self._current_psi_d_dot,
            self._current_psi_d_ddot,
            self._current_u_d_dot,
        )

    def set_heading_speed_commands(self, psi_d: float, u_d: float):
        """Set commanded heading and speed directly (goal-bearing action space)."""
        self._current_heading_cmd = float(psi_d)
        self._current_surge_cmd = self._clip_surge_cmd(u_d)
        self._current_surge_command_fraction = (
            self._current_surge_cmd / self.ego_vessel_model.v_max
            if self.ego_vessel_model.v_max
            else np.nan
        )
        self._current_yaw_rate_cmd = 0.0
        self._current_surge_accel_cmd = 0.0
        self._current_psi_d_dot = 0.0
        self._current_psi_d_ddot = 0.0
        self._current_u_d_dot = 0.0
        self._current_goal_bearing_dot = 0.0

    def hold_current_commands(self):
        """Hold carried heading and surge references without integrating action rates."""
        if self._current_heading_cmd is None or self._current_surge_cmd is None:
            self.initialize_command_references(self.sim_state)

        self._current_yaw_rate_cmd = 0.0
        self._current_surge_accel_cmd = 0.0
        self._current_psi_d_dot = 0.0
        self._current_psi_d_ddot = 0.0
        self._current_u_d_dot = 0.0
        _, self._current_goal_bearing_dot = self._goal_bearing_and_rate()
        return (
            self._current_heading_cmd,
            self._current_surge_cmd,
            self._current_psi_d_dot,
            self._current_psi_d_ddot,
            self._current_u_d_dot,
        )

    def set_current_rate_commands(self, yaw_rate_cmd: float, surge_accel_cmd: float):
        self._current_yaw_rate_cmd = float(yaw_rate_cmd)
        self._current_surge_accel_cmd = float(surge_accel_cmd)

    def _clip_surge_cmd(self, u_d: float) -> float:
        return float(
            np.clip(
                u_d,
                self.min_surge_command_mps,
                self.ego_vessel_model.v_max,
            )
        )

    def record(self):
        """Append current positions and control state to history."""
        self.history_ego.append(self.sim_state["eta"][:2].tolist())
        if self.encounter_vessel_eta is not None:
            self.history_enc.append(self.encounter_vessel_eta[:2].tolist())
        self.history_surge_command_fraction.append(self._current_surge_command_fraction)

        psi = float(self.sim_state["eta"][-1])
        u = float(self.sim_state["nu"][0])
        self.history_heading_deg.append(float(np.degrees(psi)))
        self.history_surge.append(u)
        surge_cmd = np.nan if self._current_surge_cmd is None else self._current_surge_cmd
        self.history_surge_cmd.append(float(surge_cmd))
        self.history_yaw_rate_cmd_deg_s.append(float(np.degrees(self._current_yaw_rate_cmd)))
        self.history_surge_accel_cmd.append(float(self._current_surge_accel_cmd))
        self.history_tau_surge.append(float(self._current_tau[0]))
        self.history_tau_yaw.append(float(self._current_tau[2]))

        if self._current_heading_cmd is not None:
            psi_d = self._current_heading_cmd
            self.history_heading_cmd_deg.append(float(np.degrees(psi_d)))
            self.history_heading_error_deg.append(float(np.degrees(wrap_angle(psi_d - psi))))
        else:
            self.history_heading_cmd_deg.append(np.nan)
            self.history_heading_error_deg.append(np.nan)

    def record_robustness(self, robustness, in_radius):
        """Append robustness and radius status to history."""
        self.history_rob.append(robustness)
        self.history_in_radius.append(in_radius)

    def record_maneuver_robustness(self, robustness):
        """Append maneuver spec robustness to history (None when encounter inactive)."""
        self.history_maneuver_rob.append(robustness)

    def clear_encounter(self):
        """Reset encounter state when vessel leaves monitoring radius."""
        self._encounter_active = False
        self._active_maneuver_spec = None
        self._cached_mask = np.ones(self._n_actions, dtype=bool)

    def reset(self):
        """Reset all episode state. Called at the start of each episode."""
        self.encounter_vessel_eta = None
        self._encounter_speed = None
        self._encounter_active = False
        self._active_maneuver_spec = None
        self._cached_mask = np.ones(self._n_actions, dtype=bool)
        self._current_heading_cmd = None
        self._current_surge_cmd = None
        self._current_yaw_rate_cmd = 0.0
        self._current_surge_accel_cmd = 0.0
        self._current_psi_d_dot = 0.0
        self._current_psi_d_ddot = 0.0
        self._current_u_d_dot = 0.0
        self._current_goal_bearing_dot = 0.0
        self._psi_d_offset = 0.0
        self.fallback_used = False
        self.sim_state = None
        self.in_monitoring_radius = None
        self.terminal_reason = None
        self.encounter_scenario = None
        self._goal = None
        self._nominal_path_start = None
        self.history_ego = []
        self.history_enc = []
        self.history_rob = []
        self.history_maneuver_rob = []
        self.history_in_radius = []
        self.history_surge_command_fraction = []
        self._current_surge_command_fraction = np.nan
        self.history_heading_deg = []
        self.history_heading_cmd_deg = []
        self.history_heading_error_deg = []
        self.history_surge = []
        self.history_surge_cmd = []
        self.history_yaw_rate_cmd_deg_s = []
        self.history_surge_accel_cmd = []
        self.history_tau_surge = []
        self.history_tau_yaw = []
        self._current_tau = np.zeros(3)
