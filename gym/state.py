import numpy as np
from gym.utils.geometry import propagate_vessel


class State:
    def __init__(self, monitoring_radius, n_actions):
        self.monitoring_radius = monitoring_radius

        self.encounter_vessel_eta = None
        self._encounter_speed = None

        self._encounter_active = False
        self._active_maneuver_spec = None
        self._cached_mask = np.ones(n_actions, dtype=bool)
        self._n_actions = n_actions

        self._current_heading_offset: float = 0.0
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
        self.history_speed_multiplier = []
        self._current_speed_multiplier = np.nan

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
            or np.any(np.abs(self.sim_state["nu"]) > 1e6)
        )

    def record(self):
        """Append current positions to history."""
        self.history_ego.append(self.sim_state["eta"][:2].tolist())
        if self.encounter_vessel_eta is not None:
            self.history_enc.append(self.encounter_vessel_eta[:2].tolist())
        self.history_speed_multiplier.append(self._current_speed_multiplier)

    def record_robustness(self, robustness, in_radius):
        """Append robustness and radius status to history."""
        self.history_rob.append(robustness)
        self.history_in_radius.append(in_radius)

    def record_maneuver_robustness(self, robustness):
        """Append maneuver spec robustness to history (None when encounter inactive)."""
        self.history_maneuver_rob.append(robustness)

    def set_current_heading_offset(self, offset_rad: float):
        """Cache the active heading offset (rad) for the next reward computation."""
        self._current_heading_offset = float(offset_rad)

    def set_current_speed_multiplier(self, speed_multiplier):
        """Cache the active speed multiplier for the next record() call."""
        if speed_multiplier is None:
            self._current_speed_multiplier = np.nan
        else:
            self._current_speed_multiplier = float(speed_multiplier)

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
        self._current_heading_offset = 0.0
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
        self.history_speed_multiplier = []
        self._current_speed_multiplier = np.nan
