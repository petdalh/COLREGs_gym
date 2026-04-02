import numpy as np
from gym.utils.geometry import propagate_vessel


class State:
    def __init__(self, monitoring_radius, n_actions):
        self.monitoring_radius = monitoring_radius

        self.encounter_vessel_eta = None
        self.encounter_speed = None

        self._encounter_active = False
        self._active_maneuver_spec = None
        self._cached_mask = np.ones(n_actions, dtype=bool)
        self._n_actions = n_actions

        self.sim_state = None

        self.goal = None
        self.nominal_path_start = None

        self.history_ego = []
        self.history_enc = []
        self.history_rob = []
        self.history_in_radius = []

    def update_sim(self, sim_state):
        """Store the latest sim state snapshot from get_state()."""
        self.sim_state = sim_state

    @property
    def position(self):
        """Current vessel position [x, y]."""
        return self.sim_state["eta"][:2]

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

    def record_robustness(self, robustness, in_radius):
        """Append robustness and radius status to history."""
        self.history_rob.append(robustness)
        self.history_in_radius.append(in_radius)

    def clear_encounter(self):
        """Reset encounter state when vessel leaves monitoring radius."""
        self._encounter_active = False
        self._active_maneuver_spec = None
        self._cached_mask = np.ones(self._n_actions, dtype=bool)

    def reset(self):
        """Reset all episode state. Called at the start of each episode."""
        self.encounter_vessel_eta = None
        self.encounter_speed = None
        self._encounter_active = False
        self._active_maneuver_spec = None
        self._cached_mask = np.ones(self._n_actions, dtype=bool)
        self.sim_state = None
        self.goal = None
        self.nominal_path_start = None
        self.history_ego = []
        self.history_enc = []
        self.history_rob = []
        self.history_in_radius = []
