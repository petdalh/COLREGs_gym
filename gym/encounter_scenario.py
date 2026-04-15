import numpy as np
from pacstl.common.interfaces import PACReachableSet


class EncounterScenario:
    def __init__(
        self,
        vessel_model,
        maneuver_horizon=10.0,
        monitoring_radius_safety_factor=2.0,
        masking_configuration=None,
        seed=None,
    ):
        self.v_min = getattr(vessel_model, "v_min", None)
        self.v_max = getattr(vessel_model, "v_max", None)
        self.maneuver_horizon = maneuver_horizon
        self.monitoring_radius_safety_factor = monitoring_radius_safety_factor
        self.default_masking_configuration = dict(masking_configuration or {})
        self.masking_configuration = dict(self.default_masking_configuration)
        self.start_position = None
        self.wave_conditions = None
        self.encounter_type = None
        self.separation = None
        self.target_speed = None
        self.goal_ahead_distance = None
        self.collision_radius = None
        self.simtime = None
        self.nominal_path_start = None
        self._encounter_init = None
        self.goal = None
        self.ellipsoids_Ab_dict = None
        self.reachable_tube = {}
        self.tube_time_steps = []
        self._rng = np.random.default_rng(seed)


    def set_encounter(
        self,
        start_position,
        wave_conditions,
        encounter_type="crossing",
        separation=30.0,
        target_speed=0.3,
        goal_ahead_distance=25.0,
        collision_radius=1.0,
        simtime=150.0,
        masking_configuration=None,
    ):
        if self.v_max is not None and target_speed > self.v_max:
            raise ValueError(
                f"target_speed={target_speed} exceeds vessel model v_max={self.v_max}. "
                "The pacSTL scaling assumes the obstacle speed is within [v_min, v_max]."
            )
        if self.v_min is not None and target_speed < self.v_min:
            raise ValueError(
                f"target_speed={target_speed} is below vessel model v_min={self.v_min}."
            )

        own_n, own_e, own_psi_deg = start_position

        self.start_position = start_position
        self.wave_conditions = wave_conditions
        self.encounter_type = encounter_type
        self.separation = separation
        self.target_speed = float(target_speed)
        self.goal_ahead_distance = goal_ahead_distance
        self.collision_radius = float(collision_radius)
        self.simtime = float(simtime)
        self.masking_configuration = {
            **self.default_masking_configuration,
            **dict(masking_configuration or {}),
        }
        self.nominal_path_start = np.array([own_n, own_e], dtype=float)
    
        if encounter_type == "crossing":
            self._set_crossing_encounter(own_n, own_e, own_psi_deg)
        else:
            raise ValueError(f"Unsupported encounter type: {encounter_type}")

    def _set_crossing_encounter(self, own_n, own_e, own_psi_deg, noise: np.random.Generator = None):
        if noise is None:
            noise = self._rng
        own_psi_rad = np.deg2rad(own_psi_deg)
        bearing_rad = own_psi_rad + np.deg2rad(45.0)

        pos_noise_n = noise.uniform(-2.0, 2.0)
        pos_noise_e = noise.uniform(-2.0, 2.0)
        heading_noise_deg = noise.uniform(-5.0, 5.0)

        t_n = own_n + self.separation * np.cos(bearing_rad) + pos_noise_n
        t_e = own_e + self.separation * np.sin(bearing_rad) + pos_noise_e
        t_psi_deg = (own_psi_deg - 90.0 + heading_noise_deg) % 360.0 

        goal_n = own_n + self.goal_ahead_distance * np.cos(own_psi_rad)
        goal_e = own_e + self.goal_ahead_distance * np.sin(own_psi_rad)

        self._encounter_init = np.array([t_n, t_e, np.deg2rad(t_psi_deg)])
        self.goal = (goal_n, goal_e, 1.0)

    def configure_monitoring_cache(self, ellipsoids_Ab_dict):
        self.ellipsoids_Ab_dict = ellipsoids_Ab_dict

        if ellipsoids_Ab_dict:
            self.tube_time_steps = sorted(ellipsoids_Ab_dict.keys())
            self.reachable_tube = {
                time_step: PACReachableSet(
                    time_step=time_step,
                    A_matrix=raw_tuple[0],
                    b_vector=raw_tuple[1],
                    center=raw_tuple[2],
                )
                for time_step, raw_tuple in ellipsoids_Ab_dict.items()
            }
            print(
                f"[ReachableTube] Built static cache with "
                f"{len(self.tube_time_steps)} steps: {self.tube_time_steps}"
            )
        else:
            self.reachable_tube = {}
            self.tube_time_steps = []

    def apply(self, env):
        env.start_position = np.array(
            [
                self.start_position[0],
                self.start_position[1],
                np.deg2rad(self.start_position[2]),
            ]
        )
        env.goal = self.goal
        env.wave_conditions = self.wave_conditions
        env.state.encounter_scenario = self
        env.state.goal = self.goal
        env.state.nominal_path_start = self.nominal_path_start
        env.state.encounter_speed = self.target_speed
        env.state.encounter_vessel_eta = None
        env.state._goal = self.goal
        env.state._nominal_path_start = self.nominal_path_start
        env.state._encounter_speed = self.target_speed

    def apply_to_state(self, state, encounter_vessel_eta):
        state.encounter_scenario = self
        state.goal = self.goal
        state.nominal_path_start = self.nominal_path_start
        state.encounter_vessel_eta = encounter_vessel_eta
        state.encounter_speed = self.target_speed
    
    def reset(self, state, seed=None):
        """Re-roll noise and apply the encounter to the given state."""
        if seed is not None:
            self._rng = np.random.default_rng(seed=seed)
            
        if self.start_position is not None and self._rng is not None:
            own_n, own_e, own_psi_deg = self.start_position
            self._set_crossing_encounter(own_n, own_e, own_psi_deg, noise=self._rng)

        encounter_eta = self._encounter_init.copy() if self._encounter_init is not None else None
        self.apply_to_state(state, encounter_eta)
        state.encounter_scenario = self
