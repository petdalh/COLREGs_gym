import numpy as np


class EncounterScenario:
    def __init__(
        self,
        vessel_model,
        maneuver_horizon=10.0,
        monitoring_radius_safety_factor=2.0,
    ):
        self.v_min = getattr(vessel_model, "v_min", None)
        self.v_max = getattr(vessel_model, "v_max", None)
        self.maneuver_horizon = maneuver_horizon
        self.monitoring_radius_safety_factor = monitoring_radius_safety_factor
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
        own_psi_rad = np.deg2rad(own_psi_deg)

        if encounter_type == "crossing":
            bearing_rad = own_psi_rad + np.deg2rad(45.0)
            t_n = own_n + separation * np.cos(bearing_rad)
            t_e = own_e + separation * np.sin(bearing_rad)
            t_psi_deg = (own_psi_deg - 90.0) % 360.0
        else:
            raise ValueError(f"Unsupported encounter type: {encounter_type}")

        goal_n = own_n + goal_ahead_distance * np.cos(own_psi_rad)
        goal_e = own_e + goal_ahead_distance * np.sin(own_psi_rad)

        self.start_position = start_position
        self.wave_conditions = wave_conditions
        self.encounter_type = encounter_type
        self.separation = separation
        self.target_speed = float(target_speed)
        self.goal_ahead_distance = goal_ahead_distance
        self.collision_radius = float(collision_radius)
        self.simtime = float(simtime)
        self.nominal_path_start = np.array([own_n, own_e], dtype=float)
        self._encounter_init = np.array([t_n, t_e, np.deg2rad(t_psi_deg)])
        self.goal = (goal_n, goal_e, 1.0)

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
