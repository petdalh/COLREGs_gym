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
        self.base_wave_conditions = None
        self.wave_conditions = None
        self.wave_direction_range_deg = None
        self.encounter_type = None
        self.separation = None
        self.target_speed = None
        self.target_speed_options = None
        self.target_speed_range = None
        self.time_to_conflict_s = None
        self.ego_reference_speed = None
        self.heading_noise_deg = 5.0
        self.encounter_position_noise_m = 2.0
        self.target_cross_track_noise_m = 0.0
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
        target_speeds=None,
        target_speed_range=None,
        time_to_conflict_s=None,
        ego_reference_speed=None,
        heading_noise_deg=5.0,
        encounter_position_noise_m=2.0,
        target_cross_track_noise_m=0.0,
        goal_ahead_distance=25.0,
        collision_radius=1.0,
        simtime=150.0,
        masking_configuration=None,
        wave_direction_range_deg=None,
    ):
        if (
            target_speed_range is not None or target_speeds is not None
        ) and time_to_conflict_s is None:
            raise ValueError(
                "Sampled target speeds require time_to_conflict_s so sampled speeds "
                "produce speed-aware crossing geometry."
            )

        target_speed_range = self._normalize_target_speed_range(target_speed_range)
        target_speed_options = self._normalize_target_speeds(
            target_speed=target_speed,
            target_speeds=target_speeds,
        )
        wave_conditions = self._normalize_wave_conditions(wave_conditions)
        wave_direction_range_deg = self._normalize_wave_direction_range_deg(
            wave_direction_range_deg
        )
        speeds_to_validate = (
            target_speed_range if target_speed_range is not None else target_speed_options
        )
        for speed in speeds_to_validate:
            self._validate_target_speed(speed)

        own_n, own_e, own_psi_deg = start_position

        self.start_position = start_position
        self.base_wave_conditions = wave_conditions
        self.wave_direction_range_deg = wave_direction_range_deg
        self.encounter_type = encounter_type
        self.separation = separation
        self.target_speed_range = target_speed_range
        self.target_speed_options = target_speed_options
        self.time_to_conflict_s = (
            None if time_to_conflict_s is None else float(time_to_conflict_s)
        )
        self.ego_reference_speed = (
            None if ego_reference_speed is None else float(ego_reference_speed)
        )
        self.heading_noise_deg = float(heading_noise_deg)
        self.encounter_position_noise_m = float(encounter_position_noise_m)
        self.target_cross_track_noise_m = float(target_cross_track_noise_m)
        self.target_speed = None
        self.goal_ahead_distance = goal_ahead_distance
        self.collision_radius = float(collision_radius)
        self.simtime = float(simtime)
        self.masking_configuration = {
            **self.default_masking_configuration,
            **dict(masking_configuration or {}),
        }
        self.nominal_path_start = np.array([own_n, own_e], dtype=float)

        self.prepare_reset()

    def _normalize_wave_conditions(self, wave_conditions):
        waves = np.asarray(wave_conditions, dtype=float).reshape(-1)
        if waves.size != 3:
            raise ValueError(
                "wave_conditions must contain exactly three values: "
                "(Hs, Tp, wave_dir_deg)."
            )
        if not np.all(np.isfinite(waves)):
            raise ValueError("wave_conditions must contain only finite values.")
        return tuple(float(value) for value in waves)

    def _normalize_wave_direction_range_deg(self, wave_direction_range_deg):
        if wave_direction_range_deg is None:
            return None

        direction_range = np.asarray(wave_direction_range_deg, dtype=float).reshape(-1)
        if direction_range.size != 2:
            raise ValueError(
                "wave_direction_range_deg must contain exactly two directions."
            )
        if not np.all(np.isfinite(direction_range)):
            raise ValueError(
                "wave_direction_range_deg must contain only finite values."
            )

        low, high = direction_range
        if low < 0.0 or high > 360.0:
            raise ValueError("wave_direction_range_deg must lie within [0, 360].")
        if low >= high:
            raise ValueError("wave_direction_range_deg min must be < max.")
        return (float(low), float(high))

    def _sample_wave_conditions(self):
        if self.base_wave_conditions is None:
            return None
        if self.wave_direction_range_deg is None:
            return self.base_wave_conditions

        hs, tp, _ = self.base_wave_conditions
        low, high = self.wave_direction_range_deg
        wave_dir_deg = float(self._rng.uniform(low, high))
        return (hs, tp, wave_dir_deg)

    def prepare_reset(self, seed=None):
        """Re-roll sampled encounter and wave parameters for the next reset."""
        if seed is not None:
            self._rng = np.random.default_rng(seed=seed)

        if self.start_position is None or self._rng is None:
            self.wave_conditions = self._sample_wave_conditions()
            return

        self.target_speed = self._sample_target_speed()
        own_n, own_e, own_psi_deg = self.start_position
        if self.encounter_type == "crossing":
            self._set_crossing_encounter(own_n, own_e, own_psi_deg)
        else:
            raise ValueError(f"Unsupported encounter type: {self.encounter_type}")
        self.wave_conditions = self._sample_wave_conditions()

    def _normalize_target_speeds(self, target_speed, target_speeds):
        if target_speeds is None:
            return np.array([float(target_speed)], dtype=float)

        speeds = np.asarray(target_speeds, dtype=float).reshape(-1)
        if speeds.size == 0:
            raise ValueError("target_speeds must contain at least one speed.")
        if not np.all(np.isfinite(speeds)):
            raise ValueError("target_speeds must contain only finite values.")
        return speeds

    def _normalize_target_speed_range(self, target_speed_range):
        if target_speed_range is None:
            return None

        speed_range = np.asarray(target_speed_range, dtype=float).reshape(-1)
        if speed_range.size != 2:
            raise ValueError("target_speed_range must contain exactly two speeds.")
        if not np.all(np.isfinite(speed_range)):
            raise ValueError("target_speed_range must contain only finite values.")
        if speed_range[0] > speed_range[1]:
            raise ValueError("target_speed_range min must be <= max.")
        return speed_range

    def _validate_target_speed(self, speed):
        if self.v_max is not None and speed > self.v_max:
            raise ValueError(
                f"target_speed={speed} exceeds vessel model v_max={self.v_max}. "
                "The pacSTL scaling assumes the obstacle speed is within [v_min, v_max]."
            )
        if self.v_min is not None and speed < self.v_min:
            raise ValueError(
                f"target_speed={speed} is below vessel model v_min={self.v_min}."
            )

    def _sample_target_speed(self):
        if self.target_speed_range is not None:
            low, high = self.target_speed_range
            return float(self._rng.uniform(low, high))
        if self.target_speed_options is None:
            return self.target_speed
        speed_idx = self._rng.integers(len(self.target_speed_options))
        return float(self.target_speed_options[speed_idx])

    def _set_crossing_encounter(self, own_n, own_e, own_psi_deg, noise: np.random.Generator = None):
        if noise is None:
            noise = self._rng
        own_psi_rad = np.deg2rad(own_psi_deg)

        heading_noise_deg = noise.uniform(-self.heading_noise_deg, self.heading_noise_deg)
        t_psi_deg = (own_psi_deg - 90.0 + heading_noise_deg) % 360.0
        t_psi_rad = np.deg2rad(t_psi_deg)

        goal_n = own_n + self.goal_ahead_distance * np.cos(own_psi_rad)
        goal_e = own_e + self.goal_ahead_distance * np.sin(own_psi_rad)

        if self.time_to_conflict_s is None:
            bearing_rad = own_psi_rad + np.deg2rad(45.0)
            pos_noise_n = noise.uniform(
                -self.encounter_position_noise_m,
                self.encounter_position_noise_m,
            )
            pos_noise_e = noise.uniform(
                -self.encounter_position_noise_m,
                self.encounter_position_noise_m,
            )

            t_n = own_n + self.separation * np.cos(bearing_rad) + pos_noise_n
            t_e = own_e + self.separation * np.sin(bearing_rad) + pos_noise_e
        else:
            if self.time_to_conflict_s <= 0.0:
                raise ValueError("time_to_conflict_s must be positive.")
            if self.ego_reference_speed is None or self.ego_reference_speed <= 0.0:
                raise ValueError(
                    "ego_reference_speed must be positive when time_to_conflict_s is set."
                )
            ego_conflict_distance = self.ego_reference_speed * self.time_to_conflict_s
            if ego_conflict_distance >= self.goal_ahead_distance:
                raise ValueError(
                    "ego_reference_speed * time_to_conflict_s must be less than "
                    "goal_ahead_distance so the crossing point lies before the goal."
                )

            own_unit = np.array([np.cos(own_psi_rad), np.sin(own_psi_rad)])
            target_unit = np.array([np.cos(t_psi_rad), np.sin(t_psi_rad)])
            conflict_point = (
                np.array([own_n, own_e], dtype=float)
                + ego_conflict_distance * own_unit
            )
            target_start = (
                conflict_point
                - self.target_speed * self.time_to_conflict_s * target_unit
            )
            if self.target_cross_track_noise_m > 0.0:
                lateral_unit = np.array([-target_unit[1], target_unit[0]])
                lateral_noise = noise.uniform(
                    -self.target_cross_track_noise_m,
                    self.target_cross_track_noise_m,
                )
                target_start = target_start + lateral_noise * lateral_unit
            t_n, t_e = target_start

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
    
    def reset(self, state, seed=None, prepared=False):
        """Apply prepared episode parameters to the state."""
        if not prepared:
            self.prepare_reset(seed=seed)

        encounter_eta = self._encounter_init.copy() if self._encounter_init is not None else None
        self.apply_to_state(state, encounter_eta)
        state.encounter_scenario = self
