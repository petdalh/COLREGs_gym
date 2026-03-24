"""
Minimal COLREGs encounter gym for RL training with pacSTL monitoring.

Wraps the numpy_core McGym with encounter vessel kinematics,
collision detection, and pacSTL robustness evaluation.
"""

import numpy as np
from mchorcrux.numpy_core.gym.mc_gym_csad_numpy import McGym
from pacstl.common.interfaces import PACReachableSet, TimeStampedState
from pacstl.domains.colregs.utils import VesselModel


class ColregsGym(McGym):
    def __init__(
        self,
        vessel_model: VesselModel,
        dt: float,
        grid_width: float,
        grid_height: float,
        **kwargs,
    ):
        super().__init__(
            dt=dt, grid_width=grid_width, grid_height=grid_height, **kwargs
        )

        # Encounter state
        self.encounter_type: str | None = None
        self._encounter_init: np.ndarray | None = None
        self.encounter_vessel_eta: np.ndarray | None = None
        self.encounter_speed: float = 0.0
        self.encounter_radius: float = 1.0
        self.encounter_max_time: float | None = None

        # pacSTL evaluator (set via configure_monitoring)
        self.spec = None
        self.ellipsoids_Ab_dict: dict | None = None
        self.robustness_sampling_rate: int = 10

        self.v_min = vessel_model.v_min
        self.v_max = vessel_model.v_max
        self.r_min = vessel_model.yaw_dot_min
        self.r_max = vessel_model.yaw_dot_max

        # Reward bookkeeping
        self.prev_dist_to_goal: float = 0.0

        # Step counter
        self._step_count: int = 0

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def configure_monitoring(self, spec, ellipsoids_Ab_dict, sampling_rate: int = 10):
        """Attach a pacSTL evaluator and preloaded reachable sets."""
        self.spec = spec
        self.ellipsoids_Ab_dict = ellipsoids_Ab_dict
        self.robustness_sampling_rate = sampling_rate

    def set_encounter(
        self,
        start_position: tuple,
        wave_conditions: tuple,
        encounter_type: str = "crossing",
        separation: float = 30.0,
        target_speed: float = 0.3,
        goal_ahead_distance: float = 25.0,
        collision_radius: float = 1.0,
        simtime: float = 150.0,
    ):
        """Configure a COLREGs encounter scenario."""

        if target_speed > self.v_max:
            raise ValueError(
                f"target_speed={target_speed} exceeds vessel model v_max={self.v_max}. "
                "The pacSTL scaling assumes the obstacle speed is within [v_min, v_max]."
            )
        if target_speed < self.v_min:
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

        self.encounter_type = encounter_type
        self.encounter_speed = float(target_speed)
        self.encounter_radius = float(collision_radius)
        self.encounter_max_time = float(simtime)
        self._encounter_init = np.array([t_n, t_e, np.deg2rad(t_psi_deg)])

        goal_n = own_n + goal_ahead_distance * np.cos(own_psi_rad)
        goal_e = own_e + goal_ahead_distance * np.sin(own_psi_rad)

        self.set_task(
            start_position=start_position,
            goal=(goal_n, goal_e, 1.0),
            wave_conditions=wave_conditions,
            simtime=simtime,
        )

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        super().reset()
        if self._encounter_init is not None:
            self.encounter_vessel_eta = self._encounter_init.copy()

        state = self.get_state()
        gn, ge = self.goal[:2]
        self.prev_dist_to_goal = np.hypot(gn - state["eta"][0], ge - state["eta"][1])
        self._step_count = 0
        return self._obs()

    def step(self, action):
        if self.encounter_vessel_eta is not None:
            self._propagate_encounter_vessel()

        obs, reward, done, info = super().step(action)
        self._step_count += 1

        # Evaluate pacSTL at configured rate
        if self._step_count % self.robustness_sampling_rate == 0:
            info["robustness"] = self.evaluate_robustness()

        return self._obs(), reward, done, info

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _obs(self) -> np.ndarray:
        state = self.get_state()
        eta, nu = state["eta"], state["nu"]
        gn, ge = self.goal[:2]
        tgt = (
            self.encounter_vessel_eta[:2]
            if self.encounter_vessel_eta is not None
            else np.zeros(2)
        )
        return np.concatenate([eta[:2], [eta[-1]], nu, [gn, ge], tgt])

    # ------------------------------------------------------------------
    # pacSTL monitoring
    # ------------------------------------------------------------------

    def evaluate_robustness(self):
        """Evaluate the pacSTL specification over the prediction horizon."""
        if self.spec is None or self.ellipsoids_Ab_dict is None:
            return None

        state = self.get_state()
        eta, nu = state["eta"], state["nu"]
        psi = eta[-1]
        u, v_sway = nu[0], nu[1]

        # World-frame velocities
        ego_vn = u * np.cos(psi) - v_sway * np.sin(psi)
        ego_ve = u * np.sin(psi) + v_sway * np.cos(psi)

        ego_now = np.array([eta[0], eta[1], psi, ego_vn, ego_ve, 0.0])

        # Predict ego trajectory: constant velocity, constant heading
        ego_trajectory = {}
        reachable_tube = {}
        for time_step, raw_tuple in self.ellipsoids_Ab_dict.items():
            A, b, c = raw_tuple

            predicted = ego_now.copy()
            predicted[0] += ego_vn * time_step
            predicted[1] += ego_ve * time_step

            ego_trajectory[time_step] = TimeStampedState(
                time_step=time_step, state_array=predicted
            )
            reachable_tube[time_step] = PACReachableSet(
                time_step=time_step, A_matrix=A, b_vector=b, center=c
            )

        return self.spec.evaluate(ego_trajectory, reachable_tube)

    # ------------------------------------------------------------------
    # Encounter vessel kinematics
    # ------------------------------------------------------------------

    def _propagate_encounter_vessel(self):
        n, e, psi = self.encounter_vessel_eta
        n += self.encounter_speed * np.cos(psi) * self.dt
        e += self.encounter_speed * np.sin(psi) * self.dt
        self.encounter_vessel_eta = np.array([n, e, psi])

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------

    def _check_termination(self, boat_pos):
        done, info = super()._check_termination(boat_pos)
        if done:
            return done, info

        if self.encounter_max_time and self.curr_sim_time > self.encounter_max_time:
            return True, {"reason": "time_limit"}

        if self.encounter_vessel_eta is not None:
            dist = np.hypot(
                boat_pos[0] - self.encounter_vessel_eta[0],
                boat_pos[1] - self.encounter_vessel_eta[1],
            )
            if dist < self.encounter_radius:
                return True, {"reason": "collision"}

        return False, {}
