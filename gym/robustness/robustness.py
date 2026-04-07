from gym.utils.geometry import within_monitoring_radius
from gym.utils.robustness import evaluate_robustness
import interval
from pacstl.core.evaluator import PacSTLEvaluator
from pacstl.common.interfaces import TimeStampedState, PACReachableSet
import numpy as np
from gym.utils.geometry import to_obstacle_frame, wrap_angle


class Robustness:
    def __init__(self, spec, ellipsoids_Ab_dict, sampling_rate):
        self.spec = spec
        self.ellipsoids_Ab_dict = ellipsoids_Ab_dict
        self.sampling_rate = sampling_rate

    def evaluate(self, state, sim_step_count, in_radius=None):
        """
        Evaluate robustness if within sampling rate and monitoring radius.

        Returns the robustness value or None. Also updates state encounter
        tracking and records robustness/radius to history.
        """
        if in_radius is None:
            in_radius = state.in_monitoring_radius
        if in_radius is None:
            in_radius = within_monitoring_radius(
                state.encounter_vessel_eta,
                state.monitoring_radius,
                state.sim_state,
            )

        robustness = None
        if sim_step_count % self.sampling_rate == 0:
            if in_radius:
                robustness = self.evaluate_robustness(
                    encounter_vessel_eta=state.encounter_vessel_eta,
                    state=state.sim_state,
                    encounter_speed=state.encounter_speed,
                )
            elif state._encounter_active:
                state.clear_encounter()

        state.record_robustness(robustness, in_radius)
        return robustness

    def evaluate_robustness(
        self,
        encounter_vessel_eta: tuple,
        state: dict,
        encounter_speed: float,
    ) -> interval.interval:
        """Evaluate the pacSTL specification over the prediction horizon."""
        if self.spec is None or self.ellipsoids_Ab_dict is None:
            return None
        if encounter_vessel_eta is None:
            return None

        eta, nu = state["eta"], state["nu"]
        psi_ego = eta[-1]
        u, v_sway = nu[0], nu[1]

        # World-frame velocities of ego
        ego_vn = u * np.cos(psi_ego) - v_sway * np.sin(psi_ego)
        ego_ve = u * np.sin(psi_ego) + v_sway * np.cos(psi_ego)

        # Obstacle current state
        obs_n, obs_e, obs_psi = encounter_vessel_eta
        obs_vn = encounter_speed * np.cos(obs_psi)
        obs_ve = encounter_speed * np.sin(obs_psi)

        ego_trajectory = {}
        reachable_tube = {}

        for time_step, raw_tuple in self.ellipsoids_Ab_dict.items():
            A, b, c = raw_tuple

            # Predict ego position in world frame at this time step
            ego_n = eta[0] + ego_vn * time_step
            ego_e = eta[1] + ego_ve * time_step

            # Predict obstacle position in world frame at this time step
            pred_obs_n = obs_n + obs_vn * time_step
            pred_obs_e = obs_e + obs_ve * time_step

            local_pos, local_vel = to_obstacle_frame(
                ego_pos=np.array([ego_n, ego_e]),
                ego_vel=np.array([ego_vn, ego_ve]),
                obs_pos=np.array([pred_obs_n, pred_obs_e]),
                obs_psi=obs_psi,
                obs_vel=np.array([obs_vn, obs_ve]),
            )
            local_x, local_y = local_pos

            # Relative heading (same as original)
            psi_rel = -wrap_angle(psi_ego - obs_psi)
            local_vx, local_vy = local_vel
            speed = np.hypot(local_vx, local_vy)

            # 6D state in obstacle-relative frame:
            # [p_x, p_y, psi_rel, v_x, v_y, |v|]
            state_array = np.array(
                [local_x, local_y, psi_rel, local_vx, local_vy, speed]
            )

            ego_trajectory[time_step] = TimeStampedState(
                time_step=time_step, state_array=state_array
            )
            reachable_tube[time_step] = PACReachableSet(
                time_step=time_step, A_matrix=A, b_vector=b, center=c
            )

        return self.spec.evaluate(reachable_tube, ego_trajectory)