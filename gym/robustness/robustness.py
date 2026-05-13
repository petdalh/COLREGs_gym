from gym.utils.geometry import within_monitoring_radius
import interval
from pacstl.common.interfaces import TimeStampedState, PACReachableSet
import numpy as np
from gym.utils.geometry import to_obstacle_frame


class Robustness:
    def __init__(self, spec, ellipsoids_Ab_dict, sampling_rate):
        self.spec = spec
        self.ellipsoids_Ab_dict = ellipsoids_Ab_dict
        self.sampling_rate = sampling_rate
        self._cache_use_logged = False

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
                    encounter_scenario=state.encounter_scenario,
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
        encounter_scenario,
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
        if encounter_scenario is not None and encounter_scenario.reachable_tube:
            reachable_tube = encounter_scenario.reachable_tube
            tube_time_steps = encounter_scenario.tube_time_steps
            if not self._cache_use_logged:
                print(
                    f"[ReachableTube] Robustness reusing static cache with "
                    f"{len(tube_time_steps)} steps"
                )
                self._cache_use_logged = True
        else:
            tube_time_steps = sorted(self.ellipsoids_Ab_dict.keys())
            reachable_tube = {
                time_step: PACReachableSet(
                    time_step=time_step,
                    A_matrix=raw_tuple[0],
                    b_vector=raw_tuple[1],
                    center=raw_tuple[2],
                )
                for time_step, raw_tuple in self.ellipsoids_Ab_dict.items()
            }

        for time_step in tube_time_steps:
            # Predict ego position in world frame at this time step
            ego_n = eta[0] + ego_vn * time_step
            ego_e = eta[1] + ego_ve * time_step

            # Predict obstacle position in world frame at this time step
            pred_obs_n = obs_n + obs_vn * time_step
            pred_obs_e = obs_e + obs_ve * time_step

            local_pos, local_psi, local_vel = to_obstacle_frame(
                ego_pos=np.array([ego_n, ego_e]),
                ego_psi=psi_ego,
                ego_vel=np.array([ego_vn, ego_ve]),
                obs_pos=np.array([pred_obs_n, pred_obs_e]),
                obs_psi=obs_psi,
            )
            local_x, local_y = local_pos
            local_vx, local_vy = local_vel
            speed = np.hypot(local_vx, local_vy)

            state_array = np.array(
                [local_x, local_y, local_psi, local_vx, local_vy, speed]
            )

            ego_trajectory[time_step] = TimeStampedState(
                time_step=time_step, state_array=state_array
            )

        self._log_eval_inputs(
            eta=eta,
            ego_vn=ego_vn,
            ego_ve=ego_ve,
            obs_n=obs_n,
            obs_e=obs_e,
            obs_psi=obs_psi,
            obs_vn=obs_vn,
            obs_ve=obs_ve,
            ego_trajectory=ego_trajectory,
            reachable_tube=reachable_tube,
        )
        return self.spec.evaluate(reachable_tube, ego_trajectory)

    @staticmethod
    def _log_eval_inputs(
        eta, ego_vn, ego_ve, obs_n, obs_e, obs_psi,
        obs_vn, obs_ve, ego_trajectory, reachable_tube,
    ):
        # print("[Eval] ----------------------------------------")
        # print(
        #     f"[Eval] ego NED: pos=(N={eta[0]:.2f}, E={eta[1]:.2f}), "
        #     f"psi={np.degrees(eta[2]):+.1f}deg, "
        #     f"vel=(vN={ego_vn:+.3f}, vE={ego_ve:+.3f})"
        # )
        # print(
        #     f"[Eval] obs NED: pos=(N={obs_n:.2f}, E={obs_e:.2f}), "
        #     f"psi={np.degrees(obs_psi):+.1f}deg, "
        #     f"vel=(vN={obs_vn:+.3f}, vE={obs_ve:+.3f})"
        # )
        # print("[Eval] ego trajectory (obstacle body frame):")
        # for t in sorted(ego_trajectory.keys()):
        #     s = ego_trajectory[t].state_array
        #     print(
        #         f"[Eval]   t={t:>4}: pos=({s[0]:+.2f}, {s[1]:+.2f}), "
        #         f"psi={np.degrees(s[2]):+.1f}deg, "
        #         f"vel=({s[3]:+.3f}, {s[4]:+.3f}), |v|={s[5]:.3f}"
        #     )
        # print("[Eval] reachable set centers (obstacle body frame):")
        # for t in sorted(reachable_tube.keys()):
        #     c = reachable_tube[t].center
        #     print(f"[Eval]   t={t:>4}: center={np.array2string(np.asarray(c), precision=2, suppress_small=True)}")
        return
