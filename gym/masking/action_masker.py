import numpy as np

from pacstl.common.interfaces import PACReachableSet, TimeStampedState

from gym.utils.discrete_actions import decode_discrete_action
from gym.utils.geometry import to_obstacle_frame, within_monitoring_radius, wrap_angle
from gym.utils.robustness import extract_robustness_upper


class ActionMasker:
    def __init__(self, action, vessel_model=None, config=None):
        self.heading_offsets = np.asarray(action.heading_offsets, dtype=float)
        self.speed_multipliers = np.asarray(action.speed_multipliers, dtype=float)
        self.n_actions = len(self.heading_offsets) * len(self.speed_multipliers)
        self.v_max = np.inf
        self.r_max = np.inf
        self.sim_dt = 0.5
        self.k_heading = 1.0
        self.k_speed = 0.5
        self.a_max = 0.1
        self.configure(config=config, vessel_model=vessel_model)

        self.spec = None
        self.ellipsoids_Ab_dict = None
        self.tube_time_steps = []
        self.reachable_tube = {}
        self._using_cached_tube = False

    def configure(self, config=None, vessel_model=None):
        config = dict(config or {})

        v_max = config.get("v_max", getattr(vessel_model, "v_max", np.inf))
        r_max = config.get("r_max", getattr(vessel_model, "yaw_dot_max", np.inf))

        self.v_max = np.inf if v_max is None else float(v_max)
        self.r_max = np.inf if r_max is None else float(r_max)
        self.sim_dt = float(config.get("sim_dt", self.sim_dt))
        self.k_heading = float(config.get("k_heading", self.k_heading))
        self.k_speed = float(config.get("k_speed", self.k_speed))
        self.a_max = float(config.get("a_max", self.a_max))

    def update_scenario(self, spec, ellipsoids_Ab_dict, encounter_scenario=None):
        self.spec = spec
        self.ellipsoids_Ab_dict = ellipsoids_Ab_dict

        if (
            encounter_scenario is not None
            and encounter_scenario.ellipsoids_Ab_dict is ellipsoids_Ab_dict
            and encounter_scenario.reachable_tube
        ):
            self.tube_time_steps = list(encounter_scenario.tube_time_steps)
            self.reachable_tube = encounter_scenario.reachable_tube
            self._using_cached_tube = True
            print(
                f"[ReachableTube] ActionMasker reusing static cache with "
                f"{len(self.tube_time_steps)} steps"
            )
        elif ellipsoids_Ab_dict:
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
            self._using_cached_tube = False
        else:
            self.tube_time_steps = []
            self.reachable_tube = {}
            self._using_cached_tube = False

    def get_mask(
        self,
        situation,
        ego_state,
        encounter_vessel_eta,
        encounter_speed,
        robustness_margin,
        monitoring_radius,
        state=None,
    ):
        mask = np.ones(self.n_actions, dtype=bool)
        is_fallback = False

        if self.spec is None or self.ellipsoids_Ab_dict is None:
            return mask, is_fallback

        monitoring_state = ego_state if state is None else state
        if not within_monitoring_radius(
            encounter_vessel_eta, monitoring_radius, monitoring_state
        ):
            return mask, is_fallback

        mask = np.zeros(self.n_actions, dtype=bool)
        action_robustness = np.full(self.n_actions, np.inf)
        candidates = self._candidate_actions(situation)

        for action_idx in candidates:
            psi_d, u_d = decode_discrete_action(
                action_idx=action_idx,
                sim_state=ego_state,
                heading_offsets=self.heading_offsets,
                speed_multipliers=self.speed_multipliers,
            )
            ego_trajectory = self._simulate_candidate_trajectory(
                psi_d=psi_d,
                u_d=u_d,
                ego_state=ego_state,
                encounter_vessel_eta=encounter_vessel_eta,
                encounter_speed=encounter_speed,
            )
            robustness = self.spec.evaluate(self.reachable_tube, ego_trajectory)
            rob_upper = extract_robustness_upper(robustness)

            if rob_upper is not None:
                action_robustness[action_idx] = rob_upper
                mask[action_idx] = rob_upper < -robustness_margin

        if not mask.any():
            is_fallback = True
            candidate_rob = {
                idx: action_robustness[idx]
                for idx in candidates
                if np.isfinite(action_robustness[idx])
            }

            if candidate_rob:
                min_rob = min(candidate_rob.values())
                for idx, rob in candidate_rob.items():
                    if rob <= min_rob + 1.0:
                        mask[idx] = True

                print(
                    f"[ActionMask] Fallback after "
                    f"{'cached-tube' if self._using_cached_tube else 'local-tube'} "
                    f"verification: {mask.sum()} actions allowed "
                    f"(best robustness={min_rob:.2f})"
                )
            else:
                for action_idx in candidates:
                    h_idx = action_idx // len(self.speed_multipliers)
                    if self.heading_offsets[h_idx] > 0:
                        mask[action_idx] = True
                print("[ActionMask] Emergency fallback: starboard turns only")

        return mask, is_fallback

    def _candidate_actions(self, situation):
        candidates = []
        for action_idx in range(self.n_actions):
            h_idx = action_idx // len(self.speed_multipliers)
            if situation == "crossing" and self.heading_offsets[h_idx] < 0:
                continue
            candidates.append(action_idx)
        return candidates

    def _simulate_candidate_trajectory(
        self,
        psi_d,
        u_d,
        ego_state,
        encounter_vessel_eta,
        encounter_speed,
    ):
        eta, nu = ego_state["eta"], ego_state["nu"]
        px, py, psi = eta[0], eta[1], eta[-1]
        u = nu[0]

        obs_n, obs_e, obs_psi = encounter_vessel_eta
        obs_vn = encounter_speed * np.cos(obs_psi)
        obs_ve = encounter_speed * np.sin(obs_psi)

        ego_trajectory = {}
        t = 0.0

        for target_t in self.tube_time_steps:
            while t < target_t - 1e-9:
                dt_step = min(self.sim_dt, target_t - t)

                heading_error = wrap_angle(psi_d - psi)
                omega = np.clip(self.k_heading * heading_error, -self.r_max, self.r_max)

                speed_error = u_d - u
                a = np.clip(self.k_speed * speed_error, -self.a_max, self.a_max)

                px += np.cos(psi) * u * dt_step
                py += np.sin(psi) * u * dt_step
                psi += omega * dt_step
                u += a * dt_step
                u = np.clip(u, 0.0, self.v_max)
                t += dt_step

            vn = u * np.cos(psi)
            ve = u * np.sin(psi)

            pred_obs_n = obs_n + obs_vn * target_t
            pred_obs_e = obs_e + obs_ve * target_t

            local_pos, local_vel = to_obstacle_frame(
                ego_pos=np.array([px, py]),
                ego_vel=np.array([vn, ve]),
                obs_pos=np.array([pred_obs_n, pred_obs_e]),
                obs_psi=obs_psi,
                obs_vel=np.array([obs_vn, obs_ve]),
            )

            state_array = np.array(
                [
                    local_pos[0],
                    local_pos[1],
                    -wrap_angle(psi - obs_psi),
                    local_vel[0],
                    local_vel[1],
                    np.hypot(local_vel[0], local_vel[1]),
                ]
            )

            ego_trajectory[target_t] = TimeStampedState(
                time_step=target_t,
                state_array=state_array,
            )

        return ego_trajectory
