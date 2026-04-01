from pacstl.core.evaluator import PacSTLEvaluator
from pacstl.common.interfaces import TimeStampedState, PACReachableSet
import numpy as np
from utils.geometry import within_monitoring_radius, wrap_angle, to_obstacle_frame
from utils.robustness import extract_robustness_upper
from mchorcrux.numpy_core.controllers.adaptive_seakeeping import heading_to_goal
from config import HEADING_OFFSETS, SPEED_MULTIPLIERS, N_DISCRETE_ACTIONS


class ActionMasker:
    """
    Stateful action masker for COLREGs compliance.

    The masker stores the vessel limits and cached pacSTL scenario data so the
    environment can call it repeatedly without rebuilding the same structures.
    """

    def __init__(
        self,
        v_max: float,
        r_max: float,
        sim_dt: float,
        k_heading: float = 1.0,
        k_speed: float = 0.5,
        a_max: float = 0.1,
    ):
        self.v_max = float(v_max)
        self.r_max = float(r_max)
        self.sim_dt = float(sim_dt)
        self.k_heading = float(k_heading)
        self.k_speed = float(k_speed)
        self.a_max = float(a_max)

        self.spec = None
        self.ellipsoids_Ab_dict = None
        self.tube_time_steps = []
        self.reachable_tube = {}

    def update_scenario(self, spec, ellipsoids_Ab_dict):
        """Store the current monitoring spec and pre-build tube data."""
        self.spec = spec
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
        else:
            self.tube_time_steps = []
            self.reachable_tube = {}

    def get_mask(
        self,
        situation: str,
        ego_state: dict,
        encounter_vessel_eta: tuple,
        encounter_speed: float,
        obs: np.ndarray,
        robustness_margin: float,
        monitoring_radius: float,
        state: dict | None = None,
    ) -> np.ndarray:
        """
        Evaluate all discrete actions and return a boolean mask.

        True means the action is allowed, False means it is masked out.
        """
        mask = np.ones(N_DISCRETE_ACTIONS, dtype=bool)

        if self.spec is None or self.ellipsoids_Ab_dict is None:
            return mask

        monitoring_state = ego_state if state is None else state
        if not within_monitoring_radius(
            encounter_vessel_eta, monitoring_radius, monitoring_state
        ):
            return mask

        mask = np.zeros(N_DISCRETE_ACTIONS, dtype=bool)
        action_robustness = np.full(N_DISCRETE_ACTIONS, np.inf)

        candidates = self._candidate_actions(situation)

        for action_idx in candidates:
            psi_d, u_d = decode_discrete_actions(action_idx, obs)
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
                    f"[ActionMask] Fallback: {mask.sum()} actions allowed "
                    f"(best robustness={min_rob:.2f})"
                )
            else:
                for action_idx in candidates:
                    h_idx = action_idx // len(SPEED_MULTIPLIERS)
                    if HEADING_OFFSETS[h_idx] > 0:
                        mask[action_idx] = True
                print("[ActionMask] Emergency fallback: starboard turns only")

        return mask

    def _candidate_actions(self, situation: str) -> list[int]:
        candidates = []
        for action_idx in range(N_DISCRETE_ACTIONS):
            h_idx = action_idx // len(SPEED_MULTIPLIERS)
            if situation == "crossing" and HEADING_OFFSETS[h_idx] < 0:
                continue
            candidates.append(action_idx)
        return candidates

    def _simulate_candidate_trajectory(
        self,
        psi_d: float,
        u_d: float,
        ego_state: dict,
        encounter_vessel_eta: np.ndarray,
        encounter_speed: float,
    ) -> dict:
        """
        Simulate a single candidate action forward over the cached tube times.
        """
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


_DEFAULT_MASKER: ActionMasker | None = None


def _get_default_masker(v_max: float, r_max: float, sim_dt: float) -> ActionMasker:
    global _DEFAULT_MASKER

    if (
        _DEFAULT_MASKER is None
        or _DEFAULT_MASKER.v_max != float(v_max)
        or _DEFAULT_MASKER.r_max != float(r_max)
        or _DEFAULT_MASKER.sim_dt != float(sim_dt)
    ):
        _DEFAULT_MASKER = ActionMasker(v_max=v_max, r_max=r_max, sim_dt=sim_dt)

    return _DEFAULT_MASKER


def get_action_mask(
    situation: str,
    maneuver_spec: PacSTLEvaluator,
    ellipsoids_Ab_dict: dict,
    encounter_vessel_eta: tuple,
    state: dict,
    encounter_speed: float,
    robustness_margin: float,
    monitoring_radius: float,
    obs: dict,
    robutness_margin: float,
    r_max: float,
    sim_dt: float,
    v_max: float,
) -> np.ndarray:
    """
    Backward-compatible wrapper around the stateful ActionMasker.
    """
    print("Computing action mask...")
    if ellipsoids_Ab_dict is None:
        return np.ones(N_DISCRETE_ACTIONS, dtype=bool)

    masker = _get_default_masker(v_max=v_max, r_max=r_max, sim_dt=sim_dt)
    masker.update_scenario(maneuver_spec, ellipsoids_Ab_dict)
    return masker.get_mask(
        situation=situation,
        ego_state=state,
        encounter_vessel_eta=encounter_vessel_eta,
        encounter_speed=encounter_speed,
        obs=obs,
        robustness_margin=robustness_margin,
        monitoring_radius=monitoring_radius,
        state=state,
    )


def decode_discrete_actions(action_idx: int, obs: dict) -> tuple:
    h_idx = action_idx // len(SPEED_MULTIPLIERS)
    s_idx = action_idx % len(SPEED_MULTIPLIERS)

    n, e, psi = obs[0], obs[1], obs[2]
    gn, ge = obs[6], obs[7]
    psi_goal = heading_to_goal(n, e, gn, ge)

    psi_d = psi_goal + HEADING_OFFSETS[h_idx]
    u_d = SPEED_MULTIPLIERS[s_idx]
    return psi_d, u_d


def simulate_candidate_trajectory(
    psi_d: float,
    u_d: float,
    state: dict,
    encounter_vessel_eta: np.ndarray,
    encounter_speed: float,
    r_max: float,
    ellipsoids_Ab_dict: dict,
    sim_dt: float,
    v_max: float,
) -> dict:
    """
    Backward-compatible wrapper around the class-based candidate simulation.
    """
    masker = ActionMasker(v_max=v_max, r_max=r_max, sim_dt=sim_dt)
    masker.update_scenario(None, ellipsoids_Ab_dict)
    return masker._simulate_candidate_trajectory(
        psi_d=psi_d,
        u_d=u_d,
        ego_state=state,
        encounter_vessel_eta=encounter_vessel_eta,
        encounter_speed=encounter_speed,
    )
