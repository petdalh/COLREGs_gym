from pacstl.core.evaluator import PacSTLEvaluator
from pacstl.common.interfaces import TimeStampedState, PACReachableSet
import numpy as np
from vessel_utils import simulate_candidate_trajectory, within_monitoring_radius
from robustness_utils import extract_robustness_upper
from mchorcrux.numpy_core.controllers.adaptive_seakeeping import heading_to_goal
from config import HEADING_OFFSETS, SPEED_MULTIPLIERS, N_DISCRETE_ACTIONS

def get_action_mask(situation: str, maneuver_spec: PacSTLEvaluator, ellipsoids_Ab_dict: dict, encounter_vessel_eta: tuple, state: dict, encounter_speed: float, robustness_margin: float, monitoring_radius: float, obs: dict, robutness_margin: float, r_max: float, sim_dt: float, v_max: float) -> np.ndarray:
    print("Computing action mask...")
    if ellipsoids_Ab_dict is None:
        return np.ones(N_DISCRETE_ACTIONS, dtype=bool)

    # Skip mask computation if outside monitoring radius
    if not within_monitoring_radius(encounter_vessel_eta, monitoring_radius, state):
        return np.ones(N_DISCRETE_ACTIONS, dtype=bool)

    mask = np.zeros(N_DISCRETE_ACTIONS, dtype=bool)

    # Track robustness per action for ranked fallback
    action_robustness = np.full(N_DISCRETE_ACTIONS, np.inf)

    reachable_tube = {}
    for time_step, raw_tuple in ellipsoids_Ab_dict.items():
        A, b, c = raw_tuple
        reachable_tube[time_step] = PACReachableSet(
            time_step=time_step, A_matrix=A, b_vector=b, center=c
        )

    # Pre-filter: skip actions that violate COLREGS turning direction
    candidates = []
    for action_idx in range(N_DISCRETE_ACTIONS):
        h_idx = action_idx // len(SPEED_MULTIPLIERS)
        if situation == "crossing" and HEADING_OFFSETS[h_idx] < 0:
            continue
        candidates.append(action_idx)

    # Evaluate each candidate action
    for action_idx in candidates:
        psi_d, u_d = decode_discrete_actions(action_idx, obs)
        ego_trajectory = simulate_candidate_trajectory(psi_d, u_d, state, encounter_vessel_eta, encounter_speed, r_max, ellipsoids_Ab_dict, sim_dt, v_max)
        robustness = maneuver_spec.evaluate(reachable_tube, ego_trajectory)

        # Extract the upper bound of the robustness interval
        rob_upper = extract_robustness_upper(robustness)

        if rob_upper is not None:
            action_robustness[action_idx] = rob_upper
            # Allow action if its robustness upper bound is safely below
            # the margin. Negative robustness = no encounter detected.
            # The margin pushes the threshold below zero so the agent
            # avoids actions that are *close to* triggering an encounter.
            mask[action_idx] = rob_upper < -robustness_margin

    # Fallback: if no action passes the margin, allow the least-bad ones.
    # This prevents the all-zeros mask that crashes MaskablePPO.
    if not mask.any():
        # Only consider COLREGS-directional candidates
        candidate_rob = {idx: action_robustness[idx] for idx in candidates
                        if np.isfinite(action_robustness[idx])}

        if candidate_rob:
            # Pick actions with the lowest (most negative) robustness —
            # these are furthest from violation
            min_rob = min(candidate_rob.values())
            # Allow all actions within 1.0 of the best available
            for idx, rob in candidate_rob.items():
                if rob <= min_rob + 1.0:
                    mask[idx] = True

            print(f"[ActionMask] Fallback: {mask.sum()} actions allowed "
                    f"(best robustness={min_rob:.2f})")
        else:
            # Absolute last resort: allow all starboard turns
            for action_idx in candidates:
                h_idx = action_idx // len(SPEED_MULTIPLIERS)
                if HEADING_OFFSETS[h_idx] > 0:
                    mask[action_idx] = True
            print("[ActionMask] Emergency fallback: starboard turns only")

    return mask

def decode_discrete_actions(action_idx: int, obs: dict) -> tuple:
    h_idx = action_idx // len(SPEED_MULTIPLIERS)
    s_idx = action_idx % len(SPEED_MULTIPLIERS)

    n, e, psi = obs[0], obs[1], obs[2]
    gn, ge = obs[6], obs[7]
    psi_goal = heading_to_goal(n, e, gn, ge)

    psi_d = psi_goal + HEADING_OFFSETS[h_idx]
    u_d = SPEED_MULTIPLIERS[s_idx]
    return psi_d, u_d