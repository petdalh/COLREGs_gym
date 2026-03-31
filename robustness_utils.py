from pacstl.core.evaluator import PacSTLEvaluator
from pacstl.common.interfaces import TimeStampedState, PACReachableSet
import interval
import numpy as np
from vessel_utils import to_obstacle_frame, wrap_angle

def evaluate_robustness(spec: PacSTLEvaluator, ellipsoids_Ab_dict: dict, encounter_vessel_eta: tuple, state: dict, encounter_speed: float) -> interval.interval:
    """Evaluate the pacSTL specification over the prediction horizon."""
    if spec is None or ellipsoids_Ab_dict is None:
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

    for time_step, raw_tuple in ellipsoids_Ab_dict.items():
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
        state_array = np.array([
            local_x, local_y, psi_rel,
            local_vx, local_vy, speed
        ])

        ego_trajectory[time_step] = TimeStampedState(
            time_step=time_step, state_array=state_array
        )
        reachable_tube[time_step] = PACReachableSet(
            time_step=time_step, A_matrix=A, b_vector=b, center=c
        )

    return spec.evaluate(reachable_tube, ego_trajectory)