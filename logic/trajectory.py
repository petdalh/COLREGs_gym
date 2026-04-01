import numpy as np
from pacstl.common.interfaces import TimeStampedState
from utils.geometry import wrap_angle, to_obstacle_frame


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
    eta, nu = state["eta"], state["nu"]
    px, py = eta[0], eta[1]
    psi = eta[-1]
    u = nu[0]

    # Get target info for relative transformation
    obs_n, obs_e, obs_psi = encounter_vessel_eta
    obs_vn = encounter_speed * np.cos(obs_psi)
    obs_ve = encounter_speed * np.sin(obs_psi)

    k_heading = 1.0
    k_speed = 0.5
    omega_max = r_max
    a_max = 0.1

    tube_time_steps = sorted(ellipsoids_Ab_dict.keys())
    ego_trajectory = {}
    t = 0.0

    for target_t in tube_time_steps:
        while t < target_t - 1e-9:
            dt_step = min(sim_dt, target_t - t)
            heading_error = wrap_angle(psi_d - psi)
            omega = np.clip(k_heading * heading_error, -omega_max, omega_max)
            speed_error = u_d - u
            a = np.clip(k_speed * speed_error, -a_max, a_max)

            px += np.cos(psi) * u * dt_step
            py += np.sin(psi) * u * dt_step
            psi += omega * dt_step
            u += a * dt_step
            u = np.clip(u, 0.0, v_max)
            t += dt_step

        # World-frame velocities at this predicted state
        vn = u * np.cos(psi)
        ve = u * np.sin(psi)

        # Transform to Target's Relative Frame
        pred_obs_n = obs_n + obs_vn * target_t
        pred_obs_e = obs_e + obs_ve * target_t

        local_pos, local_vel = to_obstacle_frame(
            ego_pos=np.array([px, py]),
            ego_vel=np.array([vn, ve]),
            obs_pos=np.array([pred_obs_n, pred_obs_e]),
            obs_psi=obs_psi,
            obs_vel=np.array([obs_vn, obs_ve]),
        )
        local_x, local_y = local_pos

        psi_rel = -wrap_angle(psi - obs_psi)
        local_vx, local_vy = local_vel
        speed = np.hypot(local_vx, local_vy)

        state_array = np.array([local_x, local_y, psi_rel, local_vx, local_vy, speed])

        ego_trajectory[target_t] = TimeStampedState(
            time_step=target_t, state_array=state_array
        )

    return ego_trajectory
