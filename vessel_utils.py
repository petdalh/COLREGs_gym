import numpy as np
from pacstl.common.interfaces import TimeStampedState


def wrap_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def to_obstacle_frame(
    ego_pos: np.ndarray,
    ego_vel: np.ndarray,
    obs_pos: np.ndarray,
    obs_psi: float,
    obs_vel: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform ego state to the obstacle-relative body frame."""
    rel_pos = np.asarray(ego_pos, dtype=float) - np.asarray(obs_pos, dtype=float)
    rel_vel = np.asarray(ego_vel, dtype=float) - np.asarray(obs_vel, dtype=float)

    angle = obs_psi + np.pi
    transform = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        dtype=float,
    )

    local_pos = transform @ rel_pos
    local_vel = transform @ rel_vel
    return local_pos, local_vel


def cross_track_error(
    pos_xy: np.ndarray,
    path_start: np.ndarray | None,
    goal_xy: np.ndarray,
) -> float:
    """Signed cross-track error using Fossen LOS line equation."""
    if path_start is None:
        return 0.0

    x1, y1 = np.asarray(path_start, dtype=float)
    x2, y2 = np.asarray(goal_xy, dtype=float)
    x, y = np.asarray(pos_xy, dtype=float)

    # Degenerate case: start and goal coincide.
    if np.hypot(x2 - x1, y2 - y1) < 1e-9:
        return 0.0

    pi_h = np.arctan2(y2 - y1, x2 - x1)
    y_e = -(x - x1) * np.sin(pi_h) + (y - y1) * np.cos(pi_h)
    return float(y_e)
    

def within_monitoring_radius(encounter_vessel_eta: np.ndarray, monitoring_radius: float, state: dict) -> bool:
    """Check if the encounter vessel is within monitoring range."""
    if encounter_vessel_eta is None:
        return False
    if monitoring_radius is None:
        return True  # no radius configured -> always monitor

    eta = state["eta"]
    dist = np.hypot(
        eta[0] - encounter_vessel_eta[0],
        eta[1] - encounter_vessel_eta[1],
    )
    return dist <= monitoring_radius

def propagate_vessel(vessel_eta: np.ndarray, encounter_speed: float, dt: float) -> np.ndarray:
    n, e, psi = vessel_eta
    n += encounter_speed * np.cos(psi) * dt
    e += encounter_speed * np.sin(psi) * dt
    return np.array([n, e, psi])

def simulate_candidate_trajectory(psi_d: float, u_d: float, state: dict, encounter_vessel_eta: np.ndarray, encounter_speed: float, r_max: float, ellipsoids_Ab_dict: dict, sim_dt: float, v_max: float) -> dict:
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