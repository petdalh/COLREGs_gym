import numpy as np


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
    
def compute_monitoring_radius(v_max: float, maneuver_horizon: float, ellipsoids_Ab_dict: dict | None, target_speed: float, monitoring_radius_safety_factor: float) -> float:
    """Derive a monitoring radius from vessel speeds and maneuver horizon.

    The radius must satisfy:
        radius > (v_ego_max + v_target) * (T_maneuver + T_prediction)

    so that the pacSTL evaluator has time to:
        1. detect the encounter (prediction horizon covers future risk), and
        2. the agent has room to execute a full avoidance maneuver.

    The safety factor (default 2.0) adds margin for:
        - non-straight approach geometries (crossing angles reduce closing
        rate compared to head-on)
        - decision interval delays (agent acts every N sub-steps)
        - ellipsoid prediction horizon (must overlap with encounter geometry)
    """
    max_closing_speed = v_max + target_speed

    # Prediction horizon: max time key in ellipsoid dict, or fallback
    if ellipsoids_Ab_dict is not None and len(ellipsoids_Ab_dict) > 0:
        t_prediction = max(ellipsoids_Ab_dict.keys())
    else:
        t_prediction = 2.5  # default from pacSTL paper

    t_total = maneuver_horizon + t_prediction
    radius = monitoring_radius_safety_factor * max_closing_speed * t_total

    return radius

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