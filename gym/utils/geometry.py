import numpy as np


def wrap_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def relative_polar(
    ego_pos: np.ndarray,
    ego_psi: float,
    target_pos: np.ndarray,
) -> tuple[float, float]:
    """Distance and ego-body-frame bearing to a target.

    Returns (d, beta) where:
      d    -- Euclidean distance [m]
      beta -- bearing in ego body frame [rad], positive = starboard
    """
    delta = np.asarray(target_pos, dtype=float) - np.asarray(ego_pos, dtype=float)
    d = np.hypot(delta[0], delta[1])
    bearing_world = np.arctan2(delta[1], delta[0])  # delta[1]=De, delta[0]=Dn
    beta = wrap_angle(bearing_world - ego_psi)
    return float(d), float(beta)


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


def within_monitoring_radius(
    encounter_vessel_eta: np.ndarray,
    monitoring_radius: float,
    state: dict,
) -> bool:
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