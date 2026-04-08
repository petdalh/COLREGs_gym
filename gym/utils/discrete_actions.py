import numpy as np
from gym.utils.geometry import wrap_angle


def decode_discrete_action(action_idx, sim_state, heading_offsets, speed_multipliers):
    heading_offsets = np.asarray(heading_offsets, dtype=float)
    speed_multipliers = np.asarray(speed_multipliers, dtype=float)

    h_idx = action_idx // len(speed_multipliers)
    s_idx = action_idx % len(speed_multipliers)

    eta = sim_state["eta"]
    n, e, psi = eta[0], eta[1], float(eta[-1])
    goal = sim_state["goal"]
    gn, ge = float(goal[0]), float(goal[1])

    psi_goal = np.arctan2(ge - e, gn - n)
    psi_d = psi_goal + heading_offsets[h_idx]
    u_d = speed_multipliers[s_idx]
    return psi_d, u_d
