import numpy as np

from mchorcrux.numpy_core.controllers.adaptive_seakeeping import heading_to_goal


def decode_discrete_action(action_idx, obs, heading_offsets, speed_multipliers):
    heading_offsets = np.asarray(heading_offsets, dtype=float)
    speed_multipliers = np.asarray(speed_multipliers, dtype=float)

    h_idx = action_idx // len(speed_multipliers)
    s_idx = action_idx % len(speed_multipliers)

    n, e, psi = obs[0], obs[1], obs[2]
    gn, ge = obs[6], obs[7]
    psi_goal = heading_to_goal(n, e, gn, ge)

    psi_d = psi_goal + heading_offsets[h_idx]
    u_d = speed_multipliers[s_idx]
    return psi_d, u_d
