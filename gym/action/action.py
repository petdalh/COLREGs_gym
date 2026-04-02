import numpy as np
from mchorcrux.numpy_core.controllers.adaptive_seakeeping import heading_to_goal


class Action:
    def __init__(self, config: dict):

        action_config = config["action_space"]

        self.heading_offsets_deg = np.array(action_config["heading_offsets_deg"])
        self.heading_offsets = np.deg2rad(self.heading_offsets_deg)
        self.speed_multipliers = np.array(action_config["speed_multipliers"])
        self.n_actions = len(self.heading_offsets)*len(self.speed_multipliers)
        


    def decode_discrete_actions(self, action_idx: int, obs: dict) -> tuple[float, float]:
        h_idx = action_idx // len(self.speed_multipliers)
        s_idx = action_idx % len(self.speed_multipliers)

        n, e, psi = obs[0], obs[1], obs[2]
        gn, ge = obs[6], obs[7]
        psi_goal = heading_to_goal(n, e, gn, ge)

        psi_d = psi_goal + self.heading_offsets[h_idx]
        u_d = self.speed_multipliers[s_idx]
        return psi_d, u_d