import numpy as np
from mchorcrux.numpy_core.controllers.adaptive_seakeeping import heading_to_goal


class Action:
    def __init__(self, config: dict):
        action_config = (
            config.get("action_configuration")
            or config.get("action_space")
            or config
        )

        self.heading_offsets_deg = np.array(action_config["heading_offsets_deg"])
        self.heading_offsets = np.deg2rad(self.heading_offsets_deg)
        self.speed_multipliers = np.array(action_config["speed_multipliers"])
        self.n_actions = len(self.heading_offsets) * len(self.speed_multipliers)

    def _decode_discrete_actions(self, action_idx, obs):
        h_idx = action_idx // len(self.speed_multipliers)
        s_idx = action_idx % len(self.speed_multipliers)

        n, e, psi = obs[0], obs[1], obs[2]
        gn, ge = obs[6], obs[7]
        psi_goal = heading_to_goal(n, e, gn, ge)

        psi_d = psi_goal + self.heading_offsets[h_idx]
        u_d = self.speed_multipliers[s_idx]
        return psi_d, u_d

    def decode_discrete_actions(self, action_idx, obs):
        return self._decode_discrete_actions(action_idx, obs)

    def compute(self, action, obs, sim_state, controller):
        """Decode the discrete action and compute the control torques."""
        psi_d, u_d = self._decode_discrete_actions(action, obs)
        return controller.compute_action_minimal(sim_state, psi_d, u_d)
