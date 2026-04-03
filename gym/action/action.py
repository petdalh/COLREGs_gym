import numpy as np

from gym.utils.discrete_actions import decode_discrete_action


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
        return decode_discrete_action(
            action_idx=action_idx,
            obs=obs,
            heading_offsets=self.heading_offsets,
            speed_multipliers=self.speed_multipliers,
        )

    def decode_discrete_actions(self, action_idx, obs):
        return self._decode_discrete_actions(action_idx, obs)

    def compute(self, action, obs, sim_state, controller):
        """Decode the discrete action and compute the control torques."""
        psi_d, u_d = self._decode_discrete_actions(action, obs)
        return controller.compute_action_minimal(sim_state, psi_d, u_d)
