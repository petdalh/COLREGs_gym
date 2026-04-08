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

    def decode_discrete_actions(self, action_idx, sim_state):
        return decode_discrete_action(
            action_idx=action_idx,
            sim_state=sim_state,
            heading_offsets=self.heading_offsets,
            speed_multipliers=self.speed_multipliers,
        )

    def compute(self, action, sim_state, controller):
        """Decode the discrete action and compute the control torques."""
        psi_d, u_d = self.decode_discrete_actions(action, sim_state)
        return controller.compute_action_minimal(sim_state, psi_d, u_d)
