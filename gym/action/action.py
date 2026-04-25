import numpy as np
from gym.utils.discrete_actions import decode_discrete_action


class Action:
    def __init__(self, config: dict, v_max: float):
        action_config = (
            config.get("action_configuration")
            or config.get("action_space")
            or config
        )

        self.yaw_rate_commands_deg_s = np.array(
            action_config["yaw_rate_commands_deg_s"], dtype=float
        )
        self.yaw_rate_commands = np.deg2rad(self.yaw_rate_commands_deg_s)
        self.surge_accel_commands = np.array(
            action_config["surge_accel_commands"], dtype=float
        )
        self.n_actions = len(self.yaw_rate_commands) * len(self.surge_accel_commands)
        self.v_max = v_max

    def decode_discrete_actions(self, action_idx, sim_state):
        return decode_discrete_action(
            action_idx=action_idx,
            yaw_rate_commands=self.yaw_rate_commands,
            surge_accel_commands=self.surge_accel_commands,
        )

    def compute(self, sim_state, controller, psi_d, u_d):
        """Compute control torques for the current carried references."""
        return controller.compute_action_minimal(sim_state, psi_d, u_d)
