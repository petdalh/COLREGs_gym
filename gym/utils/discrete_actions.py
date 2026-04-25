import numpy as np


def decode_discrete_action(action_idx, yaw_rate_commands, surge_accel_commands):
    yaw_rate_commands = np.asarray(yaw_rate_commands, dtype=float)
    surge_accel_commands = np.asarray(surge_accel_commands, dtype=float)

    yaw_idx = action_idx // len(surge_accel_commands)
    accel_idx = action_idx % len(surge_accel_commands)

    return yaw_rate_commands[yaw_idx], surge_accel_commands[accel_idx]
