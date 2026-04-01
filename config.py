# action_config.py
import numpy as np

HEADING_OFFSETS = np.deg2rad([-45, -30, -15, -10, -5, 0, 5, 10, 15, 30, 45])
SPEED_MULTIPLIERS = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
N_DISCRETE_ACTIONS = len(HEADING_OFFSETS) * len(SPEED_MULTIPLIERS)