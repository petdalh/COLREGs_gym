import numpy as np


class Observation:
    def __init__(self, config):
        self.config = config

    def get(self, state):
        """
        Build the observation dict/array from the current state.

        This should contain the exact same logic currently in COLREGsGym._obs().
        Move that logic here. The gym's _obs() should then delegate:

            def _obs(self):
                return self.observation.get(self.state)
        """
        eta = state.sim_state["eta"]
        nu = state.sim_state["nu"]
        goal = state.goal
        if goal is None and isinstance(state.sim_state, dict):
            goal = state.sim_state.get("goal")
        if goal is None:
            goal = np.zeros(2, dtype=float)
        goal_xy = np.asarray(goal[:2], dtype=float)
        return np.array(
            [
                eta[0],
                eta[1],
                eta[2],
                nu[0],
                nu[1],
                nu[2],
                goal_xy[0],
                goal_xy[1],
            ],
            dtype=float,
        )
