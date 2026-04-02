import numpy as np


class Observation:
    def __init__(self, config):
        self.config = config

    def get(self, state):
        eta = state.sim_state["eta"]
        nu = state.sim_state["nu"]
        goal = state.goal
        if goal is None and isinstance(state.sim_state, dict):
            goal = state.sim_state.get("goal")
        if goal is None:
            goal = np.zeros(2, dtype=float)
        goal_xy = np.asarray(goal[:2], dtype=float)
        tgt = (
            state.encounter_vessel_eta[:2]
            if state.encounter_vessel_eta is not None
            else np.zeros(2)
        )
        obs = np.concatenate([eta[:2], [eta[-1]], nu, goal_xy, tgt])

        if not np.all(np.isfinite(obs)):
            obs = np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)

        return obs.astype(np.float32)
