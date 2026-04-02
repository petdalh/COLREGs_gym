import numpy as np


class Termination:
    def is_terminated(self, state, boat_pos=None):
        if boat_pos is None:
            if state.is_diverged():
                return True, {"reason": "diverged"}
            return False, {}

        env = state
        if not np.all(np.isfinite(boat_pos)):
            print("[Termination] State diverged (non-finite position)")
            return True, {"reason": "diverged"}

        state = env.get_state()
        if not np.all(np.isfinite(state["nu"])):
            print("[Termination] State diverged (non-finite velocity)")
            return True, {"reason": "diverged"}

        max_pos = max(env.grid_width, env.grid_height) * 4
        if np.abs(boat_pos[0]) > max_pos or np.abs(boat_pos[1]) > max_pos:
            print(
                f"[Termination] Vessel far outside grid: pos=({boat_pos[0]:.1f}, {boat_pos[1]:.1f})"
            )
            return True, {"reason": "out_of_bounds"}

        if env.state.encounter_vessel_eta is not None:
            dist = np.hypot(
                boat_pos[0] - env.state.encounter_vessel_eta[0],
                boat_pos[1] - env.state.encounter_vessel_eta[1],
            )
            if dist < env.state.encounter_radius:
                return True, {"reason": "collision"}

        return False, {}
