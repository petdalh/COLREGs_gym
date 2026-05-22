import numpy as np


class Termination:
    def is_terminated(self, state, boat_pos=None):
        if boat_pos is None:
            if state.is_diverged():
                self._log_state_blowup(state)
                return True, {"reason": "state_blowup"}
            return False, {}

        env = state
        if not np.all(np.isfinite(boat_pos)):
            print("[Termination] State diverged (non-finite position)")
            return True, {"reason": "diverged"}

        state = env.get_state()
        if not np.all(np.isfinite(state["nu"])):
            print("[Termination] State diverged (non-finite velocity)")
            return True, {"reason": "diverged"}

        # max_pos = max(env.grid_width, env.grid_height) * 4
        max_width = env.grid_width * 4
        max_height = env.grid_height - 5
        if np.abs(boat_pos[0]) > max_height or np.abs(boat_pos[1]) > max_width:
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

    @staticmethod
    def _log_state_blowup(state):
        sim_state = getattr(state, "sim_state", None) or {}
        eta = np.asarray(sim_state.get("eta", []), dtype=float)
        nu = np.asarray(sim_state.get("nu", []), dtype=float)
        tau = np.asarray(getattr(state, "_current_tau", []), dtype=float)
        print(
            "[Termination] State blowup: "
            f"eta={np.array2string(eta, precision=4, suppress_small=True)}, "
            f"nu={np.array2string(nu, precision=4, suppress_small=True)}, "
            f"psi_d={getattr(state, '_current_heading_cmd', None)}, "
            f"u_d={getattr(state, '_current_surge_cmd', None)}, "
            f"psi_d_dot={getattr(state, '_current_psi_d_dot', None)}, "
            f"psi_d_ddot={getattr(state, '_current_psi_d_ddot', None)}, "
            f"u_d_dot={getattr(state, '_current_u_d_dot', None)}, "
            f"tau={np.array2string(tau, precision=4, suppress_small=True)}"
        )
