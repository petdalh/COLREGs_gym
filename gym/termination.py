class Termination:
    def is_terminated(self, state):
        """
        Check if the episode should terminate due to state issues.

        Currently only checks divergence. Returns (bool, dict).
        The dict contains the reason if terminated.
        """
        if state.is_diverged():
            return True, {"reason": "diverged"}
        return False, {}
