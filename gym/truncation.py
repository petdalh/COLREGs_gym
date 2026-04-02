class Truncation:
    def is_truncated(self, state, step_count):
        """
        Check if the episode should be truncated.

        Currently returns False — truncation is handled by the parent class.
        Add custom truncation logic here as needed (e.g., max steps).
        """
        return False, {}
