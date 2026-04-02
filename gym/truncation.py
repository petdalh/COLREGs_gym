class Truncation:
    def is_truncated(self, state, curr_sim_time):
        if state.encounter_max_time and curr_sim_time > state.encounter_max_time:
            return True, {"reason": "time_limit"}
        return False, {}
