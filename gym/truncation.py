class Truncation:
    def is_truncated(self, env):
        if env.encounter_max_time and env.curr_sim_time > env.encounter_max_time:
            return True, {"reason": "time_limit"}
        return False, {}
