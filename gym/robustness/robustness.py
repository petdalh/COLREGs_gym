from gym.utils.geometry import within_monitoring_radius
from gym.utils.robustness import evaluate_robustness


class Robustness:
    def __init__(self, spec, ellipsoids_Ab_dict, sampling_rate):
        self.spec = spec
        self.ellipsoids_Ab_dict = ellipsoids_Ab_dict
        self.sampling_rate = sampling_rate

    def evaluate(self, state, step_count):
        """
        Evaluate robustness if within sampling rate and monitoring radius.

        Returns the robustness value or None. Also updates state encounter
        tracking and records robustness/radius to history.
        """
        in_radius = within_monitoring_radius(
            state.encounter_vessel_eta,
            state.monitoring_radius,
            state.sim_state,
        )

        robustness = None
        if step_count % self.sampling_rate == 0:
            if in_radius:
                robustness = evaluate_robustness(
                    spec=self.spec,
                    ellipsoids_Ab_dict=self.ellipsoids_Ab_dict,
                    encounter_vessel_eta=state.encounter_vessel_eta,
                    state=state.sim_state,
                    encounter_speed=state.encounter_speed,
                )
            elif state._encounter_active:
                state.clear_encounter()

        state.record_robustness(robustness, in_radius)
        return robustness
