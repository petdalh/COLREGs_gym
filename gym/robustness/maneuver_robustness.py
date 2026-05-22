from gym.robustness.robustness import Robustness
from gym.utils.istl import (
    ISTL_SEMANTICS,
    normalize_istl_state_noise,
    normalize_stl_semantics,
)


class ManeuverRobustness(Robustness):
    """Evaluates the maneuver_verified spec during an active encounter.

    Inherits evaluate_robustness from Robustness (same constant-velocity
    projection + obstacle-frame transform logic). Overrides evaluate() to
    gate on encounter activation rather than monitoring radius, and records
    to state.history_maneuver_rob instead of history_rob.

    configure() must be called before the first episode to build the spec
    from the factory with T_end = full tube horizon.
    """

    def __init__(
        self,
        sampling_rate: int = 10,
        stl_semantics="pacstl",
        istl_state_noise=None,
        istl_state_noise_growth_per_s=None,
        monitoring_time_steps=None,
    ):
        super().__init__(
            spec=None,
            ellipsoids_Ab_dict=None,
            sampling_rate=sampling_rate,
            stl_semantics=stl_semantics,
            istl_state_noise=istl_state_noise,
            istl_state_noise_growth_per_s=istl_state_noise_growth_per_s,
            monitoring_time_steps=monitoring_time_steps,
        )
        self._configured = False

    def configure(
        self,
        spec_factory,
        tube_time_steps,
        ellipsoids_Ab_dict,
        stl_semantics=None,
        istl_state_noise=None,
        istl_state_noise_growth_per_s=None,
        monitoring_time_steps=None,
    ):
        """Build and cache the spec over the final reachable-tube interval."""
        if stl_semantics is not None:
            self.stl_semantics = normalize_stl_semantics(stl_semantics)
        if istl_state_noise is not None:
            self.istl_state_noise = normalize_istl_state_noise(istl_state_noise)
        if istl_state_noise_growth_per_s is not None:
            self.istl_state_noise_growth_per_s = normalize_istl_state_noise(
                istl_state_noise_growth_per_s
            )
        if monitoring_time_steps is not None:
            self.monitoring_time_steps = list(monitoring_time_steps)

        time_steps = list(tube_time_steps or self.monitoring_time_steps)
        if spec_factory is None or not time_steps:
            self._configured = False
            return
        if self.stl_semantics != ISTL_SEMANTICS and ellipsoids_Ab_dict is None:
            self._configured = False
            return
        T_end = len(time_steps) - 1
        T_start = max(0, T_end - 1)
        self.spec = spec_factory(T_start=T_start, T_end=T_end)
        self.ellipsoids_Ab_dict = ellipsoids_Ab_dict
        self.monitoring_time_steps = time_steps
        self._configured = True

    def evaluate(self, state, sim_step_count):
        """Record None when encounter is inactive; evaluate + record when active."""
        if not state._encounter_active or not self._configured:
            state.record_maneuver_robustness(None)
            return None

        if sim_step_count % self.sampling_rate != 0:
            state.record_maneuver_robustness(None)
            return None

        robustness = self.evaluate_robustness(
            encounter_vessel_eta=state.encounter_vessel_eta,
            state=state.sim_state,
            encounter_speed=state.encounter_speed,
            encounter_scenario=state.encounter_scenario,
        )
        state.record_maneuver_robustness(robustness)
        return robustness
