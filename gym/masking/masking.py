import numpy as np

from .action_masker import ActionMasker
from gym.utils.geometry import within_monitoring_radius


class Masking:
    def __init__(
        self,
        n_actions,
        robustness_margin=1.0,
        mask_recompute_interval=2,
        enabled=True,
        action=None,
        vessel_model=None,
        ego_vessel_model=None,
        action_masking_config=None,
    ):
        self.n_actions = n_actions
        self.robustness_margin = robustness_margin
        self.mask_recompute_interval = mask_recompute_interval
        self.enabled = enabled
        self._steps_since_mask_update = 0
        self._default_action_masking_config = dict(action_masking_config or {})
        self._vessel_model = vessel_model
        self._action_masker = ActionMasker(
            action=action,
            vessel_model=vessel_model,
            ego_vessel_model=ego_vessel_model,
            config=self._default_action_masking_config,
        )

    def _default_mask(self):
        return np.ones(self.n_actions, dtype=bool)

    def configure_for_scenario(self, encounter_scenario):
        scenario_cfg = getattr(encounter_scenario, "masking_configuration", None)
        self._action_masker.configure(
            config=scenario_cfg or self._default_action_masking_config,
            vessel_model=self._vessel_model,
        )

    def reset(self, env):
        self._steps_since_mask_update = 0
        self.configure_for_scenario(env.encounter_scenario)
        if env.state is not None:
            env.state._encounter_active = False
            env.state._active_maneuver_spec = None
            env.state._cached_mask = self._default_mask()
        self._action_masker.update_scenario(None, None, env.encounter_scenario)

    def update_encounter_state(self, env, robustness):
        if not self.enabled or robustness is None:
            return

        was_active = env.state._encounter_active
        trace_list = robustness[0]
        for _, rob_interval in trace_list:
            if rob_interval.u > -self.robustness_margin:
                env.state._encounter_active = True
                if env.state._active_maneuver_spec is None:
                    env.state._active_maneuver_spec = env.spec
                    if env.state._active_maneuver_spec is not None:
                        self._action_masker.update_scenario(
                            env.state._active_maneuver_spec,
                            env.ellipsoids_Ab_dict,
                            env.encounter_scenario,
                            spec_factory=getattr(env, "maneuver_spec_factory", None),
                        )
                if not was_active:
                    print(
                        f"[Encounter] Activated (rob_upper={rob_interval.u:.2f}, "
                        f"margin={self.robustness_margin:.1f})"
                    )
                    env.state._cached_mask = self._compute_mask(
                        env,
                        robustness_margin=self.robustness_margin,
                    )
                    self._steps_since_mask_update = 0
                return

        env.state._encounter_active = False
        env.state._active_maneuver_spec = None
        env.state._cached_mask = self._default_mask()
        self._action_masker.update_scenario(None, None, env.encounter_scenario)

    def action_masks(self, env):
        env.state.fallback_used = False
        if not self.enabled:
            return self._default_mask()
        if not env.state._encounter_active or env.state._active_maneuver_spec is None:
            return self._default_mask()

        in_radius = env.state.in_monitoring_radius
        if in_radius is None:
            in_radius = within_monitoring_radius(
                env.state.encounter_vessel_eta,
                env.state.monitoring_radius,
                env.state.sim_state,
            )
            env.state.in_monitoring_radius = in_radius

        if not in_radius:
            env.state._encounter_active = False
            env.state._active_maneuver_spec = None
            env.state._cached_mask = self._default_mask()
            self._action_masker.update_scenario(None, None, env.encounter_scenario)
            return env.state._cached_mask

        self._steps_since_mask_update += 1
        if self._steps_since_mask_update >= self.mask_recompute_interval:
            env.state._cached_mask = self._compute_mask(
                env,
                robustness_margin=self.robustness_margin,
            )
            self._steps_since_mask_update = 0

        return env.state._cached_mask

    def _compute_mask(self, env, robustness_margin):
        mask, is_fallback = self._action_masker.get_mask(
            situation=env.state.encounter_type,
            ego_state=env.get_state(),
            encounter_vessel_eta=env.state.encounter_vessel_eta,
            encounter_speed=env.state.encounter_speed,
            robustness_margin=robustness_margin,
            monitoring_radius=env.state.monitoring_radius,
        )
        env.state.fallback_used = is_fallback
        return mask
