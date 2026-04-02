import numpy as np

from gym.utils.geometry import within_monitoring_radius


class Masking:
    def __init__(
        self,
        n_actions,
        robustness_margin=1.0,
        mask_recompute_interval=2,
        vessel_model=None,
        sim_dt=0.5,
    ):
        self.n_actions = n_actions
        self.robustness_margin = robustness_margin
        self.mask_recompute_interval = mask_recompute_interval
        self._steps_since_mask_update = 0
        self._action_masker = None

        v_max = getattr(vessel_model, "v_max", None)
        r_max = getattr(vessel_model, "yaw_dot_max", None)

        try:
            from logic.masking import ActionMasker

            self._action_masker = ActionMasker(
                v_max=v_max,
                r_max=r_max,
                sim_dt=sim_dt,
            )
        except ModuleNotFoundError:
            self._action_masker = None

    def _default_mask(self):
        return np.ones(self.n_actions, dtype=bool)

    def reset(self, env):
        self._steps_since_mask_update = 0
        env._steps_since_mask_update = 0
        if env.state is not None:
            env.state._encounter_active = False
            env.state._active_maneuver_spec = None
            env.state._cached_mask = self._default_mask()
        env._encounter_active = False
        env._active_maneuver_spec = None
        env._cached_mask = self._default_mask()
        if self._action_masker is not None:
            self._action_masker.update_scenario(None, None)

    def configure_monitoring(self, env, spec, ellipsoids_Ab_dict):
        if env.state._active_maneuver_spec is not None and self._action_masker is not None:
            self._action_masker.update_scenario(
                env.state._active_maneuver_spec,
                ellipsoids_Ab_dict,
            )

    def update_encounter_state(self, env, robustness):
        if robustness is None:
            return

        was_active = env.state._encounter_active
        trace_list = robustness[0]
        for _, rob_interval in trace_list:
            if rob_interval.u > -self.robustness_margin:
                env.state._encounter_active = True
                env._encounter_active = True
                if env.state._active_maneuver_spec is None:
                    try:
                        from pacstl.core.factory import create as create_spec

                        env.state._active_maneuver_spec = create_spec(
                            "colregs", "crossing_detection"
                        )
                        env._active_maneuver_spec = env.state._active_maneuver_spec
                        if self._action_masker is not None:
                            self._action_masker.update_scenario(
                                env.state._active_maneuver_spec,
                                env.ellipsoids_Ab_dict,
                            )
                    except ModuleNotFoundError:
                        env.state._active_maneuver_spec = None
                        env._active_maneuver_spec = None
                if not was_active:
                    print(
                        f"[Encounter] Activated (rob_upper={rob_interval.u:.2f}, "
                        f"margin={self.robustness_margin:.1f})"
                    )
                    env.state._cached_mask = self._compute_mask(
                        env,
                        robustness_margin=self.robustness_margin,
                    )
                    env._cached_mask = env.state._cached_mask
                    self._steps_since_mask_update = 0
                    env._steps_since_mask_update = 0
                return

        env.state._encounter_active = False
        env._encounter_active = False
        env.state._active_maneuver_spec = None
        env._active_maneuver_spec = None
        env.state._cached_mask = self._default_mask()
        env._cached_mask = env.state._cached_mask
        if self._action_masker is not None:
            self._action_masker.update_scenario(None, None)

    def action_masks(self, env):
        if not env.state._encounter_active or env.state._active_maneuver_spec is None:
            return self._default_mask()

        if not within_monitoring_radius(
            env.encounter_vessel_eta, env.monitoring_radius, env.get_state()
        ):
            env.state._encounter_active = False
            env._encounter_active = False
            env.state._active_maneuver_spec = None
            env._active_maneuver_spec = None
            env.state._cached_mask = self._default_mask()
            env._cached_mask = env.state._cached_mask
            if self._action_masker is not None:
                self._action_masker.update_scenario(None, None)
            return env.state._cached_mask

        self._steps_since_mask_update += 1
        env._steps_since_mask_update = self._steps_since_mask_update
        if self._steps_since_mask_update >= self.mask_recompute_interval:
            env.state._cached_mask = self._compute_mask(
                env,
                robustness_margin=env.robustness_margin,
            )
            env._cached_mask = env.state._cached_mask
            self._steps_since_mask_update = 0
            env._steps_since_mask_update = 0

        return env.state._cached_mask

    def _compute_mask(self, env, robustness_margin):
        if self._action_masker is None:
            return self._default_mask()

        return self._action_masker.get_mask(
            situation=env.encounter_type,
            ego_state=env.get_state(),
            encounter_vessel_eta=env.encounter_vessel_eta,
            encounter_speed=env.encounter_speed,
            obs=env._obs(),
            robustness_margin=robustness_margin,
            monitoring_radius=env.monitoring_radius,
        )
