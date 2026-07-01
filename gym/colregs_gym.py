from gym.action.action import Action
from gym.callback.episode_logger import EpisodeLogger
from gym.encounter_scenario import EncounterScenario
from gym.observation.observation import Observation
from gym.reward.colregs_reward import ColregsReward
from gym.robustness import Robustness, ManeuverRobustness
from gym.masking import Masking
from gym.state import State
from gym.termination import Termination
from gym.truncation import Truncation
from gym.utils.config import resolve_config
from gym.utils.geometry import within_monitoring_radius, wrap_angle
from gym.utils.reachable_sets import select_reachable_set

from gymnasium import spaces

from mchorcrux.numpy_core.gym.mc_gym_csad_numpy import McGym

import numpy as np


class COLREGsGym(McGym):
    def __init__(
        self,
        vessel_model,
        ego_vessel_model,
        dt=None,
        grid_width=None,
        grid_height=None,
        maneuver_horizon=None,
        sim_dt=None,
        monitoring_radius=None,
        monitoring_radius_safety_factor=None,
        robustness_margin=None,
        config=None,
        seed=None,
        **kwargs,
    ):
        config = resolve_config(config)
        env_cfg = config.get("environment_configuration", config)
        action_cfg = env_cfg.get("action_configuration", config)
        encounter_cfg = env_cfg.get("encounter_configuration", {})
        masking_cfg = dict(env_cfg.get("masking_configuration", {}))
        monitoring_cfg = env_cfg.get("monitoring_configuration", {})
        reward_cfg = env_cfg.get("reward_configuration", {})
        action_constraints_cfg = env_cfg.get("action_constraints", {})
        min_surge_command_mps = float(
            env_cfg.get(
                "min_surge_command_mps",
                masking_cfg.get("min_surge_command_mps", 0.0),
            )
        )

        dt = dt if dt is not None else env_cfg.get("dt", 0.5)
        grid_width = grid_width if grid_width is not None else env_cfg.get("grid_width", 25.0)
        grid_height = grid_height if grid_height is not None else env_cfg.get("grid_height", 30.0)
        maneuver_horizon = (
            maneuver_horizon
            if maneuver_horizon is not None
            else encounter_cfg.get("maneuver_horizon", 10.0)
        )
        sim_dt = (
            sim_dt
            if sim_dt is not None
            else masking_cfg.get("sim_dt", env_cfg.get("sim_dt", dt))
        )
        decision_interval = env_cfg.get("decision_interval", 5)
        masking_cfg["sim_dt"] = sim_dt
        masking_cfg["dt"] = dt
        masking_cfg["decision_interval"] = decision_interval
        masking_cfg["min_surge_command_mps"] = min_surge_command_mps
        masking_cfg["speed_floor_enabled"] = action_constraints_cfg.get(
            "speed_floor_enabled",
            masking_cfg.get("speed_floor_enabled", min_surge_command_mps > 0.0),
        )
        monitoring_radius = (
            monitoring_radius
            if monitoring_radius is not None
            else monitoring_cfg.get("monitoring_radius")
        )
        monitoring_radius_safety_factor = (
            monitoring_radius_safety_factor
            if monitoring_radius_safety_factor is not None
            else monitoring_cfg.get("monitoring_radius_safety_factor", 2.0)
        )
        robustness_margin = (
            robustness_margin
            if robustness_margin is not None
            else monitoring_cfg.get("robustness_margin", 1.0)
        )

        controller_type = env_cfg.get("controller_type", "backstepping")

        super().__init__(
            dt=dt, grid_width=grid_width, grid_height=grid_height, **kwargs
        )

        self._controller_type = controller_type
        if controller_type == "mrac":
            from mchorcrux.numpy_core.controllers.adaptive_seakeeping import MRACShipController
            self._controller = MRACShipController(dt=dt)

        self.vessel_model = vessel_model
        self.ego_vessel_model = ego_vessel_model
        self.config = config
        self.encounter_scenario = EncounterScenario(
            vessel_model,
            maneuver_horizon=maneuver_horizon,
            monitoring_radius_safety_factor=monitoring_radius_safety_factor,
            masking_configuration=masking_cfg,
            seed=seed,
        )
        self.vessel_action = Action(action_cfg, v_max=ego_vessel_model.v_max)
        self.action_space = spaces.Discrete(self.vessel_action.n_actions)
        self.observation = Observation(env_cfg.get("observation_configuration", {}))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.observation.dim,), dtype=np.float32
        )
        self.spec = None
        self.maneuver_spec_factory = None
        self.ellipsoids_Ab_dict = None
        self._reachable_set_bank = None
        self.reward = ColregsReward(config=reward_cfg)
        self.masking = Masking(
            n_actions=self.vessel_action.n_actions,
            robustness_margin=robustness_margin,
            mask_recompute_interval=monitoring_cfg.get("mask_recompute_interval", 2),
            enabled=monitoring_cfg.get("masking_enabled", True),
            action=self.vessel_action,
            vessel_model=vessel_model,
            ego_vessel_model=ego_vessel_model,
            action_masking_config=masking_cfg,
        )
        self.state = State(
            monitoring_radius=monitoring_radius,
            n_actions=self.vessel_action.n_actions,
            ego_vessel_model=ego_vessel_model,
            initial_surge_command_fraction=env_cfg.get(
                "initial_surge_command_fraction", 0.8
            ),
            min_surge_command_mps=min_surge_command_mps,
        )
        self.robustness = Robustness(
            spec=self.spec,
            ellipsoids_Ab_dict=self.ellipsoids_Ab_dict,
            sampling_rate=monitoring_cfg.get("robustness_sampling_rate", 10),
        )
        self.maneuver_robustness = ManeuverRobustness(
            sampling_rate=monitoring_cfg.get("robustness_sampling_rate", 10),
        )
        self.termination = Termination()
        self.truncation = Truncation()
        self.callback = EpisodeLogger()
        self.decision_interval = decision_interval
        self.pre_maneuver_duration_s = float(
            env_cfg.get("pre_maneuver_duration_s", 0.0)
        )
        self.straight_until_encounter_active = bool(
            env_cfg.get("straight_until_encounter_active", True)
        )
        self._neutral_action_mask = self._build_neutral_action_mask()
        self._straight_hold_released = False
        self._step_count = 0

    def _sync_runtime_state(self):
        sim_state = self.get_state()
        self.state.update_sim(sim_state)
        self.state.in_monitoring_radius = within_monitoring_radius(
            self.state.encounter_vessel_eta,
            self.state.monitoring_radius,
            sim_state,
        )
        return sim_state

    def _encounter_distance(self):
        if self.state.encounter_vessel_eta is None or self.state.sim_state is None:
            return None
        ego_pos = self.state.sim_state["eta"][:2]
        enc_pos = self.state.encounter_vessel_eta[:2]
        return float(np.hypot(ego_pos[0] - enc_pos[0], ego_pos[1] - enc_pos[1]))

    def _build_neutral_action_mask(self):
        mask = np.zeros(self.vessel_action.n_actions, dtype=bool)
        yaw_idx = int(np.argmin(np.abs(self.vessel_action.yaw_rate_commands)))
        accel_idx = int(np.argmin(np.abs(self.vessel_action.surge_accel_commands)))
        action_idx = yaw_idx * len(self.vessel_action.surge_accel_commands) + accel_idx
        mask[action_idx] = True
        return mask

    def _pre_maneuver_active(self):
        if self._straight_hold_available():
            return False
        if self.pre_maneuver_duration_s <= 0.0:
            return False
        return self._step_count * self.dt < self.pre_maneuver_duration_s

    def _straight_hold_available(self):
        return (
            self.straight_until_encounter_active
            and self.masking.enabled
            and self.spec is not None
            and self.ellipsoids_Ab_dict is not None
        )

    def _straight_hold_active(self):
        return (
            self._straight_hold_available()
            and not self._straight_hold_released
            and not self.state._encounter_active
        )

    def set_encounter(
        self,
        start_position,
        wave_conditions,
        encounter_type="crossing",
        separation=30.0,
        target_speed=0.3,
        target_speeds=None,
        target_speed_range=None,
        time_to_conflict_s=None,
        ego_reference_speed=None,
        heading_noise_deg=5.0,
        encounter_position_noise_m=2.0,
        target_cross_track_noise_m=0.0,
        goal_ahead_distance=25.0,
        collision_radius=1.0,
        simtime=150.0,
        masking_configuration=None,
        wave_direction_range_deg=None,
    ):
        if ego_reference_speed is None and time_to_conflict_s is not None:
            ego_reference_speed = (
                self.state.initial_surge_command_fraction * self.ego_vessel_model.v_max
            )
        self.encounter_scenario.set_encounter(
            start_position=start_position,
            wave_conditions=wave_conditions,
            encounter_type=encounter_type,
            separation=separation,
            target_speed=target_speed,
            target_speeds=target_speeds,
            target_speed_range=target_speed_range,
            time_to_conflict_s=time_to_conflict_s,
            ego_reference_speed=ego_reference_speed,
            heading_noise_deg=heading_noise_deg,
            encounter_position_noise_m=encounter_position_noise_m,
            target_cross_track_noise_m=target_cross_track_noise_m,
            goal_ahead_distance=goal_ahead_distance,
            collision_radius=collision_radius,
            simtime=simtime,
            masking_configuration=masking_configuration,
            wave_direction_range_deg=wave_direction_range_deg,
        )
        self.masking.configure_for_scenario(self.encounter_scenario)
        self.encounter_scenario.apply(self)
        self._refresh_monitoring_for_current_speed()

    def step(self, action):
        terminated = False
        truncated = False
        info = {}
        requested_yaw_rate_cmd, requested_surge_accel_cmd = (
            self.vessel_action.decode_discrete_actions(int(action), self.state.sim_state)
        )
        yaw_rate_cmd = requested_yaw_rate_cmd
        surge_accel_cmd = requested_surge_accel_cmd
        psi_d = self.state._current_heading_cmd
        u_d = self.state._current_surge_cmd

        last_tau = np.zeros(3)
        straight_hold_was_active = False
        pre_maneuver_was_active = False
        min_encounter_distance_interval = np.inf
        for _ in range(self.decision_interval):
            straight_hold_active = self._straight_hold_active()
            pre_maneuver_active = self._pre_maneuver_active()
            straight_hold_was_active = (
                straight_hold_was_active or straight_hold_active
            )
            pre_maneuver_was_active = pre_maneuver_was_active or pre_maneuver_active
            if straight_hold_active or pre_maneuver_active:
                yaw_rate_cmd = 0.0
                surge_accel_cmd = 0.0
            else:
                yaw_rate_cmd = requested_yaw_rate_cmd
                surge_accel_cmd = requested_surge_accel_cmd
            self.state.set_current_rate_commands(yaw_rate_cmd, surge_accel_cmd)

            self.state.propagate_encounter(self.dt)
            sim_state = self._sync_runtime_state()
            encounter_distance = self._encounter_distance()
            if encounter_distance is not None:
                min_encounter_distance_interval = min(
                    min_encounter_distance_interval,
                    encounter_distance,
                )

            terminated, term_info = self.termination.is_terminated(self.state)
            if terminated:
                info.update(term_info)
                break

            if straight_hold_active:
                psi_d, u_d, psi_d_dot, psi_d_ddot, u_d_dot = (
                    self.state.hold_transit_heading()
                )
            elif pre_maneuver_active:
                psi_d, u_d, psi_d_dot, psi_d_ddot, u_d_dot = (
                    self.state.hold_current_commands()
                )
            else:
                psi_d, u_d, psi_d_dot, psi_d_ddot, u_d_dot = (
                    self.state.apply_rate_command(
                        yaw_rate_cmd, surge_accel_cmd, self.dt
                    )
                )
            # if self._step_count % 20 == 0:
            #     tau, debug = self.vessel_action.compute(
            #         sim_state, self._controller, psi_d, u_d, True
            #     )
            #     print("------ debug -------")
            #     print(debug)
            # else:
            #     tau = self.vessel_action.compute(
            #         sim_state, self._controller, psi_d, u_d, False
            #     )
            # last_tau = tau
            if self._controller_type == "mrac":
                tau = self.vessel_action.compute(
                    self.get_observed_state(), self._controller, psi_d, u_d, False
                )
            else:
                tau = self.vessel_action.compute_backstepping(
                    self.get_observed_state(),
                    self._controller,
                    psi_d,
                    u_d,
                    psi_d_dot,
                    psi_d_ddot,
                    u_d_dot,
                    False,
                )
            last_tau = tau
            self.state.set_current_tau(tau)
            _, _, terminated, truncated, info = super().step(tau)
            self._step_count += 1

            self._sync_runtime_state()
            encounter_distance = self._encounter_distance()
            if encounter_distance is not None:
                min_encounter_distance_interval = min(
                    min_encounter_distance_interval,
                    encounter_distance,
                )
            self.state.record()

            if terminated or truncated:
                break

        robustness = self.robustness.evaluate(
            self.state,
            self._step_count,
            in_radius=self.state.in_monitoring_radius,
        )
        if robustness is not None:
            info["robustness"] = robustness
            self._update_encounter_state(robustness)
            if self.state._encounter_active:
                self._straight_hold_released = True

        self.maneuver_robustness.evaluate(self.state, self._step_count)

        obs = self.observation.get(self.state)
        self.state.terminal_reason = info.get("reason", None)
        reward, reward_info = self.reward.get_reward(
            self.state,
            in_radius=self.state.in_monitoring_radius,
        )
        self.state.terminal_reason = None
        info.update(reward_info)
        info["encounter_active"] = bool(self.state._encounter_active)
        info["straight_hold_active"] = bool(straight_hold_was_active)
        info["straight_hold_released"] = bool(self._straight_hold_released)
        info["pre_maneuver_active"] = bool(pre_maneuver_was_active)
        info["mask_allowed_count"] = int(
            self._neutral_action_mask.sum()
            if straight_hold_was_active or pre_maneuver_was_active
            else self.state._cached_mask.sum()
        )
        info["mask_fallback"] = bool(self.state.fallback_used)
        if np.isfinite(min_encounter_distance_interval):
            info["masking/min_distance_interval"] = float(
                min_encounter_distance_interval
            )
        self._populate_masking_diagnostics_info(info)

        ctrl_sim = self.state.sim_state
        psi_d = self.state._current_heading_cmd
        u_d = self.state._current_surge_cmd
        psi = float(ctrl_sim["eta"][-1])
        u = float(ctrl_sim["nu"][0])
        info["control/heading_deg"] = float(np.degrees(psi))
        info["control/heading_cmd_deg"] = float(np.degrees(psi_d))
        info["control/heading_error_deg"] = float(np.degrees(wrap_angle(psi_d - psi)))
        info["control/surge_velocity"] = u
        info["control/surge_cmd"] = float(u_d)
        info["control/speed_error"] = float(u_d - u)
        info["control/yaw_rate_cmd_deg_s"] = float(np.degrees(yaw_rate_cmd))
        info["control/surge_accel_cmd"] = float(surge_accel_cmd)
        info["control/sway_velocity"] = float(ctrl_sim["nu"][1])
        info["control/yaw_rate_deg_s"] = float(np.degrees(ctrl_sim["nu"][2]))
        info["control/tau_surge"] = float(last_tau[0])
        info["control/tau_yaw"] = float(last_tau[2])

        if (
            (terminated or truncated)
            and info.get("reason") == "collision"
            and self.masking.enabled
        ):
            info["collision_plot_data"] = self._build_episode_plot_data()

        if (terminated or truncated) and self.callback is not None:
            self.callback.on_episode_end(info, terminated, self.reward)

        return obs, reward, terminated, truncated, info

    def _build_episode_plot_data(self):
        """Return a serializable snapshot of histories for terminal plotting."""
        goal = self.state.goal
        collision_radius = self.encounter_scenario.collision_radius
        obstacle_speed_mps = self.state.encounter_speed

        return {
            "ego_traj": [list(p) for p in self.history_ego],
            "enc_traj": [list(p) for p in self.history_enc],
            "goal": list(goal[:2]) if goal is not None else None,
            "dt": float(self.dt),
            "robustness_dt": float(self.dt * self.decision_interval),
            "collision_radius": (
                float(collision_radius) if collision_radius is not None else 0.0
            ),
            "obstacle_speed_mps": (
                float(obstacle_speed_mps)
                if obstacle_speed_mps is not None and np.isfinite(obstacle_speed_mps)
                else None
            ),
            "history_surge_command_fraction": [
                float(v) for v in self.history_surge_command_fraction
            ],
            "history_heading_deg": [float(v) for v in self.history_heading_deg],
            "history_heading_cmd_deg": [
                float(v) for v in self.history_heading_cmd_deg
            ],
            "history_heading_error_deg": [
                float(v) for v in self.history_heading_error_deg
            ],
            "history_surge": [float(v) for v in self.history_surge],
            "history_surge_cmd": [float(v) for v in self.history_surge_cmd],
            "history_tau_surge": [float(v) for v in self.history_tau_surge],
            "history_tau_yaw": [float(v) for v in self.history_tau_yaw],
            "history_rob": self._build_robustness_plot_data(self.history_rob),
            "history_maneuver_rob": self._build_robustness_plot_data(
                self.history_maneuver_rob
            ),
        }

    @staticmethod
    def _build_robustness_plot_data(history):
        plot_data = []
        for rob in history:
            if rob is None:
                plot_data.append(None)
                continue

            trace_list = rob[0]
            if not trace_list:
                plot_data.append(None)
                continue

            _, rob_interval = trace_list[0]
            plot_data.append(
                {
                    "lower": float(rob_interval.l),
                    "upper": float(rob_interval.u),
                }
            )
        return plot_data

    def _populate_masking_diagnostics_info(self, info):
        diagnostics = getattr(self.state, "last_mask_diagnostics", {}) or {}
        if not diagnostics:
            return

        field_map = {
            "effective_depth": "masking/effective_depth",
            "candidate_count": "masking/candidate_actions",
            "safe_action_count": "masking/search_safe_actions",
            "certified_action_count": "masking/certified_actions",
            "best_robustness": "masking/best_robustness",
            "elapsed_ms": "masking/search_time_ms",
            "nodes_evaluated": "masking/nodes_evaluated",
            "nodes_pruned": "masking/nodes_pruned",
            "certified_before_full_depth_count": (
                "masking/certified_before_full_depth"
            ),
            "allowed_certified_depth_mean": (
                "masking/allowed_certified_depth_mean"
            ),
            "fallback_action": "masking/fallback_action",
            "fallback_action_yaw_deg_s": "masking/fallback_action_yaw_deg_s",
            "fallback_allowed_all": "masking/fallback_allowed_all",
        }
        for diagnostics_key, info_key in field_map.items():
            value = diagnostics.get(diagnostics_key)
            if value is None:
                continue
            if isinstance(value, bool):
                info[info_key] = float(value)
                continue
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(numeric_value):
                info[info_key] = numeric_value

        fallback_mode = diagnostics.get("fallback_mode")
        if fallback_mode is not None:
            info["masking/fallback_mode"] = fallback_mode

        predicate_summary = diagnostics.get("predicate_summary", {}) or {}
        for predicate_name, summary in predicate_summary.items():
            for stat_name, value in summary.items():
                if value is None:
                    continue
                numeric_value = float(value)
                if np.isfinite(numeric_value):
                    info[f"masking/{predicate_name}_{stat_name}"] = numeric_value

    def _update_encounter_state(self, robustness):
        self.masking.update_encounter_state(self, robustness)

    def action_masks(self):
        if self._straight_hold_active() or self._pre_maneuver_active():
            return self._neutral_action_mask.copy()
        return self.masking.action_masks(self)

    def configure_maneuver_monitoring(self, spec_factory):
        """Configure ManeuverRobustness after the maneuver spec factory is set."""
        self.maneuver_robustness.configure(
            spec_factory=spec_factory,
            tube_time_steps=self.encounter_scenario.tube_time_steps,
            ellipsoids_Ab_dict=self.ellipsoids_Ab_dict,
        )

    def configure_monitoring(self, spec, ellipsoids_Ab_dict, sampling_rate=None):
        if sampling_rate is None:
            sampling_rate = self.robustness.sampling_rate
        self.spec = spec
        self.robustness.spec = spec
        self.robustness.sampling_rate = sampling_rate
        self._reachable_set_bank = (
            ellipsoids_Ab_dict
            if self._is_reachable_set_bank(ellipsoids_Ab_dict)
            else None
        )
        self.ellipsoids_Ab_dict = (
            self._select_reachable_set_for_current_speed()
            if self._reachable_set_bank is not None
            else ellipsoids_Ab_dict
        )
        self.encounter_scenario.configure_monitoring_cache(self.ellipsoids_Ab_dict)
        self.robustness.ellipsoids_Ab_dict = self.ellipsoids_Ab_dict
        if self.maneuver_spec_factory is not None:
            self.configure_maneuver_monitoring(self.maneuver_spec_factory)

    @staticmethod
    def _is_reachable_set_bank(ellipsoids_Ab_dict):
        return isinstance(ellipsoids_Ab_dict, (list, tuple))

    def _select_reachable_set_for_current_speed(self):
        if self._reachable_set_bank is None:
            return self.ellipsoids_Ab_dict
        return select_reachable_set(
            self._reachable_set_bank,
            self.encounter_scenario.target_speed,
        )

    def _refresh_monitoring_for_current_speed(self):
        if self._reachable_set_bank is None:
            return

        self.ellipsoids_Ab_dict = self._select_reachable_set_for_current_speed()
        self.encounter_scenario.configure_monitoring_cache(self.ellipsoids_Ab_dict)
        self.robustness.ellipsoids_Ab_dict = self.ellipsoids_Ab_dict
        if self.maneuver_spec_factory is not None:
            self.configure_maneuver_monitoring(self.maneuver_spec_factory)
        if self.state._active_maneuver_spec is not None:
            self.masking._action_masker.update_scenario(
                self.state._active_maneuver_spec,
                self.ellipsoids_Ab_dict,
                self.encounter_scenario,
                spec_factory=self.maneuver_spec_factory,
            )

    def _check_termination(self, boat_pos):
        terminated, truncated, info = super()._check_termination(boat_pos)
        if terminated or truncated:
            return terminated, truncated, info
        terminated, info = self.termination.is_terminated(self, boat_pos)
        if terminated:
            return True, False, info

        truncated, info = self.truncation.is_truncated(self.state, self.curr_sim_time)
        if truncated:
            return False, True, info

        return False, False, {}

    def reset(self, seed=None, options=None):
        if self.encounter_scenario.start_position is not None:
            self.encounter_scenario.prepare_reset(seed=seed)
            self.encounter_scenario.apply(self)
        obs, info = super().reset(seed=seed, options=options)
        self.state.reset()
        self.reward.reset()
        self.observation.reset()
        self.masking.reset(self)
        self.history_ego = self.state.history_ego
        self.history_enc = self.state.history_enc
        self.history_rob = self.state.history_rob
        self.history_maneuver_rob = self.state.history_maneuver_rob
        self.history_in_radius = self.state.history_in_radius
        self.history_surge_command_fraction = self.state.history_surge_command_fraction
        self.history_heading_deg = self.state.history_heading_deg
        self.history_heading_cmd_deg = self.state.history_heading_cmd_deg
        self.history_heading_error_deg = self.state.history_heading_error_deg
        self.history_surge = self.state.history_surge
        self.history_surge_cmd = self.state.history_surge_cmd
        self.history_yaw_rate_cmd_deg_s = self.state.history_yaw_rate_cmd_deg_s
        self.history_surge_accel_cmd = self.state.history_surge_accel_cmd
        self.history_tau_surge = self.state.history_tau_surge
        self.history_tau_yaw = self.state.history_tau_yaw

        self.encounter_scenario.reset(self.state, prepared=True)
        self._refresh_monitoring_for_current_speed()

        self._sync_runtime_state()
        self._step_count = 0
        self._straight_hold_released = False
        if hasattr(self._controller, "reset"):
            self._controller.reset()
        self._episode_reward = 0.0
        self.state.initialize_command_references(
            self.state.sim_state,
            use_transit_heading=self._straight_hold_available(),
        )
        self.state.record()

        return self._obs(), {}

    def compute_reward(self, action, prev_action):
        return self.reward.get_reward(self.state)[0]

    def _obs(self):
        return self.observation.get(self.state)
