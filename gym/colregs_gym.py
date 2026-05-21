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
from mcsimpy.utils import three2sixDOF

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
        self._dynamics_backend = env_cfg.get("dynamics_backend", "mchorcrux")

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
        )
        self.vessel_action = Action(action_cfg, v_max=ego_vessel_model.v_max)
        self.action_space = spaces.Discrete(self.vessel_action.n_actions)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32
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
        self.observation = Observation(env_cfg.get("observation_configuration", config))
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
        self._reference_shoebox = None
        self._reference_shoebox_sub_dt = float(masking_cfg.get("shoebox_sub_dt", 0.1))
        if self._dynamics_backend == "shoebox":
            self._init_reference_shoebox(masking_cfg)
        elif self._dynamics_backend != "mchorcrux":
            raise ValueError(
                "Unsupported dynamics_backend="
                f"{self._dynamics_backend!r}; expected 'mchorcrux' or 'shoebox'."
            )
        self._neutral_action_mask = self._build_neutral_action_mask()
        self._step_count = 0

    def _init_reference_shoebox(self, config):
        try:
            from shoeboxpy.model3dof import Shoebox
        except ImportError as exc:
            raise RuntimeError(
                "shoeboxpy is required when dynamics_backend: shoebox"
            ) from exc

        self._reference_shoebox = Shoebox(
            L=float(config.get("shoebox_L", 2.578)),
            B=float(config.get("shoebox_B", 0.444)),
            T=float(config.get("shoebox_T", 0.133)),
            rho=float(config.get("shoebox_rho", 1025.0)),
            alpha_u=float(config.get("shoebox_alpha_u", 0.11009)),
            alpha_v=float(config.get("shoebox_alpha_v", 0.86267)),
            alpha_r=float(config.get("shoebox_alpha_r", 0.58680)),
            beta_u=float(config.get("shoebox_beta_u", 0.30137)),
            beta_v=float(config.get("shoebox_beta_v", 2.36154)),
            beta_r=float(config.get("shoebox_beta_r", 2.84762)),
        )

    def _sync_runtime_state(self):
        sim_state = self.get_state()
        self.state.update_sim(sim_state)
        self.state.in_monitoring_radius = within_monitoring_radius(
            self.state.encounter_vessel_eta,
            self.state.monitoring_radius,
            sim_state,
        )
        return sim_state

    def _initial_surge_speed_mps(self):
        return float(
            np.clip(
                self.state.initial_surge_command_fraction * self.ego_vessel_model.v_max,
                self.state.min_surge_command_mps,
                self.ego_vessel_model.v_max,
            )
        )

    def _set_vessel_surge_speed(self, surge_speed):
        nu = np.asarray(self.vessel.get_nu(), dtype=float).copy()
        nu[0] = float(surge_speed)
        self.vessel.set_nu(nu)

    def _steady_surge_force(self, surge_speed):
        damping = getattr(self.vessel, "_D", None)
        if damping is None:
            return 0.0
        damping = np.asarray(damping, dtype=float)
        if damping.ndim != 2 or damping.shape[0] == 0 or damping.shape[1] == 0:
            return 0.0
        return float(damping[0, 0] * float(surge_speed))

    def _sync_observer_to_vessel(self):
        observer = getattr(self, "_observer", None)
        if observer is None:
            return

        sim_state = self.state.sim_state
        eta = np.asarray(sim_state["eta"], dtype=float)
        nu = np.asarray(sim_state["nu"][:3], dtype=float)
        eta_3dof = np.array([eta[0], eta[1], eta[-1]], dtype=float)

        observer._x_hat[:] = 0.0
        observer._x_hat[6:9] = eta_3dof
        observer._x_hat[12:15] = nu
        observer._y_hat[:] = eta_3dof

    def _warm_start_controller_for_reset(self):
        if not hasattr(self._controller, "warm_start"):
            return
        psi_d = self.state._current_heading_cmd
        u_d = self.state._current_surge_cmd
        if psi_d is None or u_d is None:
            return
        self._controller.warm_start(
            psi_d=psi_d,
            u_d=u_d,
            surge_feedforward=self._steady_surge_force(u_d),
        )

    def _initialize_reset_surge_state(self):
        surge_speed = self._initial_surge_speed_mps()
        self._set_vessel_surge_speed(surge_speed)
        self._sync_runtime_state()
        self._sync_observer_to_vessel()
        self._sync_reference_shoebox_from_state(
            psi_d=self.state._current_heading_cmd
        )
        self._warm_start_controller_for_reset()

    def _sync_reference_shoebox_from_state(self, psi_d=None):
        if self._reference_shoebox is None:
            return

        sim_state = self.state.sim_state
        eta = np.asarray(sim_state["eta"], dtype=float)
        nu = np.asarray(sim_state["nu"][:3], dtype=float)
        sb = self._reference_shoebox
        sb.eta[0] = float(eta[0])
        sb.eta[1] = float(eta[1])
        sb.eta[2] = float(eta[-1] if psi_d is None else psi_d)
        sb.nu[0] = float(nu[0])
        sb.nu[1] = float(nu[1])
        sb.nu[2] = float(nu[2])

    def _reference_shoebox_force_from_action(
        self, nu, surge_accel_cmd, yaw_rate_cmd, sub_dt
    ):
        u, v, r = nu
        sb = self._reference_shoebox
        cnu0 = -(sb.m + sb.MA[1, 1]) * v * r
        cnu1 = (sb.m + sb.MA[0, 0]) * u * r
        Dnu0 = sb.D[0, 0] * u
        Dnu1 = sb.D[1, 1] * v
        Dnu2 = sb.D[2, 2] * r
        r_dot_des = (yaw_rate_cmd - r) / sub_dt
        tau_X = sb.M_eff[0, 0] * surge_accel_cmd + Dnu0 + cnu0
        tau_Y = Dnu1 + cnu1
        tau_N = sb.M_eff[2, 2] * r_dot_des + Dnu2
        return np.array([tau_X, tau_Y, tau_N], dtype=float)

    def _set_vessel_3dof_state(self, eta_3dof, nu_3dof):
        self.vessel.set_eta(three2sixDOF(np.asarray(eta_3dof, dtype=float)))
        self.vessel.set_nu(three2sixDOF(np.asarray(nu_3dof, dtype=float)))

    def _step_reference_shoebox(
        self, psi_d, surge_accel_cmd, yaw_rate_cmd, dt
    ):
        self._sync_reference_shoebox_from_state(psi_d=psi_d)
        sb = self._reference_shoebox
        last_tau = np.zeros(3)
        n_sub = max(1, round(float(dt) / self._reference_shoebox_sub_dt))
        sub_dt = float(dt) / n_sub

        u_start = float(sb.nu[0])
        u_target = float(
            np.clip(
                u_start + float(surge_accel_cmd) * float(dt),
                self.state.min_surge_command_mps,
                self.ego_vessel_model.v_max,
            )
        )
        u_dot_eff = (u_target - u_start) / float(dt) if dt > 0.0 else 0.0

        for _ in range(n_sub):
            last_tau = self._reference_shoebox_force_from_action(
                sb.nu,
                u_dot_eff,
                float(yaw_rate_cmd),
                sub_dt,
            )
            sb.step(tau=last_tau, dt=sub_dt)
            sb.nu[0] = float(
                np.clip(
                    sb.nu[0],
                    self.state.min_surge_command_mps,
                    self.ego_vessel_model.v_max,
                )
            )

        self.curr_sim_time += float(dt)
        self._set_vessel_3dof_state(sb.eta, sb.nu)
        self._sync_runtime_state()
        self._sync_observer_to_vessel()
        boat_pos = np.array([sb.eta[0], sb.eta[1], sb.eta[2]], dtype=float)
        terminated, truncated, info = self._check_termination(boat_pos)
        return last_tau, terminated, truncated, info

    def _build_neutral_action_mask(self):
        mask = np.zeros(self.vessel_action.n_actions, dtype=bool)
        yaw_idx = int(np.argmin(np.abs(self.vessel_action.yaw_rate_commands)))
        accel_idx = int(np.argmin(np.abs(self.vessel_action.surge_accel_commands)))
        action_idx = yaw_idx * len(self.vessel_action.surge_accel_commands) + accel_idx
        mask[action_idx] = True
        return mask

    def _pre_maneuver_active(self):
        if self.pre_maneuver_duration_s <= 0.0:
            return False
        return self._step_count * self.dt < self.pre_maneuver_duration_s

    def set_encounter(
        self,
        start_position,
        wave_conditions,
        encounter_type="crossing",
        separation=30.0,
        target_speed=0.3,
        target_speeds=None,
        time_to_conflict_s=None,
        ego_reference_speed=None,
        heading_noise_deg=5.0,
        encounter_position_noise_m=2.0,
        target_cross_track_noise_m=0.0,
        goal_ahead_distance=25.0,
        collision_radius=1.0,
        simtime=150.0,
        masking_configuration=None,
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
            time_to_conflict_s=time_to_conflict_s,
            ego_reference_speed=ego_reference_speed,
            heading_noise_deg=heading_noise_deg,
            encounter_position_noise_m=encounter_position_noise_m,
            target_cross_track_noise_m=target_cross_track_noise_m,
            goal_ahead_distance=goal_ahead_distance,
            collision_radius=collision_radius,
            simtime=simtime,
            masking_configuration=masking_configuration,
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

        last_tau = np.zeros(3)
        pre_maneuver_was_active = False
        for _ in range(self.decision_interval):
            pre_maneuver_active = self._pre_maneuver_active()
            pre_maneuver_was_active = pre_maneuver_was_active or pre_maneuver_active
            if pre_maneuver_active:
                yaw_rate_cmd = 0.0
                surge_accel_cmd = 0.0
            else:
                yaw_rate_cmd = requested_yaw_rate_cmd
                surge_accel_cmd = requested_surge_accel_cmd
            self.state.set_current_rate_commands(yaw_rate_cmd, surge_accel_cmd)

            self.state.propagate_encounter(self.dt)
            sim_state = self._sync_runtime_state()

            terminated, term_info = self.termination.is_terminated(self.state)
            if terminated:
                info.update(term_info)
                break

            if pre_maneuver_active:
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
            if self._dynamics_backend == "shoebox":
                last_tau, terminated, truncated, info = self._step_reference_shoebox(
                    psi_d,
                    surge_accel_cmd,
                    yaw_rate_cmd,
                    self.dt,
                )
                self.state.set_current_tau(last_tau)
            else:
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
                self.state.set_current_tau(last_tau)
                _, _, terminated, truncated, info = super().step(tau)
            self._step_count += 1

            self._sync_runtime_state()
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
        info["pre_maneuver_active"] = bool(pre_maneuver_was_active)
        info["mask_allowed_count"] = int(
            self._neutral_action_mask.sum()
            if pre_maneuver_was_active
            else self.state._cached_mask.sum()
        )
        info["mask_fallback"] = bool(self.state.fallback_used)

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

        if terminated or truncated:
            self.callback.on_episode_end(info, terminated, self.reward)

        return obs, reward, terminated, truncated, info

    def _update_encounter_state(self, robustness):
        self.masking.update_encounter_state(self, robustness)

    def action_masks(self):
        if self._pre_maneuver_active():
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
        self.initial_surge_speed_mps = self._initial_surge_speed_mps()
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

        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.encounter_scenario.reset(self.state)
        self._refresh_monitoring_for_current_speed()

        self._sync_runtime_state()
        self._step_count = 0
        if hasattr(self._controller, "reset"):
            self._controller.reset()
        self._episode_reward = 0.0
        self.state.initialize_command_references(self.state.sim_state)
        self._initialize_reset_surge_state()
        self.state.record()

        return self._obs(), {}

    def compute_reward(self, action, prev_action):
        return self.reward.get_reward(self.state)[0]

    def _obs(self):
        return self.observation.get(self.state)
