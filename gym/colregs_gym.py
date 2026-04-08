from gym.action.action import Action
from gym.callback.episode_logger import EpisodeLogger
from gym.encounter_scenario import EncounterScenario
from gym.observation.observation import Observation
from gym.reward.reward import Reward
from gym.robustness.robustness import Robustness
from gym.masking import Masking
from gym.state import State
from gym.termination import Termination
from gym.truncation import Truncation
from gym.utils.config import resolve_config
from gym.utils.geometry import within_monitoring_radius

from gymnasium import spaces

from mchorcrux.numpy_core.gym.mc_gym_csad_numpy import McGym
from mchorcrux.numpy_core.controllers.adaptive_seakeeping import MRACShipController

import numpy as np


class COLREGsGym(McGym):
    def __init__(
        self,
        vessel_model,
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
        masking_cfg["sim_dt"] = sim_dt
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

        super().__init__(
            dt=dt, grid_width=grid_width, grid_height=grid_height, **kwargs
        )

        self.vessel_model = vessel_model
        self.config = config
        self.encounter_scenario = EncounterScenario(
            vessel_model,
            maneuver_horizon=maneuver_horizon,
            monitoring_radius_safety_factor=monitoring_radius_safety_factor,
            masking_configuration=masking_cfg,
        )
        self.vessel_action = Action(action_cfg)
        self.action_space = spaces.Discrete(self.vessel_action.n_actions)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32
        )
        self.spec = None
        self.ellipsoids_Ab_dict = None
        self.reward = Reward(config=reward_cfg)
        self.masking = Masking(
            n_actions=self.vessel_action.n_actions,
            robustness_margin=robustness_margin,
            mask_recompute_interval=monitoring_cfg.get("mask_recompute_interval", 2),
            enabled=monitoring_cfg.get("masking_enabled", True),
            action=self.vessel_action,
            vessel_model=vessel_model,
            action_masking_config=masking_cfg,
        )
        self.state = State(
            monitoring_radius=monitoring_radius,
            n_actions=self.vessel_action.n_actions,
        )
        self.observation = Observation(env_cfg.get("observation_configuration", config))
        self.robustness = Robustness(
            spec=self.spec,
            ellipsoids_Ab_dict=self.ellipsoids_Ab_dict,
            sampling_rate=monitoring_cfg.get("robustness_sampling_rate", 10),
        )
        self.termination = Termination()
        self.truncation = Truncation()
        self.callback = EpisodeLogger()
        self.decision_interval = env_cfg.get("decision_interval", 5)

    def _sync_runtime_state(self):
        sim_state = self.get_state()
        self.state.update_sim(sim_state)
        self.state.in_monitoring_radius = within_monitoring_radius(
            self.state.encounter_vessel_eta,
            self.state.monitoring_radius,
            sim_state,
        )
        return sim_state

    def set_encounter(
        self,
        start_position,
        wave_conditions,
        encounter_type="crossing",
        separation=30.0,
        target_speed=0.3,
        goal_ahead_distance=25.0,
        collision_radius=1.0,
        simtime=150.0,
        masking_configuration=None,
    ):
        self.encounter_scenario.set_encounter(
            start_position=start_position,
            wave_conditions=wave_conditions,
            encounter_type=encounter_type,
            separation=separation,
            target_speed=target_speed,
            goal_ahead_distance=goal_ahead_distance,
            collision_radius=collision_radius,
            simtime=simtime,
            masking_configuration=masking_configuration,
        )
        self.masking.configure_for_scenario(self.encounter_scenario)
        self.encounter_scenario.apply(self)

    def step(self, action):
        terminated = False
        truncated = False
        info = {}
        speed_multiplier = self.vessel_action.speed_multipliers[
            action % len(self.vessel_action.speed_multipliers)
        ]
        self.state.set_current_speed_multiplier(speed_multiplier)
        h_idx = action // len(self.vessel_action.speed_multipliers)
        self.state.set_current_heading_offset(self.vessel_action.heading_offsets[h_idx])

        for _ in range(self.decision_interval):
            self.state.propagate_encounter(self.dt)
            sim_state = self._sync_runtime_state()

            terminated, term_info = self.termination.is_terminated(self.state)
            if terminated:
                info.update(term_info)
                break

            tau = self.vessel_action.compute(
                action, sim_state, self._controller
            )
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

        obs = self.observation.get(self.state)
        reward, reward_info = self.reward.get_reward(
            self.state,
            in_radius=self.state.in_monitoring_radius,
        )
        info.update(reward_info)

        if terminated or truncated:
            self.callback.on_episode_end(info, terminated, self.reward)

        return obs, reward, terminated, truncated, info

    def _update_encounter_state(self, robustness):
        self.masking.update_encounter_state(self, robustness)

    def action_masks(self):
        return self.masking.action_masks(self)

    def configure_monitoring(self, spec, ellipsoids_Ab_dict, sampling_rate=None):
        if sampling_rate is None:
            sampling_rate = self.robustness.sampling_rate
        self.spec = spec
        self.ellipsoids_Ab_dict = ellipsoids_Ab_dict
        self.encounter_scenario.configure_monitoring_cache(ellipsoids_Ab_dict)
        self.robustness.spec = spec
        self.robustness.ellipsoids_Ab_dict = ellipsoids_Ab_dict
        self.robustness.sampling_rate = sampling_rate

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
        obs, info = super().reset(seed=seed, options=options)
        self.state.reset()
        self.reward.reset()
        self.observation.reset()
        self.masking.reset(self)
        self.history_ego = self.state.history_ego
        self.history_enc = self.state.history_enc
        self.history_rob = self.state.history_rob
        self.history_in_radius = self.state.history_in_radius
        self.history_speed_multiplier = self.state.history_speed_multiplier

        if self.encounter_scenario._encounter_init is not None:
            self.state.encounter_vessel_eta = self.encounter_scenario._encounter_init.copy()
        else:
            self.state.encounter_vessel_eta = None

        self.encounter_scenario.apply_to_state(self.state, self.state.encounter_vessel_eta)
        self.state.encounter_scenario = self.encounter_scenario
        self._sync_runtime_state()
        self._step_count = 0
        self._controller = MRACShipController(dt=self.dt)
        self._episode_reward = 0.0

        self.state.record()

        return self._obs(), {}

    def compute_reward(self, action, prev_action):
        return self.reward.get_reward(self.state)[0]

    def _obs(self):
        return self.observation.get(self.state)
