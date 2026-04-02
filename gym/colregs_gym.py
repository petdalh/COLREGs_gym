from gym.action.action import Action
from gym.callback.episode_logger import EpisodeLogger
from gym.observation.observation import Observation
from gym.reward.reward import Reward
from gym.robustness.robustness import Robustness
from gym.masking import Masking
from gym.state import State
from gym.termination import Termination
from gym.truncation import Truncation
from gym.utils.config import load_config

from gymnasium import spaces

from mchorcrux.numpy_core.gym.mc_gym_csad_numpy import MCGym
from mchorcrux.numpy_core.controllers.adaptive_seakeeping import MRACShipController

import numpy as np


class COLREGsGym(MCGym):
    def __init__(
        self,
        vessel_model,
        dt,
        grid_width,
        grid_height,
        config=load_config("configuration/config.yaml"),
        **kwargs,
    ):
        super().__init__(
            dt=dt, grid_width=grid_width, grid_height=grid_height, **kwargs
        )

        self.vessel_model = vessel_model
        self.vessel_action = Action(config)
        self.action_space = spaces.Discrete(self.vessel_action.n_actions)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )
        self.encounter_type = None
        self._encounter_init = None
        self.encounter_vessel_eta = None
        self.encounter_speed = 0.0
        self.encounter_radius = 1.0
        self.encounter_max_time = None
        self.spec = None
        self.ellipsoids_Ab_dict = None
        self.robustness_sampling_rate = 10
        self.robustness_margin = 1.0
        self._encounter_active = False
        self._active_maneuver_spec = None
        self._cached_mask = np.ones(self.vessel_action.n_actions, dtype=bool)
        self._mask_recompute_interval = 2
        self._steps_since_mask_update = 0
        self.v_min = getattr(vessel_model, "v_min", None)
        self.v_max = getattr(vessel_model, "v_max", None)
        self.r_min = getattr(vessel_model, "yaw_dot_min", None)
        self.r_max = getattr(vessel_model, "yaw_dot_max", None)
        self.masking = Masking(
            n_actions=self.vessel_action.n_actions,
            robustness_margin=self.robustness_margin,
            mask_recompute_interval=self._mask_recompute_interval,
            vessel_model=vessel_model,
            sim_dt=self.dt,
        )
        self.state = State(
            monitoring_radius=self.monitoring_radius,
            n_actions=self.vessel_action.n_actions,
        )
        self.observation = Observation(config)
        self.reward = Reward(config)
        self.robustness = Robustness(
            spec=self.spec,
            ellipsoids_Ab_dict=self.ellipsoids_Ab_dict,
            sampling_rate=self.robustness_sampling_rate,
        )
        self.termination = Termination()
        self.truncation = Truncation()
        self.callback = EpisodeLogger()
        self.decision_interval = 5

    def step(self, action):
        self._step_count += 1
        terminated = False
        truncated = False
        info = {}

        for _ in range(self.decision_interval):
            self.state.propagate_encounter(self.dt)
            self.state.update_sim(self.get_state())

            terminated, term_info = self.termination.is_terminated(self.state)
            if terminated:
                info.update(term_info)
                break

            tau = self.vessel_action.compute(
                action, self._obs(), self.state.sim_state, self._controller
            )
            _, _, terminated, truncated, info = super().step(tau)

            self.state.update_sim(self.get_state())
            self.state.record()

            if terminated or truncated:
                break

        robustness = self.robustness.evaluate(self.state, self._step_count)
        if robustness is not None:
            info["robustness"] = robustness
            self._update_encounter_state(robustness)

        obs = self.observation.get(self.state)
        reward = self.reward.get_reward(self.state)

        if terminated or truncated:
            self.callback.on_episode_end(info, terminated, self.reward)

        return obs, reward, terminated, truncated, info

    def _update_encounter_state(self, robustness):
        self.masking.update_encounter_state(self, robustness)

    def action_masks(self):
        return self.masking.action_masks(self)

    def configure_monitoring(self, spec, ellipsoids_Ab_dict, sampling_rate=10):
        self.spec = spec
        self.ellipsoids_Ab_dict = ellipsoids_Ab_dict
        self.robustness_sampling_rate = sampling_rate
        self.robustness.spec = spec
        self.robustness.ellipsoids_Ab_dict = ellipsoids_Ab_dict
        self.robustness.sampling_rate = sampling_rate
        self.masking.configure_monitoring(self, spec, ellipsoids_Ab_dict)
    
    def _check_termination(self, boat_pos):
        terminated, truncated, info = super()._check_termination(boat_pos)
        if terminated or truncated:
            return terminated, truncated, info
        terminated, info = self.termination.is_terminated(self, boat_pos)
        if terminated:
            return True, False, info

        truncated, info = self.truncation.is_truncated(self)
        if truncated:
            return False, True, info

        return False, False, {}
    
    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.state.reset()
        self.reward.reset()
        self.masking.reset(self)
        self._steps_since_mask_update = 0
        self.history_ego = self.state.history_ego
        self.history_enc = self.state.history_enc
        self.history_rob = self.state.history_rob
        self.history_in_radius = self.state.history_in_radius

        if self._encounter_init is not None:
            self.encounter_vessel_eta = self._encounter_init.copy()
        else:
            self.encounter_vessel_eta = None

        self.state.encounter_vessel_eta = self.encounter_vessel_eta
        self.state.encounter_speed = getattr(self, "encounter_speed", None)
        self.state.goal = getattr(self, "goal", None)
        self.state.nominal_path_start = getattr(self, "_nominal_path_start", None)
        self.state.update_sim(self.get_state())

        state = self.get_state()
        if self.state.goal is not None:
            gn, ge = self.state.goal[:2]
            self.reward.prev_dist_to_goal = np.hypot(
                gn - state["eta"][0], ge - state["eta"][1]
            )
        else:
            self.reward.prev_dist_to_goal = None
        self.prev_dist_to_goal = self.reward.prev_dist_to_goal
        self._step_count = 0
        self._controller = MRACShipController(dt=self.dt)
        self._episode_reward = 0.0

        self.state.record()

        return self._obs(), {}


    def compute_reward(self, action, prev_action):
        return self.reward.get_reward(self.state)

    def _obs(self):
        return self.observation.get(self.state)
