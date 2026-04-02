from gym.action.action import Action
from gym.callback.episode_logger import EpisodeLogger
from gym.observation.observation import Observation
from gym.reward.reward import Reward
from gym.robustness.robustness import Robustness
from gym.state import State
from gym.termination import Termination
from gym.truncation import Truncation
from gym.utils.config import load_config
from gymnasium import spaces
from mchorcrux.numpy_core.gym.mc_gym_csad_numpy import MCGym


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

        self.vessel_action = Action(config)
        self.action_space = spaces.Discrete(self.vessel_action.n_actions)
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

    def compute_reward(self, action, prev_action):
        return self.reward.get_reward(self.state)

    def _obs(self):
        return self.observation.get(self.state)
