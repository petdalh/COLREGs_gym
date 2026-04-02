import numpy as np
from gymnasium import spaces

from mchorcrux.numpy_core.gym.mc_gym_csad_numpy import MCGym

from gym.action.action import Action
from gym.utils.config import load_config
from gym.utils.geometry import propagate_vessel, within_monitoring_radius
from gym.utils.robustness import evaluate_robustness



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

    def step(self, action):
        total_reward = 0.0
        decision_interval = 5

        if self._step_count % 100 == 0:
            print(f"Step {self._step_count}")

        terminated = False
        truncated = False
        info = {}

        for sub in range(decision_interval):
            if self.encounter_vessel_eta is not None:
                self.encounter_vessel_eta = propagate_vessel(
                    vessel_eta=self.encounter_vessel_eta,
                    encounter_speed=self.encounter_speed,
                    dt=self.dt,
                )

            # Check for divergence BEFORE computing obs/controller.
            # A state that is finite in float64 but overflows float32
            # will corrupt the observation, controller input, and torques.
            state = self.get_state()
            if not np.all(np.isfinite(state["eta"])) or not np.all(
                np.isfinite(state["nu"])
            ):
                print(f"[Divergence] Non-finite state at step {self._step_count}")
                terminated = True
                info = {"reason": "diverged"}
                break

            obs = self._obs()
            psi_d, u_d = self._decode_discrete_actions(action, obs)
            tau = self._controller.compute_action_minimal(self.get_state(), psi_d, u_d)

            # Also catch states that are finite but too large for float32
            if np.any(np.abs(state["eta"][:2]) > 1e6) or np.any(
                np.abs(state["nu"]) > 1e6
            ):
                print(
                    f"[Divergence] State magnitude too large at step {self._step_count}"
                )
                terminated = True
                info = {"reason": "diverged"}
                break

            _, reward, terminated, truncated, info = super().step(tau)
            self._step_count += 1
            total_reward += reward

            # Record positions
            state = self.get_state()
            self.history_ego.append([state["eta"][0], state["eta"][1]])
            if self.encounter_vessel_eta is not None:
                self.history_enc.append(
                    [self.encounter_vessel_eta[0], self.encounter_vessel_eta[1]]
                )

            if terminated or truncated:
                break

        # pacSTL evaluation — only if within monitoring radius
        robustness = None
        in_radius = within_monitoring_radius(
            self.encounter_vessel_eta, self.monitoring_radius, self.get_state()
        )

        if self._step_count % self.robustness_sampling_rate == 0:
            if in_radius:
                robustness = evaluate_robustness(
                    spec=self.spec,
                    ellipsoids_Ab_dict=self.ellipsoids_Ab_dict,
                    encounter_vessel_eta=self.encounter_vessel_eta,
                    state=self.get_state(),
                    encounter_speed=self.encounter_speed,
                )
                info["robustness"] = robustness
                self._update_encounter_state(robustness)
            else:
                # Outside monitoring radius: clear any active encounter state
                if self._encounter_active:
                    self._encounter_active = False
                    self._active_maneuver_spec = None
                    self._cached_mask = np.ones(
                        len(self.action_space["size"]), dtype=bool
                    )

        # Record robustness and radius status
        self.history_rob.append(robustness)
        self.history_in_radius.append(in_radius)

        self._episode_reward += float(total_reward)
        if terminated or truncated:
            reason = info.get("reason", "goal_reached")
            status = "Terminated" if terminated else "Truncated"
            print(
                f"--- Episode {status} | Reason: {reason} | Reward: {self._episode_reward:.2f} ---"
            )

        return self._obs(), float(total_reward), terminated, truncated, info
