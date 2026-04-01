"""
Minimal COLREGs encounter gym for RL training with pacSTL monitoring.

Wraps the numpy_core McGym with encounter vessel kinematics,
collision detection, and pacSTL robustness evaluation.
"""

import numpy as np
from mchorcrux.numpy_core.gym.mc_gym_csad_numpy import McGym
from mchorcrux.numpy_core.controllers.adaptive_seakeeping import MRACShipController
from utils.geometry import cross_track_error, within_monitoring_radius, propagate_vessel
from utils.robustness import evaluate_robustness
from logic.masking import get_action_mask, decode_discrete_actions
from config import N_DISCRETE_ACTIONS
from gymnasium import spaces

class ColregsGym(McGym):
    def __init__(self, vessel_model, dt, grid_width, grid_height,
                 maneuver_horizon=10.0, sim_dt=0.5,
                 monitoring_radius=14.0, monitoring_radius_safety_factor=2.0,
                 robustness_margin=1.0,
                 w_cte=0.005, cte_clip=5.0,
                 **kwargs):
        super().__init__(dt=dt, grid_width=grid_width, grid_height=grid_height, **kwargs)

        # Override spaces for discrete RL with 10D observation
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)

        # Encounter state
        self.encounter_type = None
        self._encounter_init = None
        self.encounter_vessel_eta = None
        self.encounter_speed = 0.0
        self.encounter_radius = 1.0
        self.encounter_max_time = None

        # pacSTL
        self.spec = None
        self.ellipsoids_Ab_dict = None
        self.robustness_sampling_rate = 10

        # Vessel model limits
        self.v_min = vessel_model.v_min
        self.v_max = vessel_model.v_max
        self.r_min = vessel_model.yaw_dot_min
        self.r_max = vessel_model.yaw_dot_max

        self.prev_dist_to_goal = 0.0
        self._step_count = 0
        self.maneuver_horizon = maneuver_horizon
        self.sim_dt = sim_dt
        self.w_cte = float(w_cte)
        self.cte_clip = float(cte_clip)
        self._nominal_path_start = None

        # Controller lives inside the environment now
        self._controller = None
        self._episode_reward = 0.0

        # Encounter detection state for action masking
        self._encounter_active = False
        self._active_maneuver_spec = None

        # Mask parameters
        self._cached_mask = np.ones(N_DISCRETE_ACTIONS, dtype=bool)
        self._mask_recompute_interval = 2
        self._steps_since_mask_update = 0

        # Monitoring radius 
        self.monitoring_radius = monitoring_radius

        # Robustness margin for action masking
        self.robustness_margin = robustness_margin


    def set_encounter(
        self,
        start_position: tuple,
        wave_conditions: tuple,
        encounter_type: str = "crossing",
        separation: float = 30.0,
        target_speed: float = 0.3,
        goal_ahead_distance: float = 25.0,
        collision_radius: float = 1.0,
        simtime: float = 150.0,
    ):
        """Configure a COLREGs encounter scenario."""

        if target_speed > self.v_max:
            raise ValueError(
                f"target_speed={target_speed} exceeds vessel model v_max={self.v_max}. "
                "The pacSTL scaling assumes the obstacle speed is within [v_min, v_max]."
            )
        if target_speed < self.v_min:
            raise ValueError(
                f"target_speed={target_speed} is below vessel model v_min={self.v_min}."
            )

        own_n, own_e, own_psi_deg = start_position
        own_psi_rad = np.deg2rad(own_psi_deg)
        self._nominal_path_start = np.array([own_n, own_e], dtype=float)

        if encounter_type == "crossing":
            bearing_rad = own_psi_rad + np.deg2rad(45.0)
            t_n = own_n + separation * np.cos(bearing_rad)
            t_e = own_e + separation * np.sin(bearing_rad)
            t_psi_deg = (own_psi_deg - 90.0) % 360.0
        else:
            raise ValueError(f"Unsupported encounter type: {encounter_type}")

        self.encounter_type = encounter_type
        self.encounter_speed = float(target_speed)
        self.encounter_radius = float(collision_radius)
        self.encounter_max_time = float(simtime)
        self._encounter_init = np.array([t_n, t_e, np.deg2rad(t_psi_deg)])

        goal_n = own_n + goal_ahead_distance * np.cos(own_psi_rad)
        goal_e = own_e + goal_ahead_distance * np.sin(own_psi_rad)


        self.set_task(
            start_position=start_position,
            goal=(goal_n, goal_e, 1.0),
            wave_conditions=wave_conditions,
            simtime=simtime,
        )
    
    def configure_monitoring(self, spec, ellipsoids_Ab_dict, sampling_rate=10):
        self.spec = spec
        self.ellipsoids_Ab_dict = ellipsoids_Ab_dict
        self.robustness_sampling_rate = sampling_rate

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        if self._encounter_init is not None:
            self.encounter_vessel_eta = self._encounter_init.copy()

        state = self.get_state()
        gn, ge = self.goal[:2]
        self.prev_dist_to_goal = np.hypot(gn - state["eta"][0], ge - state["eta"][1])
        self._step_count = 0
        self._encounter_active = False
        self._active_maneuver_spec = None
        self._controller = MRACShipController(dt=self.dt)
        self._episode_reward = 0.0

        # Store history for visualization
        self.history_ego = []
        self.history_enc = []
        self.history_rob = []
        self.history_in_radius = []  # track when monitoring was active
        
        # Record starting positions
        self.history_ego.append([state["eta"][0], state["eta"][1]])
        if self.encounter_vessel_eta is not None:
            self.history_enc.append([self.encounter_vessel_eta[0], self.encounter_vessel_eta[1]])

        return self._obs(), {}

    def step(self, action: int):
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
                    dt=self.dt
                )

            # Check for divergence BEFORE computing obs/controller.
            # A state that is finite in float64 but overflows float32
            # will corrupt the observation, controller input, and torques.
            state = self.get_state()
            if not np.all(np.isfinite(state["eta"])) or not np.all(np.isfinite(state["nu"])):
                print(f"[Divergence] Non-finite state at step {self._step_count}")
                terminated = True
                info = {"reason": "diverged"}
                break

            obs = self._obs()
            psi_d, u_d = decode_discrete_actions(action, obs)
            tau = self._controller.compute_action_minimal(self.get_state(), psi_d, u_d)

            # Also catch states that are finite but too large for float32
            if np.any(np.abs(state["eta"][:2]) > 1e6) or np.any(np.abs(state["nu"]) > 1e6):
                print(f"[Divergence] State magnitude too large at step {self._step_count}")
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
                self.history_enc.append([self.encounter_vessel_eta[0], self.encounter_vessel_eta[1]])

            if terminated or truncated:
                break

        # pacSTL evaluation — only if within monitoring radius
        robustness = None
        in_radius = within_monitoring_radius(self.encounter_vessel_eta, self.monitoring_radius, self.get_state())

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
                    self._cached_mask = np.ones(N_DISCRETE_ACTIONS, dtype=bool)

        # Record robustness and radius status
        self.history_rob.append(robustness)
        self.history_in_radius.append(in_radius)

        self._episode_reward += float(total_reward)
        if terminated or truncated:
            reason = info.get("reason", "goal_reached") 
            status = "Terminated" if terminated else "Truncated"
            print(f"--- Episode {status} | Reason: {reason} | Reward: {self._episode_reward:.2f} ---")

        return self._obs(), float(total_reward), terminated, truncated, info

    def compute_reward(self, action, prev_action):
        state = self.get_state()
        pos_xy = state["eta"][:2]
        gn, ge = self.goal[:2]
        dist = np.hypot(gn - pos_xy[0], ge - pos_xy[1])
        progress = self.prev_dist_to_goal - dist
        self.prev_dist_to_goal = dist
        cte = cross_track_error(pos_xy, self._nominal_path_start, self.goal[:2])
        if not within_monitoring_radius(self.encounter_vessel_eta, self.monitoring_radius, self.get_state()): 
            cte_penalty = self.w_cte * min(abs(cte), self.cte_clip)
            return progress - cte_penalty
        else:
            return progress

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _obs(self) -> np.ndarray:
        state = self.get_state()
        eta, nu = state["eta"], state["nu"]
        gn, ge = self.goal[:2]
        tgt = (
            self.encounter_vessel_eta[:2]
            if self.encounter_vessel_eta is not None
            else np.zeros(2)
        )
        obs = np.concatenate([eta[:2], [eta[-1]], nu, [gn, ge], tgt])

        # Guard against simulation divergence: replace non-finite values
        # with zeros and clamp to float32 range. This prevents NaN from
        # propagating into the policy network and crashing MaskablePPO.
        if not np.all(np.isfinite(obs)):
            obs = np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)

        return obs.astype(np.float32)

    # ------------------------------------------------------------------
    # Action masking and decoding
    # ------------------------------------------------------------------
    
    def _update_encounter_state(self, robustness):
        if robustness is None:
            return

        was_active = self._encounter_active

        trace_list = robustness[0]
        for _, rob_interval in trace_list:
            # Activate encounter when the upper robustness bound approaches
            # zero (within the margin). This triggers masking BEFORE the
            # robustness actually becomes positive, giving the agent time
            # to maneuver while compliant actions still exist.
            if rob_interval.u > -self.robustness_margin:
                self._encounter_active = True
                if self._active_maneuver_spec is None:
                    from pacstl.core.factory import create as create_spec
                    self._active_maneuver_spec = create_spec("colregs", "crossing_detection")
                # Force mask recomputation on transition
                if not was_active:
                    print(f"[Encounter] Activated (rob_upper={rob_interval.u:.2f}, "
                          f"margin={self.robustness_margin:.1f})")
                    self._cached_mask = get_action_mask(self.encounter_type, self._active_maneuver_spec, self.ellipsoids_Ab_dict, self.encounter_vessel_eta, self.get_state(), self.encounter_speed, self.robustness_margin, self.monitoring_radius, self._obs(), self.robustness_margin, self.r_max, self.sim_dt, self.v_max)
                    self._steps_since_mask_update = 0
                return

        self._encounter_active = False
        self._active_maneuver_spec = None

    def action_masks(self) -> np.ndarray:
        if not self._encounter_active or self._active_maneuver_spec is None:
            return np.ones(N_DISCRETE_ACTIONS, dtype=bool)

        # If vessel left monitoring radius, deactivate encounter
        if not within_monitoring_radius(self.encounter_vessel_eta, self.monitoring_radius, self.get_state()):
            self._encounter_active = False
            self._active_maneuver_spec = None
            self._cached_mask = np.ones(N_DISCRETE_ACTIONS, dtype=bool)
            return self._cached_mask

        self._steps_since_mask_update += 1
        if self._steps_since_mask_update >= self._mask_recompute_interval:
            self._cached_mask = get_action_mask(self.encounter_type, self._active_maneuver_spec, self.ellipsoids_Ab_dict, self.encounter_vessel_eta, self.get_state(), self.encounter_speed, self.robustness_margin, self.monitoring_radius, self._obs(), self.robustness_margin, self.r_max, self.sim_dt, self.v_max)
            self._steps_since_mask_update = 0

        return self._cached_mask

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------

    def _check_termination(self, boat_pos):
        terminated, truncated, info = super()._check_termination(boat_pos)
        if terminated or truncated:
            return terminated, truncated, info

        # Detect simulation divergence: if the state has blown up, the
        # dynamics have gone unstable (e.g. large dt, extreme controller
        # output). Terminate early rather than feeding NaN/inf to the policy.
        if not np.all(np.isfinite(boat_pos)):
            print("[Termination] State diverged (non-finite position)")
            return True, False, {"reason": "diverged"}

        state = self.get_state()
        if not np.all(np.isfinite(state["nu"])):
            print("[Termination] State diverged (non-finite velocity)")
            return True, False, {"reason": "diverged"}

        # Also catch states that are finite but absurdly large — the vessel
        # has left the grid and the simulation is meaningless.
        max_pos = max(self.grid_width, self.grid_height) * 4
        if np.abs(boat_pos[0]) > max_pos or np.abs(boat_pos[1]) > max_pos:
            print(f"[Termination] Vessel far outside grid: pos=({boat_pos[0]:.1f}, {boat_pos[1]:.1f})")
            return True, False, {"reason": "out_of_bounds"}

        if self.encounter_max_time and self.curr_sim_time > self.encounter_max_time:
            return False, True, {"reason": "time_limit"}

        if self.encounter_vessel_eta is not None:
            dist = np.hypot(
                boat_pos[0] - self.encounter_vessel_eta[0],
                boat_pos[1] - self.encounter_vessel_eta[1],
            )
            if dist < self.encounter_radius:
                return True, False, {"reason": "collision"}

        return False, False, {}
