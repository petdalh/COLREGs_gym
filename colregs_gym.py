"""
Minimal COLREGs encounter gym for RL training with pacSTL monitoring.

Wraps the numpy_core McGym with encounter vessel kinematics,
collision detection, and pacSTL robustness evaluation.
"""

import numpy as np
from mchorcrux.numpy_core.gym.mc_gym_csad_numpy import McGym
from pacstl.common.interfaces import PACReachableSet, TimeStampedState
from pacstl.domains.colregs.utils import VesselModel
from pacstl.core.evaluator import PacSTLEvaluator
from mchorcrux.numpy_core.controllers.adaptive_seakeeping import heading_to_goal, MRACShipController


# Discrete action space
HEADING_OFFSETS = np.deg2rad([-45, -30, -15, 0, 15, 30, 45])
SPEED_MULTIPLIERS = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
N_DISCRETE_ACTIONS = len(HEADING_OFFSETS) * len(SPEED_MULTIPLIERS)

from gymnasium import spaces
from mchorcrux.numpy_core.controllers.adaptive_seakeeping import MRACShipController

class ColregsGym(McGym):
    def __init__(self, vessel_model, dt, grid_width, grid_height,
                 maneuver_horizon=10.0, sim_dt=0.5, **kwargs):
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

        # Controller lives inside the environment now
        self._controller = None

        # Encounter detection state for action masking
        self._encounter_active = False
        self._active_maneuver_spec = None

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def configure_monitoring(self, spec, ellipsoids_Ab_dict, sampling_rate: int = 10):
        """Attach a pacSTL evaluator and preloaded reachable sets."""
        self.spec = spec
        self.ellipsoids_Ab_dict = ellipsoids_Ab_dict
        self.robustness_sampling_rate = sampling_rate

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
        return self._obs(), {}

    def step(self, action: int):
        if self.encounter_vessel_eta is not None:
            self._propagate_encounter_vessel()
        
        if self._step_count % 100 == 0:
            print(f"Step {self._step_count} completed...")

        # Decode discrete action to heading/speed reference
        obs = self._obs()
        psi_d, u_d = self.decode_discrete_actions(action, obs)

        # Compute continuous torque internally
        tau = self._controller.compute_action_minimal(self.get_state(), psi_d, u_d)

        # Step the parent McGym with the continuous torque
        _, reward, terminated, truncated, info = super().step(tau)
        self._step_count += 1

        # pacSTL evaluation at configured rate
        if self._step_count % self.robustness_sampling_rate == 0:
            robustness = self.evaluate_robustness()
            info["robustness"] = robustness
            self._update_encounter_state(robustness)

        return self._obs(), float(reward), terminated, truncated, info

    def compute_reward(self, action, prev_action):
        state = self.get_state()
        gn, ge = self.goal[:2]
        dist = np.hypot(gn - state["eta"][0], ge - state["eta"][1])
        progress = self.prev_dist_to_goal - dist
        self.prev_dist_to_goal = dist
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
        return np.concatenate([eta[:2], [eta[-1]], nu, [gn, ge], tgt]).astype(np.float32)

    # ------------------------------------------------------------------
    # pacSTL monitoring
    # ------------------------------------------------------------------

    def evaluate_robustness(self):
        """Evaluate the pacSTL specification over the prediction horizon."""
        if self.spec is None or self.ellipsoids_Ab_dict is None:
            return None
        if self.encounter_vessel_eta is None:
            return None

        state = self.get_state()
        eta, nu = state["eta"], state["nu"]
        psi_ego = eta[-1]
        u, v_sway = nu[0], nu[1]

        # World-frame velocities of ego
        ego_vn = u * np.cos(psi_ego) - v_sway * np.sin(psi_ego)
        ego_ve = u * np.sin(psi_ego) + v_sway * np.cos(psi_ego)

        # Obstacle current state
        obs_n, obs_e, obs_psi = self.encounter_vessel_eta
        obs_vn = self.encounter_speed * np.cos(obs_psi)
        obs_ve = self.encounter_speed * np.sin(obs_psi)

        ego_trajectory = {}
        reachable_tube = {}

        for time_step, raw_tuple in self.ellipsoids_Ab_dict.items():
            A, b, c = raw_tuple

            # Predict ego position in world frame at this time step
            ego_n = eta[0] + ego_vn * time_step
            ego_e = eta[1] + ego_ve * time_step

            # Predict obstacle position in world frame at this time step
            pred_obs_n = obs_n + obs_vn * time_step
            pred_obs_e = obs_e + obs_ve * time_step

            # Transform ego into obstacle's local frame 
            # Translate (obstacle at origin)
            dn = ego_n - pred_obs_n
            de = ego_e - pred_obs_e

            angle = obs_psi + np.pi
            cos_o = np.cos(angle)
            sin_o = np.sin(angle)
            local_x = cos_o * dn - sin_o * de
            local_y = sin_o * dn + cos_o * de

            # Relative heading (same as original)
            psi_rel = - _wrap_angle(psi_ego - obs_psi)

            # Transform velocities with the same rotation
            dvn = ego_vn - obs_vn
            dve = ego_ve - obs_ve
            local_vx = cos_o * dvn - sin_o * dve
            local_vy = sin_o * dvn + cos_o * dve
            speed = np.sqrt(local_vx**2 + local_vy**2)

            # 6D state in obstacle-relative frame:
            # [p_x, p_y, psi_rel, v_x, v_y, |v|]
            state_array = np.array([
                local_x, local_y, psi_rel,
                local_vx, local_vy, speed
            ])

            ego_trajectory[time_step] = TimeStampedState(
                time_step=time_step, state_array=state_array
            )
            reachable_tube[time_step] = PACReachableSet(
                time_step=time_step, A_matrix=A, b_vector=b, center=c
            )

        return self.spec.evaluate(reachable_tube, ego_trajectory)

    # ------------------------------------------------------------------
    # Action masking and decoding
    # ------------------------------------------------------------------

    def get_action_mask(
        self,
        situation: str,
        maneuver_spec: PacSTLEvaluator,
    ) -> np.ndarray:
        print("Computing action mask based on pacSTL robustness...")
        if self.ellipsoids_Ab_dict is None:
            return np.ones(N_DISCRETE_ACTIONS, dtype=bool)

        obs = self._obs()
        mask = np.zeros(N_DISCRETE_ACTIONS, dtype=bool)

        reachable_tube = {}
        for time_step, raw_tuple in self.ellipsoids_Ab_dict.items():
            A, b, c = raw_tuple
            reachable_tube[time_step] = PACReachableSet(
                time_step=time_step, A_matrix=A, b_vector=b, center=c
            )

        for action_idx in range(N_DISCRETE_ACTIONS):
            h_idx = action_idx // len(SPEED_MULTIPLIERS)
            heading_offset = HEADING_OFFSETS[h_idx]

            if situation == "crossing":
                if heading_offset < 0:
                    mask[action_idx] = False
                    continue

            psi_d, u_d = self.decode_discrete_actions(action_idx, obs)
            ego_trajectory = self._simulate_candidate_trajectory(psi_d, u_d)

            robustness = maneuver_spec.evaluate(reachable_tube, ego_trajectory)

            if hasattr(robustness, 'u'):
                encounter_resolved = robustness.u < 0
            elif hasattr(robustness, '__getitem__'):
                last_rob = robustness[-1][1] if robustness else None
                encounter_resolved = (
                    last_rob is not None
                    and hasattr(last_rob, 'u')
                    and last_rob.u < 0
                )
            else:
                encounter_resolved = robustness < 0 if robustness is not None else False

            mask[action_idx] = encounter_resolved

        if not mask.any():
            for action_idx in range(N_DISCRETE_ACTIONS):
                h_idx = action_idx // len(SPEED_MULTIPLIERS)
                if HEADING_OFFSETS[h_idx] > 0:
                    mask[action_idx] = True

        return mask
    
    def _simulate_candidate_trajectory(self, psi_d: float, u_d: float) -> dict:
        state = self.get_state()
        eta, nu = state["eta"], state["nu"]
        px, py = eta[0], eta[1]
        psi = eta[-1]
        u = nu[0]

        k_heading = 1.0
        k_speed = 0.5
        omega_max = self.r_max
        a_max = 0.1

        tube_time_steps = sorted(self.ellipsoids_Ab_dict.keys())
        
        ego_trajectory = {}
        t = 0.0
        tube_idx = 0

        while tube_idx < len(tube_time_steps):
            target_t = tube_time_steps[tube_idx]
            while t < target_t - 1e-9:
                dt_step = min(self.sim_dt, target_t - t)
                heading_error = _wrap_angle(psi_d - psi)
                omega = np.clip(k_heading * heading_error, -omega_max, omega_max)
                speed_error = u_d - u
                a = np.clip(k_speed * speed_error, -a_max, a_max)

                px += np.cos(psi) * u * dt_step
                py += np.sin(psi) * u * dt_step
                psi += omega * dt_step
                u += a * dt_step
                u = np.clip(u, 0.0, self.v_max)
                t += dt_step
        
            # World-frame velocities at this predicted state
            vn = u * np.cos(psi)
            ve = u * np.sin(psi)
 
            # Build the 6D state vector matching the evaluator's expectation:
            # [px, py, psi, vn, ve, 0.0]
            state_array = np.array([px, py, psi, vn, ve, 0.0])
 
            ego_trajectory[target_t] = TimeStampedState(
                time_step=target_t, state_array=state_array
            )
            tube_idx += 1
 
        return ego_trajectory

    def _update_encounter_state(self, robustness):
        """Update encounter detection based on pacSTL robustness output."""
        if robustness is None:
            return

        trace_list = robustness[0]
        for _, rob_interval in trace_list:
            if rob_interval.u > 0:
                self._encounter_active = True
                if self._active_maneuver_spec is None:
                    from pacstl.core.factory import create as create_spec
                    self._active_maneuver_spec = create_spec("colregs", "crossing_detection")
                return

        self._encounter_active = False
        self._active_maneuver_spec = None


    @staticmethod
    def decode_discrete_actions(action_idx: int, obs: np.ndarray) -> tuple:
        h_idx = action_idx // len(SPEED_MULTIPLIERS)
        s_idx = action_idx % len(SPEED_MULTIPLIERS)

        n, e, psi = obs[0], obs[1], obs[2]
        gn, ge = obs[6], obs[7]
        psi_goal = heading_to_goal(n, e, gn, ge)

        psi_d = psi_goal + HEADING_OFFSETS[h_idx]
        u_d = SPEED_MULTIPLIERS[s_idx]
        return psi_d, u_d

    def action_masks(self) -> np.ndarray:
        """Called by MaskablePPO to get the current valid action mask."""
        if self._encounter_active and self._active_maneuver_spec is not None:
            return self.get_action_mask(self.encounter_type, self._active_maneuver_spec)
        return np.ones(N_DISCRETE_ACTIONS, dtype=bool)

    # ------------------------------------------------------------------
    # Encounter vessel kinematics
    # ------------------------------------------------------------------

    def _propagate_encounter_vessel(self):
        n, e, psi = self.encounter_vessel_eta
        n += self.encounter_speed * np.cos(psi) * self.dt
        e += self.encounter_speed * np.sin(psi) * self.dt
        self.encounter_vessel_eta = np.array([n, e, psi])

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------

    def _check_termination(self, boat_pos):
        terminated, truncated, info = super()._check_termination(boat_pos)
        if terminated or truncated:
            return terminated, truncated, info

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


def _wrap_angle(angle: float) -> float:
    """Wrap angle to [-π, π]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi