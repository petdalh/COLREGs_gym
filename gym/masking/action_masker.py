import time
from collections import deque

import numpy as np
from pacstl.common.interfaces import PACReachableSet, TimeStampedState

from gym.utils.discrete_actions import decode_discrete_action
from gym.utils.geometry import (
    bearing_to_goal,
    to_obstacle_frame,
    within_monitoring_radius,
    wrap_angle,
)
from gym.utils.robustness import extract_robustness_upper, extract_robustness_lower


class ActionMasker:
    def __init__(self, action, vessel_model=None, ego_vessel_model=None, config=None):
        self.yaw_rate_commands = np.asarray(action.yaw_rate_commands, dtype=float)
        self.yaw_rate_commands_deg_s = np.asarray(
            action.yaw_rate_commands_deg_s, dtype=float
        )
        self.surge_accel_commands = np.asarray(action.surge_accel_commands, dtype=float)
        self.n_actions = len(self.yaw_rate_commands) * len(self.surge_accel_commands)
        self.v_max = ego_vessel_model.v_max
        self.dt = 0.5
        self.sim_dt = 0.5
        self.decision_interval = 1
        self.action_hold_dt = self.dt * self.decision_interval
        self.decision_depth = 5
        self.configure(config=config, vessel_model=vessel_model)

        self.spec = None
        self._spec_factory = None
        self._spec_cache = {}
        self.ellipsoids_Ab_dict = None
        self.tube_time_steps = []
        self.reachable_tube = {}
        self._using_cached_tube = False

    def configure(self, config=None, vessel_model=None):
        config = dict(config or {})

        self.dt = float(config.get("dt", self.dt))
        self.sim_dt = float(config.get("sim_dt", self.sim_dt))
        self.decision_interval = int(
            config.get("decision_interval", self.decision_interval)
        )
        self.action_hold_dt = self.dt * self.decision_interval
        self.decision_depth = int(config.get("decision_depth", self.decision_depth))
        self.min_surge_command_mps = float(
            config.get(
                "min_surge_command_mps",
                getattr(self, "min_surge_command_mps", 0.0),
            )
        )
        self.speed_floor_enabled = bool(
            config.get(
                "speed_floor_enabled",
                self.min_surge_command_mps > 0.0,
            )
        )
        # Separate from the detection margin: threshold for certifying an action
        # as safe.  0.0 means "spec is satisfied", the spec's own physical
        # parameters (r_ego, t_h) already encode the safety margin.
        self.certification_margin = float(config.get("certification_margin", 0.0))
        # Heuristic pruning: skip expanding a node's children when its upper
        # robustness bound is below this threshold (default -inf = no pruning).
        self.pruning_threshold = float(config.get("pruning_threshold", -np.inf))

    def _aligned_tube_steps(self):
        aligned_steps = []
        tube_steps = list(self.tube_time_steps)
        for depth_idx in range(self.decision_depth):
            target_t = (depth_idx + 1) * self.action_hold_dt
            match = next(
                (
                    step
                    for step in tube_steps
                    if np.isclose(float(step), target_t, rtol=0.0, atol=1e-9)
                ),
                None,
            )
            if match is None:
                break
            aligned_steps.append(match)
        return aligned_steps

    def _decode_action(self, action_idx):
        return decode_discrete_action(
            action_idx=action_idx,
            yaw_rate_commands=self.yaw_rate_commands,
            surge_accel_commands=self.surge_accel_commands,
        )

    def update_scenario(
        self, spec, ellipsoids_Ab_dict, encounter_scenario=None, spec_factory=None
    ):
        self.spec = spec
        self._spec_factory = spec_factory
        self._spec_cache.clear()
        self.ellipsoids_Ab_dict = ellipsoids_Ab_dict

        if (
            encounter_scenario is not None
            and encounter_scenario.ellipsoids_Ab_dict is ellipsoids_Ab_dict
            and encounter_scenario.reachable_tube
        ):
            self.tube_time_steps = list(encounter_scenario.tube_time_steps)
            self.reachable_tube = encounter_scenario.reachable_tube
            self._using_cached_tube = True
            print(
                f"[ReachableTube] ActionMasker reusing static cache with "
                f"{len(self.tube_time_steps)} steps"
            )
        elif ellipsoids_Ab_dict:
            self.tube_time_steps = sorted(ellipsoids_Ab_dict.keys())
            self.reachable_tube = {
                time_step: PACReachableSet(
                    time_step=time_step,
                    A_matrix=raw_tuple[0],
                    b_vector=raw_tuple[1],
                    center=raw_tuple[2],
                )
                for time_step, raw_tuple in ellipsoids_Ab_dict.items()
            }
            self._using_cached_tube = False
        else:
            self.tube_time_steps = []
            self.reachable_tube = {}
            self._using_cached_tube = False

    def _get_spec(self, depth_idx: int):
        """Return (and cache) the spec appropriate for the trajectory depth.

        When a spec_factory is set, it is called with T_end equal to the
        real-time offset of the last trajectory step from the first step
        (converted to int when the value is a whole number).  This ensures
        that `eventually[T_end, T_end]` always refers to the final sample in
        the trajectory, regardless of the actual tube time-step values.

        Falls back to self.spec when no factory is configured.
        """
        if self._spec_factory is None:
            return self.spec

        # Compute T_end: real-time distance from the first to the current step.
        # Uses _search_tube_steps (the action-hold-aligned view of tube_time_steps)
        # so that T_end reflects the actual simulation time at this BFS depth.
        search_steps = getattr(self, "_search_tube_steps", self.tube_time_steps)
        if search_steps:
            T_end = search_steps[depth_idx] - search_steps[0]
            if isinstance(T_end, float) and T_end.is_integer():
                T_end = int(T_end)
        else:
            T_end = depth_idx

        if T_end not in self._spec_cache:
            if T_end == 0:
                T_start = 0.0
            else:
                T_start = T_end - 2
                if T_start < 0:
                    T_start = 0.0
            self._spec_cache[T_end] = self._spec_factory(T_end=T_end)
        return self._spec_cache[T_end]

    def get_mask(
        self,
        situation,
        ego_state,
        encounter_vessel_eta,
        encounter_speed,
        robustness_margin,
        monitoring_radius,
        state=None,
    ):
        mask = np.ones(self.n_actions, dtype=bool)
        is_fallback = False

        if (
            self.spec is None and self._spec_factory is None
        ) or self.ellipsoids_Ab_dict is None:
            return mask, is_fallback

        if not within_monitoring_radius(
            encounter_vessel_eta, monitoring_radius, ego_state
        ):
            return mask, is_fallback

        # Build the action-hold-aligned time step list for this search.  Stored as an
        # instance variable so _simulate_depth_step and _get_spec can read it
        # without needing a changed signature.
        self._search_tube_steps = self._aligned_tube_steps()
        effective_depth = min(self.decision_depth, len(self._search_tube_steps))
        if effective_depth == 0:
            return mask, is_fallback

        mask = np.zeros(self.n_actions, dtype=bool)
        action_robustness = np.full(self.n_actions, np.inf)
        # Sort candidates so larger (more starboard) yaw rates come first.
        candidates = sorted(
            self._candidate_actions(situation),
            key=lambda a: -self.yaw_rate_commands[a // len(self.surge_accel_commands)],
        )
        _t0 = time.perf_counter()
        any_safe_found = False  # tracks whether any first_action certified safe so far
        initial_cmd_psi = (
            float(state._current_heading_cmd)
            if state is not None and state._current_heading_cmd is not None
            else float(ego_state["eta"][-1])
        )
        initial_psi_d_offset = self._initial_psi_d_offset(
            ego_state,
            initial_cmd_psi,
        )
        initial_cmd_u = (
            float(state._current_surge_cmd)
            if state is not None and state._current_surge_cmd is not None
            else float(ego_state["nu"][0])
        )
        candidates = [
            action_idx
            for action_idx in candidates
            if not self._violates_speed_floor(
                initial_cmd_u,
                self._decode_action(action_idx)[1],
                self.action_hold_dt,
            )
        ]

        for first_action in candidates:
            best_rob = -np.inf  # track best (highest) lower bound seen
            found_safe = False
            cont_candidates = candidates
            # Once at least one safe action exists, tighten pruning to skip
            # subtrees that cannot realistically reach the certification margin.
            effective_pruning = (
                max(self.pruning_threshold, self.certification_margin - 0.5)
                if any_safe_found
                else self.pruning_threshold
            )

            queue = deque(
                [
                    {
                        "state": ego_state,
                        "trajectory": {},
                        "partial_tube": {},
                        "depth": 0,
                        "pending_action": first_action,
                        "psi_d_offset": initial_psi_d_offset,
                        "cmd_u": initial_cmd_u,
                    }
                ]
            )


            while queue and not found_safe:
                node = queue.popleft()  # BFS: shallowest nodes first

                yaw_rate_cmd, surge_accel_cmd = self._decode_action(
                    node["pending_action"]
                )
                if self._violates_speed_floor(
                    node["cmd_u"], surge_accel_cmd, self.action_hold_dt
                ):
                    continue
                next_state, rollout = self._simulate_depth_step(
                    state=node["state"],
                    psi_d_offset=node["psi_d_offset"],
                    cmd_u=node["cmd_u"],
                    yaw_rate_cmd=yaw_rate_cmd,
                    surge_accel_cmd=surge_accel_cmd,
                    depth_idx=node["depth"],
                    encounter_vessel_eta=encounter_vessel_eta,
                    encounter_speed=encounter_speed,
                )
                trajectory = node["trajectory"] | rollout

                spec = self._get_spec(node["depth"])
                # Extend the parent's already-resolved partial_tube by one key
                # instead of rebuilding it from the full trajectory each time.
                (target_t,) = rollout  # rollout always contains exactly one entry
                partial_tube = node["partial_tube"] | {
                    target_t: self.reachable_tube[target_t]
                }
                robustness = spec.evaluate(partial_tube, trajectory)
                # maneuver_verified: positive robustness = spec satisfied = safe.
                # Use upper bound (best-case obstacle realisation) to certify
                # possible safety; use certification_margin (default 0.0, not
                # the detection robustness_margin) as the threshold.
                rob_upper = extract_robustness_upper(robustness)
                rob_lower = extract_robustness_lower(robustness)

                if rob_lower is not None:
                    best_rob = max(best_rob, rob_lower)
                    if rob_lower > self.certification_margin:
                        found_safe = True
                        break
                    # Heuristic pruning: if this node's upper bound is so far
                    # below the certification margin that recovery is unlikely,
                    # skip expanding its children.  pruning_threshold defaults
                    # to -inf (disabled); effective_pruning may be tightened
                    # dynamically once a safe action has already been found.
                    if (
                        np.isfinite(effective_pruning)
                        and rob_lower < effective_pruning
                    ):
                        continue

                next_depth = node["depth"] + 1
                if next_depth < effective_depth:
                    for cont_action in cont_candidates:
                        queue.append(
                            {
                                "state": next_state,
                                "trajectory": trajectory,
                                "partial_tube": partial_tube,
                                "depth": next_depth,
                                "pending_action": cont_action,
                                "psi_d_offset": float(
                                    next_state["psi_d_offset"]
                                ),
                                "cmd_u": float(next_state["nu"][0]),
                            }
                        )

            if found_safe:
                mask[first_action] = True
                any_safe_found = True

            action_robustness[first_action] = best_rob

        print(
            f"[ActionMask] search took {time.perf_counter() - _t0:.4f}s "
            f"({mask.sum()}/{len(candidates)} actions safe)"
        )

        if not mask.any():
            is_fallback = True
            candidate_rob = {
                idx: action_robustness[idx]
                for idx in candidates
                if np.isfinite(action_robustness[idx])
            }

            if candidate_rob:
                max_rob = max(candidate_rob.values())
                # Fallback: only allow starboard turns (positive offset / rate)
                for idx, rob in candidate_rob.items():
                    yaw_idx = idx // len(self.surge_accel_commands)
                    if rob >= max_rob - 0.1 and self.yaw_rate_commands[yaw_idx] > 0:
                        mask[idx] = True

                if not mask.any():
                    for idx in candidate_rob:
                        yaw_idx = idx // len(self.surge_accel_commands)
                        if self.yaw_rate_commands[yaw_idx] > 0:
                            mask[idx] = True

                print(
                    f"[ActionMask] Fallback after "
                    f"{'cached-tube' if self._using_cached_tube else 'local-tube'} "
                    f"verification: {mask.sum()} actions allowed "
                    f"(best robustness={max_rob:.2f})"
                )
            else:
                for action_idx in candidates:
                    yaw_idx = action_idx // len(self.surge_accel_commands)
                    if self.yaw_rate_commands[yaw_idx] > 0:
                        mask[action_idx] = True
                print("[ActionMask] Emergency fallback: starboard turns only")
        
        return mask, is_fallback

    def _candidate_actions(self, situation):
        candidates = []
        for action_idx in range(self.n_actions):
            yaw_idx = action_idx // len(self.surge_accel_commands)
            if situation == "crossing" and self.yaw_rate_commands[yaw_idx] < 0:
                continue
            candidates.append(action_idx)
        return candidates

    def _simulate_depth_step(
        self,
        state,
        psi_d_offset,
        cmd_u,
        yaw_rate_cmd,
        surge_accel_cmd,
        depth_idx,
        encounter_vessel_eta,
        encounter_speed,
    ):
        """Simulate one tube-time-step of trajectory from the given state.

        Returns (next_state, {target_t: TimeStampedState}) where the TimeStampedState
        is in obstacle-relative frame. encounter_vessel_eta is the current (t=0)
        obstacle position; obstacle is linearly extrapolated to target_t.
        """
        eta = state["eta"]
        px, py = eta[0], eta[1]
        goal = state.get("goal")
        if goal is not None:
            psi = self._goal_relative_heading(px, py, goal, psi_d_offset)
        else:
            psi = float(eta[-1])
        u = float(cmd_u)

        obs_n, obs_e, obs_psi = encounter_vessel_eta
        obs_vn = encounter_speed * np.cos(obs_psi)
        obs_ve = encounter_speed * np.sin(obs_psi)

        search_steps = getattr(self, "_search_tube_steps", self.tube_time_steps)
        start_t = 0.0 if depth_idx == 0 else float(search_steps[depth_idx - 1])
        target_t = float(search_steps[depth_idx])
        t = start_t

        while t < target_t - 1e-9:
            dt_step = min(self.sim_dt, target_t - t)

            if goal is not None:
                psi_start = self._goal_relative_heading(px, py, goal, psi_d_offset)
                psi_d_offset += yaw_rate_cmd * dt_step
            else:
                psi_start = psi
                psi_d_offset += yaw_rate_cmd * dt_step

            u_next = float(
                np.clip(
                    u + surge_accel_cmd * dt_step,
                    self._min_surge_for_rollout(),
                    self.v_max,
                )
            )
            avg_u = 0.5 * (u + u_next)

            if goal is not None:
                # First-order predictor: estimate the end-of-step LOS from
                # the start heading, then use the midpoint heading for motion.
                pred_px = px + np.cos(psi_start) * avg_u * dt_step
                pred_py = py + np.sin(psi_start) * avg_u * dt_step
                psi_next = self._goal_relative_heading(
                    pred_px,
                    pred_py,
                    goal,
                    psi_d_offset,
                )
                avg_psi = psi_start + 0.5 * wrap_angle(psi_next - psi_start)
            else:
                psi_next = psi + yaw_rate_cmd * dt_step
                avg_psi = psi + 0.5 * yaw_rate_cmd * dt_step

            px += np.cos(avg_psi) * avg_u * dt_step
            py += np.sin(avg_psi) * avg_u * dt_step
            psi = psi_next
            u = u_next
            t += dt_step

        vn = u * np.cos(psi)
        ve = u * np.sin(psi)

        pred_obs_n = obs_n + obs_vn * target_t
        pred_obs_e = obs_e + obs_ve * target_t

        local_pos, local_vel = to_obstacle_frame(
            ego_pos=np.array([px, py]),
            ego_vel=np.array([vn, ve]),
            obs_pos=np.array([pred_obs_n, pred_obs_e]),
            obs_psi=obs_psi,
            obs_vel=np.array([obs_vn, obs_ve]),
        )

        state_array = np.array(
            [
                local_pos[0],
                local_pos[1],
                -wrap_angle(psi - obs_psi),
                local_vel[0],
                local_vel[1],
                np.hypot(local_vel[0], local_vel[1]),
            ]
        )

        rollout = {
            target_t: TimeStampedState(time_step=target_t, state_array=state_array)
        }
        next_state = {
            "eta": np.array([px, py, psi]),
            "nu": np.array([u, 0.0, 0.0]),
            "goal": state["goal"],
            "psi_d_offset": psi_d_offset,
        }
        return next_state, rollout

    def _initial_psi_d_offset(self, ego_state, initial_cmd_psi):
        goal = ego_state.get("goal")
        if goal is None:
            return 0.0
        goal_bearing = bearing_to_goal(ego_state["eta"][:2], goal)
        return wrap_angle(initial_cmd_psi - goal_bearing)

    def _goal_relative_heading(self, px, py, goal, psi_d_offset):
        return bearing_to_goal(np.array([px, py]), goal) + float(psi_d_offset)

    def _min_surge_for_rollout(self):
        if not self.speed_floor_enabled:
            return 0.0
        return self.min_surge_command_mps

    def _violates_speed_floor(self, cmd_u, surge_accel_cmd, dt):
        if (
            not self.speed_floor_enabled
            or self.min_surge_command_mps <= 0.0
            or surge_accel_cmd >= 0.0
        ):
            return False
        next_cmd_u = float(cmd_u) + float(surge_accel_cmd) * float(dt)
        return next_cmd_u < self.min_surge_command_mps
