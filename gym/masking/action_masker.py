import time
from collections import deque

import numpy as np
from pacstl.common.interfaces import PACReachableSet, TimeStampedState

from gym.utils.discrete_actions import decode_discrete_action
from gym.utils.geometry import to_obstacle_frame, within_monitoring_radius, wrap_angle
from gym.utils.robustness import extract_robustness_upper


class ActionMasker:
    def __init__(self, action, vessel_model=None, config=None):
        self.heading_offsets = np.asarray(action.heading_offsets, dtype=float)
        self.speed_multipliers = np.asarray(action.speed_multipliers, dtype=float)
        self.n_actions = len(self.heading_offsets) * len(self.speed_multipliers)
        self.v_max = np.inf
        self.r_max = np.inf
        self.sim_dt = 0.5
        self.k_heading = 1.0
        self.k_speed = 0.5
        self.a_max = 0.1
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

        v_max = config.get("v_max", getattr(vessel_model, "v_max", np.inf))
        r_max = config.get("r_max", getattr(vessel_model, "yaw_dot_max", np.inf))

        self.v_max = np.inf if v_max is None else float(v_max)
        self.r_max = np.inf if r_max is None else float(r_max)
        self.sim_dt = float(config.get("sim_dt", self.sim_dt))
        self.k_heading = float(config.get("k_heading", self.k_heading))
        self.k_speed = float(config.get("k_speed", self.k_speed))
        self.a_max = float(config.get("a_max", self.a_max))
        self.decision_depth = int(config.get("decision_depth", self.decision_depth))
        # Separate from the detection margin: threshold for certifying an action
        # as safe.  0.0 means "spec is satisfied", the spec's own physical
        # parameters (r_ego, t_h) already encode the safety margin.
        self.certification_margin = float(config.get("certification_margin", 0.0))
        # Subsample the reachable tube: use every tube_step_factor-th time step.
        # Larger values -> coarser temporal resolution but longer lookahead per
        # BFS depth, which often lets the maneuver take effect sooner.
        self.tube_step_factor = int(config.get("tube_step_factor", 1))
        # At depth > 0, fix the speed multiplier to the one chosen at depth 0
        # and only vary heading. Reduces branching factor by len(speed_multipliers).
        self.fix_speed_at_depth = bool(config.get("fix_speed_at_depth", False))
        # Heuristic pruning: skip expanding a node's children when its upper
        # robustness bound is below this threshold (default -inf = no pruning).
        self.pruning_threshold = float(config.get("pruning_threshold", -np.inf))

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
        # Uses _search_tube_steps (which may be a subsampled view of tube_time_steps)
        # so that T_end reflects the actual simulation time at this BFS depth.
        search_steps = getattr(self, "_search_tube_steps", self.tube_time_steps)
        if search_steps:
            T_end = search_steps[depth_idx] - search_steps[0]
            if isinstance(T_end, float) and T_end.is_integer():
                T_end = int(T_end)
        else:
            T_end = depth_idx

        if T_end not in self._spec_cache:
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

        monitoring_state = ego_state if state is None else state
        if not within_monitoring_radius(
            encounter_vessel_eta, monitoring_radius, monitoring_state
        ):
            return mask, is_fallback

        # Build the subsampled time step list for this search.  Stored as an
        # instance variable so _simulate_depth_step and _get_spec can read it
        # without needing a changed signature.
        self._search_tube_steps = self.tube_time_steps[:: self.tube_step_factor]
        effective_depth = min(self.decision_depth, len(self._search_tube_steps))
        if effective_depth == 0:
            return mask, is_fallback

        mask = np.zeros(self.n_actions, dtype=bool)
        action_robustness = np.full(self.n_actions, np.inf)
        candidates = self._candidate_actions(situation)
        _t0 = time.perf_counter()

        for first_action in candidates:
            best_rob = -np.inf  # track best (highest) lower bound seen
            found_safe = False
            # Speed index of the first action — used to fix speed at depth > 0.
            first_s_idx = first_action % len(self.speed_multipliers)

            queue = deque(
                [
                    {
                        "state": ego_state,
                        "trajectory": {},
                        "depth": 0,
                        "pending_action": first_action,
                    }
                ]
            )

            while queue and not found_safe:
                node = queue.popleft()  # BFS: shallowest nodes first

                psi_d, u_d = decode_discrete_action(
                    action_idx=node["pending_action"],
                    sim_state=node["state"],
                    heading_offsets=self.heading_offsets,
                    speed_multipliers=self.speed_multipliers,
                )
                next_state, rollout = self._simulate_depth_step(
                    state=node["state"],
                    psi_d=psi_d,
                    u_d=u_d,
                    depth_idx=node["depth"],
                    encounter_vessel_eta=encounter_vessel_eta,
                    encounter_speed=encounter_speed,
                )
                trajectory = node["trajectory"] | rollout

                spec = self._get_spec(node["depth"])
                partial_tube = {t: self.reachable_tube[t] for t in trajectory}
                robustness = spec.evaluate(partial_tube, trajectory)
                # maneuver_verified: positive robustness = spec satisfied = safe.
                # Use upper bound (best-case obstacle realisation) to certify
                # possible safety; use certification_margin (default 0.0, not
                # the detection robustness_margin) as the threshold.
                rob_upper = extract_robustness_upper(robustness)

                if rob_upper is not None:
                    best_rob = max(best_rob, rob_upper)
                    if rob_upper > self.certification_margin:
                        found_safe = True
                        break
                    # Heuristic pruning: if this node's upper bound is so far
                    # below the certification margin that recovery is unlikely,
                    # skip expanding its children.  pruning_threshold defaults
                    # to -inf (disabled).
                    if (
                        np.isfinite(self.pruning_threshold)
                        and rob_upper < self.pruning_threshold
                    ):
                        continue

                next_depth = node["depth"] + 1
                if next_depth < effective_depth:
                    # At depth > 0, optionally fix speed to the first action's
                    # speed and only vary heading.  This cuts the branching
                    # factor by len(speed_multipliers) for every inner level.
                    if self.fix_speed_at_depth and next_depth > 0:
                        cont_candidates = [
                            a
                            for a in candidates
                            if a % len(self.speed_multipliers) == first_s_idx
                        ]
                    else:
                        cont_candidates = candidates
                    for cont_action in cont_candidates:
                        queue.append(
                            {
                                "state": next_state,
                                "trajectory": trajectory,
                                "depth": next_depth,
                                "pending_action": cont_action,
                            }
                        )

            if found_safe:
                mask[first_action] = True

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
                # Fallback: only allow starboard turns (positive heading offset)
                # within 0.1 of the best robustness.  This prevents the agent
                # from choosing port or straight-ahead actions in a crossing
                # situation where no certified safe action exists.
                for idx, rob in candidate_rob.items():
                    h_idx = idx // len(self.speed_multipliers)
                    if rob >= max_rob - 0.1 and self.heading_offsets[h_idx] > 0:
                        mask[idx] = True

                if not mask.any():
                    # All starboard candidates fell outside the robustness band
                    # (e.g. the best action was straight-ahead).  Open to all
                    # starboard turns regardless of robustness ranking.
                    for idx in candidate_rob:
                        h_idx = idx // len(self.speed_multipliers)
                        if self.heading_offsets[h_idx] > 0:
                            mask[idx] = True

                print(
                    f"[ActionMask] Fallback after "
                    f"{'cached-tube' if self._using_cached_tube else 'local-tube'} "
                    f"verification: {mask.sum()} actions allowed "
                    f"(best robustness={max_rob:.2f})"
                )
            else:
                for action_idx in candidates:
                    h_idx = action_idx // len(self.speed_multipliers)
                    if self.heading_offsets[h_idx] > 0:
                        mask[action_idx] = True
                print("[ActionMask] Emergency fallback: starboard turns only")

        return mask, is_fallback

    def _candidate_actions(self, situation):
        candidates = []
        for action_idx in range(self.n_actions):
            h_idx = action_idx // len(self.speed_multipliers)
            if situation == "crossing" and self.heading_offsets[h_idx] < 0:
                continue
            candidates.append(action_idx)
        return candidates

    def _simulate_depth_step(
        self,
        state,
        psi_d,
        u_d,
        depth_idx,
        encounter_vessel_eta,
        encounter_speed,
    ):
        """Simulate one tube-time-step of trajectory from the given state.

        Returns (next_state, {target_t: TimeStampedState}) where the TimeStampedState
        is in obstacle-relative frame. encounter_vessel_eta is the current (t=0)
        obstacle position; obstacle is linearly extrapolated to target_t.
        """
        eta, nu = state["eta"], state["nu"]
        px, py, psi = eta[0], eta[1], float(eta[-1])
        u = float(nu[0])

        obs_n, obs_e, obs_psi = encounter_vessel_eta
        obs_vn = encounter_speed * np.cos(obs_psi)
        obs_ve = encounter_speed * np.sin(obs_psi)

        search_steps = getattr(self, "_search_tube_steps", self.tube_time_steps)
        start_t = 0.0 if depth_idx == 0 else float(search_steps[depth_idx - 1])
        target_t = float(search_steps[depth_idx])
        t = start_t

        while t < target_t - 1e-9:
            dt_step = min(self.sim_dt, target_t - t)

            heading_error = wrap_angle(psi_d - psi)
            omega = np.clip(self.k_heading * heading_error, -self.r_max, self.r_max)

            speed_error = u_d - u
            a = np.clip(self.k_speed * speed_error, -self.a_max, self.a_max)

            px += np.cos(psi) * u * dt_step
            py += np.sin(psi) * u * dt_step
            psi += omega * dt_step
            u += a * dt_step
            u = np.clip(u, 0.0, self.v_max)
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
        }
        return next_state, rollout
