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
from pacstl.core.factory import create as create_spec


class DebugActionMasker:
    def __init__(self, action, vessel_model=None, ego_vessel_model=None, config=None):
        self.vessel_model = vessel_model
        self.ego_vessel_model = ego_vessel_model
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
        self._shoebox = None
        self._use_3dof = False
        self._shoebox_sub_dt = 0.1
        self.configure(config=config, vessel_model=vessel_model)

        self.spec = None
        self._spec_factory = None
        self._spec_cache = {}
        self._debug_spec_cache = {}
        self.ellipsoids_Ab_dict = None
        self.tube_time_steps = []
        self.reachable_tube = {}
        self._using_cached_tube = False
        self.last_search_tree = None
        self._debug_node_counter = 0

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
        self.fallback_robustness_tolerance = float(
            config.get("fallback_robustness_tolerance", 0.5)
        )
        # Heuristic pruning: skip expanding a node's children when its upper
        # robustness bound is below this threshold (default -inf = no pruning).
        self.pruning_threshold = float(config.get("pruning_threshold", -np.inf))
        self._use_3dof = bool(config.get("use_3dof_dynamics", False))
        self._shoebox_sub_dt = float(config.get("shoebox_sub_dt", getattr(self, "_shoebox_sub_dt", 0.1)))
        if self._use_3dof:
            self._init_shoebox(config)

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
        self._debug_spec_cache.clear()
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

    @staticmethod
    def _normalize_spec_time(value):
        value = float(value)
        if value.is_integer():
            return int(value)
        return value

    def _get_spec_window(self, depth_idx: int):
        return max(0, depth_idx - 1), depth_idx

    def _get_spec(self, depth_idx: int):
        """Return (and cache) the spec appropriate for the trajectory depth.

        When a spec_factory is set, it is called with discrete trace-index
        bounds for this BFS depth.  With sparse tube samples [5, 10, 15],
        the evaluated trajectories have 1, 2, and 3 samples respectively, so
        the resulting windows are (0, 0), (0, 1), and (1, 2).

        Falls back to self.spec when no factory is configured.
        """
        if self._spec_factory is None:
            return self.spec

        T_start, T_end = self._get_spec_window(depth_idx)
        cache_key = (T_start, T_end)
        if cache_key not in self._spec_cache:
            self._spec_cache[cache_key] = self._spec_factory(
                T_start=T_start,
                T_end=T_end,
            )
        return self._spec_cache[cache_key]

    def _init_debug_tree(self):
        self._debug_node_counter = 0
        self.last_search_tree = {
            "roots": [],
            "nodes": {},
            "safe_actions": [],
            "fallback": False,
        }

    def _new_debug_node(self, parent_id, depth, first_action, pending_action):
        node_id = self._debug_node_counter
        self._debug_node_counter += 1
        yaw_rate_cmd, surge_accel_cmd = self._decode_action(pending_action)
        T_start, T_end = self._get_spec_window(depth)
        record = {
            "node_id": node_id,
            "parent_id": parent_id,
            "children": [],
            "depth": depth,
            "first_action": int(first_action),
            "pending_action": int(pending_action),
            "yaw_rate_cmd": float(yaw_rate_cmd),
            "surge_accel_cmd": float(surge_accel_cmd),
            "T_start": T_start,
            "T_end": T_end,
            "target_t": None,
            "trajectory": None,
            "partial_tube_times": [],
            "next_state": None,
            "maneuver_verified": None,
            "sub_specs": {},
            "pruned": False,
            "certified_safe": False,
            "evaluated": False,
            "rejected_reason": None,
        }
        self.last_search_tree["nodes"][node_id] = record
        if parent_id is None:
            self.last_search_tree["roots"].append(node_id)
        else:
            self.last_search_tree["nodes"][parent_id]["children"].append(node_id)
        return node_id

    def _robustness_record(self, robustness):
        return {
            "raw": robustness,
            "lower": extract_robustness_lower(robustness),
            "upper": extract_robustness_upper(robustness),
        }

    def _debug_sub_spec_window(self, name, T_start, T_end):
        if name == "collision_possible":
            T_start = self._normalize_spec_time(max(0.0, float(T_start) - 1.0))
        return T_start, T_end

    def _get_debug_spec(self, name, T_start, T_end):
        T_start, T_end = self._debug_sub_spec_window(name, T_start, T_end)
        cache_key = (name, T_start, T_end)
        if cache_key not in self._debug_spec_cache:
            kwargs = {
                "T_start": T_start,
                "T_end": T_end,
            }
            if self.vessel_model is not None:
                kwargs["vessel"] = self.vessel_model
            if self.ego_vessel_model is not None:
                kwargs["ego_vessel"] = self.ego_vessel_model
            self._debug_spec_cache[cache_key] = create_spec("colregs", name, **kwargs)
        return self._debug_spec_cache[cache_key]

    def _evaluate_debug_sub_specs(self, T_start, T_end, partial_tube, trajectory):
        results = {}
        for name in ("occupancy_clear", "collision_possible"):
            sub_T_start, sub_T_end = self._debug_sub_spec_window(
                name,
                T_start,
                T_end,
            )
            try:
                spec = self._get_debug_spec(name, T_start, T_end)
                robustness = spec.evaluate(partial_tube, trajectory)
                results[name] = {
                    "T_start": sub_T_start,
                    "T_end": sub_T_end,
                    **self._robustness_record(robustness),
                }
            except Exception as exc:
                results[name] = {
                    "T_start": sub_T_start,
                    "T_end": sub_T_end,
                    "raw": None,
                    "lower": None,
                    "upper": None,
                    "error": repr(exc),
                }
        return results

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
        self._init_debug_tree()

        if (
            self.spec is None and self._spec_factory is None
        ) or self.ellipsoids_Ab_dict is None:
            self.last_search_tree["skip_reason"] = "missing_spec_or_tube"
            return mask, is_fallback

        if not within_monitoring_radius(
            encounter_vessel_eta, monitoring_radius, ego_state
        ):
            self.last_search_tree["skip_reason"] = "outside_monitoring_radius"
            return mask, is_fallback

        # Build the action-hold-aligned time step list for this search.  Stored as an
        # instance variable so _simulate_depth_step and _get_spec can read it
        # without needing a changed signature.
        self._search_tube_steps = self._aligned_tube_steps()
        effective_depth = min(self.decision_depth, len(self._search_tube_steps))
        if effective_depth == 0:
            self.last_search_tree["skip_reason"] = "no_aligned_tube_steps"
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
        first_action_candidates = [
            action_idx
            for action_idx in candidates
            if not self._violates_speed_floor(
                initial_cmd_u,
                self._decode_action(action_idx)[1],
                self.action_hold_dt,
            )
            and self._respects_crossing_side(
                situation,
                initial_psi_d_offset,
                self._decode_action(action_idx)[0],
                self.action_hold_dt,
            )
        ]

        for first_action in first_action_candidates:
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

            root_node_id = self._new_debug_node(
                parent_id=None,
                depth=0,
                first_action=first_action,
                pending_action=first_action,
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
                        "debug_node_id": root_node_id,
                    }
                ]
            )


            while queue and not found_safe:
                node = queue.popleft()  # BFS: shallowest nodes first
                debug_record = self.last_search_tree["nodes"][node["debug_node_id"]]

                yaw_rate_cmd, surge_accel_cmd = self._decode_action(
                    node["pending_action"]
                )
                if not self._respects_crossing_side(
                    situation,
                    node["psi_d_offset"],
                    yaw_rate_cmd,
                    self.action_hold_dt,
                ):
                    debug_record["rejected_reason"] = "crossing_side"
                    continue
                if self._violates_speed_floor(
                    node["cmd_u"], surge_accel_cmd, self.action_hold_dt
                ):
                    debug_record["rejected_reason"] = "speed_floor"
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
                # Use lower bound (worst-case obstacle realisation) to certify
                # robust safety; certification_margin is independent of the
                # detection robustness_margin.
                rob_upper = extract_robustness_upper(robustness)
                rob_lower = extract_robustness_lower(robustness)
                T_start, T_end = self._get_spec_window(node["depth"])
                debug_record.update(
                    {
                        "target_t": target_t,
                        "trajectory": trajectory,
                        "partial_tube_times": list(partial_tube.keys()),
                        "next_state": next_state,
                        "maneuver_verified": self._robustness_record(robustness),
                        "sub_specs": self._evaluate_debug_sub_specs(
                            T_start,
                            T_end,
                            partial_tube,
                            trajectory,
                        ),
                        "evaluated": True,
                    }
                )

                if rob_lower is not None:
                    best_rob = max(best_rob, rob_lower)
                    if rob_lower > self.certification_margin:
                        found_safe = True
                        debug_record["certified_safe"] = True
                        break
                    # Heuristic pruning: if this node's upper bound is so far
                    # below the certification margin that recovery is unlikely,
                    # skip expanding its children.  pruning_threshold defaults
                    # to -inf (disabled); effective_pruning may be tightened
                    # dynamically once a safe action has already been found.
                    if (
                        rob_upper is not None
                        and np.isfinite(effective_pruning)
                        and rob_upper < effective_pruning
                    ):
                        debug_record["pruned"] = True
                        debug_record["rejected_reason"] = "pruning_threshold"
                        continue

                next_depth = node["depth"] + 1
                if next_depth < effective_depth:
                    for cont_action in cont_candidates:
                        child_node_id = self._new_debug_node(
                            parent_id=node["debug_node_id"],
                            depth=next_depth,
                            first_action=first_action,
                            pending_action=cont_action,
                        )
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
                                "debug_node_id": child_node_id,
                            }
                        )

            if found_safe:
                mask[first_action] = True
                any_safe_found = True
                self.last_search_tree["safe_actions"].append(int(first_action))

            action_robustness[first_action] = best_rob

        print(
            f"[ActionMask] search took {time.perf_counter() - _t0:.4f}s "
            f"({mask.sum()}/{len(first_action_candidates)} actions safe)"
        )

        if not mask.any():
            is_fallback = True
            self.last_search_tree["fallback"] = True
            candidate_rob = {
                idx: action_robustness[idx]
                for idx in first_action_candidates
                if np.isfinite(action_robustness[idx])
            }

            if candidate_rob:
                max_rob = max(candidate_rob.values())
                fallback_candidates = []
                for idx, rob in candidate_rob.items():
                    yaw_rate_cmd, _ = self._decode_action(idx)
                    if self._respects_crossing_side(
                        situation,
                        initial_psi_d_offset,
                        yaw_rate_cmd,
                        self.action_hold_dt,
                    ):
                        fallback_candidates.append((float(rob), float(yaw_rate_cmd), idx))

                if fallback_candidates:
                    if max_rob > 0.0:
                        near_best = [
                            candidate
                            for candidate in fallback_candidates
                            if candidate[0] >= max_rob - self.fallback_robustness_tolerance
                        ]
                        _, _, fallback_idx = max(
                            near_best,
                            key=lambda candidate: (candidate[1], candidate[0]),
                        )
                    else:
                        _, _, fallback_idx = max(
                            fallback_candidates,
                            key=lambda candidate: (candidate[1], candidate[0]),
                        )
                    mask[fallback_idx] = True

                print(
                    f"[ActionMask] Fallback after "
                    f"{'cached-tube' if self._using_cached_tube else 'local-tube'} "
                    f"verification: {mask.sum()} actions allowed "
                    f"(best robustness={max_rob:.2f})"
                )
            else:
                fallback_idx = None
                fallback_key = None
                for action_idx in first_action_candidates:
                    yaw_rate_cmd, _ = self._decode_action(action_idx)
                    if self._respects_crossing_side(
                        situation,
                        initial_psi_d_offset,
                        yaw_rate_cmd,
                        self.action_hold_dt,
                    ):
                        key = (float(yaw_rate_cmd), int(action_idx))
                        if fallback_key is None or key > fallback_key:
                            fallback_key = key
                            fallback_idx = action_idx
                if fallback_idx is not None:
                    mask[fallback_idx] = True
                print("[ActionMask] Emergency fallback: strongest non-port command only")
        
        return mask, is_fallback

    def _candidate_actions(self, situation):
        return list(range(self.n_actions))

    @staticmethod
    def _respects_crossing_side(
        situation,
        psi_d_offset,
        yaw_rate_cmd,
        dt,
        tolerance=1e-9,
    ):
        if situation != "crossing":
            return True
        next_offset = float(psi_d_offset) + float(yaw_rate_cmd) * float(dt)
        return next_offset >= -float(tolerance)

    def _init_shoebox(self, config):
        try:
            from shoeboxpy.model3dof import Shoebox
        except ImportError:
            raise RuntimeError(
                "shoeboxpy is not installed; set use_3dof_dynamics: false in config"
            )
        self._shoebox = Shoebox(
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

    def _force_from_action(self, nu, surge_accel_cmd, r_cmd, sub_dt):
        """Inverse dynamics: forces/moments to track commanded surge accel and yaw rate.

        Cancels Coriolis and damping; uses a deadbeat law for yaw rate so that
        r converges to r_cmd within one sub-step.
        """
        u, v, r = nu
        sb = self._shoebox
        cnu0 = -(sb.m + sb.MA[1, 1]) * v * r
        cnu1 =  (sb.m + sb.MA[0, 0]) * u * r
        Dnu0 = sb.D[0, 0] * u
        Dnu1 = sb.D[1, 1] * v
        Dnu2 = sb.D[2, 2] * r
        r_dot_des = (r_cmd - r) / sub_dt
        tau_X = sb.M_eff[0, 0] * surge_accel_cmd + Dnu0 + cnu0
        tau_Y = sb.M_eff[1, 1] * 0.0 + Dnu1 + cnu1
        tau_N = sb.M_eff[2, 2] * r_dot_des + Dnu2
        return np.array([tau_X, tau_Y, tau_N])

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

        if self._use_3dof and self._shoebox is not None:
            # 3DOF shoebox branch: full hydrodynamic forward simulation.
            nu_state = state.get("nu", np.zeros(3))
            sb = self._shoebox
            sb.eta[0], sb.eta[1], sb.eta[2] = px, py, psi
            sb.nu[0] = float(cmd_u)
            sb.nu[1] = float(nu_state[1]) if len(nu_state) > 1 else 0.0
            sb.nu[2] = float(nu_state[2]) if len(nu_state) > 2 else 0.0

            while t < target_t - 1e-9:
                dt_step = min(self.sim_dt, target_t - t)
                psi_d_offset += yaw_rate_cmd * dt_step
                u_now = sb.nu[0]
                u_next_clamped = float(
                    np.clip(
                        u_now + surge_accel_cmd * dt_step,
                        self._min_surge_for_rollout(),
                        self.v_max,
                    )
                )
                u_dot_eff = (u_next_clamped - u_now) / dt_step if dt_step > 0 else 0.0
                n_sub = max(1, round(dt_step / self._shoebox_sub_dt))
                sub_dt_inner = dt_step / n_sub
                for _ in range(n_sub):
                    tau = self._force_from_action(sb.nu, u_dot_eff, yaw_rate_cmd, sub_dt_inner)
                    sb.step(tau=tau, dt=sub_dt_inner)
                    sb.nu[0] = float(
                        np.clip(sb.nu[0], self._min_surge_for_rollout(), self.v_max)
                    )
                t += dt_step

            px, py, psi = float(sb.eta[0]), float(sb.eta[1]), float(sb.eta[2])
            u, v, r = float(sb.nu[0]), float(sb.nu[1]), float(sb.nu[2])
            vn = np.cos(psi) * u - np.sin(psi) * v
            ve = np.sin(psi) * u + np.cos(psi) * v
            nu_out = np.array([u, v, r])
        else:
            # Kinematic branch (original implementation).
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
            nu_out = np.array([u, 0.0, 0.0])

        pred_obs_n = obs_n + obs_vn * target_t
        pred_obs_e = obs_e + obs_ve * target_t

        local_pos, local_psi, local_vel = to_obstacle_frame(
            ego_pos=np.array([px, py]),
            ego_psi=psi,
            ego_vel=np.array([vn, ve]),
            obs_pos=np.array([pred_obs_n, pred_obs_e]),
            obs_psi=obs_psi,
        )

        state_array = np.array(
            [
                local_pos[0],
                local_pos[1],
                local_psi,
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
            "nu": nu_out,
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
