import numpy as np


class ActionConstraints:
    def __init__(self, action, config=None):
        self.action = action
        self.n_actions = action.n_actions
        self.min_surge_command_mps = 0.0
        self.speed_floor_enabled = False
        self.neutral_yaw_without_encounter = False
        self.dt = 0.5
        self.decision_interval = 1
        self.configure(config)

    def configure(self, config=None):
        config = dict(config or {})
        self.min_surge_command_mps = float(
            config.get("min_surge_command_mps", self.min_surge_command_mps)
        )
        self.speed_floor_enabled = bool(
            config.get(
                "speed_floor_enabled",
                self.min_surge_command_mps > 0.0,
            )
        )
        self.neutral_yaw_without_encounter = bool(
            config.get(
                "neutral_yaw_without_encounter",
                self.neutral_yaw_without_encounter,
            )
        )
        self.dt = float(config.get("dt", self.dt))
        self.decision_interval = int(
            config.get("decision_interval", self.decision_interval)
        )

    def action_masks(self, state):
        mask = np.ones(self.n_actions, dtype=bool)
        if (
            self.neutral_yaw_without_encounter
            and state is not None
            and not state._encounter_active
        ):
            self._apply_neutral_yaw_mask(mask)

        if not self.speed_floor_enabled or self.min_surge_command_mps <= 0.0:
            return mask

        cmd_u = self._current_surge_command(state)
        if cmd_u is None:
            return mask

        action_hold_dt = self.dt * self.decision_interval
        for action_idx in range(self.n_actions):
            _, surge_accel_cmd = self.action.decode_discrete_actions(
                action_idx, state.sim_state
            )
            if self._would_cross_speed_floor(cmd_u, surge_accel_cmd, action_hold_dt):
                mask[action_idx] = False
        return mask

    def _apply_neutral_yaw_mask(self, mask):
        yaw_idx = int(np.argmin(np.abs(self.action.yaw_rate_commands)))
        n_accel = len(self.action.surge_accel_commands)
        for action_idx in range(self.n_actions):
            if action_idx // n_accel != yaw_idx:
                mask[action_idx] = False

    def _current_surge_command(self, state):
        if state is not None and state._current_surge_cmd is not None:
            return float(state._current_surge_cmd)
        if state is not None and state.sim_state is not None:
            return float(state.sim_state["nu"][0])
        return None

    def _would_cross_speed_floor(self, cmd_u, surge_accel_cmd, dt):
        if surge_accel_cmd >= 0.0:
            return False
        next_cmd_u = float(cmd_u) + float(surge_accel_cmd) * float(dt)
        return next_cmd_u < self.min_surge_command_mps
