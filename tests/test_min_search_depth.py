import sys
import types
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

pacstl_module = types.ModuleType("pacstl")
pacstl_common_module = types.ModuleType("pacstl.common")
pacstl_interfaces_module = types.ModuleType("pacstl.common.interfaces")


class PACReachableSet:
    pass


class TimeStampedState:
    def __init__(self, time_step, state_array):
        self.time_step = time_step
        self.state_array = state_array


pacstl_interfaces_module.PACReachableSet = PACReachableSet
pacstl_interfaces_module.TimeStampedState = TimeStampedState
sys.modules.setdefault("pacstl", pacstl_module)
sys.modules.setdefault("pacstl.common", pacstl_common_module)
sys.modules.setdefault("pacstl.common.interfaces", pacstl_interfaces_module)

from gym.masking.action_masker import ActionMasker


class DummyAction:
    yaw_rate_commands = np.array([0.0])
    yaw_rate_commands_deg_s = np.array([0.0])
    surge_accel_commands = np.array([0.0])


class DummyVesselModel:
    v_max = 1.0


class DepthAwareSpec:
    def __init__(self, by_trace_length):
        self.by_trace_length = dict(by_trace_length)
        self.trace_lengths = []

    def evaluate(self, partial_tube, trajectory):
        trace_length = len(trajectory)
        self.trace_lengths.append(trace_length)
        return self.by_trace_length[trace_length]


def make_masker(spec, min_search_depth=2, decision_depth=2):
    masker = ActionMasker(
        DummyAction(),
        ego_vessel_model=DummyVesselModel(),
        config={
            "dt": 1.0,
            "sim_dt": 1.0,
            "decision_interval": 5,
            "decision_depth": decision_depth,
            "min_search_depth": min_search_depth,
            "use_3dof_dynamics": False,
        },
    )
    masker.spec = spec
    masker.ellipsoids_Ab_dict = {}
    masker.tube_time_steps = [5.0, 10.0]
    masker.reachable_tube = {5.0: object(), 10.0: object()}
    return masker


def get_mask(masker):
    return masker.get_mask(
        situation="head_on",
        ego_state={
            "eta": np.array([0.0, 0.0, 0.0]),
            "nu": np.array([0.2, 0.0, 0.0]),
            "goal": None,
        },
        encounter_vessel_eta=np.array([10.0, 0.0, 0.0]),
        encounter_speed=0.1,
        robustness_margin=1.0,
        monitoring_radius=100.0,
    )


class MinSearchDepthTest(unittest.TestCase):
    def test_min_search_depth_blocks_first_step_certification(self):
        spec = DepthAwareSpec({1: 1.0, 2: -1.0})
        masker = make_masker(spec, min_search_depth=2, decision_depth=2)

        mask, is_fallback = get_mask(masker)

        self.assertEqual(spec.trace_lengths, [1, 2])
        self.assertEqual(mask.tolist(), [True])
        self.assertIs(is_fallback, True)

    def test_min_search_depth_allows_certification_at_second_step(self):
        spec = DepthAwareSpec({1: 1.0, 2: 1.0})
        masker = make_masker(spec, min_search_depth=2, decision_depth=2)

        mask, is_fallback = get_mask(masker)

        self.assertEqual(spec.trace_lengths, [1, 2])
        self.assertEqual(mask.tolist(), [True])
        self.assertIs(is_fallback, False)

    def test_min_search_depth_must_not_exceed_decision_depth(self):
        with self.assertRaisesRegex(
            ValueError,
            "min_search_depth must be <= decision_depth",
        ):
            make_masker(DepthAwareSpec({1: 1.0}), min_search_depth=3, decision_depth=2)


if __name__ == "__main__":
    unittest.main()
