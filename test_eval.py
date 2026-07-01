import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import yaml

from eval import (
    EpisodeBuilder,
    EvalAccumulator,
    EvalTarget,
    _eta_seconds,
    _format_duration,
    load_manifest,
    resolve_eval_args,
    summarize_by_agent,
    _wandb_checkpoint_scalars,
)


def _target(tmp_path, agent="depth1", seed=0):
    config_path = tmp_path / f"{agent}_{seed}.yaml"
    model_path = tmp_path / f"{agent}_{seed}.zip"
    config_path.write_text("training_configuration: {}\n", encoding="utf-8")
    model_path.write_text("model", encoding="utf-8")
    return EvalTarget(agent, seed, config_path, model_path)


class EvalAggregationTests(unittest.TestCase):
    def test_episode_tracks_wrong_side_crossing_once(self):
        episode = EpisodeBuilder()
        episode.step(1.0, {"reward_wrong_side_crossing": 0.0})
        episode.step(2.0, {"reward_wrong_side_crossing": -75.0})
        episode.step(3.0, {"reward_wrong_side_crossing": -75.0})

        record = episode.finish("goal_reached", episode_index=4)

        self.assertTrue(record["wrong_side_crossing"])
        self.assertEqual(record["reward"], 6.0)
        self.assertEqual(record["length"], 3)
        self.assertEqual(record["episode_index"], 4)

    def test_checkpoint_metrics_use_episode_rates_and_encounter_fallback_rate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            accumulator = EvalAccumulator(
                target=_target(tmp_path),
                eval_steps_target=10,
                total_actions=5,
            )

            for info in [
                {
                    "encounter_active": True,
                    "mask_fallback": True,
                    "mask_allowed_count": 2,
                },
                {
                    "encounter_active": True,
                    "mask_fallback": False,
                    "mask_allowed_count": 4,
                },
                {
                    "encounter_active": False,
                    "mask_fallback": True,
                    "mask_allowed_count": 1,
                },
            ]:
                accumulator.observe_step(0.0, info)

            empty_builder = EpisodeBuilder()
            empty_sums = {key: 0.0 for key in empty_builder.reward_sums}
            empty_counts = {key: 0 for key in empty_builder.reward_counts}
            accumulator.add_episode(
                {
                    "reason": "collision",
                    "reward": -5.0,
                    "length": 3,
                    "wrong_side_crossing": True,
                    "reward_sums": dict(empty_sums),
                    "reward_counts": dict(empty_counts),
                }
            )
            accumulator.add_episode(
                {
                    "reason": "time_limit",
                    "reward": 1.0,
                    "length": 4,
                    "wrong_side_crossing": False,
                    "reward_sums": dict(empty_sums),
                    "reward_counts": dict(empty_counts),
                }
            )

            metrics = accumulator.checkpoint_metrics()

        self.assertEqual(metrics["collision_rate"], 0.5)
        self.assertEqual(metrics["goal_rate"], 0.0)
        self.assertEqual(metrics["timeout_rate"], 0.5)
        self.assertEqual(metrics["wrong_side_crossing_rate"], 0.5)
        self.assertEqual(metrics["fallback_rate"], 0.5)
        self.assertEqual(metrics["masking_allowed_actions"], 3.0)
        self.assertTrue(
            math.isclose(metrics["masking_allowed_action_fraction"], 0.6)
        )

    def test_summarize_by_agent_reports_mean_std_and_n(self):
        summary = summarize_by_agent(
            [
                {"agent": "depth1", "seed": 0, "collision_rate": 0.0},
                {"agent": "depth1", "seed": 1, "collision_rate": 1.0},
                {"agent": "depth2", "seed": 0, "collision_rate": 0.25},
            ]
        )

        by_key = {(row["agent"], row["metric"]): row for row in summary}

        self.assertEqual(by_key[("depth1", "collision_rate")]["mean"], 0.5)
        self.assertTrue(
            math.isclose(
                by_key[("depth1", "collision_rate")]["std"],
                math.sqrt(2) / 2,
            )
        )
        self.assertEqual(by_key[("depth1", "collision_rate")]["n"], 2)
        self.assertEqual(by_key[("depth2", "collision_rate")]["std"], 0.0)


class EvalManifestTests(unittest.TestCase):
    def test_load_manifest_accepts_eval_config_with_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config_path = tmp_path / "config.yaml"
            model_path = tmp_path / "model.zip"
            config_path.write_text("training_configuration: {}\n", encoding="utf-8")
            model_path.write_text("model", encoding="utf-8")
            eval_config_path = tmp_path / "eval_single.yaml"
            eval_config_path.write_text(
                yaml.safe_dump(
                    {
                        "eval_steps": 10000,
                        "wandb": True,
                        "checkpoints": [
                            {
                                "agent": "depth3",
                                "seed": 0,
                                "config_path": str(config_path),
                                "model_path": str(model_path),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            targets = load_manifest(eval_config_path)

        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0].agent, "depth3")

    def test_load_manifest_validates_duplicate_agent_seed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config_path = tmp_path / "config.yaml"
            model_path = tmp_path / "model.zip"
            config_path.write_text("training_configuration: {}\n", encoding="utf-8")
            model_path.write_text("model", encoding="utf-8")
            manifest_path = tmp_path / "manifest.yaml"
            entries = [
                {
                    "agent": "depth1",
                    "seed": 0,
                    "config_path": str(config_path),
                    "model_path": str(model_path),
                },
                {
                    "agent": "depth1",
                    "seed": 0,
                    "config_path": str(config_path),
                    "model_path": str(model_path),
                },
            ]
            manifest_path.write_text(
                yaml.safe_dump({"checkpoints": entries}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Duplicate manifest entry"):
                load_manifest(manifest_path)

    def test_load_manifest_validates_missing_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            manifest_path = tmp_path / "manifest.yaml"
            manifest_path.write_text(
                yaml.safe_dump(
                    {
                        "checkpoints": [
                            {
                                "agent": "depth1",
                                "seed": 0,
                                "config_path": str(tmp_path / "missing.yaml"),
                                "model_path": str(tmp_path / "missing.zip"),
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(FileNotFoundError, "Config path"):
                load_manifest(manifest_path)


class EvalConfigTests(unittest.TestCase):
    def test_resolve_eval_args_uses_config_as_manifest_when_it_has_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "eval_single.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "eval_steps": 10000,
                        "wandb": True,
                        "wandb_project": "colregs-ppo",
                        "wandb_run_name": "depth3_seed0_eval_10k",
                        "checkpoints": [],
                    }
                ),
                encoding="utf-8",
            )

            args = resolve_eval_args(
                SimpleNamespace(
                    config=str(config_path),
                    manifest=None,
                    eval_steps=None,
                    output_dir=None,
                    eval_seed_base=None,
                    progress_log_interval=None,
                    disable_monitoring=None,
                    wandb=None,
                    wandb_project=None,
                    wandb_run_name=None,
                    wandb_group=None,
                )
            )

        self.assertEqual(args.manifest, str(config_path))
        self.assertEqual(args.eval_steps, 10000)
        self.assertTrue(args.wandb)
        self.assertEqual(args.wandb_run_name, "depth3_seed0_eval_10k")

    def test_resolve_eval_args_prefers_cli_over_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "eval_single.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "manifest": "from_config.yaml",
                        "eval_steps": 10000,
                        "wandb": False,
                    }
                ),
                encoding="utf-8",
            )

            args = resolve_eval_args(
                SimpleNamespace(
                    config=str(config_path),
                    manifest="from_cli.yaml",
                    eval_steps=500,
                    output_dir=None,
                    eval_seed_base=None,
                    progress_log_interval=None,
                    disable_monitoring=None,
                    wandb=True,
                    wandb_project=None,
                    wandb_run_name=None,
                    wandb_group=None,
                )
            )

        self.assertEqual(args.manifest, "from_cli.yaml")
        self.assertEqual(args.eval_steps, 500)
        self.assertTrue(args.wandb)


class EvalWandbTests(unittest.TestCase):
    def test_wandb_checkpoint_scalars_are_prefixed_by_agent_and_seed(self):
        scalars = _wandb_checkpoint_scalars(
            {
                "agent": "depth 1",
                "seed": 2,
                "collision_rate": 0.25,
                "goal_rate": 0.75,
                "config_path": "configuration/config_1.yaml",
                "model_path": "model.zip",
            }
        )

        self.assertEqual(scalars["checkpoint/depth_1_seed_2/collision_rate"], 0.25)
        self.assertEqual(scalars["checkpoint/depth_1_seed_2/goal_rate"], 0.75)
        self.assertNotIn("checkpoint/depth_1_seed_2/seed", scalars)


class EvalProgressTests(unittest.TestCase):
    def test_eta_seconds_scales_remaining_time_from_progress(self):
        self.assertEqual(_eta_seconds(elapsed_s=25.0, progress=0.25), 75.0)
        self.assertIsNone(_eta_seconds(elapsed_s=25.0, progress=0.0))

    def test_format_duration_uses_compact_units(self):
        self.assertEqual(_format_duration(5.2), "5s")
        self.assertEqual(_format_duration(65), "1m05s")
        self.assertEqual(_format_duration(3661), "1h01m01s")


if __name__ == "__main__":
    unittest.main()
