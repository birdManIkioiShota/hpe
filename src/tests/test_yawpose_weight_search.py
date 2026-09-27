from __future__ import annotations

from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from experiments.common.yawpose import build_yawpose_epoch_plan
from experiments.common.yawpose_search import (
    condition_records,
    summarize_search,
    yawpose_weight_conditions,
)
from experiments.scripts import evaluate_yawpose_rear_search as evaluation
from experiments.scripts import run_yawpose_rear_search as search
from hpe.datasets.common import sha256_file, write_json_atomic
from hpe.evaluation.metrics import ERROR_COLUMNS, yaw_bin_table

ROOT = Path(__file__).resolve().parents[2]


class WeightSearchTests(unittest.TestCase):
    def test_dry_run_is_six_fixed_conditions_without_local_training_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = io.StringIO()
            with mock.patch.object(search, "ROOT", root), mock.patch(
                "sys.argv", ["search", "--search-mode", "loss-weight", "--dry-run"]
            ), redirect_stdout(output):
                search.main()
            config = json.loads(output.getvalue())
            self.assertEqual(list(root.iterdir()), [])
        rows = config["conditions"]
        self.assertEqual(
            [r["yawpose_weight"] for r in rows], [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        )
        self.assertEqual(rows[0]["subset"], "base")
        self.assertEqual([r["subset"] for r in rows[1:]], ["top060"] * 5)
        self.assertTrue(all(r["use_dad"] for r in rows))
        self.assertEqual(len({r["condition_id"] for r in rows}), 6)
        self.assertNotIn("yawpose_weight", config["shared_training"])

    def test_coefficients_reach_each_training_command(self):
        args = search.build_parser().parse_args(["--search-mode", "loss-weight"])
        search._search_config(args)
        self.assertEqual(args.run_id, "yawpose_rear_weight_search")
        for condition in yawpose_weight_conditions():
            command = search._condition_command(args, condition, resume=False)
            self.assertEqual(
                float(command[command.index("--yawpose-weight") + 1]),
                condition.yawpose_weight,
            )
            self.assertEqual(command[command.index("--subset") + 1], condition.subset)
            self.assertIn("--use-dad", command)
            self.assertNotIn("--resume", command)
            resumed = search._condition_command(args, condition, resume=True)
            self.assertIn("--resume", resumed)

    def test_weight_mode_rejects_shared_coefficient_override(self):
        args = search.build_parser().parse_args(
            [
                "--search-mode",
                "loss-weight",
                "--yawpose-weight",
                "1.2",
            ]
        )
        with self.assertRaisesRegex(ValueError, "fixed six"):
            search._search_config(args)

    def test_legacy_config_matches_recorded_run(self):
        args = search.build_parser().parse_args(["--workers", "4"])
        actual = search._search_config(args)
        expected = json.loads(
            (
                ROOT / "experiments/runs/yawpose_rear_stratified_search/config.json"
            ).read_text()
        )
        self.assertEqual(actual, expected)

    def test_previous_inputs_are_required(self):
        reference = {
            "train_manifests": {"vgg": "vgg-sha", "dad": "dad-sha"},
            "dev_manifests": {"dev": "dev-sha"},
            "yawpose_subset": {
                "path": "top060.jsonl",
                "sha256": "subset-sha",
                "count": 8433,
            },
        }
        actual = {
            "training_manifests": {
                "vgg": "vgg-sha",
                "dad": "dad-sha",
                "dev": "dev-sha",
            },
            "reliability_subsets": {
                "top060": {"path": "top060.jsonl", "sha256": "subset-sha"}
            },
        }
        search._validate_weight_inputs(actual, reference)
        changed = copy.deepcopy(actual)
        changed["training_manifests"]["dad"] = "changed"
        with self.assertRaisesRegex(ValueError, "VGG\\+DAD"):
            search._validate_weight_inputs(changed, reference)
        changed = copy.deepcopy(actual)
        changed["reliability_subsets"]["top060"]["sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "top060"):
            search._validate_weight_inputs(changed, reference)

    def test_all_positive_coefficients_use_identical_epoch_orders(self):
        records = [{"source": "synthetic", "rear_yaw_bin": "rear"} for _ in range(8433)]
        for epoch in (0, 9):
            plans = [
                build_yawpose_epoch_plan(records, seed=42, epoch=epoch)
                for _ in yawpose_weight_conditions()[1:]
            ]
            self.assertTrue(all(plan == plans[0] for plan in plans))
            self.assertEqual(plans[0].unique_samples, 8433)
            self.assertEqual(plans[0].total_draws, 8433)
            self.assertEqual(plans[0].repeated_draws, 0)

    def test_training_summary_keeps_weights_and_rejects_mismatched_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            conditions = condition_records("loss-weight")
            write_json_atomic(
                root / "config.json",
                {
                    "kind": "yawpose_rear_loss_weight_search",
                    "epochs": 10,
                    "conditions": conditions,
                    "shared_training": {"seed": 42},
                },
            )
            provenance = {
                "training_manifests": {"train": "t", "dev": "d"},
                "reliability_subsets": {
                    "top060": {"path": "top060.jsonl", "sha256": "s"}
                },
            }
            write_json_atomic(root / "provenance.json", provenance)
            metric = {
                "mean_deg": 1,
                "p90_deg": 2,
                "median_deg": 1,
                "over30_percent": 0,
                "over60_percent": 0,
                "over90_percent": 0,
            }
            for condition in conditions:
                directory = root / "conditions" / condition["condition_id"]
                directory.mkdir(parents=True)
                (directory / "metrics").mkdir()
                write_json_atomic(
                    directory / "status.json", {"status": "completed", "best_epoch": 8}
                )
                write_json_atomic(directory / "config.json", dict(condition, seed=42))
                write_json_atomic(
                    directory / "provenance.json",
                    {
                        "train_manifests": {"train": "t"},
                        "dev_manifests": {"dev": "d"},
                        "yawpose_subset": (
                            None
                            if condition["subset"] == "base"
                            else {"path": "top060.jsonl", "sha256": "s", "count": 8433}
                        ),
                    },
                )
                write_json_atomic(
                    directory / "metrics/epoch_010.json",
                    {
                        "dev": {group: metric for group in ("front", "side", "rear")},
                        "dev_yaw": {"rear": metric},
                        "train": {"yawpose_sampling": {}},
                    },
                )
            (root / "metrics").mkdir()
            result = json.loads(summarize_search(root).read_text())
            self.assertEqual(
                [r["yawpose_weight"] for r in result], [0, 0.2, 0.4, 0.6, 0.8, 1.0]
            )
            condition_dir = root / "conditions" / conditions[-1]["condition_id"]
            changed = dict(conditions[-1], yawpose_weight=0.2, seed=42)
            write_json_atomic(condition_dir / "config.json", changed)
            with self.assertRaisesRegex(ValueError, "trained condition differs"):
                summarize_search(root)
            write_json_atomic(
                condition_dir / "config.json", dict(conditions[-1], seed=43)
            )
            with self.assertRaisesRegex(ValueError, "shared training setting changed"):
                summarize_search(root)
            write_json_atomic(
                condition_dir / "config.json", dict(conditions[-1], seed=42)
            )
            inputs = json.loads((condition_dir / "provenance.json").read_text())
            inputs["yawpose_subset"]["sha256"] = "changed"
            write_json_atomic(condition_dir / "provenance.json", inputs)
            with self.assertRaisesRegex(ValueError, "top060 changed"):
                summarize_search(root)


def make_evaluation(root: Path, condition: dict, dataset: str, suffix: str) -> None:
    name = condition["condition_id"] + suffix
    directory = root / "conditions" / name
    (directory / "predictions").mkdir(parents=True)
    yaw = np.array([-179.0, -125.0, 0.0, 145.0])
    error = 10.0 - condition["yawpose_weight"] * 5.0
    prediction = np.deg2rad(yaw + error)
    frame = pd.DataFrame(
        {
            "instance_id": ["a", "b", "c", "d"],
            "gt_source_yaw_deg": yaw,
            "pred_R_02": np.sin(prediction),
            "pred_R_22": np.cos(prediction),
            **{column: np.full(4, error) for column in ERROR_COLUMNS},
        }
    )
    frame.to_csv(directory / "predictions" / f"{dataset}.csv.gz", index=False)
    write_json_atomic(
        directory / "run.json",
        {
            "run_name": name,
            "checkpoint": {"path": f"{name}/checkpoints/epoch_010.pth", "sha256": name},
            "settings": {"model": "fixture", "batch_size": 256},
            "datasets": [
                {
                    "dataset": dataset,
                    "count": 4,
                    "manifest_sha256": dataset,
                    "image_lock_sha256": dataset,
                    "overall": {"geodesic_error_deg_mean": error},
                }
            ],
        },
    )
    if dataset == "agora_hpe":
        path = directory / "datasets/agora_hpe/metrics/yaw_bins_15.json"
        path.parent.mkdir(parents=True)
        table = yaw_bin_table(frame, bin_size_deg=15, include_empty=True).astype(object)
        write_json_atomic(path, table.where(pd.notna(table), None).to_dict("records"))


class WeightEvaluationTests(unittest.TestCase):
    def test_evaluation_entry_point_uses_final_weights_and_writes_summary(self):
        conditions = condition_records("loss-weight")
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            evaluation, "_write_agora_yaw15_comparison_plot"
        ), mock.patch.object(evaluation.subprocess, "run") as inference:
            root = Path(temporary)
            run = root / "experiments/runs/weights"
            output = root / "eval/weights"
            write_json_atomic(run / "status.json", {"status": "completed"})
            write_json_atomic(
                run / "config.json",
                {
                    "kind": "yawpose_rear_loss_weight_search",
                    "epochs": 10,
                    "conditions": conditions,
                },
            )
            for condition in conditions:
                make_evaluation(output, condition, "agora_hpe", "")
                checkpoint = (
                    run
                    / "conditions"
                    / condition["condition_id"]
                    / "checkpoints/epoch_010.pth"
                )
                checkpoint.parent.mkdir(parents=True)
                checkpoint.write_bytes(condition["condition_id"].encode())
                checkpoint.with_name("best.pth").write_bytes(b"other epoch")
                metadata_path = (
                    output / "conditions" / condition["condition_id"] / "run.json"
                )
                metadata = json.loads(metadata_path.read_text())
                metadata["checkpoint"] = {
                    "path": str(checkpoint),
                    "sha256": sha256_file(checkpoint),
                }
                write_json_atomic(metadata_path, metadata)
            baseline = root / "eval/baseline_fp32/evaluations/baseline"
            shutil.copytree(
                output / "conditions" / conditions[0]["condition_id"], baseline
            )
            metadata = json.loads((baseline / "run.json").read_text())
            metadata["run_name"] = "initial"
            write_json_atomic(baseline / "run.json", metadata)
            argv = [
                "eval",
                "--run-id",
                "weights",
                "--baseline-run",
                "baseline_fp32",
                "--bootstrap-repetitions",
                "10",
            ]
            with mock.patch.object(evaluation, "ROOT", root), mock.patch(
                "sys.argv", argv
            ):
                evaluation.main()
            inference.assert_not_called()
            summary = json.loads((output / "weight_summary.json").read_text())
            rows = summary["suites"]["baseline_fp32_final"]["conditions"]
            self.assertEqual(
                [row["yawpose_weight"] for row in rows], [0, 0.2, 0.4, 0.6, 0.8, 1.0]
            )
            self.assertTrue(
                all(row["checkpoint"]["path"].endswith("epoch_010.pth") for row in rows)
            )
            self.assertEqual(
                rows[-1]["so3"][0]["delta_candidate_minus_control_deg"], -5
            )
            # A cached result for best.pth must not be accepted as epoch 10.
            metadata = json.loads(metadata_path.read_text())
            metadata["checkpoint"]["sha256"] = sha256_file(
                checkpoint.with_name("best.pth")
            )
            write_json_atomic(metadata_path, metadata)
            with mock.patch.object(evaluation, "ROOT", root), mock.patch(
                "sys.argv", argv
            ):
                with self.assertRaisesRegex(ValueError, "checkpoint differs"):
                    evaluation.main()

    def test_saved_evaluation_must_match_requested_weight_and_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "epoch_010.pth"
            checkpoint.write_bytes(b"final weights")
            metadata = {
                "checkpoint": {"sha256": sha256_file(checkpoint)},
                "settings": {"batch_size": 256},
                "datasets": [
                    {
                        "dataset": "agora_hpe",
                        "manifest_sha256": "m",
                        "image_lock_sha256": "i",
                    }
                ],
            }
            write_json_atomic(root / "run.json", metadata)
            evaluation._validate_weight_evaluation(root, checkpoint, metadata)
            changed_reference = copy.deepcopy(metadata)
            changed_reference["datasets"][0]["image_lock_sha256"] = "changed"
            with self.assertRaisesRegex(ValueError, "image locks"):
                evaluation._validate_weight_evaluation(
                    root, checkpoint, changed_reference
                )
            checkpoint.write_bytes(b"different weights")
            with self.assertRaisesRegex(ValueError, "checkpoint differs"):
                evaluation._validate_weight_evaluation(root, checkpoint, metadata)

    def test_control_comparisons_preserve_both_evaluation_suites(self):
        conditions = condition_records("loss-weight")
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            evaluation, "_write_agora_yaw15_comparison_plot"
        ):
            root = Path(temporary)
            (root / "yaw_comparisons").mkdir()
            for dataset, suffix, key in (
                ("agora_hpe", "", "baseline_fp32_final"),
                ("dad3dheads", "__dad", "baseline_dad_fp32_final__dad"),
            ):
                for condition in conditions:
                    make_evaluation(root, condition, dataset, suffix)
                evaluation._weight_comparisons(
                    root,
                    conditions,
                    suffix=suffix,
                    reference_key=key,
                    repetitions=10,
                    sample_limit=100,
                )
            summary = json.loads((root / "weight_summary.json").read_text())
            self.assertEqual(len(summary["suites"]), 2)
            rows = summary["suites"]["baseline_fp32_final"]["conditions"]
            self.assertEqual(len(rows), 6)
            self.assertEqual(rows[0]["so3"][0]["delta_candidate_minus_control_deg"], 0)
            self.assertEqual(
                rows[-1]["so3"][0]["delta_candidate_minus_control_deg"], -5
            )
            self.assertTrue(rows[-1]["checkpoint"]["path"].endswith("epoch_010.pth"))
            self.assertEqual(len(rows[-1]["so3"][0]["yaw_bins_15"]), 24)
            self.assertTrue(
                any(
                    r["count"] == 0 and r["delta_candidate_minus_control_deg"] is None
                    for r in rows[-1]["so3"][0]["yaw_bins_15"]
                )
            )
            rear = next(
                r for r in rows[-1]["head_forward_yaw"] if r["group"] == "rear_ge120"
            )
            self.assertAlmostEqual(rear["mean_delta_candidate_minus_baseline_deg"], -5)
            self.assertTrue(
                (
                    root
                    / "comparisons/Y_base_vgg_dad_vs_Y_w100_vgg_dad/comparison.json"
                ).is_file()
            )
            # Repeating the same evaluation can reuse its control comparisons.
            evaluation._weight_comparisons(
                root,
                conditions,
                suffix="",
                reference_key="baseline_fp32_final",
                repetitions=10,
                sample_limit=100,
            )
            with self.assertRaisesRegex(ValueError, "settings changed"):
                evaluation._weight_comparisons(
                    root,
                    conditions,
                    suffix="",
                    reference_key="baseline_fp32_final",
                    repetitions=20,
                    sample_limit=100,
                )

    def test_weight_evaluation_rejects_best_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "experiments/runs/weights"
            run.mkdir(parents=True)
            write_json_atomic(run / "status.json", {"status": "completed"})
            write_json_atomic(
                run / "config.json",
                {
                    "kind": "yawpose_rear_loss_weight_search",
                    "conditions": condition_records("loss-weight"),
                },
            )
            with mock.patch.object(evaluation, "ROOT", root), mock.patch(
                "sys.argv",
                [
                    "eval",
                    "--run-id",
                    "weights",
                    "--baseline-run",
                    "baseline_fp32",
                    "--checkpoint-choice",
                    "best",
                ],
            ):
                with self.assertRaisesRegex(ValueError, "fixed final checkpoint"):
                    evaluation.main()
            self.assertFalse((root / "eval").exists())


if __name__ == "__main__":
    unittest.main()
