from __future__ import annotations

import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from experiments.common.vgg_dev_evaluation import (
    EXPECTED_DEV_COUNT,
    aggregate_prediction_file,
    aggregate_predictions,
    evaluation_fingerprint,
    full_range_yaw_degrees,
    prediction_instance_ids_sha256,
    regroup_targets_by_checkpoint,
    yaw_bin_index,
    yaw_bin_record,
)
from experiments.common.vgg_dev_targets import TARGETS, validate_target_registry
from hpe.geometry.rotations import (
    circular_error_degrees,
    euler_degrees_to_matrix,
    geodesic_error_degrees,
)


def _prediction_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "instance_id": ["a", "b", "c", "d"],
            "gt_full_yaw_deg": [-179.0, -164.9, 179.0, np.nan],
            "geodesic_error_deg": [1.0, 2.0, 3.0, 4.0],
            "pitch_error_deg": [2.0, 4.0, 6.0, 8.0],
            "yaw_error_deg": [1.0, 3.0, 5.0, 7.0],
            "roll_error_deg": [3.0, 5.0, 7.0, 9.0],
            "mean_axis_error_deg": [2.0, 4.0, 6.0, 8.0],
        }
    )


class TargetRegistryTests(unittest.TestCase):
    def test_registry_has_38_logical_targets_and_36_unique_checkpoints(self) -> None:
        validate_target_registry()
        self.assertEqual(len(TARGETS), 38)
        self.assertEqual(len(regroup_targets_by_checkpoint(TARGETS)), 36)

    def test_registry_has_only_the_two_expected_shared_weight_groups(self) -> None:
        groups = regroup_targets_by_checkpoint(TARGETS)
        shared = {
            sha: sorted(target.logical_id for target in targets)
            for sha, targets in groups.items()
            if len(targets) > 1
        }
        self.assertEqual(len(shared), 2)
        self.assertIn(
            sorted(
                [
                    "experiment_7_yawpose_stratified:Y_base_vgg_dad",
                    "experiment_8_yawpose_loss_weight:Y_base_vgg_dad",
                ]
            ),
            shared.values(),
        )
        self.assertIn(
            sorted(
                [
                    "experiment_7_yawpose_stratified:Y_top60_vgg_dad",
                    "experiment_8_yawpose_loss_weight:Y_w020_vgg_dad",
                ]
            ),
            shared.values(),
        )

    def test_reference_dev_regression_values_are_fixed(self) -> None:
        values = {
            target.logical_id: target.reference_vgg_dev_so3_mean_deg
            for target in TARGETS
            if target.reference_vgg_dev_so3_mean_deg is not None
        }
        self.assertEqual(
            values,
            {
                "initial:audited_base": 21.426986640470798,
                "experiment_1_vgg_ft:best": 2.919495880048786,
            },
        )
        self.assertEqual(EXPECTED_DEV_COUNT, 51_914)


class GeometryAndBinningTests(unittest.TestCase):
    def test_identical_rotation_has_zero_so3_error(self) -> None:
        matrix = euler_degrees_to_matrix(
            torch.tensor([[17.0, 135.0, -23.0]], dtype=torch.float64)
        )
        error = geodesic_error_degrees(matrix, matrix, stable=True)
        self.assertAlmostEqual(float(error.item()), 0.0, places=12)

    def test_circular_179_minus_minus179_is_two_degrees(self) -> None:
        error = circular_error_degrees(
            torch.tensor([179.0], dtype=torch.float64),
            torch.tensor([-179.0], dtype=torch.float64),
        )
        self.assertAlmostEqual(float(error.item()), 2.0, places=12)

    def test_head_forward_yaw_uses_full_range_rotation_geometry(self) -> None:
        matrix = euler_degrees_to_matrix(
            torch.tensor([[0.0, 135.0, 0.0]], dtype=torch.float64)
        )[0].numpy()
        self.assertAlmostEqual(full_range_yaw_degrees(matrix), 135.0, places=10)

    def test_vertical_head_forward_has_undefined_full_range_yaw(self) -> None:
        matrix = euler_degrees_to_matrix(
            torch.tensor([[90.0, 0.0, 0.0]], dtype=torch.float64)
        )[0].numpy()
        self.assertTrue(math.isnan(full_range_yaw_degrees(matrix)))

    def test_all_15_degree_boundaries_are_unique_and_complete(self) -> None:
        self.assertEqual(yaw_bin_index(-180.0), 0)
        for index, boundary in enumerate(range(-165, 180, 15), start=1):
            self.assertEqual(yaw_bin_index(float(boundary)), index)
        self.assertEqual(yaw_bin_index(180.0), 23)
        self.assertIsNone(yaw_bin_index(float("nan")))
        with self.assertRaises(ValueError):
            yaw_bin_index(180.01)
        labels = [yaw_bin_record(index)["yaw_bin"] for index in range(24)]
        self.assertEqual(len(labels), 24)
        self.assertEqual(len(set(labels)), 24)
        self.assertFalse(yaw_bin_record(0)["right_inclusive"])
        self.assertTrue(yaw_bin_record(23)["right_inclusive"])


class AggregationTests(unittest.TestCase):
    def test_empty_bins_are_missing_not_zero(self) -> None:
        aggregate = aggregate_predictions(_prediction_frame())
        empty = next(row for row in aggregate["yaw_bins_15"] if row["count"] == 0)
        for key in (
            "so3_mean_deg",
            "pitch_maae_deg",
            "yaw_maae_deg",
            "roll_maae_deg",
            "mean_axis_maae_deg",
        ):
            self.assertIsNone(empty[key])

    def test_bins_plus_undefined_reconstruct_overall(self) -> None:
        aggregate = aggregate_predictions(_prediction_frame())
        self.assertEqual(aggregate["overall"]["count"], 4)
        self.assertEqual(aggregate["overall"]["defined_yaw_count"], 3)
        self.assertEqual(aggregate["overall"]["undefined_yaw_count"], 1)
        self.assertEqual(
            sum(row["count"] for row in aggregate["yaw_bins_15"])
            + aggregate["undefined_yaw"]["count"],
            4,
        )
        self.assertAlmostEqual(aggregate["overall"]["so3_mean_deg"], 2.5)
        self.assertAlmostEqual(aggregate["overall"]["pitch_maae_deg"], 5.0)
        self.assertAlmostEqual(aggregate["overall"]["yaw_maae_deg"], 4.0)
        self.assertAlmostEqual(aggregate["overall"]["roll_maae_deg"], 6.0)
        self.assertAlmostEqual(aggregate["overall"]["mean_axis_maae_deg"], 5.0)

    def test_saved_prediction_reaggregation_matches(self) -> None:
        expected = aggregate_predictions(_prediction_frame())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.csv.gz"
            _prediction_frame().to_csv(
                path,
                index=False,
                compression="gzip",
                float_format="%.17g",
            )
            actual = aggregate_prediction_file(path)
        self.assertEqual(actual["overall"]["count"], expected["overall"]["count"])
        for key in (
            "so3_mean_deg",
            "pitch_maae_deg",
            "yaw_maae_deg",
            "roll_maae_deg",
            "mean_axis_maae_deg",
        ):
            self.assertAlmostEqual(actual["overall"][key], expected["overall"][key], places=12)

    def test_instance_id_digest_is_order_sensitive(self) -> None:
        frame = _prediction_frame()
        original = prediction_instance_ids_sha256(frame)
        reordered = prediction_instance_ids_sha256(frame.iloc[::-1].reset_index(drop=True))
        self.assertNotEqual(original, reordered)

    def test_duplicate_instance_ids_are_rejected(self) -> None:
        frame = _prediction_frame()
        frame.loc[1, "instance_id"] = "a"
        with self.assertRaises(ValueError):
            aggregate_predictions(frame)


class FingerprintTests(unittest.TestCase):
    def test_fingerprint_changes_when_input_or_settings_change(self) -> None:
        base = {
            "checkpoint_sha256": "a" * 64,
            "input": {"manifest_sha256": "b" * 64},
            "settings": {"batch_size": 256, "amp": False},
        }
        same = {
            "settings": {"amp": False, "batch_size": 256},
            "input": {"manifest_sha256": "b" * 64},
            "checkpoint_sha256": "a" * 64,
        }
        changed = {
            **base,
            "settings": {"batch_size": 128, "amp": False},
        }
        self.assertEqual(evaluation_fingerprint(base), evaluation_fingerprint(same))
        self.assertNotEqual(evaluation_fingerprint(base), evaluation_fingerprint(changed))


if __name__ == "__main__":
    unittest.main()
