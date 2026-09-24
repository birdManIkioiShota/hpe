from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from hpe.data import ManifestDataset
from hpe.evaluation.metrics import ERROR_COLUMNS, metric_tables, yaw_bin_table
from hpe.geometry import (
    circular_error_degrees,
    euler_degrees_to_matrix,
    geodesic_error_degrees,
    matrix_to_euler_degrees,
    rotation_matrix_from_6d,
)
from hpe.models import SixDRepNet360, load_checkpoint


ROOT = Path(__file__).resolve().parents[2]


class RotationTests(unittest.TestCase):
    def test_six_d_rotation_is_orthonormal(self):
        matrix = rotation_matrix_from_6d(torch.randn(8, 6))
        identity = matrix.transpose(1, 2) @ matrix
        torch.testing.assert_close(identity, torch.eye(3).expand_as(identity), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(torch.linalg.det(matrix), torch.ones(8), atol=1e-5, rtol=1e-5)

    def test_rear_euler_canonicalization_preserves_rotation(self):
        source = torch.tensor([[-6.7, 117.6, -8.7], [10.0, -155.0, 35.0]])
        matrix = euler_degrees_to_matrix(source)
        canonical = matrix_to_euler_degrees(matrix)
        reconstructed = euler_degrees_to_matrix(canonical)
        error = geodesic_error_degrees(reconstructed, matrix)
        self.assertLess(float(error.max()), 0.1)

    def test_circular_error_wraps_at_180(self):
        error = circular_error_degrees(torch.tensor([179.0]), torch.tensor([-179.0]))
        self.assertEqual(float(error.item()), 2.0)


class DatasetTests(unittest.TestCase):
    def test_manifest_dataset_loads_official_input_shape(self):
        dataset = ManifestDataset(
            ROOT,
            ROOT / "datasets/prepared/aflw2000/manifest.jsonl",
            max_samples=1,
        )
        image, metadata = dataset[0]
        self.assertEqual(tuple(image.shape), (3, 224, 224))
        self.assertEqual(metadata["dataset"], "aflw2000")
        self.assertEqual(tuple(metadata["bbox_xyxy"].shape), (4,))


class ModelTests(unittest.TestCase):
    def test_published_checkpoint_loads_strictly(self):
        model = SixDRepNet360()
        info = load_checkpoint(
            model,
            ROOT / "checkpoints/6DRepNet360_Full-Rotation_300W_LP+Panoptic.pth",
        )
        self.assertEqual(info["parameter_tensors"], 320)
        self.assertEqual(
            info["sha256"],
            "3ee08f1e04b8d452a6c4a40926a6f38051894ae6d0aaa6d191fe6d8bc6e4f9c6",
        )


class MetricsTests(unittest.TestCase):
    def test_yaw_bands_use_source_yaw(self):
        frame = pd.DataFrame(
            {
                "gt_source_yaw_deg": [0.0, 60.0, 119.9, -120.0],
                "bbox_width": [100.0] * 4,
                "bbox_height": [100.0] * 4,
                "occlusion_percent": [np.nan] * 4,
                **{column: [1.0, 2.0, 3.0, 4.0] for column in ERROR_COLUMNS},
            }
        )
        bands = metric_tables(frame)["yaw_bands"].set_index("yaw_band")
        self.assertEqual(int(bands.loc["front_lt60", "count"]), 1)
        self.assertEqual(int(bands.loc["side_60_to_lt120", "count"]), 2)
        self.assertEqual(int(bands.loc["rear_ge120", "count"]), 1)


    def test_fifteen_degree_yaw_bins_cover_full_range(self):
        yaw = [
            -180.0,
            -165.0,
            -150.0,
            -135.0,
            -120.0,
            -15.0,
            0.0,
            15.0,
            120.0,
            135.0,
            150.0,
            165.0,
            179.9,
        ]
        frame = pd.DataFrame(
            {
                "gt_source_yaw_deg": yaw,
                "bbox_width": [100.0] * len(yaw),
                "bbox_height": [100.0] * len(yaw),
                "occlusion_percent": [np.nan] * len(yaw),
                **{
                    column: [1.0] * len(yaw)
                    for column in ERROR_COLUMNS
                },
            }
        )
        table = yaw_bin_table(frame, bin_size_deg=15)
        self.assertEqual(int(table["count"].sum()), len(yaw))
        self.assertIn("-180_to_-165", set(table["yaw_bin"]))
        self.assertIn("165_to_180", set(table["yaw_bin"]))


if __name__ == "__main__":
    unittest.main()
