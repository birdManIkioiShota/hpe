from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from experiments.common.checkpoints import interpolate_state_dict
from experiments.common.pose import (
    UndefinedAzimuthError,
    forward_azimuth_degrees,
    pose_band,
)
from experiments.common.run_directory import (
    ExperimentRun,
    experiment_condition_path,
    experiment_run_path,
)


class ExperimentRunTests(unittest.TestCase):
    def test_run_directory_contract_and_path_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = ExperimentRun.create(
                root,
                "candidate_a",
                config={"kind": "test"},
                provenance={"source": "synthetic"},
            )
            self.assertEqual(run.path, root / "experiments/runs/candidate_a")
            for name in ("checkpoints", "metrics", "predictions", "artifacts"):
                self.assertTrue((run.path / name).is_dir())
            self.assertEqual(
                json.loads((run.path / "status.json").read_text())["status"],
                "running",
            )
            run.complete(metric=1.0)
            status = json.loads((run.path / "status.json").read_text())
            self.assertEqual(status, {"status": "completed", "metric": 1.0})
            with self.assertRaises(FileExistsError):
                ExperimentRun.create(
                    root,
                    "candidate_a",
                    config={},
                    provenance={},
                )
        for invalid in ("", ".", "..", "../bad", "a/b", "comparisons"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                experiment_run_path(Path("/tmp"), invalid)

    def test_condition_directory_is_nested_under_one_experiment(self):
        root = Path("/workspace")
        self.assertEqual(
            experiment_condition_path(root, "rear_flip_search", "rear_005_flip_020"),
            root
            / "experiments/runs/rear_flip_search/conditions/rear_005_flip_020",
        )

    def test_checkpoint_interpolation(self):
        left = {
            "weight": torch.tensor([0.0, 2.0], dtype=torch.float32),
            "counter": torch.tensor(3, dtype=torch.int64),
        }
        right = {
            "weight": torch.tensor([2.0, 6.0], dtype=torch.float32),
            "counter": torch.tensor(3, dtype=torch.int64),
        }
        result = interpolate_state_dict(left, right, 0.25)
        torch.testing.assert_close(result["weight"], torch.tensor([0.5, 3.0]))
        self.assertEqual(result["counter"].item(), 3)
        right["counter"] = torch.tensor(4, dtype=torch.int64)
        with self.assertRaisesRegex(ValueError, "Non-floating tensor differs"):
            interpolate_state_dict(left, right, 0.25)

    def test_full_range_forward_azimuth(self):
        matrices = Rotation.from_euler(
            "xyz",
            [[0, 0, 0], [0, 90, 0], [0, 135, 0], [0, -170, 0]],
            degrees=True,
        ).as_matrix()
        values = [forward_azimuth_degrees(matrix) for matrix in matrices]
        np.testing.assert_allclose(values, [0.0, 90.0, 135.0, -170.0], atol=1e-10)
        self.assertEqual(pose_band(values[0]), "front_lt60")
        self.assertEqual(pose_band(values[1]), "side_60_to_lt120")
        self.assertEqual(pose_band(values[2]), "rear_120_to_lt150")
        self.assertEqual(pose_band(values[3]), "rear_150_to_180")

        vertical = Rotation.from_euler("x", 90, degrees=True).as_matrix()
        with self.assertRaises(UndefinedAzimuthError):
            forward_azimuth_degrees(vertical)
        with self.assertRaisesRegex(ValueError, "orthonormal"):
            forward_azimuth_degrees(np.diag([1.0, 1.0, 2.0]))


if __name__ == "__main__":
    unittest.main()
