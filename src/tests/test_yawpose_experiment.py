from __future__ import annotations

import math
import unittest

import torch

from experiments.common.yawpose import (
    circular_distance_degrees,
    reliability_records,
    signed_yaw_degrees,
    subset_records,
    yaw_loss_rad,
)
from experiments.common.yawpose_search import yawpose_conditions


class YawPoseExperimentTests(unittest.TestCase):
    def test_yaw_periodicity_and_loss(self):
        self.assertEqual(signed_yaw_degrees(270.0), -90.0)
        self.assertEqual(signed_yaw_degrees(180.0), -180.0)
        self.assertAlmostEqual(
            circular_distance_degrees(179.0, -179.0),
            2.0,
        )
        angle = math.radians(179.0)
        rotation = torch.tensor(
            [
                [math.cos(angle), 0.0, math.sin(angle)],
                [0.0, 1.0, 0.0],
                [-math.sin(angle), 0.0, math.cos(angle)],
            ],
            dtype=torch.float32,
        ).unsqueeze(0).requires_grad_()
        loss = yaw_loss_rad(
            rotation,
            torch.tensor([math.radians(-179.0)]),
        ).mean()
        self.assertAlmostEqual(
            math.degrees(float(loss.detach())),
            2.0,
            places=3,
        )
        loss.backward()
        self.assertTrue(torch.isfinite(rotation.grad).all())

    def test_reliability_ranking_and_nested_subsets(self):
        candidates = [
            {
                "instance_id": f"s{index}",
                "canonical_yaw_deg": 150.0,
                "source": "synthetic_005",
                "rear_bucket": "positive:rear_150_to_180",
            }
            for index in range(10)
        ]
        predictions = {
            "sixdrepnet360_base": {
                f"s{i}": 150.0 + i for i in range(10)
            },
            "semiuhpe_effnetv2s": {
                f"s{i}": 150.0 + i * 1.5 for i in range(10)
            },
            "whenet": {
                f"s{i}": 150.0 + i * 2.0 for i in range(10)
            },
        }
        rows = reliability_records(candidates, predictions)
        self.assertEqual(
            [row["instance_id"] for row in rows],
            [f"s{i}" for i in range(10)],
        )
        top20 = {
            row["instance_id"]
            for row in subset_records(rows, 0.2)
        }
        top40 = {
            row["instance_id"]
            for row in subset_records(rows, 0.4)
        }
        top100 = {
            row["instance_id"]
            for row in subset_records(rows, 1.0)
        }
        self.assertTrue(top20 < top40 < top100)

    def test_fixed_search_matrix(self):
        conditions = yawpose_conditions()
        self.assertEqual(
            [
                (
                    item.condition_id,
                    item.subset,
                    item.adoption_ratio,
                )
                for item in conditions
            ],
            [
                ("Y_base", "base", 0.0),
                ("Y_top20", "top020", 0.2),
                ("Y_top40", "top040", 0.4),
                ("Y_top60", "top060", 0.6),
                ("Y_top80", "top080", 0.8),
                ("Y_top100", "top100", 1.0),
            ],
        )


if __name__ == "__main__":
    unittest.main()
