from __future__ import annotations

import math
import sys
import unittest
from unittest import mock

import torch

from experiments.common.yawpose import (
    REAR_YAW_BINS,
    build_yawpose_epoch_plan,
    circular_distance_degrees,
    rear_yaw_bin,
    reliability_records,
    signed_yaw_degrees,
    subset_records,
    yaw_loss_rad,
)
from experiments.common.yawpose_search import yawpose_conditions
from experiments.scripts.infer_yawpose_semiuhpe import (
    _calibrate_sign as calibrate_semiuhpe_sign,
)

with mock.patch.dict(sys.modules, {"cv2": mock.MagicMock()}):
    from experiments.scripts.infer_yawpose_whenet import (
        _calibrate_sign as calibrate_whenet_sign,
    )


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

    def test_rear_yaw_bins_cover_signed_rear_range(self):
        cases = {
            -180.0: "negative:yaw_-180_to_lt-165",
            -165.0: "negative:yaw_-165_to_lt-150",
            -164.9: "negative:yaw_-165_to_lt-150",
            -150.0: "negative:yaw_-150_to_lt-135",
            -149.9: "negative:yaw_-150_to_lt-135",
            -135.0: "negative:yaw_-135_to_-120",
            -134.9: "negative:yaw_-135_to_-120",
            -120.0: "negative:yaw_-135_to_-120",
            120.0: "positive:yaw_120_to_lt135",
            135.0: "positive:yaw_135_to_lt150",
            150.0: "positive:yaw_150_to_lt165",
            165.0: "positive:yaw_165_to_lt180",
            179.9: "positive:yaw_165_to_lt180",
        }
        for yaw, expected in cases.items():
            with self.subTest(yaw=yaw):
                self.assertEqual(rear_yaw_bin(yaw), expected)
        self.assertEqual(len(REAR_YAW_BINS), 8)

    def test_reliability_ranking_is_stratified_and_nested(self):
        yaw_by_bin = {
            "negative:yaw_-180_to_lt-165": -172.0,
            "negative:yaw_-165_to_lt-150": -157.0,
            "negative:yaw_-150_to_lt-135": -142.0,
            "negative:yaw_-135_to_-120": -127.0,
            "positive:yaw_120_to_lt135": 127.0,
            "positive:yaw_135_to_lt150": 142.0,
            "positive:yaw_150_to_lt165": 157.0,
            "positive:yaw_165_to_lt180": 172.0,
        }
        candidates = []
        predictions = {
            "sixdrepnet360_base": {},
            "semiuhpe_effnetv2s": {},
            "whenet": {},
        }
        for bin_name, yaw in yaw_by_bin.items():
            for index in range(10):
                instance_id = f"{bin_name}:{index}"
                candidates.append(
                    {
                        "instance_id": instance_id,
                        "canonical_yaw_deg": yaw,
                        "source": "synthetic_005",
                        "rear_bucket": (
                            "negative:rear_150_to_180"
                            if yaw < -150
                            else "negative:rear_120_to_lt150"
                            if yaw < 0
                            else "positive:rear_150_to_180"
                            if yaw >= 150
                            else "positive:rear_120_to_lt150"
                        ),
                    }
                )
                predictions["sixdrepnet360_base"][instance_id] = (
                    yaw + index * 0.5
                )
                predictions["semiuhpe_effnetv2s"][instance_id] = (
                    yaw + index
                )
                predictions["whenet"][instance_id] = (
                    yaw + index * 1.5
                )

        rows = reliability_records(candidates, predictions)
        for bin_name in REAR_YAW_BINS:
            stratum = [
                row for row in rows
                if row["rear_yaw_bin"] == bin_name
            ]
            self.assertEqual(len(stratum), 10)
            self.assertEqual(
                [row["reliability_rank"] for row in stratum],
                list(range(1, 11)),
            )

        top20 = subset_records(rows, 0.2)
        top40 = subset_records(rows, 0.4)
        top100 = subset_records(rows, 1.0)
        self.assertEqual(len(top20), 16)
        self.assertEqual(len(top40), 32)
        self.assertEqual(len(top100), 80)
        ids20 = {row["instance_id"] for row in top20}
        ids40 = {row["instance_id"] for row in top40}
        ids100 = {row["instance_id"] for row in top100}
        self.assertTrue(ids20 < ids40 < ids100)
        for bin_name in REAR_YAW_BINS:
            self.assertEqual(
                sum(row["rear_yaw_bin"] == bin_name for row in top20),
                2,
            )

    def test_yawpose_epoch_plan_draws_every_sample_once(self):
        records = [
            {
                "instance_id": f"s{index}",
                "source": "synthetic_005",
                "rear_yaw_bin": REAR_YAW_BINS[index % len(REAR_YAW_BINS)],
            }
            for index in range(19)
        ]
        first = build_yawpose_epoch_plan(records, seed=42, epoch=0)
        second = build_yawpose_epoch_plan(records, seed=42, epoch=1)
        self.assertEqual(first.total_draws, len(records))
        self.assertEqual(first.unique_samples, len(records))
        self.assertEqual(first.repeated_draws, 0)
        self.assertEqual(
            sorted(index for index, _ in first.order),
            list(range(len(records))),
        )
        self.assertNotEqual(first.order, second.order)

    def test_fixed_search_matrix(self):
        conditions = yawpose_conditions()
        self.assertEqual(len(conditions), 12)
        self.assertEqual(
            {
                (item.hpe_regime, item.use_dad)
                for item in conditions
            },
            {
                ("vgg_only", False),
                ("vgg_plus_dad", True),
            },
        )
        for regime in ("vgg_only", "vgg_plus_dad"):
            subset = [
                item for item in conditions
                if item.hpe_regime == regime
            ]
            self.assertEqual(
                [item.adoption_ratio for item in subset],
                [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            )

    def test_teacher_sign_calibration_uses_direction_verified_sources(self):
        rows = []
        predictions = {}
        for index in range(10):
            for source, yaw in (
                ("intent_s004", 150.0),
                ("intent_s005", -150.0),
            ):
                instance_id = f"{source}_{index}"
                rows.append(
                    {
                        "instance_id": instance_id,
                        "label_source": source,
                        "canonical_yaw_deg": yaw,
                    }
                )
                predictions[instance_id] = yaw

        rows.append(
            {
                "instance_id": "excluded_operator_label",
                "label_source": "intent_operator_promoted",
                "canonical_yaw_deg": 150.0,
            }
        )
        predictions["excluded_operator_label"] = -150.0

        for calibrate in (calibrate_semiuhpe_sign, calibrate_whenet_sign):
            sign, validation = calibrate(rows, predictions)
            self.assertEqual(sign, 1)
            self.assertEqual(validation["count"], 20)
            self.assertEqual(validation["positive_count"], 10)
            self.assertEqual(validation["negative_count"], 10)
            self.assertEqual(
                validation["label_sources"],
                ["intent_s004", "intent_s005"],
            )


if __name__ == "__main__":
    unittest.main()
