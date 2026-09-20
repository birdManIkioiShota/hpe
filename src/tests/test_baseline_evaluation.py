from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from PIL import Image

from hpe.data.dataset import ManifestDataset
from hpe.datasets.common import sha256_file
from hpe.evaluation.comparison import _check_compatible
from hpe.evaluation.runner import EvaluationConfig, _validate_config, evaluate
from hpe.geometry import euler_degrees_to_matrix, vector_errors_degrees
from hpe.models import SixDRepNet360
from training.audit import BASE_SHA256, BENCHMARKS, audit_manifest, geometry_audit, validate_record
from training.evaluate_baseline import build_parser, run, validate_args


def record(index: int = 0) -> dict:
    return {
        "schema_version": 1, "dataset": "aflw2000", "split": "validation",
        "sample_id": str(index), "instance_id": str(index), "image_path": "image.png",
        "image_width": 80, "image_height": 60,
        "bbox_xyxy": [0, 0, 70, 50], "crop_xyxy": [-3.9, -7.1, 77.8, 93.9],
        "pitch_deg": 21.0, "yaw_deg": 153.0 if index == 0 else 0., "roll_deg": -37.0,
    }


class ConstantPose(torch.nn.Module):
    def forward(self, images):
        angles = torch.tensor([[21., 153., -37.]], device=images.device)
        return euler_degrees_to_matrix(angles).expand(len(images), 3, 3)


class AuditTests(unittest.TestCase):
    def test_analytic_geometry_and_preprocessing_contract(self):
        self.assertEqual(geometry_audit()["status"], "passed")

    def test_cropped_pixels_follow_integer_padding_resize_and_center_crop(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pixels = np.arange(60 * 80 * 3, dtype=np.uint8).reshape(60, 80, 3)
            original = Image.fromarray(pixels)
            original.save(root / "image.png")
            manifest = root / "manifest.jsonl"
            manifest.write_text(json.dumps(record()) + "\n")
            dataset = ManifestDataset(root, manifest)
            actual, _ = dataset[0]
            expected_image = original.crop((-3, -7, 77, 93)).resize(
                (256, 320), Image.Resampling.BILINEAR
            ).crop((16, 48, 240, 272))
            expected = torch.from_numpy(np.array(expected_image)).permute(2, 0, 1).float() / 255
            expected = (expected - torch.tensor([.485, .456, .406])[:, None, None]) / torch.tensor([.229, .224, .225])[:, None, None]
            torch.testing.assert_close(actual, expected)
            dataset.close()

    def test_invalid_labels_and_paths_are_rejected(self):
        root = Path("/tmp/baseline-test")
        for changes in ({"yaw_deg": float("nan")}, {"crop_xyxy": [1., 1., 1.1, 2.]},
                        {"bbox_xyxy": [0, 0, 0, 10]}, {"image_path": "../image.png"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_record({**record(), **changes}, "aflw2000", root)

    def test_manifest_hash_counts_images_and_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "datasets/prepared/aflw2000/manifest.jsonl"
            manifest.parent.mkdir(parents=True)
            Image.new("RGB", (80, 60)).save(root / "image.png")
            manifest.write_text(json.dumps(record()) + "\n")
            expected = {"sha256": sha256_file(manifest), "instances": 1, "images": 1, "split": "validation"}
            with patch.dict(BENCHMARKS, {"aflw2000": expected}):
                result = audit_manifest(root, "aflw2000", root / "lock")
                self.assertEqual(result["yaw_bands"]["rear_ge120"], 1)
                self.assertEqual(result["image_lock_sha256"], sha256_file(root / "lock/aflw2000_images.jsonl"))
                manifest.write_text(json.dumps(record()) + "\n" + json.dumps(record()) + "\n")
                with self.assertRaisesRegex(ValueError, "SHA-256"):
                    audit_manifest(root, "aflw2000", root / "lock2")
                expected.update(sha256=sha256_file(manifest), instances=2)
                with self.assertRaisesRegex(ValueError, "Duplicate instance"):
                    audit_manifest(root, "aflw2000", root / "lock3")

    def test_low_precision_head_output_is_normalized_in_fp32_without_new_weights(self):
        class HalfHead(torch.nn.Module):
            def forward(self, value):
                return torch.tensor([[1e-5, 2e-5, -1e-5, 3e-5, -1e-5, 1e-5]], dtype=torch.float16)

        model = SixDRepNet360()
        keys = set(model.state_dict())
        model.rotation_fp32 = False
        self.assertEqual(set(model.state_dict()), keys)
        model.rotation_fp32 = True
        # Isolate the precision boundary at the unchanged 6D head.
        for name in ("conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4", "avgpool"):
            setattr(model, name, torch.nn.Identity())
        model.linear_reg = HalfHead()
        matrix = model(torch.ones(1, 3, 1, 1))
        self.assertEqual(matrix.dtype, torch.float32)
        torch.testing.assert_close(matrix.transpose(1, 2) @ matrix, torch.eye(3)[None], atol=1e-6, rtol=0)
        torch.testing.assert_close(torch.linalg.det(matrix), torch.ones(1), atol=1e-6, rtol=0)


class BaselineLifecycleTests(unittest.TestCase):
    def args(self, root, *extra):
        return build_parser().parse_args(["--root", str(root), "--run-id", "test", "--datasets", "aflw2000", *extra])

    def test_explicit_dataset_and_safe_new_output_required(self):
        with self.assertRaises(SystemExit):
            build_parser().parse_args(["--run-id", "test"])
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(directory, "--audit-only")
            args.run_id = ".."
            with self.assertRaises(ValueError):
                validate_args(args)
            args.run_id = "test"
            output = validate_args(args)
            output.mkdir(parents=True)
            with self.assertRaises(FileExistsError):
                validate_args(args)
            args.run_id = "duplicate"
            args.datasets *= 2
            with self.assertRaises(ValueError):
                validate_args(args)

    def test_missing_cuda_does_not_fall_back_or_create_output(self):
        with tempfile.TemporaryDirectory() as directory, patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA"):
                validate_args(self.args(directory))
            self.assertFalse((Path(directory) / "eval").exists())

    def test_wrong_checkpoint_stops_before_audit_or_inference(self):
        with tempfile.TemporaryDirectory() as directory, patch("training.evaluate_baseline.sha256_file", return_value="wrong"), patch("training.evaluate_baseline.evaluate") as infer:
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                run(self.args(directory, "--audit-only"))
            infer.assert_not_called()
            self.assertFalse((Path(directory) / "eval").exists())

    def test_audit_only_never_runs_inference(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("training.evaluate_baseline.sha256_file", return_value=BASE_SHA256),
            patch("training.evaluate_baseline.geometry_audit", return_value={"status": "passed"}),
            patch("training.evaluate_baseline.SixDRepNet360", return_value=torch.nn.Linear(1, 1)),
            patch("training.evaluate_baseline.load_checkpoint", return_value={}),
            patch("training.evaluate_baseline.audit_manifest", return_value={"dataset": "aflw2000"}),
            patch("training.evaluate_baseline.evaluate") as infer,
        ):
            output = run(self.args(directory, "--audit-only"))
            infer.assert_not_called()
            status = json.loads((output / "status.json").read_text())
            self.assertEqual(status["status"], "completed")
            self.assertFalse(status["full_baseline"])
            self.assertFalse(status["training_performed"])

    def test_failure_preserves_status_and_does_not_start_inference(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("training.evaluate_baseline.sha256_file", return_value=BASE_SHA256),
            patch("training.evaluate_baseline.geometry_audit", side_effect=ValueError("bad geometry")),
            patch("training.evaluate_baseline.evaluate") as infer,
        ):
            with self.assertRaisesRegex(ValueError, "bad geometry"):
                run(self.args(directory, "--audit-only"))
            infer.assert_not_called()
            output = Path(directory) / "eval/test"
            self.assertEqual(json.loads((output / "status.json").read_text())["status"], "failed")
            self.assertIn("bad geometry", (output / "error.txt").read_text())

    def test_interruption_is_recorded(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("training.evaluate_baseline.sha256_file", return_value=BASE_SHA256),
            patch("training.evaluate_baseline.geometry_audit", side_effect=KeyboardInterrupt),
        ):
            with self.assertRaises(KeyboardInterrupt):
                run(self.args(directory, "--audit-only"))
            status = json.loads((Path(directory) / "eval/test/status.json").read_text())
            self.assertEqual(status["status"], "interrupted")


class RunnerTests(unittest.TestCase):
    def test_versioned_evaluation_writes_consistent_metrics_and_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (80, 60)).save(root / "image.png")
            manifest = root / "datasets/prepared/aflw2000/manifest.jsonl"
            manifest.parent.mkdir(parents=True)
            manifest.write_text("".join(json.dumps(record(i)) + "\n" for i in range(2)))
            events = []
            config = EvaluationConfig(root, root / "dummy.pth", "synthetic", output_root=root / "eval",
                                      datasets=("aflw2000",), device="cpu", batch_size=1, workers=0,
                                      amp=False, protocol_version=2)
            with patch("hpe.evaluation.runner.SixDRepNet360", return_value=ConstantPose()), patch("hpe.evaluation.runner.load_checkpoint", return_value={"sha256": "test"}):
                output = evaluate(config, event_callback=events.append)
            predictions = pd.read_csv(output / "predictions/aflw2000.csv.gz")
            self.assertEqual(len(predictions), 2)
            self.assertLess(predictions.iloc[0]["geodesic_error_deg"], 1e-5)
            self.assertAlmostEqual(predictions.iloc[1]["geodesic_error_deg"], 153., places=4)
            self.assertIn("pred_R_00", predictions)
            self.assertEqual(events[-1]["event"], "dataset_complete")
            self.assertEqual(events[-2]["processed"], 2)
            info = json.loads((output / "run.json").read_text())
            self.assertEqual(info["protocol_version"], 2)
            self.assertEqual(info["settings"]["vector_definition"], "rows")
            metrics = json.loads((output / "summary.json").read_text())[0]
            self.assertEqual(metrics["geodesic_error_deg_gt90_count"], 1)
            self.assertEqual(metrics["geodesic_error_deg_gt90_percent"], 50.)
            legacy = copy.deepcopy(info)
            legacy["protocol_version"] = 1
            with self.assertRaisesRegex(ValueError, "protocol"):
                _check_compatible(legacy, info)
            mixed_precision = copy.deepcopy(info)
            mixed_precision["settings"]["amp"] = True
            with self.assertRaisesRegex(ValueError, "amp"):
                _check_compatible(info, mixed_precision)
            different_images = copy.deepcopy(info)
            different_images["datasets"][0]["image_lock_sha256"] = "changed"
            with self.assertRaisesRegex(ValueError, "image locks"):
                _check_compatible(info, different_images)

    def test_invalid_rotation_fails_without_deleting_partial_results(self):
        class InvalidPose(torch.nn.Module):
            def forward(self, images):
                return torch.zeros(len(images), 3, 3)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (80, 60)).save(root / "image.png")
            manifest = root / "datasets/prepared/aflw2000/manifest.jsonl"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(json.dumps(record()) + "\n")
            config = EvaluationConfig(
                root, root / "dummy", "bad", output_root=root / "eval",
                datasets=("aflw2000",), device="cpu", workers=0,
                protocol_version=2, preserve_partial=True,
            )
            with (
                patch("hpe.evaluation.runner.SixDRepNet360", return_value=InvalidPose()),
                patch("hpe.evaluation.runner.load_checkpoint", return_value={}),
                self.assertRaisesRegex(RuntimeError, "Invalid predicted rotation"),
            ):
                evaluate(config)
            self.assertTrue((root / "eval/.bad.partial").exists())
            self.assertFalse((root / "eval/bad").exists())

    def test_legacy_vector_metrics_remain_column_based(self):
        pred = euler_degrees_to_matrix(torch.tensor([[21., 153., -37.]], dtype=torch.float64))
        gt = euler_degrees_to_matrix(torch.tensor([[7., -52., 11.]], dtype=torch.float64))
        expected = torch.rad2deg(torch.acos((pred * gt).sum(dim=1).clamp(-1, 1)))
        torch.testing.assert_close(vector_errors_degrees(pred, gt), expected)
        self.assertGreater(float((expected - vector_errors_degrees(pred, gt, rows=True)).abs().max()), 1.)

    def test_incomplete_runs_are_not_deleted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            partial = root / ".baseline.partial"
            partial.mkdir()
            (partial / "keep.txt").write_text("keep")
            config = EvaluationConfig(root, root / "dummy", "baseline", output_root=root, device="cpu")
            with self.assertRaises(FileExistsError):
                evaluate(config)
            self.assertEqual((partial / "keep.txt").read_text(), "keep")

    def test_invalid_evaluation_configs_are_rejected(self):
        for changes in ({"run_name": ".."}, {"datasets": ()}, {"datasets": ("aflw2000", "aflw2000")},
                        {"max_samples": 0}, {"protocol_version": 3}):
            config = EvaluationConfig(**{"project_root": Path("/tmp"), "checkpoint": Path("dummy"),
                                         "run_name": "valid", "device": "cpu", **changes})
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                _validate_config(config)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
