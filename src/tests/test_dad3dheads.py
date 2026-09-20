"""Synthetic validation archives and pose conventions; no real-data inference."""
from __future__ import annotations

import io
import importlib
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial.transform import Rotation
import torch

from hpe.data.dataset import ManifestDataset
from hpe.datasets.dad3dheads import full_range_euler, rotation_from_model_view
from hpe.evaluation import EvaluationConfig, compare_runs, evaluate
from hpe.evaluation.runner import _validate_config
from training.audit import BENCHMARKS, audit_manifest
from training.prepare_dad3dheads import extract_validation, item_file, prepare


def make_archive(path: Path, target: np.ndarray, *, bad_rotation=False):
    image = io.BytesIO()
    Image.new("RGB", (80, 60), (50, 100, 150)).save(image, format="PNG")
    matrix = np.eye(4)
    matrix[:3, :3] = target.T
    matrix[:3, 3] = [10, -20, -30]
    if bad_rotation:
        matrix[0, 0] = 9
    items = [{
        "item_id": str(i),
        "img_path": f"DAD-3DHeadsDataset/val/images/{i}.png",
        "annotation_path": f"DAD-3DHeadsDataset/val/annotations/{i}.json",
        "bbox": [5, 10, 50, 40],
    } for i in range(2)]
    files = {"val/val.json": json.dumps(items).encode(), "._val": b"ignored"}
    for i in range(2):
        files[f"val/images/{i}.png"] = image.getvalue()
        files[f"val/annotations/{i}.json"] = json.dumps({"model_view_matrix": matrix.tolist()}).encode()
    with tarfile.open(path, "w") as stream:
        for name, raw in files.items():
            entry = tarfile.TarInfo(name)
            entry.size = len(raw)
            stream.addfile(entry, io.BytesIO(raw))


class DADTests(unittest.TestCase):
    def test_coordinate_conversion_matches_official_dad_and_hpe_euler(self):
        flip = np.diag([1., -1., -1.])
        for angles in ([0, 0, 0], [20, 153, -37], [-30, -170, 15], [0, 90, 0], [0, -90, 0]):
            with self.subTest(angles=angles):
                target = Rotation.from_euler("xyz", angles, degrees=True).as_matrix()
                model_view = np.eye(4)
                model_view[:3, :3] = target.T
                model_view[:3, 3] = [12, -10, -90]
                actual = rotation_from_model_view(model_view)
                np.testing.assert_allclose(actual, target, atol=1e-12)
                np.testing.assert_allclose(
                    Rotation.from_euler("xyz", full_range_euler(actual), degrees=True).as_matrix(),
                    target, atol=1e-12)
                if abs(angles[1]) > 120:
                    self.assertGreaterEqual(abs(full_range_euler(actual)[1]), 120)
                # Independent upstream DAD-to-Euler recipe, including the 180° pitch shift.
                if abs(angles[1]) != 90:
                    official = flip @ model_view[:3, :3]
                    upstream_euler = Rotation.from_matrix(official.T).as_euler("xyz", degrees=True)
                    upstream_euler[0] -= 180
                    reference = Rotation.from_euler("xyz", upstream_euler, degrees=True).as_matrix()
                    np.testing.assert_allclose(actual, reference, atol=1e-12)

    def test_invalid_rotations_and_annotation_splits_are_rejected(self):
        for rotation in (np.zeros((4, 4)), np.diag([-1., 1, 1, 1]), np.full((4, 4), np.nan)):
            with self.assertRaises(ValueError):
                rotation_from_model_view(rotation)
        with self.assertRaises(ValueError):
            item_file({"item_id": "a", "img_path": "train/images/a.png"}, "img_path", "images", ".png", Path("/tmp"))

    def test_archive_rejects_traversal_links_and_other_splits_before_extracting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, kind in (("../escape", tarfile.REGTYPE), ("/escape", tarfile.REGTYPE),
                               ("train/image.png", tarfile.REGTYPE), ("val/link", tarfile.SYMTYPE)):
                with self.subTest(name=name):
                    archive = root / "bad.tar"
                    with tarfile.open(archive, "w") as stream:
                        entry = tarfile.TarInfo(name)
                        entry.type = kind
                        entry.linkname = "/tmp"
                        stream.addfile(entry)
                    with self.assertRaises(ValueError):
                        extract_validation(archive, root / "output")
                    self.assertFalse((root / "output").exists())

    def test_validation_prepare_audit_evaluate_and_compare(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "val.tar"
            target = Rotation.from_euler("xyz", [21, 153, -37], degrees=True).as_matrix()
            make_archive(archive, target)
            output = root / "datasets/prepared/dad_data"
            manifest = prepare(root, archive, output, expected_count=2)
            self.assertTrue(archive.is_file())
            rows = [json.loads(line) for line in manifest.read_text().splitlines()]
            self.assertEqual({row["split"] for row in rows}, {"validation"})
            self.assertEqual(rows[0]["crop_xyxy"], [5, 10, 55, 50])
            self.assertEqual(rows[0]["attributes"], {})
            self.assertFalse((output / "train").exists())
            dataset = ManifestDataset(root, manifest)
            image, metadata = dataset[0]
            self.assertEqual(tuple(image.shape), (3, 224, 224))
            np.testing.assert_allclose(metadata["rotation_matrix"].numpy(), target)
            self.assertTrue(np.isnan(metadata["occlusion_percent"]))
            dataset.close()
            with patch.dict(BENCHMARKS, dad3dheads={"instances": 2, "images": 2, "split": "validation"}):
                audit = audit_manifest(root, "dad3dheads", root / "locks", manifest=manifest)
            self.assertEqual(audit["yaw_bands"]["rear_ge120"], 2)

            class ConstantPose(torch.nn.Module):
                def forward(self, images):
                    return torch.from_numpy(target).to(images.device).expand(len(images), 3, 3)

            with (patch("hpe.evaluation.runner.SixDRepNet360", return_value=ConstantPose()),
                  patch("hpe.evaluation.runner.load_checkpoint", return_value={"sha256": "fixture"}),
                  patch("hpe.evaluation.runner.euler_degrees_to_matrix", side_effect=AssertionError("Must use matrix GT"))):
                results = [evaluate(EvaluationConfig(
                    project_root=root, checkpoint=root / "unused.pth", run_name=name,
                    output_root=root / "eval/reference/evaluations", datasets=("dad3dheads",), device="cpu", workers=0,
                    amp=False, protocol_version=2, manifest_paths={"dad3dheads": manifest},
                    image_lock_sha256={"dad3dheads": audit["image_lock_sha256"]},
                )) for name in ("baseline", "candidate")]
            predictions = pd.read_csv(results[0] / "predictions/dad3dheads.csv.gz")
            self.assertLess(predictions["geodesic_error_deg"].max(), 1e-8)
            self.assertTrue((predictions["gt_source_yaw_deg"].abs() >= 120).all())
            comparison = compare_runs(*results, root / "eval", bootstrap_repetitions=10)
            self.assertTrue(comparison.is_dir())
            reference = root / "eval/reference"
            (reference / "status.json").write_text(json.dumps({"status": "completed", "scope": "full"}))
            (reference / "audit.json").write_text(json.dumps({"datasets": [audit]}))
            trained = root / "ft_runs/trained"
            trained.mkdir(parents=True)
            (trained / "status.json").write_text(json.dumps({"status": "completed"}))
            post = importlib.import_module("training.evaluate")
            with (patch.object(post, "ROOT", root),
                  patch.object(post, "evaluate", return_value=results[1]) as inference,
                  patch.object(post, "compare_runs", return_value=comparison),
                  patch("torch.cuda.is_available", return_value=True),
                  patch("sys.argv", ["evaluate", "--run-id", "trained", "--baseline-run", "reference"])):
                post.main()
            config = inference.call_args.args[0]
            self.assertEqual(config.datasets, ("dad3dheads",))
            self.assertEqual(config.manifest_paths, {"dad3dheads": manifest})
            self.assertEqual(config.protocol_version, 2)
            self.assertFalse(config.amp)
            before = manifest.read_bytes()
            with self.assertRaises(FileExistsError):
                prepare(root, archive, output, expected_count=2)
            self.assertEqual(before, manifest.read_bytes())

    def test_failed_preparation_preserves_archive_and_reports_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "val.tar"
            make_archive(archive, np.eye(3), bad_rotation=True)
            output = root / "datasets/prepared/bad"
            with self.assertRaises(ValueError):
                prepare(root, archive, output, expected_count=2)
            self.assertEqual(json.loads((output / "status.json").read_text())["status"], "failed")
            self.assertTrue(archive.is_file())

    def test_dad_requires_protocol_v2(self):
        config = EvaluationConfig(Path("/tmp"), Path("/tmp/unused.pth"), "test", datasets=("dad3dheads",), device="cpu")
        with self.assertRaisesRegex(ValueError, "protocol v2"):
            _validate_config(config)


if __name__ == "__main__":
    unittest.main()
