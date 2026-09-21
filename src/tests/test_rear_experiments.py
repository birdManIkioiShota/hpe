from __future__ import annotations

import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import torch
from torch import nn

from experiments.common.sampling import build_epoch_order, scan_pose_buckets
from experiments.common.training import configure_update_scope, selection_score
from experiments.scripts.prepare_dad3dheads_train import prepare


def _write_manifest(path: Path, yaws: list[float]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for index, yaw in enumerate(yaws):
            rotation = Rotation.from_euler("xyz", [0, yaw, 0], degrees=True).as_matrix()
            stream.write(json.dumps({
                "rotation_matrix": rotation.tolist(),
                "instance_id": str(index),
            }) + "\n")


def _make_dad_train_archive(path: Path, count: int = 30) -> None:
    image = io.BytesIO()
    Image.new("RGB", (40, 40), (80, 120, 160)).save(image, format="PNG")
    items = []
    files: dict[str, bytes] = {}
    for index in range(count):
        item_id = str(index)
        yaw = -170 + (340 * index / max(1, count - 1))
        rotation = Rotation.from_euler("xyz", [0, yaw, 0], degrees=True).as_matrix()
        model_view = np.eye(4)
        model_view[:3, :3] = rotation.T
        items.append({
            "item_id": item_id,
            "img_path": f"DAD-3DHeadsDataset/train/images/{item_id}.png",
            "annotation_path": f"DAD-3DHeadsDataset/train/annotations/{item_id}.json",
            "bbox": [2, 3, 30, 30],
            "attributes": {"pose": "atypical" if abs(yaw) >= 120 else "front"},
        })
        files[f"DAD-3DHeadsDataset/train/images/{item_id}.png"] = image.getvalue()
        files[f"DAD-3DHeadsDataset/train/annotations/{item_id}.json"] = json.dumps({
            "model_view_matrix": model_view.tolist()
        }).encode()
    files["DAD-3DHeadsDataset/train/train.json"] = json.dumps(items).encode()
    with tarfile.open(path, "w") as stream:
        for directory in ("DAD-3DHeadsDataset", "DAD-3DHeadsDataset/train"):
            entry = tarfile.TarInfo(directory)
            entry.type = tarfile.DIRTYPE
            stream.addfile(entry)
        for name, raw in files.items():
            entry = tarfile.TarInfo(name)
            entry.size = len(raw)
            stream.addfile(entry, io.BytesIO(raw))


class RearExperimentTests(unittest.TestCase):
    def test_pose_balanced_epoch_order(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "train.jsonl"
            _write_manifest(
                manifest,
                [-170, -160, -140, -130, -80, 0, 80, 130, 140, 160, 170],
            )
            buckets = scan_pose_buckets([manifest])
            self.assertEqual(
                {name: len(values) for name, values in buckets.rear.items()},
                {
                    "negative:rear_120_to_lt150": 2,
                    "negative:rear_150_to_180": 2,
                    "positive:rear_120_to_lt150": 2,
                    "positive:rear_150_to_180": 2,
                },
            )
            order = build_epoch_order(
                buckets,
                num_samples=40,
                rear_fraction=0.4,
                seed=7,
                epoch=3,
            )
            self.assertEqual(len(order), 40)
            rear_indices = {index for values in buckets.rear.values() for index in values}
            self.assertEqual(sum(index in rear_indices for index, _ in order), 16)
            self.assertEqual(order, build_epoch_order(
                buckets, num_samples=40, rear_fraction=0.4, seed=7, epoch=3
            ))

    def test_dad_train_preparation_is_separate_from_official_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "train.tar"
            _make_dad_train_archive(archive)
            output = root / "datasets/prepared/dad_train"
            output.parent.mkdir(parents=True)
            train, dev = prepare(
                root,
                archive,
                output,
                seed=42,
                dev_fraction=0.2,
                expected_count=30,
            )
            train_rows = [json.loads(line) for line in train.read_text().splitlines()]
            dev_rows = [json.loads(line) for line in dev.read_text().splitlines()]
            self.assertEqual(len(train_rows) + len(dev_rows), 30)
            self.assertTrue(train_rows and dev_rows)
            self.assertEqual({row["dataset"] for row in train_rows + dev_rows}, {"dad3dheads"})
            self.assertEqual({row["split"] for row in train_rows}, {"train"})
            self.assertEqual({row["split"] for row in dev_rows}, {"dev"})
            self.assertFalse((output / "val").exists())
            metadata = json.loads((output / "metadata.json").read_text())
            self.assertEqual(metadata["role"], "experiment_training")
            self.assertEqual(metadata["source_split"], "train")

    def test_update_scope_and_selection_constraint(self):
        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer3 = nn.Linear(2, 2)
                self.layer4 = nn.Linear(2, 2)
                self.linear_reg = nn.Linear(2, 6)

        model = Tiny()
        configure_update_scope(model, "layer4")
        trainable = {name for name, value in model.named_parameters() if value.requires_grad}
        self.assertTrue(all(name.startswith(("layer4.", "linear_reg.")) for name in trainable))
        self.assertTrue(any(name.startswith("layer4.") for name in trainable))
        self.assertTrue(any(name.startswith("linear_reg.") for name in trainable))

        baseline = {
            "rear": {"p90_deg": 80.0, "over90_percent": 8.0, "mean_deg": 38.0},
            "retention": {"mean_deg": 6.0},
        }
        better = {
            "rear": {"p90_deg": 70.0, "over90_percent": 6.0, "mean_deg": 34.0},
            "retention": {"mean_deg": 5.9},
        }
        degraded = {
            "rear": {"p90_deg": 60.0, "over90_percent": 4.0, "mean_deg": 30.0},
            "retention": {"mean_deg": 6.2},
        }
        base_score = selection_score(baseline, baseline, retention_tolerance_deg=0.0)
        self.assertLess(selection_score(better, baseline, retention_tolerance_deg=0.0), base_score)
        self.assertGreater(selection_score(degraded, baseline, retention_tolerance_deg=0.0), base_score)


if __name__ == "__main__":
    unittest.main()
