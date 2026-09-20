"""Three focused checks; no real-data training or benchmark inference."""
from __future__ import annotations

import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import torch
from torch import nn
from torch.utils.data import Dataset

from hpe.datasets.common import sha256_file
from hpe.geometry.rotations import rotation_matrix_from_6d
from hpe.models import SixDRepNet360, load_checkpoint
from training.prepare_data import prepare, source_group, split_group
from training.train import (atomic_save, make_optimizer, make_scheduler, optimizer_update,
                            restore_rng, rng_state, rotation_loss, training_mode, verify_data,
                            verify_image_files, make_loader)
from training.vgg import VGGDataset, annotation_heads


class VGGTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_annotation_dataset_and_group_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # Independently build FLAME from HPE, including rear and singular rotations.
            target = Rotation.from_euler("xyz", [[0, 0, 0], [21, 153, -37], [12, 90, 8]], degrees=True).as_matrix()
            flame = np.diag([1., -1., -1.]) @ target.transpose(0, 2, 1)
            params = np.zeros((3, 1, 413))
            params[:, 0, 400:403] = [123, 456, 789]  # Jaw must never enter the rotation.
            params[:, 0, 403:409] = np.concatenate([flame[:, :, 0], flame[:, :, 1]], axis=1)
            annotation = root / "pose.npz"
            np.savez(annotation, **{"3dmm_params": params, "extended_bbox": [[10, 20, 50, 60]] * 3})
            heads = annotation_heads(annotation)
            self.assertEqual(heads[0]["crop_xyxy"], [10, 20, 60, 80])
            np.testing.assert_allclose([head["rotation_matrix"] for head in heads], target, atol=1e-12)
            Image.new("RGB", (100, 100), (50, 100, 150)).save(root / "image.png")
            row = {"dataset": "vggheads", "image_path": "image.png", "instance_id": "test#1",
                   "image_sha256": sha256_file(root / "image.png"), **heads[1]}
            (root / "manifest.jsonl").write_text(json.dumps(row) + "\n")
            integrity_log = root / "integrity.jsonl"
            dataset = VGGDataset(root, root / "manifest.jsonl", augment=True, seed=42,
                                 integrity_log=integrity_log)
            image, matrix = dataset[(0, 2)]
            self.assertEqual(tuple(image.shape), (3, 224, 224))
            torch.testing.assert_close(image, dataset[(0, 2)][0])
            rng = random.Random("42:2:test#1")
            expected = torch.tensor(target[1], dtype=torch.float32)
            if rng.random() < .5:
                mirror = torch.diag(torch.tensor([-1., 1., 1.]))
                expected = mirror @ expected @ mirror
            torch.testing.assert_close(matrix, expected)
            Image.new("RGB", (100, 100), (200, 10, 10)).save(root / "image.png")
            with self.assertRaisesRegex(ValueError, "after 3 reads"):
                dataset[(0, 2)]
            self.assertEqual(len(integrity_log.read_text().splitlines()), 3)
            dataset.close()
        root = Path("/dataset")
        group = source_group(root / "small/shard1/annotations/image.npz", root)
        self.assertEqual(group, source_group(root / "small/shard2/annotations/image.npz", root))
        self.assertEqual(split_group(group, 42), split_group(group, 42))

    def test_actual_model_one_update_and_eval_compatibility(self):
        torch.manual_seed(3)
        model = SixDRepNet360()
        training_mode(model, head_only=False)
        config = {"backbone_lr": 1e-5, "head_lr": 1e-4, "weight_decay": 1e-4}
        optimizer = make_optimizer(model, config)
        scheduler = make_scheduler(optimizer, 10, 1)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        old_head, old_backbone = model.linear_reg.weight.detach().clone(), model.conv1.weight.detach().clone()
        bn_mean = model.bn1.running_mean.clone()
        prediction = model(torch.randn(2, 3, 224, 224))
        target = torch.from_numpy(Rotation.from_euler("xyz", [[20, 150, 10], [-10, -50, 12]], degrees=True).as_matrix()).float()
        loss = rotation_loss(prediction, target).mean()
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        norm = optimizer_update(model, optimizer, scheduler, scaler, 1.)
        self.assertTrue(np.isfinite(norm))
        self.assertFalse(torch.equal(old_head, model.linear_reg.weight))
        self.assertFalse(torch.equal(old_backbone, model.conv1.weight))
        torch.testing.assert_close(bn_mean, model.bn1.running_mean)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "best.pth"
            atomic_save(checkpoint, {"state_dict": model.state_dict(), "epoch": 1})
            loaded = SixDRepNet360()
            load_checkpoint(loaded, checkpoint)  # Existing benchmark loader, strict=True.
            torch.testing.assert_close(loaded.linear_reg.weight, model.linear_reg.weight)

    def test_optimizer_scheduler_rng_resume(self):
        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Linear(3, 8)
                self.linear_reg = nn.Linear(8, 6)

            def forward(self, values):
                return rotation_matrix_from_6d(self.linear_reg(self.backbone(values).tanh()))

        torch.manual_seed(1)
        config = {"backbone_lr": 1e-5, "head_lr": 1e-4, "weight_decay": 1e-4}
        model = Tiny()
        optimizer = make_optimizer(model, config)
        scheduler = make_scheduler(optimizer, 10, 2)
        scaler = torch.amp.GradScaler("cuda", enabled=False)

        def update(m, o, s):
            loss = rotation_loss(m(torch.randn(2, 3)), torch.eye(3).repeat(2, 1, 1)).mean()
            loss.backward()
            optimizer_update(m, o, s, scaler, 1.)

        update(model, optimizer, scheduler)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            atomic_save(path, {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                              "scheduler": scheduler.state_dict(), "rng": rng_state()})
            update(model, optimizer, scheduler)
            loaded = torch.load(path, weights_only=True)
            resumed = Tiny()
            resumed.load_state_dict(loaded["state_dict"], strict=True)
            optimizer2 = make_optimizer(resumed, config)
            scheduler2 = make_scheduler(optimizer2, 10, 2)
            optimizer2.load_state_dict(loaded["optimizer"])
            scheduler2.load_state_dict(loaded["scheduler"])
            restore_rng(loaded["rng"])
            update(resumed, optimizer2, scheduler2)
            for actual, expected in zip(resumed.parameters(), model.parameters()):
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            self.assertEqual(scheduler.state_dict(), scheduler2.state_dict())

        class Indexed(Dataset):
            def __len__(self):
                return 11

            def __getitem__(self, key):
                return key

        loader_config = {"seed": 42, "batch_size": 2, "workers": 0}
        complete = list(make_loader(Indexed(), loader_config, epoch=4))
        resumed_batches = list(make_loader(Indexed(), loader_config, epoch=4, start_batch=2))
        self.assertEqual(len(complete), 6)
        for resumed_batch, original_batch in zip(resumed_batches, complete[2:]):
            torch.testing.assert_close(resumed_batch[0], original_batch[0])
            torch.testing.assert_close(resumed_batch[1], original_batch[1])

    def test_prepare_without_benchmarks_and_manifest_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vgg = root / "datasets/VGGHeads"
            annotations, images = vgg / "small/annotations", vgg / "small/images"
            annotations.mkdir(parents=True)
            images.mkdir()
            names = {}
            for i in range(1000):
                name = f"source_{i}"
                names.setdefault(split_group(f"small:{name}", 42), name)
                if len(names) == 3:
                    break
            self.assertEqual(set(names), {"train", "dev", "holdout"})
            # Identical bytes in distinct source IDs must not trigger image matching.
            pixels = np.random.default_rng(1).integers(0, 256, (80, 80, 3), dtype=np.uint8)
            params = np.zeros((2, 1, 413))
            params[:, 0, 403:409] = [1, 0, 0, 0, -1, 0]
            for name in names.values():
                Image.fromarray(pixels).save(images / f"{name}.png")
                np.savez(annotations / f"{name}.npz", **{
                    "3dmm_params": params, "extended_bbox": [[5, 10, 50, 40]] * 2})
            output = root / "datasets/prepared/vgg_data"
            prepare(root, output, 42)
            metadata = verify_data(output, root)
            self.assertEqual(verify_image_files(output, root, metadata, workers=2), 3)
            self.assertEqual(metadata["duplicate_screening"], "not_performed")
            self.assertNotIn("benchmarks", metadata)
            self.assertFalse((output / "benchmark_signatures.jsonl").exists())
            self.assertEqual({split: item["heads"] for split, item in metadata["splits"].items()},
                             {"train": 2, "dev": 2, "holdout": 2})
            self.assertEqual(len({row["image_sha256"] for split in names
                                  for row in map(json.loads, (output / f"{split}.jsonl").read_text().splitlines())}), 1)
            before = (output / "train.jsonl").read_bytes()
            with self.assertRaises(FileExistsError):
                prepare(root, output, 42)
            self.assertEqual((output / "train.jsonl").read_bytes(), before)

            rows = list(map(json.loads, (output / "dev.jsonl").read_text().splitlines()))
            for row in rows:
                row["dataset"] = "dad3dheads"
            (output / "dev.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            metadata["splits"]["dev"]["sha256"] = sha256_file(output / "dev.jsonl")
            (output / "metadata.json").write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "Unexpected data source"):
                verify_data(output, root)


if __name__ == "__main__":
    unittest.main()
