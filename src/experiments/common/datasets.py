from __future__ import annotations

import random
from bisect import bisect_right
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from experiments.common.pose import (
    UndefinedAzimuthError,
    azimuth_side,
    forward_azimuth_degrees,
    pose_band,
)
from hpe.data.dataset import ManifestDataset, evaluation_transform, read_rgb_image


_MIRROR = torch.diag(torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float32))


def validate_rotation(value: Any) -> torch.Tensor:
    rotation = torch.as_tensor(value, dtype=torch.float32)
    if rotation.shape != (3, 3) or not torch.isfinite(rotation).all():
        raise ValueError("rotation_matrix must be a finite 3x3 matrix")
    identity = torch.eye(3, dtype=rotation.dtype)
    if not torch.allclose(rotation.T @ rotation, identity, atol=1e-4, rtol=0):
        raise ValueError("rotation_matrix is not orthonormal")
    if not torch.isclose(torch.linalg.det(rotation), torch.tensor(1.0), atol=1e-4, rtol=0):
        raise ValueError("rotation_matrix determinant is not +1")
    return rotation


class PoseManifestDataset(ManifestDataset):
    """Training adapter for prepared manifests that contain direct rotation matrices."""

    def __init__(
        self,
        project_root: Path,
        manifest_path: Path,
        *,
        augment: bool,
        seed: int,
        allowed_datasets: set[str] | None = None,
        read_error_dir: Path | None = None,
    ) -> None:
        super().__init__(project_root, manifest_path, read_error_dir=read_error_dir)
        self.augment = augment
        self.seed = seed
        self.allowed_datasets = allowed_datasets
        self.transform = evaluation_transform()

    def __getitem__(self, key):
        index, epoch = key if isinstance(key, tuple) else (key, 0)
        record = self._record(index)
        dataset = str(record["dataset"])
        if self.allowed_datasets is not None and dataset not in self.allowed_datasets:
            raise ValueError(f"Unexpected training dataset: {dataset}")

        image_path = self.project_root / record["image_path"]
        image = read_rgb_image(
            image_path,
            expected_sha256=record.get("image_sha256"),
            error_dir=self.read_error_dir,
        )
        image = image.crop(tuple(int(value) for value in record["crop_xyxy"]))
        target = validate_rotation(record["rotation_matrix"])

        if self.augment:
            rng = random.Random(f"{self.seed}:{epoch}:{record['instance_id']}")
            if rng.random() < 0.5:
                image = TF.hflip(image)
                target = _MIRROR @ target @ _MIRROR
            for operation in (TF.adjust_brightness, TF.adjust_contrast, TF.adjust_saturation):
                image = operation(image, rng.uniform(0.8, 1.2))

        try:
            azimuth = forward_azimuth_degrees(target.double().numpy())
        except UndefinedAzimuthError:
            azimuth = float("nan")
            band = "undefined"
            side = "undefined"
        else:
            band = pose_band(azimuth)
            side = azimuth_side(azimuth)
        metadata = {
            "dataset": dataset,
            "instance_id": str(record["instance_id"]),
            "image_path": str(record["image_path"]),
            "azimuth_deg": float(azimuth),
            "pose_band": band,
            "azimuth_side": side,
            "is_rear": band.startswith("rear_"),
        }
        return self.transform(image), target, metadata


class MultiPoseDataset(Dataset):
    def __init__(self, datasets: list[PoseManifestDataset]) -> None:
        if not datasets:
            raise ValueError("At least one dataset is required")
        self.datasets = datasets
        self.ends: list[int] = []
        total = 0
        for dataset in datasets:
            total += len(dataset)
            self.ends.append(total)

    def __len__(self) -> int:
        return self.ends[-1]

    def __getitem__(self, key):
        index, epoch = key if isinstance(key, tuple) else (key, 0)
        dataset_index = bisect_right(self.ends, index)
        start = 0 if dataset_index == 0 else self.ends[dataset_index - 1]
        return self.datasets[dataset_index][(index - start, epoch)]

    def close(self) -> None:
        for dataset in self.datasets:
            dataset.close()
