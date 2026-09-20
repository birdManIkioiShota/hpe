"""VGGHeads adapter; no FLAME assets or benchmark labels are needed for training.

Annotation convention reference (pinned upstream):
https://github.com/ThomasAston/VGGHeads/tree/10190e90d7d047f7848472d8c913f430f7027cef
See yolo_head/flame.py, dataset_parsing.py and evaluation/evaluate_pose.py.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import random
from pathlib import Path
import time

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from hpe.data.dataset import ManifestDataset, evaluation_transform
from hpe.geometry.rotations import rotation_matrix_from_6d


def annotation_heads(path: Path) -> list[dict]:
    with np.load(path, allow_pickle=False) as data:
        params = np.asarray(data["3dmm_params"], dtype=np.float64)
        boxes = np.asarray(data["extended_bbox"], dtype=np.float64)
    if params.ndim != 3 or params.shape[1:] != (1, 413):
        raise ValueError(f"Unexpected 3dmm_params shape: {params.shape}")
    if boxes.shape != (len(params), 4) or not np.isfinite(params).all():
        raise ValueError("Invalid bbox/parameter shape or non-finite parameters")
    # 300 shape + 100 expression + 3 jaw, followed by two rotation columns.
    six = torch.from_numpy(params[:, 0, 403:409])
    if ((six[:, :3].norm(dim=1) < 1e-6) |
        (torch.cross(six[:, :3], six[:, 3:], dim=1).norm(dim=1) < 1e-6)).any():
        raise ValueError("Degenerate rotation basis")
    flame = rotation_matrix_from_6d(six)
    # Upstream: Euler_xyz(R_flame.T), then pitch -= 180 degrees.
    # The matrix form retains rear poses (no upstream >135-degree suppression).
    rotations = flame.transpose(1, 2) @ torch.diag(six.new_tensor([1, -1, -1]))
    if (not torch.isfinite(rotations).all() or
        not torch.allclose(rotations.transpose(1, 2) @ rotations,
                           torch.eye(3, dtype=rotations.dtype).expand_as(rotations), atol=1e-6, rtol=0)):
        raise ValueError("Rotation is not a finite orthonormal basis")
    heads = []
    for index, (box, rotation) in enumerate(zip(boxes, rotations)):
        x, y, w, h = box.tolist()  # Both small and large annotations use xywh.
        if not np.isfinite(box).all() or min(w, h) <= 0:
            raise ValueError(f"Invalid head bbox at {index}")
        crop = [int(x), int(y), int(x + w), int(y + h)]
        if crop[2] <= crop[0] or crop[3] <= crop[1]:
            raise ValueError("Empty integer crop")
        heads.append({"head_index": index, "crop_xyxy": crop,
                      "rotation_matrix": rotation.tolist()})
    if not heads:
        raise ValueError("No annotated heads")
    return heads


class VGGDataset(ManifestDataset):
    def __init__(self, root: Path, manifest: Path, *, augment: bool, seed: int,
                 integrity_log: Path | None = None, hash_attempts: int = 3):
        super().__init__(root, manifest)
        self.augment = augment
        self.seed = seed
        self.integrity_log = integrity_log
        if hash_attempts <= 0:
            raise ValueError("hash_attempts must be positive")
        self.hash_attempts = hash_attempts
        self.transform = evaluation_transform()

    def _integrity_event(self, payload: dict) -> None:
        if self.integrity_log is None:
            return
        payload = {"time_ns": time.time_ns(), **payload}
        # Each worker uses one O_APPEND write, so records do not overwrite each other.
        line = (json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
        descriptor = os.open(self.integrity_log, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o664)
        try:
            os.write(descriptor, line)
        finally:
            os.close(descriptor)

    def _verified_bytes(self, image_path: Path, record: dict) -> bytes:
        expected = record["image_sha256"]
        observations = []
        for attempt in range(1, self.hash_attempts + 1):
            raw = image_path.read_bytes()
            observed = hashlib.sha256(raw).hexdigest()
            stat = image_path.stat()
            observations.append(observed)
            if observed == expected:
                if attempt > 1:
                    self._integrity_event({
                        "event": "image_hash_recovered", "image_path": record["image_path"],
                        "attempt": attempt, "expected_sha256": expected,
                        "observed_sha256": observations, "size": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns,
                    })
                return raw
            self._integrity_event({
                "event": "image_hash_mismatch", "image_path": record["image_path"],
                "attempt": attempt, "expected_sha256": expected,
                "observed_sha256": observed, "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            })
            if attempt < self.hash_attempts:
                time.sleep(0.05 * attempt)
        raise ValueError(
            f"Image hash mismatch after {self.hash_attempts} reads: {record['image_path']} "
            f"expected={expected}, observed={observations}"
        )

    def __getitem__(self, key):
        index, epoch = key if isinstance(key, tuple) else (key, 0)
        record = self._record(index)

        if record["dataset"] != "vggheads":
            raise ValueError("The training adapter accepts only VGGHeads")

        image_path = self.project_root / record["image_path"]
        raw = self._verified_bytes(image_path, record)
        with Image.open(io.BytesIO(raw)) as source:
            image = source.convert("RGB").crop(record["crop_xyxy"])

        target = torch.tensor(record["rotation_matrix"], dtype=torch.float32)

        if self.augment:
            # Stateless augmentation gives identical resumed epochs with any worker count.
            rng = random.Random(f"{self.seed}:{epoch}:{record['instance_id']}")
            if rng.random() < 0.5:
                image = TF.hflip(image)
                mirror = torch.diag(torch.tensor([-1.0, 1.0, 1.0]))
                target = mirror @ target @ mirror

            for operation in (
                TF.adjust_brightness,
                TF.adjust_contrast,
                TF.adjust_saturation,
            ):
                image = operation(image, rng.uniform(0.8, 1.2))

        return self.transform(image), target


def read_jsonl(path: Path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)
