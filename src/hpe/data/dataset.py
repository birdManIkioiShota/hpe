from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import time
from typing import Any, BinaryIO

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def read_rgb_image(path: Path, *, expected_sha256: str | None = None,
                   error_dir: Path | None = None, attempts: int = 3) -> Image.Image:
    """Decode verified bytes, reopening after read/decode failures; never accept truncation."""
    if attempts < 1:
        raise ValueError("attempts must be positive")

    def record(kind: str, **details: Any) -> None:
        if error_dir is not None:
            error_dir.mkdir(parents=True, exist_ok=True)
            # A separate file per process avoids interleaving DataLoader-worker writes.
            with (error_dir / f"worker_{os.getpid()}.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"time_ns": time.time_ns(), "event": kind,
                    "image_path": str(path), "expected_sha256": expected_sha256, **details}) + "\n")

    for attempt in range(1, attempts + 1):
        raw, digest = None, None
        try:
            raw = path.read_bytes()
            if expected_sha256 is not None:
                digest = hashlib.sha256(raw).hexdigest()
                if digest != expected_sha256:
                    raise OSError("Image SHA-256 differs from the reference image lock")
            with Image.open(io.BytesIO(raw)) as source:
                image = source.convert("RGB")  # Force full decoding before returning.
                image.load()
        except (OSError, ValueError, SyntaxError) as error:
            if raw is not None and digest is None:
                digest = hashlib.sha256(raw).hexdigest()
            try:
                stat = path.stat()
                file_state = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
            except OSError:
                file_state = {}
            record("image_read_failed", attempt=attempt, observed_sha256=digest,
                   bytes_read=None if raw is None else len(raw), error=str(error), **file_state)
            if attempt == attempts:
                raise OSError(f"Cannot decode image after {attempts} reads: {path}; "
                              f"expected_sha256={expected_sha256}, observed_sha256={digest}; {error}") from error
            time.sleep(.05 * attempt)
        else:
            if attempt > 1:
                record("image_read_recovered", attempt=attempt,
                       observed_sha256=digest or hashlib.sha256(raw).hexdigest())
            return image
    raise AssertionError("Unreachable")


def evaluation_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


class ManifestDataset(Dataset[tuple[torch.Tensor, dict[str, Any]]]):
    """Random-access JSONL dataset without holding a large manifest in memory."""

    def __init__(self, project_root: Path, manifest_path: Path, max_samples: int | None = None,
                 *, image_hashes: dict[str, str] | None = None,
                 read_error_dir: Path | None = None) -> None:
        self.project_root = project_root.resolve()
        self.manifest_path = manifest_path.resolve()
        self.transform = evaluation_transform()
        self.image_hashes = image_hashes
        self.read_error_dir = read_error_dir
        self._stream: BinaryIO | None = None
        offsets: list[int] = []
        with self.manifest_path.open("rb") as stream:
            while stream.readline():
                offsets.append(stream.tell())
        self.offsets = [0, *offsets[:-1]] if offsets else []
        if max_samples is not None:
            if max_samples <= 0:
                raise ValueError("max_samples must be positive.")
            self.offsets = self.offsets[:max_samples]

    def __len__(self) -> int:
        return len(self.offsets)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_stream"] = None
        return state

    def close(self) -> None:
        stream = getattr(self, "_stream", None)
        if stream is not None:
            stream.close()
            self._stream = None

    def __del__(self) -> None:
        self.close()

    def _record(self, index: int) -> dict[str, Any]:
        if self._stream is None:
            self._stream = self.manifest_path.open("rb")
        self._stream.seek(self.offsets[index])
        return json.loads(self._stream.readline())

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, Any]]:
        record = self._record(index)
        image_path = self.project_root / record["image_path"]
        expected = self.image_hashes[record["image_path"]] if self.image_hashes is not None else None
        image = read_rgb_image(image_path, expected_sha256=expected, error_dir=self.read_error_dir)
        crop = tuple(int(value) for value in record["crop_xyxy"])
        tensor = self.transform(image.crop(crop))

        bbox = record.get("bbox_xyxy")
        if bbox is None:
            x, y, width, height = record["bbox_xywh"]
            bbox = [x, y, x + width, y + height]
        metadata: dict[str, Any] = {
            "dataset": record["dataset"],
            "sample_id": str(record["sample_id"]),
            "instance_id": str(record["instance_id"]),
            "image_path": record["image_path"],
            "pitch_deg": float(record["pitch_deg"]),
            "yaw_deg": float(record["yaw_deg"]),
            "roll_deg": float(record["roll_deg"]),
            "bbox_xyxy": torch.tensor(bbox, dtype=torch.float32),
            "crop_xyxy": torch.tensor(record["crop_xyxy"], dtype=torch.float32),
            "occlusion_percent": float(record.get("occlusion_percent", float("nan"))),
        }
        if record["dataset"] == "dad3dheads":
            if record["split"] != "validation":
                raise ValueError("DAD-3DHeads benchmark accepts only validation")
            from hpe.datasets.dad3dheads import validate_rotation
            metadata["rotation_matrix"] = torch.from_numpy(validate_rotation(record["rotation_matrix"]).copy())
        return tensor, metadata
