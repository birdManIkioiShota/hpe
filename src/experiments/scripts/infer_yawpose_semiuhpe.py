"""Generate fixed SemiUHPE EfficientNetV2-S yaw predictions for YawPose rear candidates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
from scipy.spatial.transform import Rotation
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

from experiments.common.yawpose import signed_yaw_degrees
from hpe.data.dataset import read_rgb_image
from hpe.datasets.common import sha256_file, write_json_atomic, write_jsonl_atomic
from training.prepare_data import ROOT, prepared_data_path


REQUIRED_REVISION = "c8f67102bf5aba8869b3f23453ac67599f21aa1f"
FIXTURE_YAWS = (0.0, 90.0, -90.0, 150.0, -150.0)


class _Dataset(Dataset):
    def __init__(self, manifest: Path) -> None:
        with manifest.open(encoding="utf-8") as stream:
            self.rows = [json.loads(line) for line in stream if line.strip()]
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image = read_rgb_image(ROOT / row["image_path"], expected_sha256=row.get("image_sha256"))
        return self.transform(image), str(row["instance_id"])


def _model() -> nn.Module:
    model = models.efficientnet_v2_s(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(1280, 512),
        nn.BatchNorm1d(512),
        nn.ReLU6(inplace=True),
        nn.Linear(512, 128),
        nn.BatchNorm1d(128),
        nn.ReLU6(inplace=True),
        nn.Linear(128, 9),
    )
    return model


def _project_matrix(raw: torch.Tensor) -> torch.Tensor:
    matrix = raw.reshape(-1, 3, 3).float()
    u, _, vh = torch.linalg.svd(matrix)
    sign = torch.linalg.det(u @ vh)
    u = u.clone()
    u[:, :, 2] *= sign[:, None]
    return u @ vh


def _matrix_to_yaw(rotation: np.ndarray) -> np.ndarray:
    euler = Rotation.from_matrix(np.transpose(rotation, (0, 2, 1))).as_euler("xyz", degrees=True)
    return np.asarray([signed_yaw_degrees(value) for value in euler[:, 1]], dtype=np.float64)


def _fixture_validation() -> dict:
    observed = []
    for yaw in FIXTURE_YAWS:
        expected_hpe = Rotation.from_euler("xyz", [0.0, yaw, 0.0], degrees=True).as_matrix()
        semi_matrix = expected_hpe.T[None, ...]
        restored = float(_matrix_to_yaw(semi_matrix)[0])
        if abs(signed_yaw_degrees(restored - yaw)) > 1e-5:
            raise ValueError(f"SemiUHPE yaw adapter fixture failed at {yaw}")
        observed.append(yaw)
    return {"passed": True, "yaw_degrees": observed, "method": "published SemiUHPE transpose+xyz yaw adapter"}


def _git_revision(repo: Path) -> str:
    return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-id", default="yawpose_rear")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="datasets/prepared/yawpose_teachers/semiuhpe_effnetv2s.jsonl")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    if _git_revision(repo) != REQUIRED_REVISION:
        raise ValueError(f"SemiUHPE repository must be at {REQUIRED_REVISION}")
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    manifest = prepared_data_path(ROOT, args.data_id) / "rear_candidates.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("SemiUHPE inference requires CUDA")

    network = _model()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = payload.get("model_state_dict", payload)
    network.load_state_dict(state, strict=True)
    network.to(device).eval()
    dataset = _Dataset(manifest)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    rows = []
    with torch.inference_mode():
        for images, instance_ids in tqdm(loader, desc="SemiUHPE teacher", unit="batch"):
            raw = network(images.to(device, non_blocking=True))
            rotation = _project_matrix(raw).cpu().numpy()
            yaw = _matrix_to_yaw(rotation)
            rows.extend({"instance_id": str(instance_id), "yaw_deg": float(value), "valid": True} for instance_id, value in zip(instance_ids, yaw))

    output = (ROOT / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(output, rows)
    sidecar = {
        "teacher_id": "semiuhpe_effnetv2s",
        "model": "SemiUHPE EfficientNetV2-S full-range",
        "implementation": "hnuzhy/SemiUHPE",
        "implementation_revision": REQUIRED_REVISION,
        "repository_path": str(repo),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "input_preprocess": "RGB resize 224x224, ImageNet normalization",
        "yaw_adapter": "matrix-Fisher SVD projection; transpose; scipy xyz Euler yaw; wrap [-180,180)",
        "fixture_validation": _fixture_validation(),
        "candidate_manifest_sha256": sha256_file(manifest),
        "prediction_sha256": sha256_file(output),
        "count": len(rows),
    }
    write_json_atomic(output.with_suffix(".manifest.json"), sidecar)
    print(output)


if __name__ == "__main__":
    main()
