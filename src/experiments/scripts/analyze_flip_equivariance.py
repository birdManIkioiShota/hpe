"""Measure horizontal-flip equivariance by pose band for a compatible checkpoint."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from experiments.common.datasets import PoseManifestDataset
from experiments.common.run_directory import ExperimentRun
from hpe.datasets.common import sha256_file, write_json_atomic
from hpe.geometry.rotations import geodesic_error_degrees
from hpe.models import SixDRepNet360, load_checkpoint
from training.prepare_data import ROOT, prepared_data_path


MIRROR = torch.diag(torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64))


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-id", default="vgg_data")
    parser.add_argument("--split", choices=("train", "dev", "holdout"), default="dev")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    manifest = prepared_data_path(ROOT, args.data_id) / f"{args.split}.jsonl"
    checkpoint = _resolve(args.checkpoint)

    run = ExperimentRun.create(
        ROOT,
        args.run_id,
        config={
            "kind": "horizontal_flip_equivariance",
            "checkpoint": str(checkpoint),
            "data_id": args.data_id,
            "split": args.split,
            "batch_size": args.batch_size,
            "workers": args.workers,
            "device": args.device,
        },
        provenance={
            "checkpoint_sha256": sha256_file(checkpoint),
            "manifest": str(manifest.relative_to(ROOT)),
            "manifest_sha256": sha256_file(manifest),
        },
    )
    dataset = PoseManifestDataset(
        ROOT,
        manifest,
        augment=False,
        seed=0,
        allowed_datasets={"vggheads"},
        read_error_dir=run.path / "artifacts" / "image_read_errors",
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    model = SixDRepNet360(rotation_fp32=True)
    load_checkpoint(model, checkpoint)
    model.to(device).eval()
    mirror = MIRROR.to(device)

    rows = []
    try:
        with torch.inference_mode():
            for images, _target, metadata in tqdm(loader, desc="Flip equivariance", unit="batch"):
                images = images.to(device, non_blocking=True)
                original = model(images).double()
                flipped = model(torch.flip(images, dims=[3])).double()
                restored = mirror @ flipped @ mirror
                error = geodesic_error_degrees(original, restored, stable=True).cpu().tolist()
                for index, value in enumerate(error):
                    rows.append(
                        {
                            "dataset": metadata["dataset"][index],
                            "instance_id": metadata["instance_id"][index],
                            "pose_band": metadata["pose_band"][index],
                            "azimuth_side": metadata["azimuth_side"][index],
                            "azimuth_deg": float(metadata["azimuth_deg"][index]),
                            "equivariance_error_deg": float(value),
                        }
                    )

        output = run.path / "predictions" / "flip_equivariance.csv.gz"
        with gzip.open(output, "wt", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        groups: dict[str, list[float]] = {}
        for row in rows:
            names = (
                "overall",
                row["pose_band"],
                f"{row['azimuth_side']}:{row['pose_band']}",
            )
            for name in names:
                groups.setdefault(name, []).append(row["equivariance_error_deg"])
        summary = {}
        for name, values in groups.items():
            tensor = torch.tensor(values, dtype=torch.float64)
            summary[name] = {
                "count": len(values),
                "mean_deg": float(tensor.mean()),
                "median_deg": float(tensor.quantile(0.5)),
                "p90_deg": float(tensor.quantile(0.9)),
                "over10_percent": float((tensor > 10).double().mean() * 100),
            }
        write_json_atomic(run.path / "metrics" / "flip_equivariance.json", summary)
        run.complete(samples=len(rows), checkpoint_sha256=sha256_file(checkpoint))
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except BaseException as error:
        run.fail(error)
        raise
    finally:
        dataset.close()


if __name__ == "__main__":
    main()
