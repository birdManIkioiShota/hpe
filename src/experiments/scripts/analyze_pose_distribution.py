"""Audit full-range pose distribution in prepared rotation-matrix manifests."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import json
from pathlib import Path

from hpe.datasets.common import sha256_file, write_json_atomic
from experiments.common.pose import (
    UndefinedAzimuthError,
    azimuth_side,
    forward_azimuth_degrees,
    pose_band,
)
from experiments.common.run_directory import ExperimentRun
from training.prepare_data import prepared_data_path
from training.vgg import read_jsonl


ROOT = Path(__file__).resolve().parents[3]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--data-id", default="vgg_data")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "dev", "holdout"],
        choices=("train", "dev", "holdout"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if len(set(args.splits)) != len(args.splits):
        raise ValueError("splits must not contain duplicates")

    data_dir = prepared_data_path(ROOT, args.data_id)
    manifests = {split: data_dir / f"{split}.jsonl" for split in args.splits}
    missing = [str(path) for path in manifests.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Prepared manifests are missing: {missing}")

    config = {
        "kind": "pose_distribution_audit",
        "data_id": args.data_id,
        "splits": args.splits,
        "azimuth_definition": "atan2((R @ +Z).x, (R @ +Z).z)",
        "bands_deg": [0, 60, 120, 150, 180],
    }
    provenance = {
        "manifests": {
            split: {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}
            for split, path in manifests.items()
        }
    }
    run = ExperimentRun.create(
        ROOT, args.run_id, config=config, provenance=provenance
    )
    counts: Counter[tuple[str, str, str]] = Counter()
    undefined: Counter[str] = Counter()
    total: Counter[str] = Counter()
    sample_path = run.path / "artifacts" / "pose_samples.csv.gz"

    try:
        with gzip.open(sample_path, "wt", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=(
                    "split",
                    "instance_id",
                    "image_path",
                    "azimuth_deg",
                    "abs_azimuth_deg",
                    "pose_band",
                    "azimuth_side",
                    "crop_width",
                    "crop_height",
                ),
            )
            writer.writeheader()
            for split, manifest in manifests.items():
                for row in read_jsonl(manifest):
                    total[split] += 1
                    try:
                        azimuth = forward_azimuth_degrees(row["rotation_matrix"])
                    except UndefinedAzimuthError:
                        undefined[split] += 1
                        continue
                    band = pose_band(azimuth)
                    side = azimuth_side(azimuth)
                    counts[(split, band, side)] += 1
                    x1, y1, x2, y2 = row["crop_xyxy"]
                    writer.writerow(
                        {
                            "split": split,
                            "instance_id": row["instance_id"],
                            "image_path": row["image_path"],
                            "azimuth_deg": f"{azimuth:.8f}",
                            "abs_azimuth_deg": f"{abs(azimuth):.8f}",
                            "pose_band": band,
                            "azimuth_side": side,
                            "crop_width": x2 - x1,
                            "crop_height": y2 - y1,
                        }
                    )

        rows = []
        for split in args.splits:
            for band in (
                "front_lt60",
                "side_60_to_lt120",
                "rear_120_to_lt150",
                "rear_150_to_180",
            ):
                for side in ("negative", "center", "positive"):
                    count = counts[(split, band, side)]
                    if count:
                        rows.append(
                            {
                                "split": split,
                                "pose_band": band,
                                "azimuth_side": side,
                                "count": count,
                            }
                        )
        summary = {
            "totals": dict(total),
            "undefined_azimuth": dict(undefined),
            "groups": rows,
        }
        write_json_atomic(run.path / "metrics" / "pose_distribution.json", summary)
        run.complete(
            samples=sum(total.values()),
            undefined_azimuth=sum(undefined.values()),
            artifact=str(sample_path.relative_to(ROOT)),
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except BaseException as error:
        run.fail(error)
        raise


if __name__ == "__main__":
    main()
