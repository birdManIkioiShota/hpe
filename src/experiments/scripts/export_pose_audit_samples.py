"""Export deterministic rear-pose crop samples with their rotation metadata."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from hpe.data.dataset import read_rgb_image
from hpe.datasets.common import sha256_file, write_json_atomic
from experiments.common.pose import azimuth_side, forward_azimuth_degrees, pose_band
from experiments.common.run_directory import ExperimentRun
from training.prepare_data import ROOT, prepared_data_path
from training.vgg import read_jsonl


BUCKETS = (
    "negative:rear_120_to_lt150",
    "negative:rear_150_to_180",
    "positive:rear_120_to_lt150",
    "positive:rear_150_to_180",
)


def _rank(seed: int, instance_id: str) -> str:
    return hashlib.sha256(f"{seed}:{instance_id}".encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--data-id", default="vgg_data")
    parser.add_argument("--split", choices=("train", "dev", "holdout"), default="dev")
    parser.add_argument("--samples-per-bucket", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.samples_per_bucket <= 0:
        raise ValueError("samples-per-bucket must be positive")

    manifest = prepared_data_path(ROOT, args.data_id) / f"{args.split}.jsonl"
    candidates: dict[str, list[tuple[str, dict, float]]] = {name: [] for name in BUCKETS}
    for row in read_jsonl(manifest):
        azimuth = forward_azimuth_degrees(row["rotation_matrix"])
        band = pose_band(azimuth)
        if not band.startswith("rear_"):
            continue
        key = f"{azimuth_side(azimuth)}:{band}"
        if key not in candidates:
            raise ValueError(f"Unsupported rear pose bucket: {key}")
        candidates[key].append((_rank(args.seed, str(row["instance_id"])), row, azimuth))

    selected = {}
    for key, values in candidates.items():
        if not values:
            raise ValueError(f"No samples in rear pose bucket: {key}")
        selected[key] = sorted(values, key=lambda value: value[0])[:args.samples_per_bucket]

    run = ExperimentRun.create(
        ROOT,
        args.run_id,
        config={
            "kind": "pose_audit_sample_export",
            "data_id": args.data_id,
            "split": args.split,
            "samples_per_bucket": args.samples_per_bucket,
            "seed": args.seed,
        },
        provenance={
            "manifest": str(manifest.relative_to(ROOT)),
            "manifest_sha256": sha256_file(manifest),
        },
    )
    manifest_output = run.path / "artifacts" / "rear_samples.jsonl"
    exported = 0
    try:
        with manifest_output.open("x", encoding="utf-8") as stream:
            for bucket, values in selected.items():
                bucket_dir = run.path / "artifacts" / "rear_samples" / bucket.replace(":", "_")
                bucket_dir.mkdir(parents=True)
                for position, (_digest, row, azimuth) in enumerate(values):
                    image = read_rgb_image(
                        ROOT / row["image_path"],
                        expected_sha256=row.get("image_sha256"),
                        error_dir=run.path / "artifacts" / "image_read_errors",
                    )
                    crop = image.crop(tuple(int(value) for value in row["crop_xyxy"]))
                    safe_id = hashlib.sha256(str(row["instance_id"]).encode()).hexdigest()[:12]
                    output = bucket_dir / f"{position:03d}_{safe_id}.png"
                    crop.save(output)
                    record = {
                        "bucket": bucket,
                        "instance_id": row["instance_id"],
                        "source_image": row["image_path"],
                        "crop_image": output.relative_to(ROOT).as_posix(),
                        "crop_xyxy": row["crop_xyxy"],
                        "azimuth_deg": azimuth,
                        "rotation_matrix": row["rotation_matrix"],
                    }
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
                    exported += 1
        summary = {
            "exported": exported,
            "bucket_counts": {key: len(values) for key, values in selected.items()},
            "manifest": str(manifest_output.relative_to(ROOT)),
        }
        write_json_atomic(run.path / "metrics" / "summary.json", summary)
        run.complete(**summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except BaseException as error:
        run.fail(error)
        raise


if __name__ == "__main__":
    main()
