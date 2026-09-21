"""Run the fixed 3x3 rear-sampling and flip-consistency experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from tqdm import tqdm

from experiments.common.rear_flip_search import (
    condition_records,
    search_conditions,
    summarize_search,
)
from experiments.common.run_directory import (
    ExperimentRun,
    SEARCH_SUBDIRECTORIES,
    experiment_condition_path,
    experiment_run_path,
)
from experiments.common.sampling import scan_pose_buckets
from hpe.datasets.common import sha256_file
from training.audit import BASE_CHECKPOINT, BASE_SHA256
from training.prepare_data import ROOT, prepared_data_path


SOURCE_FILES = (
    "src/experiments/common/datasets.py",
    "src/experiments/common/pose.py",
    "src/experiments/common/rear_flip_search.py",
    "src/experiments/common/run_directory.py",
    "src/experiments/common/sampling.py",
    "src/experiments/common/training.py",
    "src/experiments/scripts/run_rear_flip_search.py",
    "src/experiments/scripts/train_rear_balanced.py",
    "src/hpe/data/dataset.py",
    "src/hpe/geometry/rotations.py",
    "src/hpe/models/checkpoint.py",
    "src/hpe/models/sixdrepnet360.py",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="rear_flip_search")
    parser.add_argument("--vgg-data-id", default="vgg_data")
    parser.add_argument("--dad-data-id", default="dad3dheads_train")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--update-scope", choices=("head", "layer4", "all"), default="layer4")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--samples-per-epoch", type=int)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--accumulation", type=int, default=2)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--backbone-lr", type=float, default=1e-6)
    parser.add_argument("--head-lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-updates", type=int, default=100)
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--rear-supervised-weight", type=float, default=1.0)
    parser.add_argument("--replay-supervised-weight", type=float, default=1.0)
    parser.add_argument("--retain-distill-weight", type=float, default=1.0)
    parser.add_argument("--retention-tolerance-deg", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _manifests(vgg_data_id: str, dad_data_id: str) -> tuple[list[Path], list[Path]]:
    vgg = prepared_data_path(ROOT, vgg_data_id)
    dad = prepared_data_path(ROOT, dad_data_id)
    train = [vgg / "train.jsonl", dad / "train.jsonl"]
    dev = [vgg / "dev.jsonl", dad / "dev.jsonl"]
    for path in train + dev:
        if not path.is_file():
            raise FileNotFoundError(path)
    return train, dev


def _condition_command(args: argparse.Namespace, condition, *, resume: bool) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "experiments.scripts.train_rear_balanced",
        "--run-id",
        args.run_id,
        "--condition-id",
        condition.condition_id,
        "--vgg-data-id",
        args.vgg_data_id,
        "--dad-data-id",
        args.dad_data_id,
        "--device",
        args.device,
        "--precision",
        args.precision,
        "--update-scope",
        args.update_scope,
        "--epochs",
        str(args.epochs),
        "--rear-fraction",
        repr(condition.rear_fraction),
        "--rear-bucket-policy",
        "proportional",
        "--batch-size",
        str(args.batch_size),
        "--accumulation",
        str(args.accumulation),
        "--workers",
        str(args.workers),
        "--backbone-lr",
        repr(args.backbone_lr),
        "--head-lr",
        repr(args.head_lr),
        "--weight-decay",
        repr(args.weight_decay),
        "--warmup-updates",
        str(args.warmup_updates),
        "--clip-norm",
        repr(args.clip_norm),
        "--rear-supervised-weight",
        repr(args.rear_supervised_weight),
        "--replay-supervised-weight",
        repr(args.replay_supervised_weight),
        "--retain-distill-weight",
        repr(args.retain_distill_weight),
        "--rear-flip-consistency-weight",
        repr(condition.flip_consistency_weight),
        "--retention-tolerance-deg",
        repr(args.retention_tolerance_deg),
        "--seed",
        str(args.seed),
    ]
    if args.samples_per_epoch is not None:
        command.extend(("--samples-per-epoch", str(args.samples_per_epoch)))
    if resume:
        command.append("--resume")
    return command


def main() -> None:
    args = build_parser().parse_args()
    train_manifests, dev_manifests = _manifests(args.vgg_data_id, args.dad_data_id)
    buckets = scan_pose_buckets(train_manifests)
    conditions = search_conditions(buckets.natural_rear_fraction)
    config = {
        "kind": "rear_sampling_flip_consistency_search",
        "epochs": args.epochs,
        "sampling_policy": "proportional_to_source_rear_bucket_counts",
        "natural_rear_fraction": buckets.natural_rear_fraction,
        "conditions": condition_records(conditions),
        "shared_training": {
            key: value
            for key, value in vars(args).items()
            if key not in {"run_id", "dry_run"}
        },
    }
    if args.dry_run:
        print(json.dumps(config, ensure_ascii=False, indent=2))
        return

    checkpoint = ROOT / BASE_CHECKPOINT
    if sha256_file(checkpoint) != BASE_SHA256:
        raise ValueError("Base checkpoint SHA-256 mismatch")
    provenance = {
        "base_checkpoint": {"path": BASE_CHECKPOINT, "sha256": BASE_SHA256},
        "train_manifests": {
            str(path.relative_to(ROOT)): sha256_file(path) for path in train_manifests
        },
        "dev_manifests": {
            str(path.relative_to(ROOT)): sha256_file(path) for path in dev_manifests
        },
        "source_sha256": {path: sha256_file(ROOT / path) for path in SOURCE_FILES},
        "training_bucket_counts": {
            "retention": len(buckets.retention),
            **{name: len(values) for name, values in buckets.rear.items()},
        },
    }
    run_path = experiment_run_path(ROOT, args.run_id)
    if run_path.exists():
        run = ExperimentRun.open(run_path)
        if json.loads((run_path / "config.json").read_text()) != config:
            raise ValueError("Experiment config changed; resume refused")
        if json.loads((run_path / "provenance.json").read_text()) != provenance:
            raise ValueError("Experiment inputs or source changed; resume refused")
        status = json.loads((run_path / "status.json").read_text())
        if status.get("status") == "completed":
            print(f"Experiment already completed: {run_path.relative_to(ROOT)}")
            return
        run.write_status("running", resumed=True)
        run.event("resumed")
    else:
        run = ExperimentRun.create_at(
            run_path,
            config=config,
            provenance=provenance,
            subdirectories=SEARCH_SUBDIRECTORIES,
        )

    completed = 0
    progress = tqdm(conditions, desc="Rear/flip search", unit="condition")
    try:
        for condition in progress:
            progress.set_postfix(condition=condition.condition_id, completed=completed)
            condition_path = experiment_condition_path(
                ROOT, args.run_id, condition.condition_id
            )
            resume = False
            if condition_path.exists():
                condition_status = json.loads(
                    (condition_path / "status.json").read_text()
                )
                if condition_status.get("status") == "completed":
                    completed += 1
                    continue
                resume = True
            run.event(
                "condition_started",
                condition_id=condition.condition_id,
                resume=resume,
            )
            subprocess.run(_condition_command(args, condition, resume=resume), check=True)
            completed += 1
            run.write_status(
                "running",
                completed_conditions=completed,
                total_conditions=len(conditions),
                last_completed_condition=condition.condition_id,
            )
            run.event("condition_completed", condition_id=condition.condition_id)

        summary_json, summary_csv, effects_json = summarize_search(run.path)
        run.complete(
            completed_conditions=completed,
            total_conditions=len(conditions),
            comparison=str(summary_json.relative_to(ROOT)),
            comparison_csv=str(summary_csv.relative_to(ROOT)),
            effects=str(effects_json.relative_to(ROOT)),
        )
    except BaseException as error:
        run.fail(error)
        raise


if __name__ == "__main__":
    main()
