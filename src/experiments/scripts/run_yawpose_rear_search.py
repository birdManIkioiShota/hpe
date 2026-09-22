"""Run the fixed YawPose rear-yaw adoption search."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys

from tqdm import tqdm

from experiments.common.run_directory import (
    ExperimentRun,
    SEARCH_SUBDIRECTORIES,
    experiment_condition_path,
    experiment_run_path,
)
from experiments.common.yawpose_search import (
    condition_records,
    summarize_search,
    yawpose_conditions,
)
from hpe.datasets.common import sha256_file
from training.audit import BASE_CHECKPOINT, BASE_SHA256
from training.prepare_data import ROOT, prepared_data_path


SOURCE_FILES = (
    "src/experiments/common/yawpose.py",
    "src/experiments/common/yawpose_search.py",
    "src/experiments/scripts/run_yawpose_rear_search.py",
    "src/experiments/scripts/train_yawpose_rear.py",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="yawpose_rear_search")
    parser.add_argument("--reliability-run", default="yawpose_reliability")
    parser.add_argument("--vgg-data-id", default="vgg_data")
    parser.add_argument("--dad-data-id", default="dad3dheads_train")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument(
        "--update-scope",
        choices=("head", "layer4", "all"),
        default="layer4",
    )
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
    parser.add_argument("--retain-distill-weight", type=float, default=1.0)
    parser.add_argument("--rear-flip-consistency-weight", type=float, default=0.2)
    parser.add_argument("--yawpose-weight", type=float, default=0.2)
    parser.add_argument("--retention-tolerance-deg", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _condition_command(
    args: argparse.Namespace,
    condition,
    *,
    resume: bool,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "experiments.scripts.train_yawpose_rear",
        "--run-id",
        args.run_id,
        "--condition-id",
        condition.condition_id,
        "--subset",
        condition.subset,
        "--reliability-run",
        args.reliability_run,
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
        "--retain-distill-weight",
        repr(args.retain_distill_weight),
        "--rear-flip-consistency-weight",
        repr(args.rear_flip_consistency_weight),
        "--yawpose-weight",
        repr(args.yawpose_weight),
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
    conditions = yawpose_conditions()
    reliability_dir = experiment_run_path(ROOT, args.reliability_run)
    reliability_status = reliability_dir / "status.json"
    if (
        not reliability_status.is_file()
        or json.loads(reliability_status.read_text()).get("status") != "completed"
    ):
        raise ValueError("completed reliability precompute run is required")
    reliability_summary = reliability_dir / "metrics" / "summary.json"
    config = {
        "kind": "yawpose_rear_adoption_search",
        "epochs": args.epochs,
        "conditions": condition_records(),
        "reliability_run": args.reliability_run,
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
        raise ValueError("base checkpoint SHA-256 mismatch")
    vgg_dir = prepared_data_path(ROOT, args.vgg_data_id)
    dad_dir = prepared_data_path(ROOT, args.dad_data_id)
    training_manifests = [
        vgg_dir / "train.jsonl",
        vgg_dir / "dev.jsonl",
        dad_dir / "train.jsonl",
        dad_dir / "dev.jsonl",
    ]
    for manifest in training_manifests:
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
    subset_paths = {
        condition.subset: (
            reliability_dir
            / "predictions"
            / "subsets"
            / f"{condition.subset}.jsonl"
        )
        for condition in conditions
        if condition.subset != "base"
    }
    for subset_path in subset_paths.values():
        if not subset_path.is_file():
            raise FileNotFoundError(subset_path)

    provenance = {
        "base_checkpoint": {
            "path": BASE_CHECKPOINT,
            "sha256": BASE_SHA256,
        },
        "training_manifests": {
            str(manifest.relative_to(ROOT)): sha256_file(manifest)
            for manifest in training_manifests
        },
        "reliability_subsets": {
            name: {
                "path": str(subset_path.relative_to(ROOT)),
                "sha256": sha256_file(subset_path),
            }
            for name, subset_path in sorted(subset_paths.items())
        },
        "reliability_run": {
            "run_id": args.reliability_run,
            "config_sha256": sha256_file(reliability_dir / "config.json"),
            "provenance_sha256": sha256_file(
                reliability_dir / "provenance.json"
            ),
            "summary_sha256": sha256_file(reliability_summary),
        },
        "source_sha256": {
            path: sha256_file(ROOT / path)
            for path in SOURCE_FILES
        },
    }
    run_path = experiment_run_path(ROOT, args.run_id)
    if run_path.exists():
        run = ExperimentRun.open(run_path)
        if json.loads((run_path / "config.json").read_text()) != config:
            raise ValueError("experiment config changed; resume refused")
        if json.loads((run_path / "provenance.json").read_text()) != provenance:
            raise ValueError("experiment inputs or source changed; resume refused")
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
    progress = tqdm(
        conditions,
        desc="YawPose rear search",
        unit="condition",
    )
    try:
        for condition in progress:
            progress.set_postfix(
                condition=condition.condition_id,
                completed=completed,
            )
            condition_path = experiment_condition_path(
                ROOT,
                args.run_id,
                condition.condition_id,
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
            subprocess.run(
                _condition_command(args, condition, resume=resume),
                check=True,
            )
            completed += 1
            run.write_status(
                "running",
                completed_conditions=completed,
                total_conditions=len(conditions),
                last_completed_condition=condition.condition_id,
            )
            run.event(
                "condition_completed",
                condition_id=condition.condition_id,
            )
        comparison = summarize_search(run.path)
        run.complete(
            completed_conditions=completed,
            total_conditions=len(conditions),
            comparison=str(comparison.relative_to(ROOT)),
        )
    except BaseException as error:
        run.fail(error)
        raise


if __name__ == "__main__":
    main()
