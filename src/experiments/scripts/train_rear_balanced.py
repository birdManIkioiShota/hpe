"""Train a rear-balanced SixDRepNet360 candidate with base-model retention distillation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os

# PyTorch deterministic CUDA matmul requires this to be set before CUDA/cuBLAS
# is initialized. Set it before importing torch in this entry-point module.
_CUBLAS_WORKSPACE_CONFIGS = {":4096:8", ":16:8"}
if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
elif os.environ["CUBLAS_WORKSPACE_CONFIG"] not in _CUBLAS_WORKSPACE_CONFIGS:
    raise RuntimeError(
        "CUBLAS_WORKSPACE_CONFIG must be ':4096:8' or ':16:8' for deterministic CUDA"
    )

from pathlib import Path
import random
import shutil

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from experiments.common.datasets import (
    MultiPoseDataset,
    PoseManifestDataset,
    verify_manifest_partitions,
)
from experiments.common.run_directory import ExperimentRun, experiment_run_path
from experiments.common.sampling import build_epoch_order, scan_pose_buckets
from experiments.common.training import (
    autocast_context,
    evaluate_pose_model,
    make_optimizer,
    make_scheduler,
    rotation_loss_rad,
    selection_score,
    training_mode,
)
from hpe.datasets.common import sha256_file, write_json_atomic
from hpe.models import SixDRepNet360, load_checkpoint
from training.audit import BASE_CHECKPOINT, BASE_SHA256
from training.prepare_data import ROOT, prepared_data_path


SOURCE_FILES = (
    "src/experiments/common/datasets.py",
    "src/experiments/common/sampling.py",
    "src/experiments/common/training.py",
    "src/experiments/scripts/train_rear_balanced.py",
    "src/hpe/data/dataset.py",
    "src/hpe/geometry/rotations.py",
    "src/hpe/models/checkpoint.py",
    "src/hpe/models/sixdrepnet360.py",
)


def _reset_failed_pretraining_run(path: Path) -> None:
    """Remove only a failed run that has not completed any optimizer update."""
    if not path.exists():
        return
    status_path = path / "status.json"
    if not status_path.is_file():
        raise FileExistsError(f"Experiment run already exists without status: {path}")
    status = json.loads(status_path.read_text())
    if status.get("status") not in {"failed", "interrupted"}:
        raise FileExistsError(f"Experiment run already exists: {path}")

    last_checkpoint = path / "checkpoints" / "last.pt"
    train_log = path / "metrics" / "train.jsonl"
    dev_log = path / "metrics" / "dev.jsonl"
    epoch_metrics = list((path / "metrics").glob("epoch_*.json"))
    if last_checkpoint.exists() or train_log.exists() or dev_log.exists() or epoch_metrics:
        raise FileExistsError(
            "Failed run contains optimizer-update or epoch outputs; use a new run-id instead "
            f"of overwriting it: {path}"
        )

    best_checkpoint = path / "checkpoints" / "best.pth"
    baseline_metrics = path / "metrics" / "dev_baseline.json"
    if best_checkpoint.exists() or baseline_metrics.exists():
        if not (best_checkpoint.is_file() and baseline_metrics.is_file()):
            raise FileExistsError(
                "Failed run has an incomplete baseline snapshot; use a new run-id instead "
                f"of overwriting it: {path}"
            )
        baseline_checkpoint = torch.load(
            best_checkpoint,
            map_location="cpu",
            weights_only=True,
        )
        if baseline_checkpoint.get("epoch") != 0:
            raise FileExistsError(
                "Failed run contains a non-baseline best checkpoint; use a new run-id instead "
                f"of overwriting it: {path}"
            )

    allowed_baseline_files = {best_checkpoint.resolve(), baseline_metrics.resolve()}
    for directory_name in ("checkpoints", "metrics", "predictions", "artifacts"):
        directory = path / directory_name
        if not directory.exists():
            continue
        unexpected = [
            item
            for item in directory.rglob("*")
            if item.is_file() and item.resolve() not in allowed_baseline_files
        ]
        if unexpected:
            raise FileExistsError(
                "Failed run contains diagnostic artifacts; use a new run-id instead of "
                f"overwriting it: {path}"
            )
    shutil.rmtree(path)


def _append_jsonl(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")


def _atomic_torch_save(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _line_count(path: Path) -> int:
    with path.open(encoding="utf-8") as stream:
        return sum(1 for _ in stream)


def _manifest_set(vgg_data_id: str, dad_data_id: str) -> tuple[list[Path], list[Path]]:
    vgg = prepared_data_path(ROOT, vgg_data_id)
    dad = prepared_data_path(ROOT, dad_data_id)
    train = [vgg / "train.jsonl", dad / "train.jsonl"]
    dev = [vgg / "dev.jsonl", dad / "dev.jsonl"]
    for path in train + dev:
        if not path.is_file():
            raise FileNotFoundError(path)
    return train, dev


def _make_dataset(manifests: list[Path], *, augment: bool, seed: int, error_dir: Path):
    datasets = [
        PoseManifestDataset(
            ROOT,
            manifest,
            augment=augment,
            seed=seed,
            allowed_datasets={"vggheads", "dad3dheads"},
            read_error_dir=error_dir,
        )
        for manifest in manifests
    ]
    return MultiPoseDataset(datasets)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--vgg-data-id", default="vgg_data")
    parser.add_argument("--dad-data-id", default="dad3dheads_train")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--update-scope", choices=("head", "layer4", "all"), default="layer4")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--samples-per-epoch", type=int)
    parser.add_argument("--rear-fraction", type=float, default=0.25)
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
    args = parser.parse_args()

    if (
        args.epochs <= 0
        or args.batch_size <= 0
        or args.accumulation <= 0
        or args.workers < 0
        or args.backbone_lr <= 0
        or args.head_lr <= 0
        or args.weight_decay < 0
        or args.warmup_updates < 0
        or args.clip_norm <= 0
        or args.rear_supervised_weight < 0
        or args.replay_supervised_weight < 0
        or args.retain_distill_weight < 0
        or args.retention_tolerance_deg < 0
    ):
        raise ValueError("Invalid training hyperparameter")
    if not 0.0 < args.rear_fraction < 1.0:
        raise ValueError("rear-fraction must be between 0 and 1")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Training requires CUDA")
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is unavailable")

    checkpoint = ROOT / BASE_CHECKPOINT
    if sha256_file(checkpoint) != BASE_SHA256:
        raise ValueError("Base checkpoint SHA-256 mismatch")
    train_manifests, dev_manifests = _manifest_set(args.vgg_data_id, args.dad_data_id)
    partition_summary = verify_manifest_partitions(train_manifests, dev_manifests)
    buckets = scan_pose_buckets(train_manifests)
    if any(not buckets.rear[name] for name in buckets.rear):
        raise ValueError("All four rear azimuth buckets must contain training samples")

    raw_samples = sum(_line_count(path) for path in train_manifests)
    effective_batch = args.batch_size * args.accumulation
    requested_samples = args.samples_per_epoch or raw_samples
    samples_per_epoch = (requested_samples // effective_batch) * effective_batch
    if samples_per_epoch < effective_batch:
        raise ValueError("samples-per-epoch is smaller than one effective batch")

    config = {
        "kind": "rear_balanced_distillation_training",
        "vgg_data_id": args.vgg_data_id,
        "dad_data_id": args.dad_data_id,
        "device": args.device,
        "precision": args.precision,
        "update_scope": args.update_scope,
        "epochs": args.epochs,
        "samples_per_epoch": samples_per_epoch,
        "rear_fraction": args.rear_fraction,
        "batch_size": args.batch_size,
        "accumulation": args.accumulation,
        "effective_batch": effective_batch,
        "workers": args.workers,
        "backbone_lr": args.backbone_lr,
        "head_lr": args.head_lr,
        "weight_decay": args.weight_decay,
        "warmup_updates": args.warmup_updates,
        "clip_norm": args.clip_norm,
        "rear_supervised_weight": args.rear_supervised_weight,
        "replay_supervised_weight": args.replay_supervised_weight,
        "retain_distill_weight": args.retain_distill_weight,
        "retention_tolerance_deg": args.retention_tolerance_deg,
        "seed": args.seed,
        "selection": "retention constraint, then rear p90, >90 rate, mean",
    }
    provenance = {
        "base_checkpoint": {"path": BASE_CHECKPOINT, "sha256": BASE_SHA256},
        "train_manifests": {
            str(path.relative_to(ROOT)): sha256_file(path) for path in train_manifests
        },
        "dev_manifests": {
            str(path.relative_to(ROOT)): sha256_file(path) for path in dev_manifests
        },
        "source_sha256": {
            path: sha256_file(ROOT / path) for path in SOURCE_FILES
        },
        "partition_summary": partition_summary,
        "training_bucket_counts": {
            "retention": len(buckets.retention),
            **{name: len(values) for name, values in buckets.rear.items()},
        },
    }
    run_path = experiment_run_path(ROOT, args.run_id)
    _reset_failed_pretraining_run(run_path)
    run = ExperimentRun.create(ROOT, args.run_id, config=config, provenance=provenance)
    error_dir = run.path / "artifacts" / "image_read_errors"
    train_data = _make_dataset(train_manifests, augment=True, seed=args.seed, error_dir=error_dir)
    dev_data = _make_dataset(dev_manifests, augment=False, seed=args.seed, error_dir=error_dir)
    dev_loader = DataLoader(
        dev_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    try:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        torch.backends.cudnn.conv.fp32_precision = "ieee"
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

        student = SixDRepNet360(rotation_fp32=True)
        load_checkpoint(student, checkpoint)
        student.to(device)
        teacher = SixDRepNet360(rotation_fp32=True)
        load_checkpoint(teacher, checkpoint)
        teacher.to(device).eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)

        training_mode(student, args.update_scope)
        optimizer = make_optimizer(
            student,
            backbone_lr=args.backbone_lr,
            head_lr=args.head_lr,
            weight_decay=args.weight_decay,
        )
        updates_per_epoch = samples_per_epoch // effective_batch
        scheduler = make_scheduler(
            optimizer,
            total_updates=updates_per_epoch * args.epochs,
            warmup_updates=args.warmup_updates,
        )

        baseline = evaluate_pose_model(student, dev_loader, device)
        run.event("dev_baseline", rear=baseline.get("rear"), retention=baseline.get("retention"))
        if "rear" not in baseline or "retention" not in baseline:
            raise ValueError("Internal dev must contain both rear and retention samples")
        write_json_atomic(run.path / "metrics" / "dev_baseline.json", baseline)
        best_score = selection_score(
            baseline,
            baseline,
            retention_tolerance_deg=args.retention_tolerance_deg,
        )
        best_metrics = baseline
        _atomic_torch_save(
            run.path / "checkpoints" / "best.pth",
            {
                "state_dict": student.state_dict(),
                "epoch": 0,
                "dev": baseline,
                "selection_score": best_score,
                "selection": config["selection"],
            },
        )

        global_step = 0
        for epoch in range(args.epochs):
            training_mode(student, args.update_scope)
            order = build_epoch_order(
                buckets,
                num_samples=samples_per_epoch,
                rear_fraction=args.rear_fraction,
                seed=args.seed,
                epoch=epoch,
            )
            loader = DataLoader(
                train_data,
                batch_size=args.batch_size,
                sampler=order,
                num_workers=args.workers,
                pin_memory=True,
                drop_last=False,
            )
            optimizer.zero_grad(set_to_none=True)
            totals = {
                "samples": 0,
                "loss": 0.0,
                "supervised": 0.0,
                "distill": 0.0,
                "rear_samples": 0,
            }
            bar = tqdm(loader, desc=f"Train {epoch + 1}/{args.epochs}", unit="batch")
            for batch_index, (images, target, metadata) in enumerate(bar):
                images = images.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                rear_mask = torch.as_tensor(metadata["is_rear"], device=device, dtype=torch.bool)

                retention_mask = ~rear_mask
                teacher_prediction = None
                if retention_mask.any() and args.retain_distill_weight:
                    with torch.no_grad(), torch.autocast(
                        device_type=device.type, enabled=False
                    ):
                        teacher_prediction = teacher(images[retention_mask])

                with autocast_context(device, args.precision):
                    prediction = student(images)
                    supervised_each = rotation_loss_rad(prediction, target)
                    weights = torch.where(
                        rear_mask,
                        torch.full_like(supervised_each, args.rear_supervised_weight),
                        torch.full_like(supervised_each, args.replay_supervised_weight),
                    )
                    supervised = (supervised_each * weights).mean()
                    if teacher_prediction is not None:
                        distill = rotation_loss_rad(
                            prediction[retention_mask],
                            teacher_prediction,
                        ).mean()
                    else:
                        distill = prediction.sum() * 0.0
                    loss = supervised + args.retain_distill_weight * distill
                    scaled_loss = loss / args.accumulation

                if not torch.isfinite(loss):
                    raise ValueError("Non-finite training loss")
                scaled_loss.backward()

                totals["samples"] += len(images)
                totals["rear_samples"] += int(rear_mask.sum())
                totals["loss"] += float(loss.detach()) * len(images)
                totals["supervised"] += float(supervised.detach()) * len(images)
                totals["distill"] += float(distill.detach()) * len(images)

                if (batch_index + 1) % args.accumulation == 0:
                    norm = torch.nn.utils.clip_grad_norm_(
                        student.parameters(),
                        args.clip_norm,
                        error_if_nonfinite=True,
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    if global_step % 100 == 0:
                        _append_jsonl(
                            run.path / "metrics" / "train.jsonl",
                            {
                                "time": datetime.now(timezone.utc).isoformat(),
                                "epoch": epoch + 1,
                                "step": global_step,
                                "samples": totals["samples"],
                                "loss_rad": totals["loss"] / totals["samples"],
                                "supervised_rad": totals["supervised"] / totals["samples"],
                                "distill_rad": totals["distill"] / totals["samples"],
                                "rear_fraction_observed": totals["rear_samples"] / totals["samples"],
                                "gradient_norm": float(norm),
                                "lr": [group["lr"] for group in optimizer.param_groups],
                            },
                        )
                bar.set_postfix(
                    loss=f"{totals['loss'] / totals['samples']:.4f}",
                    rear=f"{totals['rear_samples'] / totals['samples']:.3f}",
                )

            dev = evaluate_pose_model(student, dev_loader, device)
            score = selection_score(
                dev,
                baseline,
                retention_tolerance_deg=args.retention_tolerance_deg,
            )
            improved = score < best_score
            if improved:
                best_score = score
                best_metrics = dev
                _atomic_torch_save(
                    run.path / "checkpoints" / "best.pth",
                    {
                        "state_dict": student.state_dict(),
                        "epoch": epoch + 1,
                        "dev": dev,
                        "selection_score": score,
                        "selection": config["selection"],
                    },
                )
            epoch_result = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "train": {
                    "samples": totals["samples"],
                    "loss_rad": totals["loss"] / totals["samples"],
                    "supervised_rad": totals["supervised"] / totals["samples"],
                    "distill_rad": totals["distill"] / totals["samples"],
                    "rear_fraction_observed": totals["rear_samples"] / totals["samples"],
                },
                "dev": dev,
                "selection_score": score,
                "improved": improved,
            }
            write_json_atomic(
                run.path / "metrics" / f"epoch_{epoch + 1:03d}.json",
                epoch_result,
            )
            run.event(
                "epoch_completed",
                epoch=epoch + 1,
                global_step=global_step,
                improved=improved,
                rear=dev.get("rear"),
                retention=dev.get("retention"),
            )
            _append_jsonl(run.path / "metrics" / "dev.jsonl", epoch_result)
            _atomic_torch_save(
                run.path / "checkpoints" / "last.pt",
                {
                    "state_dict": student.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "config": config,
                    "best_score": best_score,
                },
            )

        run.complete(
            epochs=args.epochs,
            global_step=global_step,
            best_selection_score=list(best_score),
            best_rear=best_metrics["rear"],
            best_retention=best_metrics["retention"],
        )
    except BaseException as error:
        run.fail(error)
        raise
    finally:
        train_data.close()
        dev_data.close()


if __name__ == "__main__":
    main()
