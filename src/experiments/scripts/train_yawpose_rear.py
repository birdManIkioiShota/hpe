"""Train one fixed YawPose rear-yaw adoption condition."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import shutil

_CUBLAS_WORKSPACE_CONFIGS = {":4096:8", ":16:8"}
if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
elif os.environ["CUBLAS_WORKSPACE_CONFIG"] not in _CUBLAS_WORKSPACE_CONFIGS:
    raise RuntimeError("invalid CUBLAS_WORKSPACE_CONFIG for deterministic CUDA")

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from experiments.common.datasets import (
    MultiPoseDataset,
    PoseManifestDataset,
    verify_manifest_partitions,
)
from experiments.common.run_directory import (
    ExperimentRun,
    experiment_condition_path,
    experiment_run_path,
    validate_condition_id,
)
from experiments.common.sampling import build_epoch_plan, scan_pose_buckets
from experiments.common.training import (
    REAR_SELECTION_GROUPS,
    autocast_context,
    evaluate_pose_model,
    flip_consistency_loss_rad,
    make_optimizer,
    make_scheduler,
    rotation_loss_rad,
    selection_score,
    training_mode,
)
from experiments.common.yawpose import (
    YawPoseDataset,
    build_yawpose_epoch_plan,
    evaluate_forward_yaw_model,
    yaw_loss_rad,
)
from hpe.datasets.common import sha256_file, write_json_atomic
from hpe.models import SixDRepNet360, load_checkpoint
from training.audit import BASE_CHECKPOINT, BASE_SHA256
from training.prepare_data import ROOT, prepared_data_path


SOURCE_FILES = (
    "src/experiments/common/datasets.py",
    "src/experiments/common/pose.py",
    "src/experiments/common/run_directory.py",
    "src/experiments/common/sampling.py",
    "src/experiments/common/training.py",
    "src/experiments/common/yawpose.py",
    "src/experiments/scripts/train_yawpose_rear.py",
    "src/hpe/data/dataset.py",
    "src/hpe/geometry/rotations.py",
    "src/hpe/models/checkpoint.py",
    "src/hpe/models/sixdrepnet360.py",
)


def _append_jsonl(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")


def _atomic_torch_save(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _rng_state() -> dict:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [
            numpy_state[0],
            numpy_state[1].tolist(),
            numpy_state[2],
            numpy_state[3],
            numpy_state[4],
        ],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:])
    )
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def _line_count(path: Path) -> int:
    with path.open(encoding="utf-8") as stream:
        return sum(1 for _ in stream)


def _manifest_set(
    vgg_data_id: str,
    dad_data_id: str,
    *,
    use_dad: bool,
) -> tuple[list[Path], list[Path], Path]:
    vgg = prepared_data_path(ROOT, vgg_data_id)
    vgg_train = vgg / "train.jsonl"
    train = [vgg_train]
    if use_dad:
        dad = prepared_data_path(ROOT, dad_data_id)
        train.append(dad / "train.jsonl")
    # Checkpoint selection always uses the same VGGHeads dev split so that
    # enabling DAD changes training data, not the internal comparison set.
    dev = [vgg / "dev.jsonl"]
    for manifest in train + dev:
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
    return train, dev, vgg_train


def _spread_batch_positions(
    total_batches: int,
    selected_batches: int,
) -> tuple[int, ...]:
    if selected_batches < 0 or total_batches <= 0:
        raise ValueError("invalid batch schedule size")
    if selected_batches == 0:
        return ()
    if selected_batches > total_batches:
        raise ValueError(
            "YawPose has more batches than the existing-HPE epoch"
        )
    positions = tuple(
        index * total_batches // selected_batches
        for index in range(selected_batches)
    )
    if len(set(positions)) != selected_batches:
        raise ValueError("YawPose batch schedule contains duplicate positions")
    return positions


def _make_pose_dataset(
    manifests: list[Path],
    *,
    augment: bool,
    seed: int,
    error_dir: Path,
):
    return MultiPoseDataset(
        [
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
    )


def _reset_pretraining_failure(path: Path, *, allow_running: bool = False) -> None:
    if not path.exists():
        return
    status_path = path / "status.json"
    if not status_path.is_file():
        raise FileExistsError(path)
    status = json.loads(status_path.read_text())
    allowed = {"failed", "interrupted"} | ({"running"} if allow_running else set())
    if status.get("status") not in allowed:
        raise FileExistsError(path)
    if (
        (path / "checkpoints" / "last.pt").is_file()
        or list((path / "metrics").glob("epoch_*.json"))
    ):
        raise FileExistsError(
            "run contains training progress; resume it or use a new run-id"
        )
    shutil.rmtree(path)


def _read_subset(reliability_run: str, subset: str) -> tuple[Path | None, list[dict]]:
    if subset == "base":
        return None, []
    allowed = {"top020", "top040", "top060", "top080", "top100"}
    if subset not in allowed:
        raise ValueError(f"unsupported YawPose subset: {subset}")
    run_dir = experiment_run_path(ROOT, reliability_run)
    status = json.loads((run_dir / "status.json").read_text())
    if status.get("status") != "completed":
        raise ValueError("reliability run must be completed before training")
    path = run_dir / "predictions" / "subsets" / f"{subset}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows:
        raise ValueError("empty YawPose subset")
    return path, rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--condition-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--subset",
        choices=("base", "top020", "top040", "top060", "top080", "top100"),
        required=True,
    )
    parser.add_argument("--reliability-run", default="yawpose_reliability_stratified15")
    parser.add_argument("--vgg-data-id", default="vgg_data")
    parser.add_argument("--dad-data-id", default="dad3dheads_train")
    parser.add_argument("--use-dad", action="store_true")
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
    parser.add_argument("--progress-position", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
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
        or args.retain_distill_weight < 0
        or args.rear_flip_consistency_weight < 0
        or args.yawpose_weight < 0
        or args.retention_tolerance_deg < 0
        or args.progress_position < 0
    ):
        raise ValueError("invalid training hyperparameter")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("training requires CUDA")
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is unavailable")

    checkpoint = ROOT / BASE_CHECKPOINT
    if sha256_file(checkpoint) != BASE_SHA256:
        raise ValueError("base checkpoint SHA-256 mismatch")
    train_manifests, dev_manifests, vgg_train_manifest = _manifest_set(
        args.vgg_data_id,
        args.dad_data_id,
        use_dad=args.use_dad,
    )
    partition_summary = verify_manifest_partitions(
        train_manifests,
        dev_manifests,
    )
    buckets = scan_pose_buckets(train_manifests)
    rear_fraction = buckets.natural_rear_fraction
    # Keep the existing-HPE training budget fixed to the VGGHeads-only
    # train count so that DAD inclusion changes the pool, not update count.
    vgg_raw_samples = _line_count(vgg_train_manifest)
    effective_batch = args.batch_size * args.accumulation
    requested_samples = args.samples_per_epoch or vgg_raw_samples
    samples_per_epoch = requested_samples // effective_batch * effective_batch
    if samples_per_epoch < effective_batch:
        raise ValueError("samples-per-epoch is smaller than one effective batch")

    subset_path, subset_rows = _read_subset(args.reliability_run, args.subset)
    yawpose_enabled = args.subset != "base"
    config = {
        "kind": "yawpose_rear_yaw_training",
        "condition_id": args.condition_id,
        "subset": args.subset,
        "reliability_run": args.reliability_run,
        "vgg_data_id": args.vgg_data_id,
        "dad_data_id": args.dad_data_id if args.use_dad else None,
        "use_dad": args.use_dad,
        "hpe_regime": "vgg_plus_dad" if args.use_dad else "vgg_only",
        "device": args.device,
        "precision": args.precision,
        "update_scope": args.update_scope,
        "epochs": args.epochs,
        "samples_per_epoch": samples_per_epoch,
        "existing_rear_fraction": rear_fraction,
        "existing_rear_bucket_policy": "proportional",
        "batch_size": args.batch_size,
        "yawpose_batch_size": args.batch_size,
        "accumulation": args.accumulation,
        "effective_batch": effective_batch,
        "workers": args.workers,
        "backbone_lr": args.backbone_lr,
        "head_lr": args.head_lr,
        "weight_decay": args.weight_decay,
        "warmup_updates": args.warmup_updates,
        "clip_norm": args.clip_norm,
        "retain_distill_weight": args.retain_distill_weight,
        "rear_flip_consistency_weight": args.rear_flip_consistency_weight,
        "yawpose_weight": args.yawpose_weight if yawpose_enabled else 0.0,
        "retention_tolerance_deg": args.retention_tolerance_deg,
        "seed": args.seed,
        "selection": (
            "front/side mean and p90 retention constraints, then worst rear-group "
            "p90, rear p90, worst rear-group >90 rate, rear mean"
        ),
    }
    provenance = {
        "base_checkpoint": {"path": BASE_CHECKPOINT, "sha256": BASE_SHA256},
        "train_manifests": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in train_manifests
        },
        "dev_manifests": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in dev_manifests
        },
        "checkpoint_selection_dataset": "vggheads_dev",
        "partition_summary": partition_summary,
        "training_bucket_counts": {
            "retention": len(buckets.retention),
            **{name: len(values) for name, values in buckets.rear.items()},
        },
        "yawpose_subset": (
            None
            if subset_path is None
            else {
                "path": str(subset_path.relative_to(ROOT)),
                "sha256": sha256_file(subset_path),
                "count": len(subset_rows),
            }
        ),
        "source_sha256": {
            path: sha256_file(ROOT / path)
            for path in SOURCE_FILES
        },
    }

    if args.condition_id is None:
        run_path = experiment_run_path(ROOT, args.run_id)
    else:
        validate_condition_id(args.condition_id)
        parent = experiment_run_path(ROOT, args.run_id)
        if not parent.is_dir():
            raise FileNotFoundError(parent)
        run_path = experiment_condition_path(ROOT, args.run_id, args.condition_id)

    resuming = False
    if args.resume and run_path.exists():
        status = json.loads((run_path / "status.json").read_text())
        if status.get("status") == "completed":
            raise ValueError("condition is already completed")
        last_checkpoint = run_path / "checkpoints" / "last.pt"
        if last_checkpoint.is_file():
            run = ExperimentRun.open(run_path)
            if (
                json.loads((run_path / "config.json").read_text()) != config
                or json.loads((run_path / "provenance.json").read_text()) != provenance
            ):
                raise ValueError(
                    "condition config, data, subset, or source changed; resume refused"
                )
            run.write_status("running", resumed=True)
            resuming = True
        else:
            _reset_pretraining_failure(run_path, allow_running=True)
    elif run_path.exists():
        _reset_pretraining_failure(run_path)
    if not resuming:
        run = ExperimentRun.create_at(
            run_path,
            config=config,
            provenance=provenance,
        )

    error_dir = run.path / "artifacts" / "image_read_errors"
    train_data = _make_pose_dataset(
        train_manifests,
        augment=True,
        seed=args.seed,
        error_dir=error_dir,
    )
    dev_data = _make_pose_dataset(
        dev_manifests,
        augment=False,
        seed=args.seed,
        error_dir=error_dir,
    )
    dev_loader = DataLoader(
        dev_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    yawpose_data = None
    if yawpose_enabled:
        assert subset_path is not None
        yawpose_data = YawPoseDataset(
            ROOT,
            subset_path,
            augment=True,
            seed=args.seed,
            read_error_dir=error_dir,
        )
        if len(yawpose_data) != len(subset_rows):
            raise ValueError("YawPose subset count changed")

    epoch_progress = None
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

        baseline_path = run.path / "metrics" / "dev_baseline.json"
        baseline_yaw_path = run.path / "metrics" / "dev_baseline_yaw.json"
        if resuming:
            baseline = json.loads(baseline_path.read_text())
            baseline_yaw = json.loads(baseline_yaw_path.read_text())
        else:
            baseline = evaluate_pose_model(
                student,
                dev_loader,
                device,
                progress_position=args.progress_position + 1,
                progress_leave=False,
                progress_desc="Baseline pose dev",
            )
            baseline_yaw = evaluate_forward_yaw_model(
                student,
                dev_loader,
                device,
                progress_position=args.progress_position + 1,
                progress_leave=False,
                progress_desc="Baseline yaw dev",
            )
            write_json_atomic(baseline_path, baseline)
            write_json_atomic(baseline_yaw_path, baseline_yaw)

        missing = {
            "front",
            "side",
            "rear",
            "retention",
            *REAR_SELECTION_GROUPS,
        }.difference(baseline)
        if missing:
            raise ValueError(
                "internal dev is missing checkpoint-selection groups: "
                + ", ".join(sorted(missing))
            )

        start_epoch = 0
        global_step = 0
        if resuming:
            state = torch.load(
                run.path / "checkpoints" / "last.pt",
                map_location="cpu",
                weights_only=True,
            )
            if state.get("schema") != 1 or state.get("config") != config:
                raise ValueError("resume checkpoint/config mismatch")
            student.load_state_dict(state["state_dict"], strict=True)
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            start_epoch = int(state["epoch"])
            global_step = int(state["global_step"])
            best_score = tuple(float(value) for value in state["best_score"])
            best_metrics = state["best_metrics"]
            best_epoch = int(state["best_epoch"])
            _restore_rng(state["rng"])
            run.event(
                "resumed",
                next_epoch=start_epoch + 1,
                global_step=global_step,
            )
        else:
            best_score = selection_score(
                baseline,
                baseline,
                retention_tolerance_deg=args.retention_tolerance_deg,
            )
            best_metrics = baseline
            best_epoch = 0
            _atomic_torch_save(
                run.path / "checkpoints" / "best.pth",
                {
                    "state_dict": student.state_dict(),
                    "epoch": 0,
                    "dev": baseline,
                    "selection_score": best_score,
                    "selection": config["selection"],
                    "dev_yaw": baseline_yaw,
                },
            )
            _atomic_torch_save(
                run.path / "checkpoints" / "last.pt",
                {
                    "schema": 1,
                    "state_dict": student.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": 0,
                    "global_step": 0,
                    "config": config,
                    "best_score": best_score,
                    "best_metrics": best_metrics,
                    "best_epoch": best_epoch,
                    "rng": _rng_state(),
                },
            )

        epoch_progress = tqdm(
            total=args.epochs,
            initial=start_epoch,
            desc="Epoch",
            unit="epoch",
            position=args.progress_position,
            leave=True,
            dynamic_ncols=True,
        )
        for epoch in range(start_epoch, args.epochs):
            epoch_progress.set_postfix(epoch=f"{epoch + 1}/{args.epochs}")
            training_mode(student, args.update_scope)
            hpe_plan = build_epoch_plan(
                buckets,
                num_samples=samples_per_epoch,
                rear_fraction=rear_fraction,
                seed=args.seed,
                epoch=epoch,
                rear_bucket_policy="proportional",
            )
            hpe_loader = DataLoader(
                train_data,
                batch_size=args.batch_size,
                sampler=hpe_plan.order,
                num_workers=args.workers,
                pin_memory=True,
                drop_last=False,
            )
            yaw_plan = None
            yaw_loader = None
            yaw_batch_positions: tuple[int, ...] = ()
            if yawpose_data is not None:
                yaw_plan = build_yawpose_epoch_plan(
                    subset_rows,
                    seed=args.seed,
                    epoch=epoch,
                )
                yaw_loader = DataLoader(
                    yawpose_data,
                    batch_size=args.batch_size,
                    sampler=yaw_plan.order,
                    num_workers=args.workers,
                    pin_memory=True,
                    drop_last=False,
                )
                yaw_batch_positions = _spread_batch_positions(
                    len(hpe_loader),
                    len(yaw_loader),
                )
            yaw_iterator = (
                iter(yaw_loader) if yaw_loader is not None else None
            )
            yaw_position_set = set(yaw_batch_positions)

            optimizer.zero_grad(set_to_none=True)
            totals = {
                "samples": 0,
                "loss": 0.0,
                "supervised": 0.0,
                "distill": 0.0,
                "flip": 0.0,
                "yawpose": 0.0,
                "yawpose_samples": 0,
                "yawpose_batches": 0,
                "rear_samples": 0,
            }
            bar = tqdm(
                hpe_loader,
                desc="Train",
                unit="batch",
                position=args.progress_position + 1,
                leave=False,
                dynamic_ncols=True,
            )
            for batch_index, (images, target, metadata) in enumerate(bar):
                images = images.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                rear_mask = torch.as_tensor(
                    metadata["is_rear"],
                    device=device,
                    dtype=torch.bool,
                )
                retention_mask = ~rear_mask
                teacher_prediction = None
                if retention_mask.any() and args.retain_distill_weight:
                    with torch.no_grad(), torch.autocast(
                        device_type=device.type,
                        enabled=False,
                    ):
                        teacher_prediction = teacher(images[retention_mask])

                yaw_images = yaw_target = None
                if (
                    yaw_iterator is not None
                    and batch_index in yaw_position_set
                ):
                    yaw_images, yaw_target, _ = next(yaw_iterator)
                    yaw_images = yaw_images.to(device, non_blocking=True)
                    yaw_target = yaw_target.to(device, non_blocking=True)

                yaw_errors = None
                with autocast_context(device, args.precision):
                    prediction = student(images)
                    supervised = rotation_loss_rad(
                        prediction,
                        target,
                    ).mean()
                    if teacher_prediction is not None:
                        distill = rotation_loss_rad(
                            prediction[retention_mask],
                            teacher_prediction,
                        ).mean()
                    else:
                        distill = prediction.sum() * 0.0
                    if (
                        rear_mask.any()
                        and args.rear_flip_consistency_weight
                    ):
                        flipped_prediction = student(
                            torch.flip(images[rear_mask], dims=[3])
                        )
                        flip_loss = flip_consistency_loss_rad(
                            prediction[rear_mask],
                            flipped_prediction,
                        ).mean()
                    else:
                        flip_loss = prediction.sum() * 0.0
                    if yaw_images is not None:
                        yaw_prediction = student(yaw_images)
                        yaw_errors = yaw_loss_rad(
                            yaw_prediction,
                            yaw_target,
                        )
                        # Full YawPose batches retain the previous mean-loss
                        # scale. A final partial batch is scaled by its actual
                        # sample count so every selected sample contributes
                        # exactly one equal draw per epoch.
                        yawpose_loss = (
                            yaw_errors.sum() / args.batch_size
                        )
                    else:
                        yawpose_loss = prediction.sum() * 0.0
                    loss = (
                        supervised
                        + args.retain_distill_weight * distill
                        + args.rear_flip_consistency_weight * flip_loss
                        + (
                            args.yawpose_weight
                            if yawpose_enabled
                            else 0.0
                        )
                        * yawpose_loss
                    )
                    scaled_loss = loss / args.accumulation

                if not torch.isfinite(loss):
                    raise ValueError("non-finite training loss")
                scaled_loss.backward()

                count = len(images)
                totals["samples"] += count
                totals["rear_samples"] += int(rear_mask.sum())
                totals["loss"] += float(loss.detach()) * count
                totals["supervised"] += (
                    float(supervised.detach()) * count
                )
                totals["distill"] += float(distill.detach()) * count
                totals["flip"] += float(flip_loss.detach()) * count
                if yaw_errors is not None:
                    totals["yawpose"] += float(yaw_errors.detach().sum())
                    totals["yawpose_samples"] += len(yaw_errors)
                    totals["yawpose_batches"] += 1

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
                        yaw_mean = (
                            totals["yawpose"]
                            / totals["yawpose_samples"]
                            if totals["yawpose_samples"]
                            else 0.0
                        )
                        _append_jsonl(
                            run.path / "metrics" / "train.jsonl",
                            {
                                "time": datetime.now(
                                    timezone.utc
                                ).isoformat(),
                                "epoch": epoch + 1,
                                "step": global_step,
                                "samples": totals["samples"],
                                "yawpose_samples": totals[
                                    "yawpose_samples"
                                ],
                                "loss_rad": (
                                    totals["loss"] / totals["samples"]
                                ),
                                "supervised_rad": (
                                    totals["supervised"]
                                    / totals["samples"]
                                ),
                                "distill_rad": (
                                    totals["distill"]
                                    / totals["samples"]
                                ),
                                "flip_consistency_rad": (
                                    totals["flip"]
                                    / totals["samples"]
                                ),
                                "yawpose_rad": yaw_mean,
                                "gradient_norm": float(norm),
                                "lr": [
                                    group["lr"]
                                    for group in optimizer.param_groups
                                ],
                            },
                        )
                bar.set_postfix(
                    loss=f"{totals['loss'] / totals['samples']:.4f}"
                )

            if yaw_plan is not None:
                if totals["yawpose_samples"] != yaw_plan.total_draws:
                    raise ValueError(
                        "not all selected YawPose samples were drawn"
                    )
                if totals["yawpose_batches"] != len(yaw_batch_positions):
                    raise ValueError(
                        "YawPose batch schedule was not fully consumed"
                    )

            dev = evaluate_pose_model(
                student,
                dev_loader,
                device,
                progress_position=args.progress_position + 1,
                progress_leave=False,
                progress_desc="Pose dev",
            )
            dev_yaw = evaluate_forward_yaw_model(
                student,
                dev_loader,
                device,
                progress_position=args.progress_position + 1,
                progress_leave=False,
                progress_desc="Yaw dev",
            )
            score = selection_score(
                dev,
                baseline,
                retention_tolerance_deg=args.retention_tolerance_deg,
            )
            improved = score < best_score
            if improved:
                best_score = score
                best_metrics = dev
                best_epoch = epoch + 1
            epoch_result = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "train": {
                    "samples": totals["samples"],
                    "loss_rad": totals["loss"] / totals["samples"],
                    "supervised_rad": (
                        totals["supervised"] / totals["samples"]
                    ),
                    "distill_rad": totals["distill"] / totals["samples"],
                    "distill_weighted_rad": (
                        args.retain_distill_weight
                        * totals["distill"]
                        / totals["samples"]
                    ),
                    "flip_consistency_rad": (
                        totals["flip"] / totals["samples"]
                    ),
                    "flip_consistency_weighted_rad": (
                        args.rear_flip_consistency_weight
                        * totals["flip"]
                        / totals["samples"]
                    ),
                    "yawpose_samples": totals["yawpose_samples"],
                    "yawpose_batches": totals["yawpose_batches"],
                    "yawpose_rad": (
                        totals["yawpose"] / totals["yawpose_samples"]
                        if totals["yawpose_samples"]
                        else 0.0
                    ),
                    "yawpose_weighted_rad": (
                        (
                            args.yawpose_weight
                            if yawpose_enabled
                            else 0.0
                        )
                        * totals["yawpose"]
                        / totals["yawpose_samples"]
                        if totals["yawpose_samples"]
                        else 0.0
                    ),
                    "rear_fraction_observed": (
                        totals["rear_samples"] / totals["samples"]
                    ),
                    "existing_sampling": {
                        "rear_fraction_requested": (
                            hpe_plan.rear_fraction_requested
                        ),
                        "rear_fraction_planned": (
                            hpe_plan.rear_fraction_observed
                        ),
                        "rear_bucket_policy": hpe_plan.rear_bucket_policy,
                        "rear_bucket_draws": hpe_plan.bucket_draws,
                        "unique_samples": hpe_plan.unique_samples,
                        "repeated_draws": hpe_plan.repeated_draws,
                    },
                    "yawpose_sampling": (
                        None
                        if yaw_plan is None
                        else {
                            "selected_unique_samples": len(subset_rows),
                            "unique_samples_drawn": yaw_plan.unique_samples,
                            "total_draws": yaw_plan.total_draws,
                            "repeated_draws": yaw_plan.repeated_draws,
                            "repeat_ratio": (
                                yaw_plan.repeated_draws / yaw_plan.total_draws
                            ),
                            "source_draws": yaw_plan.source_draws,
                            "rear_yaw_bin_draws": (
                                yaw_plan.rear_yaw_bin_draws
                            ),
                            "batch_positions": list(
                                yaw_batch_positions
                            ),
                        }
                    ),
                },
                "dev": dev,
                "dev_yaw": dev_yaw,
                "selection_score": score,
                "improved": improved,
            }
            write_json_atomic(
                run.path / "metrics" / f"epoch_{epoch + 1:03d}.json",
                epoch_result,
            )
            _append_jsonl(
                run.path / "metrics" / "dev.jsonl",
                epoch_result,
            )
            run.event(
                "epoch_completed",
                epoch=epoch + 1,
                global_step=global_step,
                improved=improved,
                rear=dev.get("rear"),
                rear_yaw=dev_yaw.get("rear"),
            )
            _atomic_torch_save(
                run.path / "checkpoints" / "last.pt",
                {
                    "schema": 1,
                    "state_dict": student.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "config": config,
                    "best_score": best_score,
                    "best_metrics": best_metrics,
                    "best_epoch": best_epoch,
                    "rng": _rng_state(),
                },
            )
            if improved:
                _atomic_torch_save(
                    run.path / "checkpoints" / "best.pth",
                    {
                        "state_dict": student.state_dict(),
                        "epoch": best_epoch,
                        "dev": best_metrics,
                        "dev_yaw": dev_yaw,
                        "selection_score": best_score,
                        "selection": config["selection"],
                    },
                )
            epoch_progress.update(1)
            epoch_progress.set_postfix(
                epoch=f"{epoch + 1}/{args.epochs}",
                best=best_epoch,
            )

        final = json.loads(
            (
                run.path
                / "metrics"
                / f"epoch_{args.epochs:03d}.json"
            ).read_text()
        )
        final_checkpoint = (
            run.path / "checkpoints" / f"epoch_{args.epochs:03d}.pth"
        )
        _atomic_torch_save(
            final_checkpoint,
            {
                "state_dict": student.state_dict(),
                "epoch": args.epochs,
                "dev": final["dev"],
                "dev_yaw": final["dev_yaw"],
                "selection_score": final["selection_score"],
                "selection": config["selection"],
            },
        )
        run.complete(
            epochs=args.epochs,
            global_step=global_step,
            best_epoch=best_epoch,
            final_checkpoint=str(final_checkpoint.relative_to(ROOT)),
            best_selection_score=list(best_score),
            final_rear=final["dev"]["rear"],
            final_rear_yaw=final["dev_yaw"]["rear"],
        )
    except BaseException as error:
        run.fail(error)
        raise
    finally:
        if epoch_progress is not None:
            epoch_progress.close()
        train_data.close()
        dev_data.close()


if __name__ == "__main__":
    main()
