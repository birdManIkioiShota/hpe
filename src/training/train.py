"""Supervised VGGHeads fine-tuning of the fixed 6DRepNet360 checkpoint."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from hpe.datasets.common import sha256_file, write_json_atomic
from hpe.geometry.rotations import geodesic_error_degrees
from hpe.models import SixDRepNet360, load_checkpoint
from training.audit import BASE_CHECKPOINT, BASE_SHA256
from training.evaluate_baseline import source_hashes
from training.prepare_data import ROOT, SCHEMA, prepared_data_path, run_output_path
from training.vgg import VGGDataset, read_jsonl


TRAINING_SOURCE_FILES = (
    "src/hpe/data/dataset.py", "src/hpe/datasets/common.py",
    "src/hpe/geometry/rotations.py", "src/hpe/models/checkpoint.py",
    "src/hpe/models/sixdrepnet360.py", "src/training/audit.py",
    "src/training/prepare_data.py", "src/training/train.py", "src/training/vgg.py",
    "pyproject.toml", "uv.lock",
)


def event(stream, kind: str, **values):
    stream.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(), "event": kind, **values}, allow_nan=False) + "\n")
    stream.flush()


def atomic_save(path: Path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def training_source_hashes(root: Path) -> dict[str, str]:
    return {name: sha256_file(root / name) for name in TRAINING_SOURCE_FILES}


def rng_state():
    state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [state[0], state[1].tolist(), state[2], state[3], state[4]],
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np_state = state["numpy"]
    np.random.set_state((np_state[0], np.asarray(np_state[1], dtype=np.uint32), *np_state[2:]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def make_optimizer(model, config):
    head = list(model.linear_reg.parameters())
    head_ids = {id(value) for value in head}
    backbone = [value for value in model.parameters() if id(value) not in head_ids]
    return torch.optim.AdamW([
        {"params": backbone, "lr": config["backbone_lr"], "name": "backbone"},
        {"params": head, "lr": config["head_lr"], "name": "head"},
    ], weight_decay=config["weight_decay"], betas=(.9, .999), eps=1e-8)


def make_scheduler(optimizer, total_updates, warmup_updates):
    def factor(step):
        if step < warmup_updates:
            return (step + 1) / max(1, warmup_updates)
        progress = (step - warmup_updates) / max(1, total_updates - warmup_updates)
        return .5 * (1 + math.cos(math.pi * min(progress, 1.)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def training_mode(model, *, head_only: bool):
    model.train()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(not head_only or name.startswith("linear_reg."))
    # Keep pretrained running mean/variance. BN affine parameters train after warmup.
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def rotation_loss(prediction, target):
    # Autocast must also be disabled for the relative-matrix multiplication.
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        return torch.deg2rad(geodesic_error_degrees(prediction.float(), target.float(), stable=True))


def optimizer_update(model, optimizer, scheduler, scaler, clip_norm):
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm, error_if_nonfinite=True)
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return float(norm)


@torch.inference_mode()
def evaluate_dev(model, loader, device):
    model.eval()
    errors = []
    bar = tqdm(loader, desc="VGG dev (FP32)", unit="batch")
    running_sum, count = 0., 0
    for images, target in bar:
        prediction = model(images.to(device, non_blocking=True))
        value = geodesic_error_degrees(prediction.double(), target.to(device).double(), stable=True)
        if not torch.isfinite(value).all():
            raise ValueError("Non-finite dev metrics")
        errors.append(value.cpu())
        running_sum += value.sum().item()
        count += len(value)
        bar.set_postfix(geo_deg=f"{running_sum / count:.3f}")
    values = torch.cat(errors)
    return {"count": count, "geodesic_mean_deg": values.mean().item(),
            "geodesic_median_deg": values.quantile(.5).item(),
            "geodesic_p90_deg": values.quantile(.9).item(),
            "over90_percent": (values > 90).double().mean().item() * 100}


def verify_data(data: Path, root: Path = ROOT):
    metadata = json.loads((data / "metadata.json").read_text())
    if metadata["schema"] != SCHEMA or metadata["status"] != "completed":
        raise ValueError("VGGHeads preparation is incomplete or incompatible")
    image_groups, instance_ids = {}, set()
    for split in ("train", "dev", "holdout"):
        manifest = data / f"{split}.jsonl"
        if sha256_file(manifest) != metadata["splits"][split]["sha256"]:
            raise ValueError(f"Changed {split} manifest")
        count = 0
        for row in tqdm(read_jsonl(manifest), total=metadata["splits"][split]["heads"],
                        desc=f"Check {split} membership", unit="head"):
            if row["dataset"] != "vggheads" or row["split"] != split:
                raise ValueError("Unexpected data source/split")
            if not (root / row["image_path"]).resolve().is_relative_to((root / "datasets/VGGHeads").resolve()):
                raise ValueError("Training manifests may reference only datasets/VGGHeads")
            if row["instance_id"] in instance_ids:
                raise ValueError("Repeated instance ID")
            instance_ids.add(row["instance_id"])
            for key in ("group:" + row["group_id"], "image:" + str((root / row["image_path"]).resolve())):
                if key in image_groups and image_groups[key] != split:
                    raise ValueError("Cross-split source/image leakage")
                image_groups[key] = split
            count += 1
        if count != metadata["splits"][split]["heads"] or count == 0:
            raise ValueError(f"Unexpected {split} size")
    return metadata


def verify_image_files(data: Path, root: Path, metadata: dict, workers: int) -> int:
    """Read each unique training/dev/holdout image once before allocating the model."""
    expected: dict[str, str] = {}
    for split in ("train", "dev", "holdout"):
        for row in read_jsonl(data / f"{split}.jsonl"):
            previous = expected.setdefault(row["image_path"], row["image_sha256"])
            if previous != row["image_sha256"]:
                raise ValueError(f"Conflicting image hashes in manifests: {row['image_path']}")

    def check(item: tuple[str, str]) -> tuple[str, str, str] | None:
        relative, wanted = item
        try:
            observed = sha256_file(root / relative)
        except OSError as error:
            return relative, wanted, f"read-error:{error}"
        return None if observed == wanted else (relative, wanted, observed)

    problems = []
    with ThreadPoolExecutor(max_workers=max(1, min(8, workers or 1))) as pool:
        results = pool.map(check, expected.items())
        for result in tqdm(results, total=len(expected), desc="Preflight image hashes", unit="image"):
            if result is not None:
                problems.append(result)
    if problems:
        preview = "; ".join(f"{path} expected={wanted} observed={observed}"
                            for path, wanted, observed in problems[:5])
        raise ValueError(f"Image preflight found {len(problems)} mismatch(es): {preview}")
    return len(expected)


def make_loader(dataset, config, *, epoch=None, start_batch: int = 0):
    generator = torch.Generator().manual_seed(config["seed"] + (epoch or 0))
    if epoch is None:
        sampler = None
    else:
        order = torch.randperm(len(dataset), generator=generator).tolist()
        offset = start_batch * config["batch_size"]
        if offset > len(order):
            raise ValueError("Resume batch is beyond the end of the epoch")
        sampler = [(index, epoch) for index in order[offset:]]
    return DataLoader(dataset, batch_size=config["batch_size"], sampler=sampler,
                      num_workers=config["workers"], pin_memory=True, generator=generator,
                      persistent_workers=False, drop_last=False)


def run(root: Path, run_dir: Path, config: dict, *, resume: bool):
    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Training requires CUDA; CPU fallback is disabled")
    torch.cuda.set_device(device)
    if config["precision"] == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is unavailable; explicitly choose --precision fp32")
    checkpoint = root / BASE_CHECKPOINT
    if sha256_file(checkpoint) != BASE_SHA256:
        raise ValueError("Initial checkpoint differs from the fixed baseline model")
    data = prepared_data_path(root, config["data_id"])
    metadata = verify_data(data, root)
    verified_images = verify_image_files(data, root, metadata, config["workers"])
    hashes = source_hashes(root)
    critical_hashes = training_source_hashes(root)
    if resume:
        expected_critical = config.get("training_source_hashes")
        sources_changed = (critical_hashes != expected_critical if expected_critical is not None
                           else hashes != config["source_hashes"])
        if metadata != config["data_metadata"] or sources_changed:
            raise ValueError("Data or source changed since this run; resume refused")
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        config.update(data_metadata=metadata, source_hashes=hashes,
                      training_source_hashes=critical_hashes, checkpoint_sha256=BASE_SHA256,
                      torch_version=str(torch.__version__), gpu=torch.cuda.get_device_name(device),
                      effective_batch=config["batch_size"] * config["accumulation"],
                      preflight_unique_images=verified_images)
        write_json_atomic(run_dir / "config.json", config)
    write_json_atomic(run_dir / "status.json", {"status": "running"})
    with (run_dir / "events.jsonl").open("a") as log:
        try:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            random.seed(config["seed"])
            np.random.seed(config["seed"])
            torch.manual_seed(config["seed"])
            torch.cuda.manual_seed_all(config["seed"])
            torch.backends.cuda.matmul.fp32_precision = "ieee"
            torch.backends.cudnn.conv.fp32_precision = "ieee"
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True)
            integrity_log = run_dir / "integrity_events.jsonl"
            integrity_log.touch(exist_ok=True)
            train = VGGDataset(root, data / "train.jsonl", augment=True, seed=config["seed"],
                               integrity_log=integrity_log)
            dev = VGGDataset(root, data / "dev.jsonl", augment=False, seed=config["seed"],
                             integrity_log=integrity_log)
            dev_loader = make_loader(dev, config)
            model = SixDRepNet360(rotation_fp32=True)
            load_checkpoint(model, checkpoint)
            model.to(device)
            optimizer = make_optimizer(model, config)
            updates_per_epoch = math.ceil(math.ceil(len(train) / config["batch_size"]) / config["accumulation"])
            scheduler = make_scheduler(optimizer, updates_per_epoch * config["epochs"], updates_per_epoch)
            scaler = torch.amp.GradScaler("cuda", enabled=False)  # BF16 needs no loss scaling.
            start_epoch, start_batch, step, best = 0, 0, 0, float("inf")
            resume_accumulator = {"total_loss": 0.0, "seen": 0}

            def save_last(resume_epoch, next_batch=0, accumulator=None):
                atomic_save(run_dir / "last.pt", {"schema": SCHEMA, "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(), "rng": rng_state(), "next_epoch": resume_epoch,
                    "next_batch": next_batch,
                    "epoch_accumulator": accumulator or {"total_loss": 0.0, "seen": 0},
                    "global_step": step, "best_dev_deg": best, "config": config})

            if resume:
                state = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=True)
                if state["config"] != config or state["schema"] != SCHEMA:
                    raise ValueError("Resume checkpoint/config mismatch")
                model.load_state_dict(state["state_dict"], strict=True)
                optimizer.load_state_dict(state["optimizer"])
                scheduler.load_state_dict(state["scheduler"])
                scaler.load_state_dict(state["scaler"])
                restore_rng(state["rng"])
                start_epoch, step, best = state["next_epoch"], state["global_step"], state["best_dev_deg"]
                start_batch = state.get("next_batch", 0)
                resume_accumulator = state.get("epoch_accumulator", resume_accumulator)
                del state
                event(log, "resume", next_epoch=start_epoch, next_batch=start_batch, step=step)
            else:
                # This is VGG dev, never one of the protected benchmarks.
                baseline = evaluate_dev(model, dev_loader, device)
                best = baseline["geodesic_mean_deg"]
                write_json_atomic(run_dir / "dev_baseline.json", baseline)
                atomic_save(run_dir / "best.pth", {"state_dict": model.state_dict(), "epoch": 0,
                    "dev": baseline, "selection": "VGGHeads dev geodesic_mean_deg"})
                save_last(0)
                event(log, "dev_baseline", **baseline)
            dtype = torch.bfloat16 if config["precision"] == "bf16" else torch.float32
            for epoch in range(start_epoch, config["epochs"]):
                training_mode(model, head_only=epoch < config["head_warmup_epochs"])
                epoch_start_batch = start_batch if epoch == start_epoch else 0
                loader = make_loader(train, config, epoch=epoch, start_batch=epoch_start_batch)
                if epoch == start_epoch and epoch_start_batch:
                    total_loss = float(resume_accumulator["total_loss"])
                    seen = int(resume_accumulator["seen"])
                else:
                    total_loss, seen = 0., 0
                seen_at_start = seen
                full_batches = math.ceil(len(train) / config["batch_size"])
                optimizer.zero_grad(set_to_none=True)
                start = time.monotonic()
                bar = tqdm(loader, desc=f"Train epoch {epoch + 1}/{config['epochs']}", unit="batch")
                for local_batch, (images, target) in enumerate(bar):
                    batch_index = epoch_start_batch + local_batch
                    images, target = images.to(device, non_blocking=True), target.to(device, non_blocking=True)
                    # Weight partial final batches by samples, not by microbatch count.
                    window_start = (batch_index // config["accumulation"]) * config["accumulation"] * config["batch_size"]
                    window_samples = min(config["effective_batch"], len(train) - window_start)
                    context = torch.autocast("cuda", dtype=dtype) if dtype != torch.float32 else nullcontext()
                    with context:
                        prediction = model(images)
                        losses = rotation_loss(prediction, target)
                        loss = losses.sum() / window_samples
                    if not torch.isfinite(loss):
                        raise ValueError("Non-finite training loss")
                    scaler.scale(loss).backward()
                    total_loss += losses.detach().sum().item()
                    seen += len(images)
                    if (batch_index + 1) % config["accumulation"] == 0 or batch_index + 1 == full_batches:
                        norm = optimizer_update(model, optimizer, scheduler, scaler, config["clip_norm"])
                        step += 1
                        if step % 100 == 0:
                            event(log, "training_progress", epoch=epoch + 1, step=step,
                                  samples=seen, loss_rad=total_loss / seen, gradient_norm=norm,
                                  lr=[group["lr"] for group in optimizer.param_groups])
                        if step % config["checkpoint_interval_updates"] == 0:
                            accumulator = {"total_loss": total_loss, "seen": seen}
                            save_last(epoch, batch_index + 1, accumulator)
                            event(log, "checkpoint_saved", epoch=epoch + 1,
                                  next_batch=batch_index + 1, step=step)
                    bar.set_postfix(loss=f"{total_loss / seen:.4f}", lr=f"{optimizer.param_groups[1]['lr']:.2e}",
                                    heads_s=f"{(seen - seen_at_start) / max(time.monotonic() - start, .001):.1f}")
                metrics = evaluate_dev(model, dev_loader, device)
                improved = metrics["geodesic_mean_deg"] < best
                if improved:
                    best = metrics["geodesic_mean_deg"]
                    atomic_save(run_dir / "best.pth", {"state_dict": model.state_dict(), "epoch": epoch + 1,
                        "dev": metrics, "selection": "VGGHeads dev geodesic_mean_deg"})
                save_last(epoch + 1)
                result = {"epoch": epoch + 1, "step": step, "train_loss_rad": total_loss / seen,
                          "dev": metrics, "best_dev_deg": best, "improved": improved}
                write_json_atomic(run_dir / f"epoch_{epoch + 1:03d}.json", result)
                event(log, "epoch_completed", **result)
                tqdm.write(f"epoch {epoch + 1}: dev {metrics['geodesic_mean_deg']:.3f}°, best {best:.3f}°")
                start_batch = 0
                resume_accumulator = {"total_loss": 0.0, "seen": 0}
            write_json_atomic(run_dir / "status.json", {"status": "completed", "epochs": config["epochs"],
                                                       "step": step, "best_dev_deg": best})
            event(log, "completed", best_dev_deg=best, step=step)
        except BaseException as error:
            status = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            write_json_atomic(run_dir / "status.json", {"status": status, "error": str(error),
                "resume": "last.pt restarts at the latest completed optimizer update"})
            event(log, status, error=str(error))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--data-id", default="vgg_data")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--head-warmup-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--accumulation", type=int, default=2)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--clip-norm", type=float, default=1.)
    parser.add_argument("--checkpoint-interval-updates", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_dir = run_output_path(ROOT, args.run_id)
    config = vars(args).copy()
    config.pop("resume")
    if args.resume:
        # Resume uses the original schedule/settings verbatim, never silently overrides them.
        import sys
        allowed = {"--run-id", "--resume"}
        if any(value.startswith("--") and value.split("=")[0] not in allowed for value in sys.argv[1:]):
            parser.error("--resume accepts only --run-id; all settings come from config.json")
        config = json.loads((run_dir / "config.json").read_text())
    if (min(config["epochs"], config["batch_size"], config["accumulation"],
            config["checkpoint_interval_updates"]) <= 0 or config["workers"] < 0
        or not 0 <= config["head_warmup_epochs"] < config["epochs"]
        or min(config["backbone_lr"], config["head_lr"], config["clip_norm"]) <= 0
        or config["weight_decay"] < 0 or not 0 <= config["seed"] < 2**32
        or not all(math.isfinite(config[key]) for key in ("backbone_lr", "head_lr", "clip_norm", "weight_decay"))):
        parser.error("Invalid training parameters")
    run(ROOT, run_dir, config, resume=args.resume)


if __name__ == "__main__":
    main()
