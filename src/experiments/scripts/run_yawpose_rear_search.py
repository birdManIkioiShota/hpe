"""Run a fixed YawPose adoption or top060 loss-weight comparison."""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import json
import os
import queue
import subprocess
import sys
import threading
import time
from typing import BinaryIO, Callable

from tqdm import tqdm

from experiments.common.run_directory import (
    ExperimentRun,
    SEARCH_SUBDIRECTORIES,
    experiment_condition_path,
    experiment_run_path,
)
from experiments.common.yawpose_search import (
    WEIGHT_SEARCH_KIND,
    WEIGHT_SEARCH_RUN_ID,
    condition_records,
    summarize_search,
    yawpose_conditions,
    yawpose_weight_conditions,
)
from hpe.datasets.common import sha256_file
from training.audit import BASE_CHECKPOINT, BASE_SHA256
from training.prepare_data import ROOT, prepared_data_path


PROGRESS_PREFIX = "HPE_PROGRESS "
SOURCE_FILES = (
    "src/experiments/common/yawpose.py",
    "src/experiments/common/yawpose_search.py",
    "src/experiments/scripts/run_yawpose_rear_search.py",
    "src/experiments/scripts/train_yawpose_rear.py",
)


@dataclass(frozen=True)
class ProcessTask:
    condition_id: str
    command: list[str]
    resume: bool


@dataclass
class ActiveProcess:
    task: ProcessTask
    process: subprocess.Popen
    stdout_thread: threading.Thread
    stderr_thread: threading.Thread
    stdout_tail: deque[str]
    stderr_tail: deque[str]


@dataclass(frozen=True)
class PoolResult:
    completed: tuple[str, ...]
    max_active: int


class ConditionProcessError(RuntimeError):
    def __init__(self, condition_id: str, returncode: int, details: str) -> None:
        message = f"condition failed: {condition_id} (exit {returncode})"
        if details:
            message += f"\n{details}"
        super().__init__(message)
        self.condition_id = condition_id
        self.returncode = returncode


class SearchProgress:
    def __init__(
        self,
        condition_ids: list[str],
        epochs: int,
        *,
        interactive: bool | None = None,
        output=None,
    ) -> None:
        self.epochs = epochs
        self.interactive = (
            sys.stderr.isatty() if interactive is None else interactive
        )
        self.output = output or sys.stderr
        self.completed_ids: set[str] = set()
        self.overall = None
        self.epoch_bars: dict[str, tqdm] = {}
        self.batch_bars: dict[str, tqdm] = {}
        self.batch_totals: dict[str, int] = {}
        if self.interactive:
            self.overall = tqdm(
                total=len(condition_ids),
                desc="YawPose rear search",
                unit="condition",
                position=0,
                leave=True,
                dynamic_ncols=True,
                file=self.output,
            )
            width = max(len(value) for value in condition_ids)
            for index, condition_id in enumerate(condition_ids):
                epoch_bar = tqdm(
                    total=epochs,
                    desc=f"{condition_id:<{width}}",
                    unit="epoch",
                    position=1 + index * 2,
                    leave=True,
                    dynamic_ncols=True,
                    file=self.output,
                )
                epoch_bar.set_postfix_str("pending", refresh=False)
                batch_bar = tqdm(
                    total=1,
                    desc=f"{'':<{width}} └ batch",
                    unit="batch",
                    position=2 + index * 2,
                    leave=True,
                    dynamic_ncols=True,
                    file=self.output,
                )
                batch_bar.set_postfix_str("pending", refresh=False)
                self.epoch_bars[condition_id] = epoch_bar
                self.batch_bars[condition_id] = batch_bar

    def _line(self, condition_id: str, message: str) -> None:
        if not self.interactive:
            print(
                f"[{condition_id}] {message}",
                file=self.output,
                flush=True,
            )

    def started(self, condition_id: str, *, resume: bool) -> None:
        status = "resuming" if resume else "starting"
        if self.interactive:
            self.epoch_bars[condition_id].set_postfix_str(status)
            self.batch_bars[condition_id].set_postfix_str("waiting")
        else:
            self._line(condition_id, status)

    def handle_event(self, condition_id: str, payload: dict) -> None:
        event = payload.get("event")
        if event == "training_started":
            start_epoch = int(payload["start_epoch"])
            if self.interactive:
                epoch_bar = self.epoch_bars[condition_id]
                epoch_bar.n = min(start_epoch, self.epochs)
                epoch_bar.set_postfix_str("running")
                epoch_bar.refresh()
                batches = max(1, int(payload["batches_per_epoch"]))
                self.batch_totals[condition_id] = batches
                batch_bar = self.batch_bars[condition_id]
                batch_bar.reset(total=batches)
                batch_bar.set_postfix_str("waiting")
            return
        if event == "epoch_started":
            epoch = int(payload["epoch"])
            if self.interactive:
                epoch_bar = self.epoch_bars[condition_id]
                epoch_bar.n = max(0, epoch - 1)
                epoch_bar.set_postfix_str(
                    f"epoch={epoch}/{self.epochs} training"
                )
                epoch_bar.refresh()
                batch_bar = self.batch_bars[condition_id]
                batch_bar.reset(
                    total=self.batch_totals.get(condition_id, 1)
                )
                batch_bar.set_postfix_str("preparing")
            return
        if event == "batch_progress":
            epoch = int(payload["epoch"])
            batch = int(payload["batch"])
            batches = int(payload["batches"])
            loss = float(payload["loss"])
            if self.interactive:
                epoch_bar = self.epoch_bars[condition_id]
                epoch_bar.n = max(0, epoch - 1)
                epoch_bar.set_postfix_str(
                    f"epoch={epoch}/{self.epochs} training"
                )
                epoch_bar.refresh()
                batch_bar = self.batch_bars[condition_id]
                if self.batch_totals.get(condition_id) != batches:
                    self.batch_totals[condition_id] = batches
                    batch_bar.reset(total=batches)
                batch_bar.n = min(batch, batches)
                batch_bar.set_postfix_str(f"loss={loss:.4f}")
                batch_bar.refresh()
            return
        if event == "phase":
            if self.interactive:
                self.epoch_bars[condition_id].set_postfix_str(
                    str(payload.get("phase", "running"))
                )
            return
        if event == "epoch_completed":
            epoch = int(payload["epoch"])
            if self.interactive:
                epoch_bar = self.epoch_bars[condition_id]
                epoch_bar.n = min(epoch, self.epochs)
                epoch_bar.set_postfix_str(
                    f"epoch={epoch}/{self.epochs} "
                    f"best={int(payload['best_epoch'])}"
                )
                epoch_bar.refresh()
                batch_bar = self.batch_bars[condition_id]
                batch_bar.n = batch_bar.total or batch_bar.n
                batch_bar.set_postfix_str("completed")
                batch_bar.refresh()
            else:
                self._line(
                    condition_id,
                    f"epoch {epoch}/{self.epochs} completed",
                )

    def completed(self, condition_id: str) -> None:
        if condition_id in self.completed_ids:
            return
        self.completed_ids.add(condition_id)
        if self.interactive:
            epoch_bar = self.epoch_bars[condition_id]
            epoch_bar.n = self.epochs
            epoch_bar.set_postfix_str("completed")
            epoch_bar.refresh()
            batch_bar = self.batch_bars[condition_id]
            batch_bar.n = batch_bar.total or batch_bar.n
            batch_bar.set_postfix_str("completed")
            batch_bar.refresh()
            assert self.overall is not None
            self.overall.update(1)
            self.overall.set_postfix(
                completed=len(self.completed_ids),
                refresh=True,
            )
        else:
            self._line(condition_id, "completed")

    def failed(self, condition_id: str, returncode: int) -> None:
        if self.interactive:
            self.epoch_bars[condition_id].set_postfix_str(
                f"failed exit={returncode}"
            )
            self.batch_bars[condition_id].set_postfix_str(
                f"failed exit={returncode}"
            )
        else:
            self._line(condition_id, f"failed exit={returncode}")

    def close(self) -> None:
        if self.interactive:
            condition_ids = list(self.epoch_bars)
            for condition_id in reversed(condition_ids):
                self.batch_bars[condition_id].close()
                self.epoch_bars[condition_id].close()
        if self.overall is not None:
            self.overall.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    parser.add_argument(
        "--search-mode", choices=("adoption", "loss-weight"), default="adoption",
    )
    parser.add_argument(
        "--reliability-run",
        default="yawpose_reliability_stratified15",
    )
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
    parser.add_argument("--yawpose-weight", type=float)
    parser.add_argument("--retention-tolerance-deg", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--parallel-conditions", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _condition_command(
    args: argparse.Namespace,
    condition,
    *,
    resume: bool,
) -> list[str]:
    yawpose_weight = getattr(condition, "yawpose_weight", args.yawpose_weight)
    if yawpose_weight is None:
        yawpose_weight = 0.2
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
        repr(yawpose_weight),
        "--retention-tolerance-deg",
        repr(args.retention_tolerance_deg),
        "--seed",
        str(args.seed),
        "--progress-events",
    ]
    if args.samples_per_epoch is not None:
        command.extend(("--samples-per-epoch", str(args.samples_per_epoch)))
    if condition.use_dad:
        command.append("--use-dad")
    if resume:
        command.append("--resume")
    return command


def _decode_line(value: bytes) -> str:
    return value.decode("utf-8", errors="replace").rstrip("\r")


def _read_stdout(
    stream: BinaryIO,
    condition_id: str,
    messages: queue.Queue,
    tail: deque[str],
) -> None:
    buffer = b""
    prefix = PROGRESS_PREFIX.encode("utf-8")
    try:
        while True:
            chunk = os.read(stream.fileno(), 4096)
            if not chunk:
                break
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                if line.startswith(prefix):
                    try:
                        payload = json.loads(line[len(prefix):])
                    except (json.JSONDecodeError, UnicodeDecodeError) as error:
                        tail.append(f"invalid progress event: {error}")
                    else:
                        messages.put((condition_id, payload))
                elif line:
                    tail.append(_decode_line(line))
        if buffer:
            tail.append(_decode_line(buffer))
    finally:
        stream.close()


def _read_stderr(stream: BinaryIO, tail: deque[str]) -> None:
    try:
        while True:
            chunk = os.read(stream.fileno(), 4096)
            if not chunk:
                break
            for line in chunk.replace(b"\r", b"\n").splitlines():
                if line:
                    tail.append(_decode_line(line))
    finally:
        stream.close()


def _start_process(
    task: ProcessTask,
    messages: queue.Queue,
) -> ActiveProcess:
    process = subprocess.Popen(
        task.command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )
    assert process.stdout is not None
    assert process.stderr is not None
    stdout_tail: deque[str] = deque(maxlen=40)
    stderr_tail: deque[str] = deque(maxlen=80)
    stdout_thread = threading.Thread(
        target=_read_stdout,
        args=(process.stdout, task.condition_id, messages, stdout_tail),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=_read_stderr,
        args=(process.stderr, stderr_tail),
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()
    return ActiveProcess(
        task=task,
        process=process,
        stdout_thread=stdout_thread,
        stderr_thread=stderr_thread,
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
    )


def _failure_details(active: ActiveProcess) -> str:
    lines = list(active.stdout_tail) + list(active.stderr_tail)
    return "\n".join(lines[-40:])


def _drain_progress(
    messages: queue.Queue,
    display: SearchProgress,
) -> None:
    while True:
        try:
            condition_id, payload = messages.get_nowait()
        except queue.Empty:
            return
        display.handle_event(condition_id, payload)


def run_process_pool(
    tasks: list[ProcessTask],
    *,
    parallel_conditions: int,
    display: SearchProgress,
    on_started: Callable[[ProcessTask], None] | None = None,
    on_completed: Callable[[ProcessTask], None] | None = None,
    on_failed: Callable[[ProcessTask, int], None] | None = None,
) -> PoolResult:
    if parallel_conditions <= 0:
        raise ValueError("parallel-conditions must be positive")
    pending = deque(tasks)
    active: dict[str, ActiveProcess] = {}
    messages: queue.Queue = queue.Queue()
    completed: list[str] = []
    failures: list[ConditionProcessError] = []
    max_active = 0

    while pending or active:
        while pending and len(active) < parallel_conditions and not failures:
            task = pending.popleft()
            display.started(task.condition_id, resume=task.resume)
            if on_started is not None:
                on_started(task)
            execution = _start_process(task, messages)
            active[task.condition_id] = execution
            max_active = max(max_active, len(active))

        _drain_progress(messages, display)
        finished = [
            condition_id
            for condition_id, execution in active.items()
            if execution.process.poll() is not None
        ]
        if not finished:
            if failures and not active:
                break
            time.sleep(0.05)
            continue

        for condition_id in finished:
            execution = active.pop(condition_id)
            returncode = execution.process.wait()
            execution.stdout_thread.join()
            execution.stderr_thread.join()
            _drain_progress(messages, display)
            if returncode == 0:
                completed.append(condition_id)
                display.completed(condition_id)
                if on_completed is not None:
                    on_completed(execution.task)
            else:
                display.failed(condition_id, returncode)
                if on_failed is not None:
                    on_failed(execution.task, returncode)
                failures.append(
                    ConditionProcessError(
                        condition_id,
                        returncode,
                        _failure_details(execution),
                    )
                )

        if failures and not active:
            break

    if failures:
        raise failures[0]
    return PoolResult(tuple(completed), max_active)


def _search_config(args: argparse.Namespace) -> dict:
    """Resolve mode-specific defaults without changing legacy run records."""
    weight_search = args.search_mode == "loss-weight"
    if weight_search and args.yawpose_weight is not None:
        raise ValueError("loss-weight mode uses its fixed six coefficients")
    if args.run_id is None:
        args.run_id = (
            WEIGHT_SEARCH_RUN_ID if weight_search else "yawpose_rear_stratified_search"
        )
    if args.yawpose_weight is None and not weight_search:
        args.yawpose_weight = 0.2
    excluded = {"run_id", "dry_run", "parallel_conditions", "search_mode"}
    if weight_search:
        excluded.add("yawpose_weight")
    return {
        "kind": WEIGHT_SEARCH_KIND if weight_search else "yawpose_rear_adoption_search",
        "epochs": args.epochs,
        "conditions": condition_records(args.search_mode),
        "reliability_run": args.reliability_run,
        "shared_training": {
            key: value for key, value in vars(args).items() if key not in excluded
        },
    }


def _validate_weight_inputs(provenance: dict, reference: dict) -> None:
    expected_manifests = {**reference["train_manifests"], **reference["dev_manifests"]}
    if provenance["training_manifests"] != expected_manifests:
        raise ValueError("loss-weight comparison requires the previous VGG+DAD inputs")
    selected = provenance["reliability_subsets"]["top060"]
    previous = reference["yawpose_subset"]
    if previous["count"] != 8433 or any(
        selected[key] != previous[key] for key in ("path", "sha256")
    ):
        raise ValueError("loss-weight comparison requires the previous top060 subset")


def main() -> None:
    args = build_parser().parse_args()
    if args.parallel_conditions <= 0:
        raise ValueError("parallel-conditions must be positive")
    config = _search_config(args)
    conditions = (
        yawpose_weight_conditions()
        if args.search_mode == "loss-weight"
        else yawpose_conditions()
    )
    reliability_dir = experiment_run_path(ROOT, args.reliability_run)
    reliability_summary = reliability_dir / "metrics" / "summary.json"
    if args.dry_run:
        print(
            json.dumps(
                {
                    **config,
                    "orchestration": {
                        "parallel_conditions": args.parallel_conditions,
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    reliability_status = reliability_dir / "status.json"
    if (
        not reliability_status.is_file()
        or json.loads(reliability_status.read_text()).get("status") != "completed"
    ):
        raise ValueError("completed reliability precompute run is required")

    checkpoint = ROOT / BASE_CHECKPOINT
    if sha256_file(checkpoint) != BASE_SHA256:
        raise ValueError("base checkpoint SHA-256 mismatch")
    vgg_dir = prepared_data_path(ROOT, args.vgg_data_id)
    dad_dir = prepared_data_path(ROOT, args.dad_data_id)
    training_manifests = [
        vgg_dir / "train.jsonl",
        vgg_dir / "dev.jsonl",
        dad_dir / "train.jsonl",
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
    if args.search_mode == "loss-weight":
        reference = (
            experiment_run_path(ROOT, "yawpose_rear_stratified_search")
            / "conditions" / "Y_top60_vgg_dad" / "provenance.json"
        )
        _validate_weight_inputs(provenance, json.loads(reference.read_text()))
    run_path = experiment_run_path(ROOT, args.run_id)
    resumed_parent = run_path.exists()
    if resumed_parent:
        run = ExperimentRun.open(run_path)
        if json.loads((run_path / "config.json").read_text()) != config:
            raise ValueError("experiment config changed; resume refused")
        if json.loads((run_path / "provenance.json").read_text()) != provenance:
            raise ValueError("experiment inputs or source changed; resume refused")
        status = json.loads((run_path / "status.json").read_text())
        if status.get("status") == "completed":
            print(f"Experiment already completed: {run_path.relative_to(ROOT)}")
            return
    else:
        run = ExperimentRun.create_at(
            run_path,
            config=config,
            provenance=provenance,
            subdirectories=SEARCH_SUBDIRECTORIES,
        )

    display = SearchProgress(
        [condition.condition_id for condition in conditions],
        args.epochs,
    )
    completed = 0
    tasks: list[ProcessTask] = []
    try:
        for condition in conditions:
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
                    display.completed(condition.condition_id)
                    continue
                resume = True
            tasks.append(
                ProcessTask(
                    condition_id=condition.condition_id,
                    command=_condition_command(
                        args,
                        condition,
                        resume=resume,
                    ),
                    resume=resume,
                )
            )

        run.write_status(
            "running",
            resumed=resumed_parent,
            parallel_conditions=args.parallel_conditions,
            completed_conditions=completed,
            total_conditions=len(conditions),
        )
        run.event(
            "resumed" if resumed_parent else "search_started",
            parallel_conditions=args.parallel_conditions,
            completed_conditions=completed,
        )

        def on_started(task: ProcessTask) -> None:
            run.event(
                "condition_started",
                condition_id=task.condition_id,
                resume=task.resume,
            )

        def on_completed(task: ProcessTask) -> None:
            nonlocal completed
            completed += 1
            run.write_status(
                "running",
                parallel_conditions=args.parallel_conditions,
                completed_conditions=completed,
                total_conditions=len(conditions),
                last_completed_condition=task.condition_id,
            )
            run.event(
                "condition_completed",
                condition_id=task.condition_id,
            )

        def on_failed(task: ProcessTask, returncode: int) -> None:
            run.event(
                "condition_failed",
                condition_id=task.condition_id,
                returncode=returncode,
            )

        result = run_process_pool(
            tasks,
            parallel_conditions=args.parallel_conditions,
            display=display,
            on_started=on_started,
            on_completed=on_completed,
            on_failed=on_failed,
        )
        comparison = summarize_search(run.path)
        run.complete(
            completed_conditions=completed,
            total_conditions=len(conditions),
            parallel_conditions=args.parallel_conditions,
            max_parallel_conditions_observed=result.max_active,
            comparison=str(comparison.relative_to(ROOT)),
        )
    except BaseException as error:
        run.fail(error)
        raise
    finally:
        display.close()


if __name__ == "__main__":
    main()
