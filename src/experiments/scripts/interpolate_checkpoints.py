"""Create a weight-space interpolation checkpoint without changing model structure."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from hpe.datasets.common import sha256_file
from hpe.models import SixDRepNet360, load_checkpoint
from training.audit import BASE_CHECKPOINT

from experiments.common.checkpoints import (
    atomic_save_state_dict,
    interpolate_state_dict,
    load_state_dict,
)
from experiments.common.run_directory import ExperimentRun


ROOT = Path(__file__).resolve().parents[3]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--base-checkpoint", type=Path, default=BASE_CHECKPOINT)
    parser.add_argument(
        "--candidate-checkpoint",
        type=Path,
        default=Path("ft_runs/vgg_ft/best.pth"),
    )
    return parser


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    args = build_parser().parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    base_path = _resolve(args.base_checkpoint)
    candidate_path = _resolve(args.candidate_checkpoint)
    base, base_info = load_state_dict(base_path)
    candidate, candidate_info = load_state_dict(candidate_path)

    config = {
        "kind": "checkpoint_interpolation",
        "alpha": args.alpha,
        "formula": "(1-alpha)*base + alpha*candidate",
        "base_checkpoint": str(base_path.relative_to(ROOT)) if base_path.is_relative_to(ROOT) else str(base_path),
        "candidate_checkpoint": (
            str(candidate_path.relative_to(ROOT))
            if candidate_path.is_relative_to(ROOT)
            else str(candidate_path)
        ),
    }
    run = ExperimentRun.create(
        ROOT,
        args.run_id,
        config=config,
        provenance={"base": base_info, "candidate": candidate_info},
    )
    output = run.path / "checkpoints" / "best.pth"
    try:
        interpolated = interpolate_state_dict(base, candidate, args.alpha)
        atomic_save_state_dict(
            output,
            interpolated,
            metadata={
                "kind": "checkpoint_interpolation",
                "alpha": args.alpha,
                "base_sha256": base_info["sha256"],
                "candidate_sha256": candidate_info["sha256"],
            },
        )
        # Strict-load the generated state dict into the unchanged target architecture.
        model = SixDRepNet360()
        load_checkpoint(model, output)
        digest = sha256_file(output)
        run.complete(checkpoint=str(output.relative_to(ROOT)), checkpoint_sha256=digest)
        print(
            json.dumps(
                {
                    "run": str(run.path.relative_to(ROOT)),
                    "checkpoint": str(output.relative_to(ROOT)),
                    "sha256": digest,
                    "alpha": args.alpha,
                },
                indent=2,
            )
        )
    except BaseException as error:
        run.fail(error)
        raise


if __name__ == "__main__":
    main()
