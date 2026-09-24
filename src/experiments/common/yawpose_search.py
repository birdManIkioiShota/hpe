from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from hpe.datasets.common import write_json_atomic


@dataclass(frozen=True)
class YawPoseCondition:
    condition_id: str
    subset: str
    adoption_ratio: float
    hpe_regime: str
    use_dad: bool


def yawpose_conditions() -> list[YawPoseCondition]:
    subsets = (
        ("Y_base", "base", 0.0),
        ("Y_top20", "top020", 0.2),
        ("Y_top40", "top040", 0.4),
        ("Y_top60", "top060", 0.6),
        ("Y_top80", "top080", 0.8),
        ("Y_top100", "top100", 1.0),
    )
    conditions: list[YawPoseCondition] = []
    for suffix, use_dad, regime in (
        ("vgg", False, "vgg_only"),
        ("vgg_dad", True, "vgg_plus_dad"),
    ):
        for base_id, subset, ratio in subsets:
            conditions.append(
                YawPoseCondition(
                    condition_id=f"{base_id}_{suffix}",
                    subset=subset,
                    adoption_ratio=ratio,
                    hpe_regime=regime,
                    use_dad=use_dad,
                )
            )
    return conditions


def condition_records() -> list[dict[str, Any]]:
    return [asdict(condition) for condition in yawpose_conditions()]


def summarize_search(run_dir: Path) -> Path:
    config = json.loads((run_dir / "config.json").read_text())
    rows = []
    for condition in config["conditions"]:
        condition_dir = run_dir / "conditions" / condition["condition_id"]
        status = json.loads((condition_dir / "status.json").read_text())
        if status.get("status") != "completed":
            raise ValueError(
                f"condition is not completed: {condition['condition_id']}"
            )
        final = json.loads(
            (
                condition_dir
                / "metrics"
                / f"epoch_{int(config['epochs']):03d}.json"
            ).read_text()
        )
        rows.append(
            {
                **condition,
                "epoch": int(config["epochs"]),
                "best_epoch": status["best_epoch"],
                "rear_mean_deg": final["dev"]["rear"]["mean_deg"],
                "rear_p90_deg": final["dev"]["rear"]["p90_deg"],
                "front_mean_deg": final["dev"]["front"]["mean_deg"],
                "side_mean_deg": final["dev"]["side"]["mean_deg"],
                "rear_yaw_mean_deg": final["dev_yaw"]["rear"]["mean_deg"],
                "rear_yaw_median_deg": final["dev_yaw"]["rear"][
                    "median_deg"
                ],
                "rear_yaw_p90_deg": final["dev_yaw"]["rear"]["p90_deg"],
                "rear_yaw_over30_percent": final["dev_yaw"]["rear"][
                    "over30_percent"
                ],
                "rear_yaw_over60_percent": final["dev_yaw"]["rear"][
                    "over60_percent"
                ],
                "rear_yaw_over90_percent": final["dev_yaw"]["rear"][
                    "over90_percent"
                ],
                "yawpose_sampling": final["train"]["yawpose_sampling"],
            }
        )
    output = run_dir / "metrics" / "comparison.json"
    write_json_atomic(output, rows)
    return output
