from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VggDevEvaluationTarget:
    experiment_id: str
    condition_id: str
    checkpoint_path: Path
    checkpoint_sha256: str
    reference_run_path: Path
    epoch: int | None = None
    reference_vgg_dev_so3_mean_deg: float | None = None

    @property
    def logical_id(self) -> str:
        return f"{self.experiment_id}:{self.condition_id}"


def _target(
    experiment_id: str,
    condition_id: str,
    checkpoint_path: str,
    checkpoint_sha256: str,
    reference_run_path: str,
    *,
    epoch: int | None = None,
    reference_vgg_dev_so3_mean_deg: float | None = None,
) -> VggDevEvaluationTarget:
    return VggDevEvaluationTarget(
        experiment_id=experiment_id,
        condition_id=condition_id,
        checkpoint_path=Path(checkpoint_path),
        checkpoint_sha256=checkpoint_sha256,
        reference_run_path=Path(reference_run_path),
        epoch=epoch,
        reference_vgg_dev_so3_mean_deg=reference_vgg_dev_so3_mean_deg,
    )


TARGETS: tuple[VggDevEvaluationTarget, ...] = (
    _target(
        "initial",
        "audited_base",
        "checkpoints/6DRepNet360_Full-Rotation_300W_LP+Panoptic.pth",
        "3ee08f1e04b8d452a6c4a40926a6f38051894ae6d0aaa6d191fe6d8bc6e4f9c6",
        "eval/baseline_fp32/evaluations/baseline/run.json",
        epoch=0,
        reference_vgg_dev_so3_mean_deg=21.426986640470798,
    ),
    _target(
        "experiment_1_vgg_ft",
        "best",
        "ft_runs/vgg_ft/best.pth",
        "f36abf2e366ce8cf13049202c2c970d8ea6bf723ee10e7ce9551e5d40f792ea4",
        "eval/vgg_ft/run.json",
        epoch=19,
        reference_vgg_dev_so3_mean_deg=2.919495880048786,
    ),
    _target(
        "experiment_2_weight_interpolation",
        "alpha_025",
        "experiments/runs/weight_interp_a025/checkpoints/best.pth",
        "c7b60cecdca9dcc9fcbcb8c05d856f7a9164521aeab1824ddb42a0bdb0fcc650",
        "eval/weight_interp_a025/run.json",
    ),
    _target(
        "experiment_3_rear_distillation",
        "best",
        "experiments/runs/dad_vgg_rear_distill/checkpoints/best.pth",
        "b9a38260a57bae6d4e2502d25cbe95096f4aa3222f565d7953308fe0473444eb",
        "eval/dad_vgg_rear_distill/run.json",
        epoch=7,
    ),
    _target(
        "experiment_4_rear_flip_consistency",
        "best",
        "experiments/runs/dad_vgg_rear_flip_consistency/checkpoints/best.pth",
        "1f1bc2ec8f6e92615d001b125437171151dd8d6eadae741d468cdbee456f359a",
        "eval/dad_vgg_rear_flip_consistency_bench/run.json",
        epoch=8,
    ),
)

_REAR_FLIP_ROWS = (
    ("rear_natural_flip_000", "9e85b7a492f5fa4459aa2a4cd2e5304616384668d0b42af1cb30eb125a130faa"),
    ("rear_natural_flip_020", "d55aa613311cef890a1664e768dd15ccec511a740220867c22a13c593b942ca8"),
    ("rear_natural_flip_100", "e19c5654f3633117e5aa7d26d0c3fadef78390860326fe69553fd8ae85920868"),
    ("rear_005_flip_000", "916eb72361647dafd9fee6dcfdcb585fbee46433de2f27f82f68475523620622"),
    ("rear_005_flip_020", "c2b4569a6ed885f166281cdbed9c01425efca6bb5f5cb308609eda6c43fc4fd5"),
    ("rear_005_flip_100", "0126c5ba4489db90776883cca6133acd78368a543f5abdd833559c78ee5fd196"),
    ("rear_025_flip_000", "4a94f2fc95eccc2c3b66fdf70c416bede8264677044469652fd939ec757c7ba4"),
    ("rear_025_flip_020", "082568cc8e273542a333a22b6b7906eb252097012a3f2364c2929b324a618b4e"),
    ("rear_025_flip_100", "bf8c743fa3fde228aee00cb66daf6ae6e935802759c3a58c7b91876582477940"),
)
TARGETS += tuple(
    _target(
        "experiment_5_rear_fraction_flip_weight",
        condition,
        f"experiments/runs/rear_flip_search/conditions/{condition}/checkpoints/epoch_010.pth",
        sha256,
        f"eval/rear_flip_search/conditions/{condition}/run.json",
        epoch=10,
    )
    for condition, sha256 in _REAR_FLIP_ROWS
)

_YAWPOSE_RELIABILITY_ROWS = (
    ("Y_base", "d6ca00665976edd3149cb68d64a16515ae964568aa07e7559bdd007becbe5441"),
    ("Y_top20", "8df7331515fe99eba2eda0dc5b4452031f1ace3a9dd372730d1f40a692440d44"),
    ("Y_top40", "8f00b85ef296742561c2935fbaa69d098d88bc70e1716b675206a7f9afd5ae0c"),
    ("Y_top60", "e3ac174529bbe75dcf08088f25fc1b9ce026f11b5e926c25419545575570ba0f"),
    ("Y_top80", "85cd78c6d967bd4a9e95e2e49790599d49fba14c32fbd82c7345e064ff39f98e"),
    ("Y_top100", "b21ca84ecc2424705d1da385f5900e96ae9953cfcdb310f7c9b5177130e38614"),
)
TARGETS += tuple(
    _target(
        "experiment_6_yawpose_reliability",
        condition,
        f"experiments/runs/yawpose_rear_search/conditions/{condition}/checkpoints/epoch_010.pth",
        sha256,
        f"eval/yawpose_rear_search/conditions/{condition}/run.json",
        epoch=10,
    )
    for condition, sha256 in _YAWPOSE_RELIABILITY_ROWS
)

_YAWPOSE_STRATIFIED_ROWS = (
    ("Y_base_vgg", "3eaffc0b0ef4ca17958ff1234fae1ba35b174345d44c79312cabc393f2d07024"),
    ("Y_top20_vgg", "7565e2917f54fe0cd7df383e9f209b5334bb53e19c59aed4cf851850c5480480"),
    ("Y_top40_vgg", "986e669f7e680bcc855644324a3519807aecee8e3c4a76d6a53bc831946d4179"),
    ("Y_top60_vgg", "ffd82b9b3610d8c3913f93167b359ed6c37ddf0c48866e1f7d59ad00c6807c73"),
    ("Y_top80_vgg", "2e2704923dbc510690717c4097c1b5c0ec4fdf64d7aa8da3d1de03ec86b50c0e"),
    ("Y_top100_vgg", "57d6da2db7b6f6218a81e1d99e420ee8d31060b626185c225e186351b94bb27e"),
    ("Y_base_vgg_dad", "f26bbe4b5540a22aec916797523a0673462c9517d73daa01fd33cbed22b7637f"),
    ("Y_top20_vgg_dad", "9aaa5eac8235dbd8022457ad301a9b727c61be804fcf0e54653d566596d4fdad"),
    ("Y_top40_vgg_dad", "2020290cfbe34b091715c1be0565588f18181a8681d839b99abf46a072d79e33"),
    ("Y_top60_vgg_dad", "e7a8b8788ea0d9c206332e0ace8078f4b8bc579a111afed176b3d9b37a19ccd2"),
    ("Y_top80_vgg_dad", "83fe774d03c8735dc9c79e958eb1b2494154c966ab74dfd64796c96a941bffca"),
    ("Y_top100_vgg_dad", "adce7efe6322c54070a31026b8055862b605dab63fe510b8d0d2afa89f152853"),
)
TARGETS += tuple(
    _target(
        "experiment_7_yawpose_stratified",
        condition,
        f"experiments/runs/yawpose_rear_stratified_search/conditions/{condition}/checkpoints/epoch_010.pth",
        sha256,
        f"eval/yawpose_rear_stratified_search/conditions/{condition}/run.json",
        epoch=10,
    )
    for condition, sha256 in _YAWPOSE_STRATIFIED_ROWS
)

_YAWPOSE_WEIGHT_ROWS = (
    ("Y_base_vgg_dad", "f26bbe4b5540a22aec916797523a0673462c9517d73daa01fd33cbed22b7637f"),
    ("Y_w020_vgg_dad", "e7a8b8788ea0d9c206332e0ace8078f4b8bc579a111afed176b3d9b37a19ccd2"),
    ("Y_w040_vgg_dad", "a9190bf96981bf1e0f22dd9ebc76b7bc41e5a93aeb1a947e82ab17452d7138e5"),
    ("Y_w060_vgg_dad", "91f35cc218ba348ba1a9d2fe352149807ab96e82d2f6d8d9c31b254bb14f5feb"),
    ("Y_w080_vgg_dad", "f2c9c9cef4b934882e4eb89b86662e842af80bb51407bb01f70b1c4bffcd34c6"),
    ("Y_w100_vgg_dad", "c949e6c50ec1b74690e7e20c8af00cc6062eb6207802926911e3f70337cdbe7b"),
)
TARGETS += tuple(
    _target(
        "experiment_8_yawpose_loss_weight",
        condition,
        f"experiments/runs/yawpose_rear_weight_search/conditions/{condition}/checkpoints/epoch_010.pth",
        sha256,
        f"eval/yawpose_rear_weight_search/conditions/{condition}/run.json",
        epoch=10,
    )
    for condition, sha256 in _YAWPOSE_WEIGHT_ROWS
)


def validate_target_registry(
    targets: tuple[VggDevEvaluationTarget, ...] = TARGETS,
) -> None:
    if len(targets) != 38:
        raise ValueError(f"Expected 38 logical evaluation targets, found {len(targets)}")
    logical_ids = [target.logical_id for target in targets]
    if len(set(logical_ids)) != len(logical_ids):
        raise ValueError("Duplicate logical evaluation target")
    unique_checkpoints = {target.checkpoint_sha256 for target in targets}
    if len(unique_checkpoints) != 36:
        raise ValueError(
            f"Expected 36 unique checkpoints after deduplication, found {len(unique_checkpoints)}"
        )
    by_sha: dict[str, list[str]] = {}
    for target in targets:
        by_sha.setdefault(target.checkpoint_sha256, []).append(target.logical_id)
    shared = sorted(sorted(values) for values in by_sha.values() if len(values) > 1)
    expected = sorted(
        [
            sorted(
                [
                    "experiment_7_yawpose_stratified:Y_base_vgg_dad",
                    "experiment_8_yawpose_loss_weight:Y_base_vgg_dad",
                ]
            ),
            sorted(
                [
                    "experiment_7_yawpose_stratified:Y_top60_vgg_dad",
                    "experiment_8_yawpose_loss_weight:Y_w020_vgg_dad",
                ]
            ),
        ]
    )
    if shared != expected:
        raise ValueError(f"Unexpected shared-checkpoint groups: {shared}")


validate_target_registry()
