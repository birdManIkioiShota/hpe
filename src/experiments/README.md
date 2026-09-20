# Experiment tooling

This package contains experiment-only orchestration. Reusable model, geometry, dataset,
loss, and evaluation primitives belong in their normal production modules.

Generated experiment state is stored outside the source tree:

```text
experiments/runs/<run-id>/
  config.json
  provenance.json
  status.json
  events.jsonl
  checkpoints/
  metrics/
  predictions/
  artifacts/
```

The source tree must not encode conversational history, temporary phase numbers, or
experiment conclusions in module, class, or function names. Run conditions belong in
configuration/provenance files and generated run directories.

## Pose-distribution audit

Prepared rotation-matrix manifests can be audited with a full-range head-forward
azimuth that does not depend on canonical Euler yaw:

```bash
uv run python -m experiments.scripts.analyze_pose_distribution \
  --run-id vgg_pose_distribution \
  --data-id vgg_data
```

The script writes sample-level azimuths under `artifacts/` and aggregate counts under
`metrics/`.

## Checkpoint interpolation

An unchanged SixDRepNet360 checkpoint can be created between the audited base weights
and another compatible checkpoint:

```bash
uv run python -m experiments.scripts.interpolate_checkpoints \
  --run-id weight_interp_a025 \
  --alpha 0.25 \
  --candidate-checkpoint ft_runs/vgg_ft/best.pth
```

The resulting strict-loadable checkpoint is
`experiments/runs/<run-id>/checkpoints/best.pth`.

## Final benchmark evaluation

A completed experiment run with `checkpoints/best.pth` can be evaluated against an
existing audited full baseline:

```bash
uv run python -m experiments.scripts.evaluate_candidate \
  --run-id weight_interp_a025 \
  --baseline-run baseline_fp32
```

Final benchmark output remains under `eval/<run-id>/`, with paired comparisons under
`eval/comparisons/`.
