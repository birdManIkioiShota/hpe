# Experiment outputs

Generated experiment runs are stored under `experiments/runs/<run-id>/`.

Source code does not belong here. Experiment orchestration is implemented under
`src/experiments/`. Large checkpoint binaries under `runs/` are excluded from Git,
while JSON/JSONL/CSV metadata and metrics may be retained for reproducibility.
