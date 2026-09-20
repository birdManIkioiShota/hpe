"""DAD-3DHeads annotation conventions for the shared evaluator."""
from __future__ import annotations

import numpy as np
import torch
from hpe.geometry import matrix_to_euler_degrees

SCHEMA = "dad3dheads-validation-v1"
VALIDATION_COUNT = 4312
REFERENCES = {
    "annotation_format": "https://github.com/PinataFarms/DAD-3DHeads",
    "official_rotation": "https://github.com/PinataFarms/DAD-3DHeads/blob/main/dad_3dheads_benchmark/benchmark.py",
    "rotation_to_hpe": "https://github.com/hnuzhy/SemiUHPE/blob/main/src/datasets/dataset_DAD3DHeads.py",
}


def validate_rotation(value) -> np.ndarray:
    rotation = np.asarray(value, dtype=np.float64)
    if (rotation.shape != (3, 3) or not np.isfinite(rotation).all()
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4, rtol=0)
            or not np.isclose(np.linalg.det(rotation), 1, atol=1e-4, rtol=0)):
        raise ValueError("Invalid proper rotation matrix")
    return rotation


def rotation_from_model_view(value) -> np.ndarray:
    """Convert DAD model-view rotation to the existing 6DRepNet360 convention.

    Official DAD: R_dad = D @ M, D = diag(1,-1,-1).
    HPE Euler conversion: R_hpe = R_dad.T @ D = M.T.
    Translation and projection are not head rotation.
    """
    matrix = np.asarray(value, dtype=np.float64)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6, rtol=0)):
        raise ValueError("Invalid model_view_matrix")
    return validate_rotation(matrix[:3, :3].T).copy()


def full_range_euler(rotation: np.ndarray) -> list[float]:
    """Derived RzRyRx labels for grouping: choose the branch with |pitch| <= 90.

    DAD supplies matrices, not Euler labels. This branch retains rear yaw; it
    is not a unique physical yaw for inverted/gimbal-lock poses. Metrics use R.
    """
    angles = matrix_to_euler_degrees(torch.from_numpy(rotation)[None])[0].numpy()
    if abs(angles[0]) > 90:
        angles = np.array([angles[0] + 180, 180 - angles[1], angles[2] + 180])
        angles = (angles + 180) % 360 - 180
    return angles.tolist()
