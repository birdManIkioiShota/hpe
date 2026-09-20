from __future__ import annotations

import math

import numpy as np


POSE_BANDS = (
    "front_lt60",
    "side_60_to_lt120",
    "rear_120_to_lt150",
    "rear_150_to_180",
)


class UndefinedAzimuthError(ValueError):
    """Raised when a valid rotation has no stable horizontal head-forward direction."""


def forward_azimuth_degrees(rotation_matrix: np.ndarray, *, eps: float = 1e-8) -> float:
    """Return full-range azimuth of the rotated local +Z head-forward axis.

    This is deliberately not an Euler yaw decomposition. It projects the head-forward
    direction into the camera XZ plane, so rear-facing directions remain distinguishable
    across the full [-180, 180] degree range.
    """
    rotation = np.asarray(rotation_matrix, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError("rotation_matrix must be a finite 3x3 matrix")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5, rtol=0):
        raise ValueError("rotation_matrix is not orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-5):
        raise ValueError("rotation_matrix determinant is not +1")

    direction = rotation[:, 2]
    horizontal_norm = math.hypot(float(direction[0]), float(direction[2]))
    if horizontal_norm < eps:
        raise UndefinedAzimuthError(
            "Head-forward azimuth is undefined for a near-vertical direction"
        )
    return math.degrees(math.atan2(float(direction[0]), float(direction[2])))


def pose_band(azimuth_deg: float) -> str:
    absolute = abs(float(azimuth_deg))
    if absolute > 180.0 + 1e-9:
        raise ValueError("azimuth must be within [-180, 180]")
    if absolute < 60.0:
        return "front_lt60"
    if absolute < 120.0:
        return "side_60_to_lt120"
    if absolute < 150.0:
        return "rear_120_to_lt150"
    return "rear_150_to_180"


def azimuth_side(azimuth_deg: float, *, eps: float = 1e-9) -> str:
    if azimuth_deg < -eps:
        return "negative"
    if azimuth_deg > eps:
        return "positive"
    return "center"
