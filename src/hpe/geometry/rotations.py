from __future__ import annotations

import math

import torch
from torch.nn import functional as functional


def rotation_matrix_from_6d(poses: torch.Tensor) -> torch.Tensor:
    if poses.ndim != 2 or poses.shape[1] != 6:
        raise ValueError(f"Expected [batch, 6], got {tuple(poses.shape)}")
    x = functional.normalize(poses[:, :3], dim=1, eps=1e-8)
    z = functional.normalize(torch.cross(x, poses[:, 3:], dim=1), dim=1, eps=1e-8)
    y = torch.cross(z, x, dim=1)
    return torch.stack((x, y, z), dim=2)


def matrix_to_euler_degrees(rotation_matrices: torch.Tensor) -> torch.Tensor:
    """Return pitch, yaw, roll using the published 6DRepNet360 convention."""
    matrices = rotation_matrices
    sy = torch.sqrt(matrices[:, 0, 0].square() + matrices[:, 1, 0].square())
    singular = sy < 1e-6
    pitch = torch.atan2(matrices[:, 2, 1], matrices[:, 2, 2])
    yaw = torch.atan2(-matrices[:, 2, 0], sy)
    roll = torch.atan2(matrices[:, 1, 0], matrices[:, 0, 0])
    pitch_singular = torch.atan2(-matrices[:, 1, 2], matrices[:, 1, 1])
    pitch = torch.where(singular, pitch_singular, pitch)
    roll = torch.where(singular, torch.zeros_like(roll), roll)
    return torch.stack((pitch, yaw, roll), dim=1) * (180.0 / math.pi)


def euler_degrees_to_matrix(euler_degrees: torch.Tensor) -> torch.Tensor:
    """Build Rz @ Ry @ Rx from columns [pitch, yaw, roll]."""
    if euler_degrees.ndim != 2 or euler_degrees.shape[1] != 3:
        raise ValueError(f"Expected [batch, 3], got {tuple(euler_degrees.shape)}")
    angles = torch.deg2rad(euler_degrees)
    pitch, yaw, roll = angles.unbind(dim=1)
    cx, sx = torch.cos(pitch), torch.sin(pitch)
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    cz, sz = torch.cos(roll), torch.sin(roll)
    rows = (
        torch.stack((cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx), dim=1),
        torch.stack((sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx), dim=1),
        torch.stack((-sy, cy * sx, cy * cx), dim=1),
    )
    return torch.stack(rows, dim=1)


def circular_error_degrees(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.remainder(prediction - target + 180.0, 360.0).sub(180.0).abs()


def geodesic_error_degrees(
    prediction: torch.Tensor, target: torch.Tensor, *, stable: bool = False
) -> torch.Tensor:
    relative = target.transpose(1, 2) @ prediction
    cosine = ((relative.diagonal(dim1=1, dim2=2).sum(dim=1) - 1.0) / 2.0).clamp(-1.0, 1.0)
    if stable:
        # atan2 avoids the acos endpoint sensitivity near 0 and 180 degrees.
        skew = torch.stack(
            (relative[:, 2, 1] - relative[:, 1, 2],
             relative[:, 0, 2] - relative[:, 2, 0],
             relative[:, 1, 0] - relative[:, 0, 1]), dim=1,
        )
        sine = torch.linalg.vector_norm(skew, dim=1) / 2.0
        return torch.rad2deg(torch.atan2(sine, cosine))
    return torch.rad2deg(torch.acos(cosine))


def vector_errors_degrees(
    prediction: torch.Tensor, target: torch.Tensor, *, rows: bool = False
) -> torch.Tensor:
    if rows:
        # The upstream test.py uses R[:, i], i.e. rows. Normalize away roundoff.
        prediction = functional.normalize(prediction, dim=2)
        target = functional.normalize(target, dim=2)
        sine = torch.linalg.vector_norm(torch.cross(prediction, target, dim=2), dim=2)
        cosine = (prediction * target).sum(dim=2).clamp(-1.0, 1.0)
        return torch.rad2deg(torch.atan2(sine, cosine))
    cosine = (prediction * target).sum(dim=1).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cosine))
