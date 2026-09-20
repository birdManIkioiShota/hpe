from __future__ import annotations

import builtins
import math
import pickle
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import pandas as pd
from pandas.core.indexes.base import Index, _new_Index
from pandas.core.internals.managers import BlockManager

from hpe.datasets.common import (
    SCHEMA_VERSION,
    relative_posix,
    sha256_file,
    write_json_atomic,
    write_jsonl_atomic,
)


# These constants and transformations reproduce DirectMHP's AGORA-HPE label
# construction. The input joints are the camera-coordinate joints stored in
# AGORA's official *_withjv.pkl files, so the alignment is performed directly
# in camera coordinates.
_KP_IDX_AGORA = np.asarray([0, 4, 9, 5, 28, 25, 22, 19, 18, 14, 37, 31, 40])
_KP_IDX_MODEL = np.asarray([38, 34, 33, 29, 13, 17, 25, 21, 54, 50, 43, 39, 45])
_FACE_START = 56 + 4 + 10 + 6

_REFERENCE_HEAD = np.asarray(
    [
        [-7.308957, 0.913869, 0.000000], [-6.775290, -0.730814, -0.012799],
        [-5.665918, -3.286078, 1.022951], [-5.011779, -4.876396, 1.047961],
        [-4.056931, -5.947019, 1.636229], [-1.833492, -7.056977, 4.061275],
        [0.000000, -7.415691, 4.070434], [1.833492, -7.056977, 4.061275],
        [4.056931, -5.947019, 1.636229], [5.011779, -4.876396, 1.047961],
        [5.665918, -3.286078, 1.022951], [6.775290, -0.730814, -0.012799],
        [7.308957, 0.913869, 0.000000], [5.311432, 5.485328, 3.987654],
        [4.461908, 6.189018, 5.594410], [3.550622, 6.185143, 5.712299],
        [2.542231, 5.862829, 4.687939], [1.789930, 5.393625, 4.413414],
        [2.693583, 5.018237, 5.072837], [3.530191, 4.981603, 4.937805],
        [4.490323, 5.186498, 4.694397], [-5.311432, 5.485328, 3.987654],
        [-4.461908, 6.189018, 5.594410], [-3.550622, 6.185143, 5.712299],
        [-2.542231, 5.862829, 4.687939], [-1.789930, 5.393625, 4.413414],
        [-2.693583, 5.018237, 5.072837], [-3.530191, 4.981603, 4.937805],
        [-4.490323, 5.186498, 4.694397], [1.330353, 7.122144, 6.903745],
        [2.533424, 7.878085, 7.451034], [4.861131, 7.878672, 6.601275],
        [6.137002, 7.271266, 5.200823], [6.825897, 6.760612, 4.402142],
        [-1.330353, 7.122144, 6.903745], [-2.533424, 7.878085, 7.451034],
        [-4.861131, 7.878672, 6.601275], [-6.137002, 7.271266, 5.200823],
        [-6.825897, 6.760612, 4.402142], [-2.774015, -2.080775, 5.048531],
        [-0.509714, -1.571179, 6.566167], [0.000000, -1.646444, 6.704956],
        [0.509714, -1.571179, 6.566167], [2.774015, -2.080775, 5.048531],
        [0.589441, -2.958597, 6.109526], [0.000000, -3.116408, 6.097667],
        [-0.589441, -2.958597, 6.109526], [-0.981972, 4.554081, 6.301271],
        [-0.973987, 1.916389, 7.654050], [-2.005628, 1.409845, 6.165652],
        [-1.930245, 0.424351, 5.914376], [-0.746313, 0.348381, 6.263227],
        [0.000000, 0.000000, 6.763430], [0.746313, 0.348381, 6.263227],
        [1.930245, 0.424351, 5.914376], [2.005628, 1.409845, 6.165652],
        [0.973987, 1.916389, 7.654050], [0.981972, 4.554081, 6.301271],
    ],
    dtype=np.float64,
).T

_E_REF = np.asarray(
    [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 50], [0, 0, 0, 1]],
    dtype=np.float64,
)


def _sphere_points() -> np.ndarray:
    points = []
    for theta_deg in range(0, 360, 10):
        for phi_deg in range(0, 180, 10):
            theta = math.radians(theta_deg)
            phi = math.radians(phi_deg)
            points.append(
                [
                    18 * math.cos(theta) * math.sin(phi),
                    18 * math.sin(theta) * math.sin(phi),
                    18 * math.cos(phi),
                ]
            )
    return (np.asarray(points, dtype=np.float64) + [0, 5, -5]).T


_SPHERE = _sphere_points()


class _RestrictedAgoraUnpickler(pickle.Unpickler):
    """Unpickle only the exact pandas/numpy constructors used by AGORA files."""

    _ALLOWED = {
        ("pandas.core.frame", "DataFrame"): pd.DataFrame,
        ("pandas.core.internals.managers", "BlockManager"): BlockManager,
        ("pandas.core.indexes.base", "_new_Index"): _new_Index,
        ("pandas.core.indexes.base", "Index"): Index,
        ("pandas.core.indexes.numeric", "Int64Index"): pd.Index,
        ("numpy.core.multiarray", "_reconstruct"): np.core.multiarray._reconstruct,
        ("numpy.core.multiarray", "scalar"): np.core.multiarray.scalar,
        ("numpy", "ndarray"): np.ndarray,
        ("numpy", "dtype"): np.dtype,
        ("numpy.core.numeric", "_frombuffer"): np.core.numeric._frombuffer,
        ("builtins", "slice"): builtins.slice,
    }

    def find_class(self, module: str, name: str):
        try:
            return self._ALLOWED[(module, name)]
        except KeyError as exc:
            raise pickle.UnpicklingError(
                f"Blocked unexpected pickle global: {module}.{name}"
            ) from exc


def _load_dataframe(stream: BinaryIO) -> pd.DataFrame:
    value = _RestrictedAgoraUnpickler(stream, encoding="iso-8859-1").load()
    if not isinstance(value, pd.DataFrame):
        raise TypeError(f"Expected pandas.DataFrame, got {type(value)!r}")
    return value


def _align_similarity(model: np.ndarray, data: np.ndarray):
    model_centered = model - model.mean(axis=1, keepdims=True)
    data_centered = data - data.mean(axis=1, keepdims=True)
    covariance = np.zeros((3, 3), dtype=np.float64)
    for column in range(model.shape[1]):
        covariance += np.outer(model_centered[:, column], data_centered[:, column])
    u, _, vh = np.linalg.svd(covariance.T)
    reflection = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vh) < 0:
        reflection[2, 2] = -1
    rotation = u @ reflection @ vh
    rotated_model = rotation @ model_centered
    scale = sum(
        np.dot(data_centered[:, i], rotated_model[:, i])
        for i in range(data_centered.shape[1])
    ) / sum(np.linalg.norm(model_centered[:, i]) ** 2 for i in range(model.shape[1]))
    translation = data.mean(axis=1, keepdims=True) - scale * rotation @ model.mean(
        axis=1, keepdims=True
    )
    return rotation, translation, float(scale)


def _inverse_rotate_zyx(matrix: np.ndarray):
    if np.linalg.norm(matrix[:3, :3].T @ matrix[:3, :3] - np.eye(3)) > 1e-5:
        raise ValueError("Matrix is not a rotation")
    if abs(matrix[0, 2]) > 0.9999999:
        z = 0.0
        if matrix[0, 2] > 0:
            y = -math.pi / 2
            x = math.atan2(-matrix[1, 0], -matrix[2, 0])
        else:
            y = math.pi / 2
            x = math.atan2(matrix[1, 0], matrix[2, 0])
        result = np.asarray([x, y, z])
        return result, result.copy()
    y0 = math.asin(-matrix[0, 2])
    y1 = math.pi - y0
    cy0, cy1 = math.cos(y0), math.cos(y1)
    x0 = math.atan2(matrix[1, 2] / cy0, matrix[2, 2] / cy0)
    x1 = math.atan2(matrix[1, 2] / cy1, matrix[2, 2] / cy1)
    z0 = math.atan2(matrix[0, 1] / cy0, matrix[0, 0] / cy0)
    z1 = math.atan2(matrix[0, 1] / cy1, matrix[0, 0] / cy1)
    return np.asarray([x0, y0, z0]), np.asarray([x1, y1, z1])


def _select_euler(two_sets: tuple[np.ndarray, np.ndarray]):
    for angles in two_sets:
        pitch, yaw, roll = np.degrees(angles)
        if yaw > 180:
            yaw -= 360
        if abs(roll) < 90 and abs(pitch) < 90:
            return float(pitch), float(-yaw), float(-roll)
    return None


def _camera_matrix(image_name: str, width: int, height: int) -> np.ndarray:
    if "hdri" in image_name:
        focal_length = 50
    elif any(camera in image_name for camera in ("cam00", "cam01", "cam02", "cam03")):
        focal_length = 18
    else:
        focal_length = 28
    fx = focal_length / 36 * width
    fy = focal_length / 20.25 * height
    return np.asarray([[fx, 0, width / 2], [0, fy, height / 2], [0, 0, 1]])


def _project(points: np.ndarray, camera: np.ndarray) -> np.ndarray:
    if np.any(points[2] <= 0):
        raise ValueError("Head sphere contains points behind the camera")
    normalized = points[:2] / points[2]
    projected = np.ones((3, points.shape[1]), dtype=np.float64)
    projected[:2] = normalized
    return camera @ projected


def _head_label(face_2d: np.ndarray, face_3d: np.ndarray, image_name: str):
    face_2d = np.asarray(face_2d, dtype=np.float64)
    face_3d = np.asarray(face_3d, dtype=np.float64)
    if face_2d.shape[0] < _FACE_START + 51 or face_3d.shape[0] < _FACE_START + 51:
        raise ValueError(f"Unexpected AGORA joint shape: {face_2d.shape}, {face_3d.shape}")
    face_2d = face_2d[_FACE_START:]
    face_3d = face_3d[_FACE_START:].T
    rotation, translation, scale = _align_similarity(
        _REFERENCE_HEAD[:, _KP_IDX_MODEL], face_3d[:, _KP_IDX_AGORA]
    )

    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3:] = translation
    compound = transform @ np.linalg.inv(_E_REF)
    euler = _select_euler(_inverse_rotate_zyx(compound))
    if euler is None:
        return None

    sphere_camera = scale * rotation @ _SPHERE + translation
    projected = _project(sphere_camera, _camera_matrix(image_name, 1280, 720))
    x_min = int(max(projected[0].min(), 0))
    y_min = int(max(projected[1].min(), 0))
    x_max = int(min(projected[0].max(), 1280))
    y_max = int(min(projected[1].max(), 720))
    width, height = x_max - x_min, y_max - y_min
    if not (
        x_min < x_max
        and y_min < y_max
        and height / width < 1.5
        and width / height < 1.5
        and width < 1280 * 0.7
        and height < 720 * 0.7
    ):
        return None
    pitch, yaw, roll = euler
    return {
        "bbox_xywh": [x_min, y_min, width, height],
        "crop_xyxy": [x_min, y_min, x_max, y_max],
        "pitch_deg": pitch,
        "yaw_deg": yaw,
        "roll_deg": roll,
    }


def prepare_agora(
    *,
    project_root: Path,
    images_dir: Path,
    pickle_dir: Path,
    output_dir: Path,
    overwrite: bool,
) -> dict[str, Any]:
    manifest_path = output_dir / "manifest.jsonl"
    metadata_path = output_dir / "metadata.json"
    if not overwrite and (manifest_path.exists() or metadata_path.exists()):
        raise FileExistsError("Prepared AGORA output exists; pass --overwrite")

    pickle_paths = sorted(
        pickle_dir.glob("validation_*_withjv.pkl"),
        key=lambda path: int(path.stem.split("_")[1]),
    )
    if len(pickle_paths) != 10:
        raise ValueError(f"Expected 10 AGORA pickle files, found {len(pickle_paths)}")
    image_names = {path.name for path in images_dir.glob("*.png")}
    if len(image_names) != 1225:
        raise ValueError(f"Expected 1225 AGORA images, found {len(image_names)}")

    sequence_indices: dict[str, int] = {}
    labeled_images = 0
    kept_images: set[str] = set()
    angle_min = {axis: float("inf") for axis in ("pitch", "yaw", "roll")}
    angle_max = {axis: float("-inf") for axis in ("pitch", "yaw", "roll")}

    def records():
        nonlocal labeled_images
        for pickle_path in pickle_paths:
            with pickle_path.open("rb") as stream:
                dataframe = _load_dataframe(stream)
            required = {
                "age", "gt_joints_2d", "gt_joints_3d", "imgPath", "isValid",
                "kid", "occlusion",
            }
            missing = required - set(dataframe.columns)
            if missing:
                raise ValueError(f"{pickle_path.name} lacks columns: {sorted(missing)}")
            labeled_images += len(dataframe)

            for _, row in dataframe.iterrows():
                image_name = str(row["imgPath"])
                if not image_name.endswith("_1280x720.png"):
                    image_name = image_name.replace(".png", "_1280x720.png")
                if image_name not in image_names:
                    raise ValueError(f"AGORA image referenced by labels is missing: {image_name}")

                sequence_name = image_name.removesuffix("_1280x720.png")
                frame = int(sequence_name[-5:])
                sequence_key = sequence_name[:-6]
                sequence_index = sequence_indices.setdefault(
                    sequence_key, len(sequence_indices) + 1
                )
                image_id = 2_000_000_000 + sequence_index * 100_000 + frame
                image_path = images_dir / image_name
                labels_for_image = []
                for person_index, (occlusion, joints_2d, joints_3d) in enumerate(
                    zip(row["occlusion"], row["gt_joints_2d"], row["gt_joints_3d"])
                ):
                    if float(occlusion) >= 90:
                        continue
                    label = _head_label(joints_2d, joints_3d, image_name)
                    if label is None:
                        continue
                    label["person_index"] = person_index
                    label["occlusion_percent"] = float(occlusion)
                    labels_for_image.append(label)

                for instance_index, label in enumerate(labels_for_image):
                    kept_images.add(image_name)
                    for axis in ("pitch", "yaw", "roll"):
                        value = label[f"{axis}_deg"]
                        angle_min[axis] = min(angle_min[axis], value)
                        angle_max[axis] = max(angle_max[axis], value)
                    yield {
                        "schema_version": SCHEMA_VERSION,
                        "dataset": "agora_hpe",
                        "split": "validation",
                        "sample_id": str(image_id),
                        "instance_id": str(image_id * 100 + instance_index),
                        "person_index": label["person_index"],
                        "image_path": relative_posix(image_path, project_root),
                        "image_width": 1280,
                        "image_height": 720,
                        "bbox_xywh": label["bbox_xywh"],
                        "crop_xyxy": label["crop_xyxy"],
                        "pitch_deg": label["pitch_deg"],
                        "yaw_deg": label["yaw_deg"],
                        "roll_deg": label["roll_deg"],
                        "occlusion_percent": label["occlusion_percent"],
                        "sequence_key": sequence_key,
                    }

    count = write_jsonl_atomic(manifest_path, records())
    if labeled_images != 1077:
        raise RuntimeError(f"Expected 1077 labeled AGORA images, found {labeled_images}")
    if len(kept_images) != 1070 or count != 7505:
        raise RuntimeError(
            "AGORA preprocessing did not reproduce the published split: "
            f"images={len(kept_images)} (expected 1070), "
            f"instances={count} (expected 7505)"
        )

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "dataset": "agora_hpe",
        "split": "validation",
        "source_images": len(image_names),
        "labeled_images": labeled_images,
        "images": len(kept_images),
        "instances": count,
        "image_root": relative_posix(images_dir, project_root),
        "manifest": relative_posix(manifest_path, project_root),
        "source_pickles": [
            {
                "path": relative_posix(path, project_root),
                "sha256": sha256_file(path),
            }
            for path in pickle_paths
        ],
        "angle_range_deg": {
            axis: {"min": angle_min[axis], "max": angle_max[axis]}
            for axis in angle_min
        },
    }
    write_json_atomic(metadata_path, metadata)
    return metadata
