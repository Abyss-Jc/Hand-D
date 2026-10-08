"""Shared Hand-D Feature Transform v1 (69 float32 values).

Pure NumPy: no camera, MediaPipe, Torch, GUI, or mutable runtime state.
Input handedness is the RAW MediaPipe label, not the user-facing role.
"""

from __future__ import annotations

import numpy as np


FEATURE_TRANSFORM_ID = 'world-landmarks-v1-69'
FEATURE_COUNT = 69
LANDMARK_COUNT = 21


def canonicalize_world_landmarks(
    landmarks_3d: np.ndarray, raw_mp_handedness: str
) -> np.ndarray | None:
    """Map world-space MediaPipe hand landmarks to the legacy 69-vector.

    Raise ValueError for an invalid structural contract (shape/handedness).
    Return None for a numerically degenerate or nonfinite observation.
    Valid numeric inputs preserve the original v1 wrist/MCP/palm transform.
    """
    if raw_mp_handedness not in ('Left', 'Right'):
        raise ValueError('raw_mp_handedness must be Left or Right')
    # Preserve legacy floating-point evaluation order (including float32).
    # Non-float inputs are converted so subtraction/division remain numeric.
    points = np.asarray(landmarks_3d)
    if not np.issubdtype(points.dtype, np.floating):
        points = points.astype(np.float64)
    if points.shape != (LANDMARK_COUNT, 3):
        raise ValueError(f'expected world landmarks shape (21, 3), got {points.shape}')
    if not np.isfinite(points).all():
        return None

    # Translation, raw-MediaPipe-Left selfie-mirror correction.
    wrist = points[0]
    pts = points - wrist
    if raw_mp_handedness == 'Left':
        pts[:, 0] = -pts[:, 0]

    # Mean wrist-to-MCP distance; same thresholds and order as legacy.
    mcps = pts[[5, 9, 13, 17]]
    scale = np.mean(np.linalg.norm(mcps - pts[0], axis=1))
    if scale < 1e-3:
        return None
    pts = pts / scale

    p_wrist = pts[0]
    p_index_mcp = pts[5]
    p_pinky_mcp = pts[17]
    p_middle_mcp = pts[9]

    global_y = p_middle_mcp - p_wrist
    norm_y = np.linalg.norm(global_y)
    if norm_y < 1e-3:
        return None
    global_y = global_y / norm_y

    vec1 = p_index_mcp - p_wrist
    vec2 = p_pinky_mcp - p_wrist
    cross = np.cross(vec1, vec2)
    if np.linalg.norm(cross) < 1e-3:
        return None
    global_z = cross / np.linalg.norm(cross)

    global_x = np.cross(global_y, global_z)
    norm_x = np.linalg.norm(global_x)
    if norm_x < 1e-3:
        return None
    global_x = global_x / norm_x

    global_y = np.cross(global_z, global_x)
    global_y = global_y / np.linalg.norm(global_y)

    rotation_matrix = np.stack([global_x, global_y, global_z], axis=1)
    canonical = np.dot(pts, rotation_matrix)
    if np.any(np.abs(canonical) > 4.0):
        return None

    # 63 canonical XYZ + 3 global Y + 3 global Z; no handedness bit.
    features = np.concatenate([canonical.flatten(), global_y, global_z])
    return features.astype(np.float32)
