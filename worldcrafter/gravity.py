"""Gravity-aware camera frames in the model's right/down/forward convention."""
import numpy as np


def gravity_frame(world_up):
    """Camera-to-level-world rotation; world_up is UP in the initial camera."""
    up = np.asarray(world_up, dtype=np.float64)
    if up.shape != (3,) or not np.isfinite(up).all():
        raise ValueError("world_up must contain three finite values")
    norm = np.linalg.norm(up)
    if abs(norm - 1) > .01:
        raise ValueError("world_up must be a unit vector")
    up = up / norm
    if abs(up[2]) > np.sin(np.deg2rad(75)):
        raise ValueError("Gravity alignment supports elevations below 75 degrees")
    down = -up
    forward = np.array([0., 0., 1.]) - down[2] * down
    forward /= np.linalg.norm(forward)
    return np.stack([np.cross(down, forward), down, forward])


def align_trajectory(poses, world_up):
    """Reorient an uncalibrated level-authored path about its initial camera.

    Conjugate relative rotations AND rotate translations. A common left world
    transform alone would disappear when the model computes relative poses.
    Real calibrated paths should bypass this operation.
    """
    source = np.asarray(poses)
    if source.ndim not in (3, 4) or source.shape[-2:] not in ((3, 4), (4, 4)):
        raise ValueError("Camera poses must be [T,3/4,4] or [B,T,3/4,4]")
    if not np.issubdtype(source.dtype, np.floating) or not np.isfinite(source).all():
        raise ValueError("Camera poses must be finite floating-point values")
    full = np.zeros((*source.shape[:-2], 4, 4), dtype=np.float64)
    full[..., :3, :4] = source[..., :3, :4]
    full[..., 3, 3] = 1
    first = full[..., :1, :, :]
    relative = np.linalg.inv(first) @ full
    frame = np.eye(4)
    frame[:3, :3] = gravity_frame(world_up)
    aligned = first @ np.linalg.inv(frame) @ relative @ frame
    return aligned[..., :source.shape[-2], :].astype(source.dtype)
