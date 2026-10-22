"""Manifest validation and target selection for WorldCrafter dataset preparation."""
import json
from pathlib import Path
import numpy as np

def rotation_degrees(rotation):
    return np.degrees(np.arccos(np.clip((np.trace(rotation, axis1=-2, axis2=-1) - 1) / 2, -1, 1)))

def keep_chunk(poses, mask, *, smooth=True):
    poses = np.asarray(poses, dtype=np.float64)
    if poses.shape != (9, 4, 4) or not np.all(mask) or not np.isfinite(poses).all():
        return False
    relative = np.linalg.inv(poses[0]) @ poses
    positions = relative[:, :3, 3]
    t_peak = np.linalg.norm(positions, axis=1).max()
    r_peak = rotation_degrees(relative[:, :3, :3]).max()
    if not (t_peak <= 10 and (r_peak >= 15 or (t_peak >= .8 and r_peak >= 5))):
        return False
    if not smooth:
        return True
    steps = np.diff(positions, axis=0)
    lengths = np.linalg.norm(steps, axis=1)
    chord = np.linalg.norm(positions[-1] - positions[0])
    if chord < 1e-5 or lengths.sum() / chord > 2 or lengths.max() / max(np.median(lengths), 1e-12) > 4:
        return False
    good = (lengths[:-1] > 1e-12) & (lengths[1:] > 1e-12)
    if good.any():
        turns = np.degrees(np.arccos(np.clip(np.sum(steps[:-1][good] * steps[1:][good], axis=1) /
                                            (lengths[:-1][good] * lengths[1:][good]), -1, 1)))
        if turns.max() > 120 or turns.mean() > 60:
            return False
    rotation_steps = np.swapaxes(poses[:-1, :3, :3], -1, -2) @ poses[1:, :3, :3]
    return bool(rotation_degrees(rotation_steps).max() <= 75)

def inside(root, path):
    path = Path(path)
    if path.is_absolute():
        raise ValueError('Manifest paths must be relative to --dataset-root')
    result = (root / path).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError('Manifest path escapes dataset root')
    return result


def normalize_record(row):
    """Accept the released manifest or a captioned custom video with cut/crop metadata."""
    from .data import find_nearest_length_bucket
    row = dict(row)
    sample_id = row['sample_id']
    if not isinstance(sample_id, str) or not sample_id or Path(sample_id).name != sample_id or sample_id in ('.', '..'):
        raise ValueError('sample_id must be a nonempty filename without directories')
    if not isinstance(row['caption'], str) or not row['caption'].strip():
        raise ValueError(f'{sample_id}: a nonempty caption is required')
    cut = row['cut']
    crop = row['crop']
    if (len(cut) != 2 or any(type(v) is not int for v in cut) or not 0 <= cut[0] < cut[1]):
        raise ValueError('cut must be [start_frame, end_frame), with nonnegative integer indices')
    if (len(crop) != 4 or any(type(v) is not int for v in crop)
            or not (0 <= crop[0] < crop[1] and 0 <= crop[2] < crop[3])):
        raise ValueError('crop must be [left, right, top, bottom] in pixels')
    if not np.isfinite(row['fps']) or row['fps'] <= 0:
        raise ValueError('fps must be positive')
    row.setdefault('stride', 1)
    stride = row['stride']
    if type(stride) is not int or stride not in (1, 2):
        raise ValueError('stride must be 1 or 2')
    if 'bucket_num_frame' not in row:
        count = find_nearest_length_bucket((cut[1] - cut[0] + stride - 1) // stride, stride)
        if count < 121:
            raise ValueError('Custom videos need at least 121 sampled frames')
        row.update(bucket_num_frame=count, bucket_height=384, bucket_width=640,
                   selected_start_frame=cut[0], selected_end_frame=cut[0] + (count - 1) * stride + 1)
    keys = ('bucket_num_frame', 'bucket_height', 'bucket_width', 'selected_start_frame', 'selected_end_frame')
    if any(type(row[k]) is not int for k in keys):
        raise ValueError('Frame intervals and bucket dimensions must be integers')
    if (row['bucket_height'], row['bucket_width']) != (384, 640) or row['bucket_num_frame'] < 66:
        raise ValueError('Training requires 640x384 video and at least two 33-frame chunks')
    start, end = row['selected_start_frame'], row['selected_end_frame']
    if not cut[0] <= start < end <= cut[1] or len(range(start, end, stride)) != row['bucket_num_frame']:
        raise ValueError('Selected frame interval does not match cut/stride/frame count')
    row.setdefault('feature_name', sample_id + '.pt')
    name = row['feature_name']
    if Path(name).name != name or not name.endswith('.pt'):
        raise ValueError('feature_name must be a .pt basename')
    return row


def validate_camera(row, camera):
    if camera['source_video_path'] != row['video_path'] or camera['sample_id'] != row['sample_id']:
        raise ValueError('Sample/camera mapping mismatch')
    for key in ('selected_start_frame', 'selected_end_frame', 'stride', 'bucket_num_frame', 'bucket_height', 'bucket_width'):
        if camera[key] != row[key]:
            raise ValueError('Sample/camera preprocessing mismatch: ' + key)
    if camera['prompt_raw'] != row['caption']:
        raise ValueError('Caption differs from the camera record')
    if 'crop' in camera and camera['crop'] != row['crop']:
        raise ValueError('Sample/camera crop mismatch')
    chunks = row['bucket_num_frame'] // 33
    if camera['camera_num_chunks'] != chunks:
        raise ValueError('Camera chunk count mismatch')
    grid = np.arange(chunks)[:, None] * 33 + np.arange(9)[None, :] * 4
    if not np.array_equal(camera['camera_anchor_processed_frame_idx'], grid):
        raise ValueError('Processed anchor frame mismatch')
    if not np.array_equal(camera['camera_anchor_frame_idx'], row['selected_start_frame'] + grid * row['stride']):
        raise ValueError('Source anchor frame mismatch')
    arrays = {}
    for key, shape in [('camera_pose_rel_video_c2w', (chunks, 9, 4, 4)),
                       ('camera_pose_rel_chunk_c2w', (chunks, 9, 4, 4)),
                       ('camera_K_norm', (chunks, 9, 4)), ('camera_valid_mask', (chunks, 9))]:
        value = np.asarray(camera[key])
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f'Invalid camera field {key}: expected finite {shape}, got {value.shape}')
        arrays[key] = value
    if not np.isin(arrays['camera_valid_mask'], [False, True]).all():
        raise ValueError('camera_valid_mask must contain booleans')
    if np.any(arrays['camera_K_norm'][..., :2] <= 0):
        raise ValueError('Normalized focal lengths must be positive')
    for key in ('camera_pose_rel_video_c2w', 'camera_pose_rel_chunk_c2w'):
        pose = arrays[key]
        if not np.allclose(pose[..., 3, :], [0, 0, 0, 1], atol=1e-4):
            raise ValueError('Camera poses must be homogeneous camera-to-world transforms')
        rotation = pose[..., :3, :3]
        if not (np.allclose(np.swapaxes(rotation, -1, -2) @ rotation, np.eye(3), atol=5e-3)
                and np.allclose(np.linalg.det(rotation), 1, atol=5e-3)):
            raise ValueError('Camera rotations must be orthonormal with determinant +1')
    poses = arrays['camera_pose_rel_video_c2w']
    if not np.allclose(np.linalg.inv(poses[:, :1]) @ poses, arrays['camera_pose_rel_chunk_c2w'], atol=2e-3):
        raise ValueError('Video-relative and chunk-relative camera poses disagree')
    return camera


def target_chunks(row, camera):
    chunks = row['bucket_num_frame'] // 33
    supplied = row.get('valid_chunk_indices')
    if supplied is not None:
        if any(type(i) is not int or not 0 <= i < chunks for i in supplied) or len(set(supplied)) != len(supplied):
            raise ValueError('Invalid target chunk indices')
        return supplied
    return [i for i, (pose, mask) in enumerate(zip(camera['camera_pose_rel_chunk_c2w'], camera['camera_valid_mask']))
            if keep_chunk(pose, mask)]
