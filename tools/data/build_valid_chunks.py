#!/usr/bin/env python3
"""Build motion/smoothness target indices from DA3 camera JSON files.

Existing OSP-v2 users should use the released index, which additionally includes
historical topic filtering. This tool accepts any user-provided captions unchanged.
"""
import argparse
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--camera-root', type=Path, required=True)
    p.add_argument('--feature-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--no-smooth-filter', action='store_true')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = {}
    for path in sorted(args.camera_root.glob('*.json')):
        data = json.loads(path.read_text())
        poses, masks = data['camera_pose_rel_chunk_c2w'], data['camera_valid_mask']
        if len(poses) != len(masks):
            raise ValueError(f'Pose/mask chunk mismatch: {path}')
        name = path.with_suffix('.pt').name
        if not (args.feature_root / name).is_file():
            raise FileNotFoundError(args.feature_root / name)
        chunks = [i for i, (pose, mask) in enumerate(zip(poses, masks))
                  if keep_chunk(pose, mask, smooth=not args.no_smooth_filter)]
        if chunks:
            result[name] = chunks
    if not result:
        raise ValueError('No chunks passed motion filtering; inspect cameras and input motion')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'format': 'worldcrafter_custom_motion_v1',
        'topic_filter': False, 'smooth_filter': not args.no_smooth_filter,
        'valid_chunk_indices': result}, indent=2) + '\n')
    print(json.dumps({'files': len(result), 'chunks': sum(map(len, result.values()))}))


if __name__ == '__main__':
    main()
