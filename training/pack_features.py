#!/usr/bin/env python3
"""Losslessly shard complete .pt files selected by a valid-chunk index."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tarfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--feature-root', required=True, type=Path)
    p.add_argument('--valid-chunks', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--shard-gib', type=float, default=2.0, help='Uncompressed target size; a file is never split')
    args = p.parse_args()
    if args.shard_gib <= 0:
        raise ValueError('--shard-gib must be positive')
    raw = json.loads(args.valid_chunks.read_text())
    valid = raw.get('valid_chunk_indices', raw)
    names = sorted(name for name, chunks in valid.items() if chunks)
    if not names or any(Path(n).name != n or not n.endswith('.pt') for n in names):
        raise ValueError('Expected a nonempty index of .pt basenames')
    for name in names:
        if not (args.feature_root / name).is_file():
            raise FileNotFoundError(args.feature_root / name)
    args.output.mkdir(parents=True, exist_ok=False)
    groups, group, total = [], [], 0
    for name in names:
        size = (args.feature_root / name).stat().st_size
        if group and total + size > args.shard_gib * 2**30:
            groups.append(group)
            group, total = [], 0
        group.append(name)
        total += size
    if group:
        groups.append(group)
    manifest = {'format': 'worldcrafter_features_v1', 'files': [], 'shards': []}
    for i, group in enumerate(groups):
        path = args.output / f'features-{i:05d}.tar.gz'
        temporary = path.with_suffix(path.suffix + '.tmp')
        with tarfile.open(temporary, 'w:gz', compresslevel=6) as archive:
            for name in group:
                source = args.feature_root / name
                with source.open('rb') as f:
                    digest = hashlib.file_digest(f, 'sha256').hexdigest()
                archive.add(source, arcname=name, recursive=False)
                manifest['files'].append({'name': name, 'bytes': source.stat().st_size, 'sha256': digest, 'shard': path.name})
        # Verify uncompressed bytes before publishing each shard.
        with tarfile.open(temporary, 'r:gz') as archive:
            expected = {r['name']: r['sha256'] for r in manifest['files'] if r['shard'] == path.name}
            for member in archive:
                with archive.extractfile(member) as f:
                    if hashlib.file_digest(f, 'sha256').hexdigest() != expected[member.name]:
                        raise RuntimeError(f'Shard verification failed: {member.name}')
        os.rename(temporary, path)
        with path.open('rb') as f:
            digest = hashlib.file_digest(f, 'sha256').hexdigest()
        manifest['shards'].append({'name': path.name, 'bytes': path.stat().st_size, 'sha256': digest})
        print(path.name, flush=True)
    (args.output / 'valid_chunk_indices.json').write_text(json.dumps(raw, indent=2) + '\n')
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
