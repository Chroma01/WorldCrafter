#!/usr/bin/env python3
"""Download and verify the gated WorldCrafter dataset without sharing access tokens."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def safe_path(root, relative):
    p = PurePosixPath(relative)
    if p.is_absolute() or '..' in p.parts:
        raise ValueError(f'Unsafe archive path: {relative}')
    target = root.joinpath(*p.parts)
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'Path leaves dataset directory: {relative}')
    return target


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--repo-id', default='Drexubery/WorldCrafter')
    p.add_argument('--revision', default='main', help='Prefer a release commit for reproducible downloads')
    p.add_argument('--download', action='store_true', help='Use your existing Hugging Face login')
    p.add_argument('--verify', action='store_true')
    p.add_argument('--extract', action='store_true')
    p.add_argument('--shard', action='append', help='Select exact archive paths; repeat to download a subset')
    args = p.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    if not (args.download or args.verify or args.extract):
        p.error('Choose --download, --verify, or --extract')
    fetch = None
    if args.download:
        from huggingface_hub import hf_hub_download, HfApi
        # Resolve once so metadata and all archives come from the same snapshot.
        revision = HfApi().dataset_info(args.repo_id, revision=args.revision).sha
        def fetch(name):
            return hf_hub_download(args.repo_id, name, repo_type='dataset', revision=revision, local_dir=args.root)
        for name in ['metadata/shards.json', 'metadata/videos.jsonl', 'metadata/cameras.jsonl',
                     'metadata/samples.jsonl', 'metadata/valid_chunk_indices.json']:
            fetch(name)
    manifest = json.loads((args.root / 'metadata/shards.json').read_text())
    shards = manifest['shards']
    if args.shard:
        wanted = set(args.shard)
        if wanted - {s['path'] for s in shards}:
            raise ValueError('Requested archive is absent from the manifest')
        shards = [s for s in shards if s['path'] in wanted]
    members = {}
    for name in ['videos', 'cameras']:
        for line in (args.root / f'metadata/{name}.jsonl').read_text().splitlines():
            row = json.loads(line)
            if row['path'] in members:
                raise ValueError('Duplicate manifest path')
            members[row['path']] = row
    for shard in shards:
        path = safe_path(args.root, shard['path'])
        if fetch:
            fetch(shard['path'])
        if path.stat().st_size != shard['bytes'] or sha256(path) != shard['sha256']:
            raise ValueError(f'Archive checksum mismatch: {shard["path"]}')
        if args.verify or args.extract:
            seen = set()
            with tarfile.open(path, 'r:*') as archive:
                for info in archive:
                    if not info.isfile() or info.name not in members or info.name in seen:
                        raise ValueError(f'Unexpected archive member: {info.name}')
                    seen.add(info.name)
                    expected = members[info.name]
                    if info.size != expected['bytes']:
                        raise ValueError(f'Member length mismatch: {info.name}')
                    dest = safe_path(args.root, info.name)
                    if args.extract and dest.exists():
                        if sha256(dest) != expected['sha256']:
                            raise ValueError(f'Existing file differs; refusing overwrite: {dest}')
                        # Still read and verify the archive member below.
                    tmp = dest.with_suffix(dest.suffix + '.partial')
                    output = None
                    if args.extract and not dest.exists():
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        output = tmp.open('wb')
                    h = hashlib.sha256()
                    try:
                        with archive.extractfile(info) as stream:
                            for block in iter(lambda: stream.read(1024**2), b''):
                                h.update(block)
                                if output:
                                    output.write(block)
                    finally:
                        if output:
                            output.close()
                    if h.hexdigest() != expected['sha256']:
                        raise ValueError(f'Member checksum mismatch: {info.name}')
                    if output:
                        tmp.replace(dest)
            if len(seen) != shard['member_count']:
                raise ValueError('Archive member count mismatch')
        print('verified', shard['path'], flush=True)


if __name__ == '__main__':
    main()
