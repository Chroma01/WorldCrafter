#!/usr/bin/env python3
"""Export a complete Base transformer + finetuned adapters for inference.py."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-model', required=True, type=Path)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--copy-shared', action='store_true', help='Copy VAE/text/RepEncoder for a standalone package')
    args = p.parse_args()
    if not (args.checkpoint / 'COMPLETE').is_file():
        raise ValueError('Export requires a complete training checkpoint')
    if args.output.exists():
        raise FileExistsError(args.output)
    identity = json.loads((args.checkpoint / 'state.json').read_text())['identity']
    if identity['base_checksums'] != (args.base_model / 'SHA256SUMS').read_text():
        raise ValueError('Checkpoint was trained from a different Base package')
    source_config = json.loads((args.base_model / 'inference_config.json').read_text())
    shared = (args.base_model / source_config.get('shared_components', '.')).resolve()
    staging = args.output.with_name(args.output.name + '.incomplete')
    staging.mkdir(parents=True, exist_ok=False)
    shutil.copytree(args.base_model / 'transformer', staging / 'transformer')
    shutil.copytree(args.checkpoint / 'adapter', staging / 'adapter')
    if args.copy_shared:
        for name in ('repencoder', 'vae', 'text_encoder', 'tokenizer', 'scheduler'):
            shutil.copytree(shared / name, staging / name)
        shared_ref = '.'
    else:
        shared_ref = os.path.relpath(shared, args.output.resolve())
    config = {'format': 'worldcrafter_base_v1', 'shared_components': shared_ref}
    (staging / 'inference_config.json').write_text(json.dumps(config, indent=2) + '\n')
    if (args.base_model / 'model_index.json').is_file():
        shutil.copy2(args.base_model / 'model_index.json', staging / 'model_index.json')
    metadata = json.loads((args.checkpoint / 'state.json').read_text())
    (staging / 'finetuning.json').write_text(json.dumps(metadata, indent=2) + '\n')
    checksums = []
    for path in sorted(staging.rglob('*')):
        if path.is_file():
            with path.open('rb') as f:
                digest = hashlib.file_digest(f, 'sha256').hexdigest()
            checksums.append(f'{digest}  {path.relative_to(staging)}')
    (staging / 'SHA256SUMS').write_text('\n'.join(checksums) + '\n')
    os.rename(staging, args.output)
    print(args.output)


if __name__ == '__main__':
    main()
