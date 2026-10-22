#!/usr/bin/env python3
"""Prepare video latents, caption embeddings and cameras for WorldCrafter fine-tuning."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.utils.data import encode_prompt, read_cut_crop_and_resize
from training.utils.preparation import inside, normalize_record, target_chunks, validate_camera
from training.utils.sampling import eligible_target_chunks


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.partial')
    tmp.write_text(json.dumps(payload, indent=2) + '\n')
    tmp.replace(path)


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read_frames(root, row):
    return read_cut_crop_and_resize(str(inside(root, row['video_path'])), row['bucket_num_frame'],
        row['bucket_height'], row['bucket_width'], row['stride'], row['selected_start_frame'],
        row['selected_end_frame'], row['crop'])


def camera_path(args, row):
    if row.get('camera_path'):
        return inside(args.dataset_root, row['camera_path'])
    return args.output / 'cameras' / (row['sample_id'] + '.json')


def prepare_cameras(args, rows):
    """Estimate missing cameras before loading encoders, so both phases fit one GPU."""
    from training.utils.pose_estimation import PersistentDA3Client, estimate_camera_sparse_for_video, camera_payload_to_jsonable
    estimator = None
    settings = SimpleNamespace(da3_python=args.camera_python, da3_model_dir=str(args.camera_model.resolve()),
        da3_device=args.device, tmp_dir=str(args.tmp_dir), process_res=504,
        process_res_method='upper_bound_resize', ref_view_strategy='middle', use_ray_pose=False, latent_window_size=9)
    recipe = {'model_path': settings.da3_model_dir, 'process_res': 504,
              'process_res_method': 'upper_bound_resize', 'ref_view_strategy': 'middle', 'use_ray_pose': False}
    try:
        for _, row in rows:
            path = camera_path(args, row)
            if path.exists():
                camera = validate_camera(row, json.loads(path.read_text()))
                if not row.get('camera_path'):
                    row_hash = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
                    if camera.get('estimation') != recipe or camera.get('preprocessing_sha256') != row_hash:
                        raise ValueError(f'Existing camera estimation settings or inputs differ: {path}')
                continue
            if row.get('camera_path'):
                raise FileNotFoundError(f'Supplied camera file is missing: {path}')
            if estimator is None:
                args.tmp_dir.mkdir(parents=True, exist_ok=True)
                estimator = PersistentDA3Client(settings)
            frames = read_frames(args.dataset_root, row)
            sample = dict(row, uttid=row['sample_id'], prompt=row['caption'],
                video_path=str(inside(args.dataset_root, row['video_path'])),
                cut_start_frame=row['cut'][0], cut_end_frame=row['cut'][1])
            payload = estimate_camera_sparse_for_video(frames, sample, settings, estimator)
            camera = camera_payload_to_jsonable(sample, payload)
            camera['source_video_path'] = row['video_path']
            camera['estimation'] = recipe
            camera['preprocessing_sha256'] = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
            camera['crop'] = row['crop']
            camera['video_sha256'] = digest(inside(args.dataset_root, row['video_path']))
            validate_camera(row, camera)
            write_json(path, camera)
            del frames, payload
    finally:
        if estimator is not None:
            estimator.close()


def encode_record(row, frames, camera, tokenizer, text_encoder, vae, device, seed):
    # Preserve the released recipe's bf16 normalization constants and posterior sampling.
    torch.manual_seed(seed)
    mean = torch.tensor(vae.config.latents_mean, dtype=torch.bfloat16, device=device).view(1, vae.config.z_dim, 1, 1, 1)
    inv_std = 1.0 / torch.tensor(vae.config.latents_std, dtype=torch.bfloat16, device=device).view(1, vae.config.z_dim, 1, 1, 1)
    latents = []
    with torch.inference_mode():
        for start in range(0, len(frames) - 32, 33):
            pixels = frames[start:start + 33].permute(1, 0, 2, 3).unsqueeze(0).to(device, vae.dtype)
            latents.append(((vae.encode(pixels).latent_dist.sample() - mean) * inv_std)[0].cpu())
        embeds, masks = encode_prompt(tokenizer, text_encoder, [row['caption']], device=device)
    data = {'vae_latent': torch.stack(latents), 'prompt_embed': embeds[0].cpu(),
            'prompt_attention_mask': masks[0].cpu(), 'prompt_raw': row['caption'],
            'source_video_path': row['video_path'], 'caption_record': {'path': row['video_path'],
                'cut': row['cut'], 'crop': row['crop'], 'fps': row['fps'], 'cap': [row['caption']]},
            'selected_start_frame': row['selected_start_frame'], 'selected_end_frame': row['selected_end_frame'],
            'stride': row['stride'], 'fps': row['fps']}
    for key in ('camera_pose_rel_video_c2w', 'camera_pose_rel_chunk_c2w', 'camera_K_norm', 'camera_pose_conf'):
        if key in camera:
            data[key] = torch.as_tensor(camera[key], dtype=torch.float32)
    data['camera_valid_mask'] = torch.as_tensor(camera['camera_valid_mask'], dtype=torch.bool)
    for key in ('camera_anchor_frame_idx', 'camera_anchor_processed_frame_idx'):
        data[key] = torch.as_tensor(camera[key], dtype=torch.int32)
    for key in ('camera_num_video_raw_frames', 'camera_pose_scale_factor', 'camera_pose_is_metric'):
        if key in camera:
            data[key] = camera[key]
    return data


def validate_features(data, chunks):
    for key, shape in [('vae_latent', (chunks, 16, 9, 48, 80)), ('prompt_embed', (512, 4096)),
                       ('prompt_attention_mask', (512,))]:
        value = data[key]
        if tuple(value.shape) != shape or not torch.isfinite(value).all():
            raise ValueError(f'Invalid encoded {key}: expected finite {shape}, got {tuple(value.shape)}')


def merge_indices(output, num_shards):
    merged = {}
    preparation = None
    for shard in range(num_shards):
        path = output / f'valid_chunk_indices.shard-{shard:05d}-of-{num_shards:05d}.json'
        payload = json.loads(path.read_text())
        if payload['shard_index'] != shard or payload['num_shards'] != num_shards:
            raise ValueError('Shard identity mismatch')
        if preparation is not None and preparation != payload['preparation']:
            raise ValueError('Shards were prepared with different manifests, seeds or models')
        preparation = payload['preparation']
        part = payload['valid_chunk_indices']
        if merged.keys() & part.keys():
            raise ValueError('Duplicate feature names across shards')
        for name in part:
            if Path(name).name != name or not (output / 'features' / name).is_file():
                raise ValueError(f'Missing or invalid feature: {name}')
        merged.update(part)
    if not merged:
        raise ValueError('No eligible training samples in shard indices')
    write_json(output / 'valid_chunk_indices.json', {'valid_chunk_indices': merged})
    print('merged_training_samples', len(merged))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, help='Defaults to DATASET_ROOT/metadata/samples.jsonl')
    parser.add_argument('--model-path', type=Path, default=Path('weights/WorldCrafter-Base'))
    parser.add_argument('--output', type=Path, help='Prepared dataset directory; defaults to --dataset-root')
    parser.add_argument('--use_da3', '--use-da3', '-use_da3', action='store_true',
                        help='Estimate missing cameras using Depth Anything 3')
    parser.add_argument('--camera-model', type=Path, default=Path('weights/DA3NESTED-GIANT-LARGE-1.1'))
    parser.add_argument('--camera-python', default='uvenv/data/.venv/bin/python')
    parser.add_argument('--tmp-dir', type=Path, default=Path(tempfile.gettempdir()) / 'worldcrafter-preparation')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--merge-index', action='store_true', help='Merge completed shard indices without encoding')
    parser.add_argument('--max-samples', type=int)
    parser.add_argument('--sample-id')
    parser.add_argument('--check-only', action='store_true', help='Validate inputs without loading models or writing files')
    args = parser.parse_args()
    args.dataset_root = args.dataset_root.resolve()
    args.output = (args.output or args.dataset_root).resolve()
    args.tmp_dir = args.tmp_dir.resolve()
    if not 0 <= args.shard_index < args.num_shards:
        parser.error('Require 0 <= shard-index < num-shards')
    if args.merge_index:
        merge_indices(args.output, args.num_shards)
        return
    manifest = args.manifest or args.dataset_root / 'metadata/samples.jsonl'
    rows = [normalize_record(json.loads(line)) for line in manifest.read_text().splitlines() if line.strip()]
    for key in ('sample_id', 'feature_name'):
        if len({row[key] for row in rows}) != len(rows):
            raise ValueError(f'Duplicate {key} in manifest')
    rows = [(i, row) for i, row in enumerate(rows) if i % args.num_shards == args.shard_index
            and (args.sample_id is None or row['sample_id'] == args.sample_id)]
    if args.max_samples is not None:
        if args.max_samples < 1:
            parser.error('--max-samples must be positive')
        rows = rows[:args.max_samples]
    if not rows:
        raise ValueError('No samples selected')
    for _, row in rows:
        if not inside(args.dataset_root, row['video_path']).is_file():
            raise FileNotFoundError(row['video_path'])
        path = camera_path(args, row)
        if path.exists():
            validate_camera(row, json.loads(path.read_text()))
        elif row.get('camera_path') or not args.use_da3:
            raise ValueError(f"{row['sample_id']}: camera annotations missing; add --use_da3 to estimate them")
    if args.check_only:
        print('checked_samples', len(rows))
        return
    if args.use_da3:
        prepare_cameras(args, rows)
    from diffusers import AutoencoderKLWan
    from transformers import AutoTokenizer, UMT5EncoderModel
    shared = args.model_path.resolve()
    config = shared / 'inference_config.json'
    if config.exists():
        shared = (shared / json.loads(config.read_text()).get('shared_components', '.')).resolve()
    tokenizer = AutoTokenizer.from_pretrained(shared / 'tokenizer')
    text_encoder = UMT5EncoderModel.from_pretrained(shared / 'text_encoder', dtype=torch.bfloat16).to(args.device).eval()
    vae = AutoencoderKLWan.from_pretrained(shared / 'vae', torch_dtype=torch.float32).to(args.device).eval()
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    model_files = [p for name in ('vae', 'text_encoder', 'tokenizer') for p in sorted((shared / name).rglob('*')) if p.is_file()]
    model_identity = {str(p.relative_to(shared)): [p.stat().st_size, p.stat().st_mtime_ns] for p in model_files}
    model_signature = hashlib.sha256(json.dumps(model_identity, sort_keys=True).encode()).hexdigest()
    feature_root = args.output / 'features'
    feature_root.mkdir(parents=True, exist_ok=True)
    valid = {}
    video_hashes = {}
    for index, row in rows:
        path = camera_path(args, row)
        camera = validate_camera(row, json.loads(path.read_text()))
        video_path = inside(args.dataset_root, row['video_path'])
        if video_path not in video_hashes:
            video_hashes[video_path] = digest(video_path)
        if camera.get('video_sha256', video_hashes[video_path]) != video_hashes[video_path]:
            raise ValueError(f'Video changed since camera estimation: {video_path}')
        signature = {'schema_version': 2, 'seed': args.seed + index,
            'sample_sha256': hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest(),
            'camera_sha256': digest(path), 'video_sha256': video_hashes[video_path],
            'model_path': str(shared), 'model_files_signature': model_signature}
        dest = feature_root / row['feature_name']
        if dest.exists():
            data = torch.load(dest, map_location='cpu', weights_only=True)
            if data.get('preparation') != signature:
                raise ValueError(f'Existing feature has different provenance: {dest}; use a new --output directory')
            print('existing_verified', dest.name, flush=True)
        else:
            frames = read_frames(args.dataset_root, row)
            data = encode_record(row, frames, camera, tokenizer, text_encoder, vae, args.device, args.seed + index)
            data['preparation'] = signature
            del frames
        validate_features(data, camera['camera_num_chunks'])
        selected = eligible_target_chunks(data, target_chunks(row, camera))
        if not dest.exists():
            tmp = dest.with_suffix('.pt.partial')
            torch.save(data, tmp)
            tmp.replace(dest)
            print('encoded', dest.name, tuple(data['vae_latent'].shape), flush=True)
        if selected:
            valid[dest.name] = list(selected)
        else:
            print('no_eligible_targets', dest.name, flush=True)
        del data
    if not valid and args.num_shards == 1:
        raise ValueError('No eligible training targets; inspect video motion and camera validity')
    name = ('valid_chunk_indices.json' if args.num_shards == 1 else
            f'valid_chunk_indices.shard-{args.shard_index:05d}-of-{args.num_shards:05d}.json')
    write_json(args.output / name, {'valid_chunk_indices': valid, 'num_shards': args.num_shards,
        'shard_index': args.shard_index, 'preparation': {'manifest_sha256': digest(manifest),
        'seed': args.seed, 'model_path': str(shared), 'model_files_signature': model_signature}})
    print('training_index', args.output / name, 'files', len(valid), 'chunks', sum(map(len, valid.values())))


if __name__ == '__main__':
    main()
