#!/usr/bin/env python3
"""Encode released videos and supplied captions while reusing the provided DA3 poses."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from diffusers import AutoencoderKLWan
from transformers import AutoTokenizer, UMT5EncoderModel

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from worldcrafter.training.utils.data import read_cut_crop_and_resize
from worldcrafter.training.utils.data import encode_prompt


def inside(root, path):
    result = (root / path).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError('Manifest path escapes dataset root')
    return result


def validate_record(root, row):
    video = inside(root, row['video_path'])
    camera_path = inside(root, row['camera_path'])
    camera = json.loads(camera_path.read_text())
    if camera['source_video_path'] != row['video_path'] or camera['sample_id'] != row['sample_id']:
        raise ValueError('Sample/camera mapping mismatch')
    for key in ['selected_start_frame', 'selected_end_frame', 'stride', 'bucket_num_frame', 'bucket_height', 'bucket_width']:
        if camera[key] != row[key]:
            raise ValueError('Sample/camera preprocessing mismatch: ' + key)
    chunks = row['bucket_num_frame'] // 33
    grid = np.arange(chunks)[:, None] * 33 + np.arange(9)[None, :] * 4
    if not np.array_equal(camera['camera_anchor_processed_frame_idx'], grid):
        raise ValueError('Processed anchor frame mismatch')
    if not np.array_equal(camera['camera_anchor_frame_idx'], row['selected_start_frame'] + grid * row['stride']):
        raise ValueError('Source anchor frame mismatch')
    if not isinstance(row['caption'], str) or not row['caption'].strip():
        raise ValueError('An existing nonempty caption is required')
    if row['caption'] != camera['prompt_raw']:
        raise ValueError('Caption differs from the camera record')
    if Path(row['feature_name']).name != row['feature_name'] or not row['feature_name'].endswith('.pt'):
        raise ValueError('Invalid feature filename')
    return video, camera, camera_path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root', type=Path, required=True)
    p.add_argument('--model-path', type=Path, required=True,
                   help='Local WorldCrafter-Base directory or shared components with vae/tokenizer/text_encoder')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num-shards', type=int, default=1)
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--max-samples', type=int)
    p.add_argument('--sample-id', help='Encode one exact sample ID')
    p.add_argument('--check-only', action='store_true', help='Check frame alignment and mappings without loading models')
    args = p.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        p.error('Require 0 <= shard-index < num-shards')
    rows = [json.loads(line) for line in (args.dataset_root / 'metadata/samples.jsonl').read_text().splitlines()]
    rows = [(i, row) for i, row in enumerate(rows) if i % args.num_shards == args.shard_index
            and (args.sample_id is None or row['sample_id'] == args.sample_id)]
    if args.max_samples is not None:
        if args.max_samples < 1:
            p.error('--max-samples must be positive')
        rows = rows[:args.max_samples]
    if not rows:
        raise ValueError('No samples selected')
    if args.check_only:
        for _, row in rows:
            video, _, _ = validate_record(args.dataset_root, row)
            if not video.is_file():
                raise FileNotFoundError(video)
        print('checked_samples', len(rows)); return
    shared = args.model_path.resolve()
    config = shared / 'inference_config.json'
    if config.exists():
        shared = (shared / json.loads(config.read_text()).get('shared_components', '.')).resolve()
    tokenizer = AutoTokenizer.from_pretrained(shared / 'tokenizer')
    text_encoder = UMT5EncoderModel.from_pretrained(shared / 'text_encoder', torch_dtype=torch.bfloat16).to(args.device).eval()
    vae = AutoencoderKLWan.from_pretrained(shared / 'vae', torch_dtype=torch.float32).to(args.device).eval()
    vae.requires_grad_(False); text_encoder.requires_grad_(False)
    # Match the original OSP precomputation's normalization and posterior sampling.
    mean = torch.tensor(vae.config.latents_mean, dtype=torch.bfloat16, device=args.device).view(1, vae.config.z_dim, 1, 1, 1)
    inv_std = 1.0 / torch.tensor(vae.config.latents_std, dtype=torch.bfloat16, device=args.device).view(1, vae.config.z_dim, 1, 1, 1)
    args.output.mkdir(parents=True, exist_ok=True)
    for index, row in rows:
        video_path, camera, camera_path = validate_record(args.dataset_root, row)
        dest = args.output / row['feature_name']
        signature = {'schema_version': 1, 'seed': args.seed + index,
                     'sample_sha256': hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest(),
                     'camera_sha256': hashlib.sha256(camera_path.read_bytes()).hexdigest(),
                     'model_path': str(args.model_path.resolve())}
        if dest.exists():
            old = torch.load(dest, map_location='cpu', weights_only=False)
            if old.get('release_encoding') != signature:
                raise ValueError(f'Existing feature has different provenance: {dest}')
            print('existing_verified', dest.name, flush=True); continue
        torch.manual_seed(args.seed + index)
        frames = read_cut_crop_and_resize(video_path=str(video_path), f_prime=row['bucket_num_frame'],
                    h_prime=row['bucket_height'], w_prime=row['bucket_width'], stride=row['stride'],
                    start_frame=row['selected_start_frame'], end_frame=row['selected_end_frame'], crop=row['crop'])
        latents = []
        with torch.inference_mode():
            for start in range(0, len(frames) - 32, 33):
                pixels = frames[start:start+33].permute(1, 0, 2, 3).unsqueeze(0).to(args.device, vae.dtype)
                latent = (vae.encode(pixels).latent_dist.sample() - mean) * inv_std
                latents.append(latent[0].cpu())
            embeds, masks = encode_prompt(tokenizer, text_encoder, [row['caption']], device=args.device)
        data = {'vae_latent': torch.stack(latents), 'prompt_embed': embeds[0].cpu(),
                'prompt_attention_mask': masks[0].cpu(), 'prompt_raw': row['caption'],
                'source_video_path': row['video_path'], 'caption_record': {'path': row['video_path'],
                    'cut': row['cut'], 'crop': row['crop'], 'fps': row['fps'], 'cap': [row['caption']]},
                'selected_start_frame': row['selected_start_frame'], 'selected_end_frame': row['selected_end_frame'],
                'stride': row['stride'], 'fps': row['fps'], 'release_encoding': signature}
        for key in ['camera_pose_rel_video_c2w', 'camera_pose_rel_chunk_c2w', 'camera_K_norm', 'camera_pose_conf']:
            if key in camera:
                data[key] = torch.as_tensor(camera[key], dtype=torch.float32)
        data['camera_valid_mask'] = torch.as_tensor(camera['camera_valid_mask'], dtype=torch.bool)
        for key in ['camera_anchor_frame_idx', 'camera_anchor_processed_frame_idx']:
            data[key] = torch.as_tensor(camera[key], dtype=torch.int32)
        for key in ['camera_num_video_raw_frames', 'camera_pose_scale_factor', 'camera_pose_is_metric']:
            if key in camera:
                data[key] = camera[key]
        if tuple(data['vae_latent'].shape) != (camera['camera_num_chunks'], 16, 9, 48, 80):
            raise ValueError('Encoded latent layout does not match supplied cameras')
        if not torch.isfinite(data['vae_latent']).all() or not torch.isfinite(data['prompt_embed']).all():
            raise ValueError('Non-finite encoded features')
        tmp = dest.with_suffix('.pt.partial'); torch.save(data, tmp); tmp.replace(dest)
        print('encoded', dest.name, tuple(data['vae_latent'].shape), flush=True)


if __name__ == '__main__':
    main()
