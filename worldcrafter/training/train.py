"""Finetune the released WorldCrafter-Base LoRA and camera adapter."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
import yaml
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed, broadcast
from torch.utils.data import DataLoader, DistributedSampler
from worldcrafter.repencoder import RepEncoder
from .checkpoint import fingerprint, save_checkpoint, load_checkpoint
from .dataset import OSPFeatureDataset
from .model import load_base, EXPECTED_GROUPS
from .distributed_audit import write_rank_record, verify_parameter_probes
from .objective import flow_loss, make_scheduler, sample_sequential_history_mode


def build_dataset(config):
    root = Path(config['feature_root']).resolve()
    index_path = Path(config['valid_chunks']).resolve()
    raw = json.loads(index_path.read_text())
    valid = raw.get('valid_chunk_indices', raw)
    if not isinstance(valid, dict) or not valid:
        raise ValueError('valid_chunks must contain a nonempty filename -> chunk indices mapping')
    if any(Path(name).name != name or not name.endswith('.pt') for name in valid):
        raise ValueError('Index keys must be .pt basenames, not absolute paths')
    names = sorted(valid)
    if config.get('pool'):
        names = [line.strip() for line in Path(config['pool']).read_text().splitlines()
                 if line.strip() and not line.startswith('#')]
        if len(names) != len(set(names)) or any(name not in valid for name in names):
            raise ValueError('Pool must contain unique feature basenames present in the valid index')
    files = [str(root / name) for name in names]
    missing = [path for path in files if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f'{len(missing)} features missing; first: {missing[0]}')
    dataset = OSPFeatureDataset(root, feature_files=files, valid_chunk_indices_json=index_path,
                               dataset_source_name='osp_istock_v2', max_chunk_delta_t=10.0,
                               intrinsics_contract=config['intrinsics_contract'], seed=config['seed'])
    return dataset, fingerprint({name: valid[name] for name in names})


def select_history_mode(accelerator, forced=None):
    names = ('t2v', 'v2v', 'none')
    value = torch.zeros((), device=accelerator.device, dtype=torch.int64)
    if accelerator.is_main_process:
        mode = forced or sample_sequential_history_mode(t2v_ratio=0.4, v2v_ratio=0.4)
        value.fill_(names.index(mode))
    return names[int(broadcast(value, from_process=0).item())]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='scripts/training/base_osp_v2.yaml')
    parser.add_argument('--model-path')
    parser.add_argument('--feature-root')
    parser.add_argument('--valid-chunks')
    parser.add_argument('--pool')
    parser.add_argument('--output-dir')
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--gradient-accumulation-steps', type=int)
    parser.add_argument('--checkpoint-every', type=int)
    parser.add_argument('--resume', help='Complete checkpoint directory, or latest')
    parser.add_argument('--check-data', action='store_true', help='Read and sample all selected features without loading models')
    parser.add_argument('--smoke-modes', action='store_true', help='Single-GPU bounded smoke: cycle t2v/v2v/full; changes mode sampling')
    parser.add_argument('--audit-distributed', action='store_true',
                        help='Record rank/sample allocation and compare trainable parameter probes after updates')
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    for key in ('model_path', 'feature_root', 'valid_chunks', 'pool', 'output_dir', 'max_steps',
                'gradient_accumulation_steps', 'checkpoint_every'):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    if config['gradient_accumulation_steps'] < 1 or config['max_steps'] < 1 or config['checkpoint_every'] < 1:
        raise ValueError('Steps, accumulation and checkpoint interval must be positive')
    if config['intrinsics_contract'] not in ('native', 'osp_v2_to_osp_v1_legacy'):
        raise ValueError('Unknown intrinsics contract')
    dataset, data_hash = build_dataset(config)
    if args.check_data:
        for i in range(len(dataset)):
            sample = dataset[i]
            for field in ('input_latents', 'context', 'pose', 'lagernvs_source_latents', 'lagernvs_target_c2w_metric'):
                if not torch.isfinite(sample[field]).all():
                    raise ValueError(f'Nonfinite {field} in {sample["feature_path"]}')
        print(json.dumps({'sampled_files': len(dataset), 'excluded_no_history': len(dataset.no_history_excluded_samples), 'data_hash': data_hash}))
        return
    accelerator = Accelerator(mixed_precision='bf16', gradient_accumulation_steps=config['gradient_accumulation_steps'],
                              kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False)])
    if accelerator.device.type != 'cuda':
        raise RuntimeError('Training requires CUDA')
    if args.smoke_modes and (accelerator.num_processes != 1 or config['max_steps'] > 6):
        raise ValueError('--smoke-modes is restricted to at most six steps on one GPU')
    set_seed(config['seed'], device_specific=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, paths, groups = load_base(config['model_path'], accelerator.device)
    accelerator.print(json.dumps({'trainable_groups': EXPECTED_GROUPS,
                                  'effective_batch': accelerator.num_processes * config['gradient_accumulation_steps']}))
    repencoder = RepEncoder.from_pretrained(paths['repencoder'], device=accelerator.device, compute_dtype='bf16', target_microbatch=4)
    repencoder.requires_grad_(False).eval()
    optimizer = torch.optim.AdamW([{'params': [p for _, p in group], 'name': name} for name, group in groups.items()],
                                 lr=float(config['learning_rate']), betas=(0.9, 0.999), weight_decay=0.01, eps=1e-8)
    model, optimizer = accelerator.prepare(model, optimizer)
    scheduler = make_scheduler()
    output = Path(config['output_dir']).resolve()
    # Separate generator keeps DataLoader iterator construction from consuming model RNG on resume.
    sampler = DistributedSampler(dataset, num_replicas=accelerator.num_processes, rank=accelerator.process_index,
                                 shuffle=True, seed=config['seed'], drop_last=False)
    loader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0,
                        generator=torch.Generator().manual_seed(config['seed']), pin_memory=True)
    frozen_recipe = {k: v for k, v in config.items() if k not in ('max_steps', 'checkpoint_every', 'output_dir', 'model_path', 'feature_root', 'valid_chunks', 'pool')}
    identity = {'recipe': frozen_recipe, 'data_hash': data_hash, 'world_size': accelerator.num_processes,
                'base_checksums': (Path(config['model_path']) / 'SHA256SUMS').read_text(),
                'repencoder_sha256': repencoder.report['model_sha256'], 'smoke_modes': args.smoke_modes}
    progress = {'step': 0, 'epoch': 0, 'batch_in_epoch': 0}
    resume = args.resume
    if resume == 'latest':
        candidates = [p for p in output.glob('checkpoint-*') if (p / 'COMPLETE').is_file()]
        if not candidates:
            raise FileNotFoundError(f'No complete checkpoint under {output}')
        resume = max(candidates, key=lambda p: int(p.name.split('-')[-1]))
    if resume:
        progress = load_checkpoint(accelerator, model, optimizer, resume, identity)
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Choose a fresh output directory or use --resume: {output}')
    # Every rank must finish the fresh-directory check before rank 0 writes
    # metadata; otherwise slower ranks mistake this run's files for an old run.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        output.mkdir(parents=True, exist_ok=True)
        (output / 'config.json').write_text(json.dumps(config, indent=2))
        (output / 'identity.json').write_text(json.dumps(identity, indent=2))
    accelerator.wait_for_everyone()
    if args.audit_distributed:
        write_rank_record(accelerator, output, {'event': 'startup', 'device': str(accelerator.device),
                          'dataset_size': len(dataset), 'sampler_size': len(sampler),
                          'gradient_accumulation_steps': config['gradient_accumulation_steps'],
                          'resume_step': progress['step']})
        verify_parameter_probes(accelerator, model, output, progress['step'])
    if config.get('checkpoint_at_step_zero', True) and not resume:
        save_checkpoint(accelerator, model, optimizer, output, progress, identity)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    microstep = 0
    mode = None
    while progress['step'] < config['max_steps']:
        dataset.set_epoch(progress['epoch'])
        sampler.set_epoch(progress['epoch'])
        for batch_index, batch in enumerate(loader):
            if batch_index < progress['batch_in_epoch']:
                continue
            started = time.monotonic()
            if microstep % config['gradient_accumulation_steps'] == 0:
                forced = ('t2v', 'v2v', 'none')[progress['step'] % 3] if args.smoke_modes else None
                mode = select_history_mode(accelerator, forced)
                accumulated_loss = torch.zeros((), device=accelerator.device)
            if args.audit_distributed:
                write_rank_record(accelerator, output, {'event': 'batch', 'epoch': progress['epoch'],
                                  'batch_in_epoch': batch_index, 'step': progress['step'] + 1,
                                  'history_mode': mode, 'feature_paths': batch['feature_path'],
                                  'target_chunk_indices': batch['target_chunk_index'].tolist()})
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    loss = flow_loss(model, accelerator.unwrap_model(model), repencoder, scheduler,
                                     batch, mode, accelerator.device, progress['step'] + 1)
                if not torch.isfinite(loss).all():
                    raise FloatingPointError(f'Nonfinite loss at step {progress["step"]}')
                accumulated_loss += loss.detach().float()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    # Audit both trainable groups and the frozen boundary before every update.
                    unwrapped = accelerator.unwrap_model(model)
                    for name, p in unwrapped.named_parameters():
                        if not p.requires_grad and p.grad is not None:
                            raise RuntimeError(f'Frozen parameter received a gradient: {name}')
                        if p.grad is not None and not torch.isfinite(p.grad).all():
                            raise FloatingPointError(f'Nonfinite gradient: {name}')
                    for name, group in groups.items():
                        if not any(p.grad is not None and torch.count_nonzero(p.grad).item() for _, p in group):
                            raise RuntimeError(f'No nonzero gradients in {name}')
                    if any(p.grad is not None for p in repencoder.parameters()):
                        raise RuntimeError('Frozen RepEncoder received gradients')
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            microstep += 1
            progress['batch_in_epoch'] = batch_index + 1
            if accelerator.sync_gradients:
                progress['step'] += 1
                if args.audit_distributed:
                    verify_parameter_probes(accelerator, model, output, progress['step'])
                record = {**progress, 'loss': float(accelerator.reduce(accumulated_loss, reduction='mean') / config['gradient_accumulation_steps']),
                          'history_mode': mode, 'seconds_last_microbatch': time.monotonic() - started,
                          'peak_gpu_gib': torch.cuda.max_memory_allocated() / 2**30}
                accelerator.print(json.dumps(record))
                if accelerator.is_main_process:
                    with (output / 'metrics.jsonl').open('a') as f:
                        f.write(json.dumps(record) + '\n')
                if progress['step'] % config['checkpoint_every'] == 0 or progress['step'] == config['max_steps']:
                    save_checkpoint(accelerator, model, optimizer, output, progress, identity)
                if progress['step'] >= config['max_steps']:
                    break
        else:
            progress['epoch'] += 1
            progress['batch_in_epoch'] = 0
    accelerator.wait_for_everyone()
    accelerator.print(f'Finished: {output / ("checkpoint-" + str(progress["step"]))}')


if __name__ == '__main__':
    main()
