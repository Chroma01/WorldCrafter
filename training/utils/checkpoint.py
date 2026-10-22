"""Checkpoint persistence and distributed diagnostics."""

import hashlib
import json
import os
from pathlib import Path
import random
import numpy as np
import torch
from safetensors.torch import save_file
from safetensors.torch import load_file
from ..model import save_adapters


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def save_checkpoint(accelerator, model, optimizer, output, progress, identity):
    destination = Path(output) / f"checkpoint-{progress['step']}"
    staging = destination.with_name(destination.name + '.incomplete')
    if accelerator.is_main_process:
        if destination.exists() or staging.exists():
            raise FileExistsError(f'Refusing to overwrite {destination} or {staging}')
        staging.mkdir(parents=True)
    accelerator.wait_for_everyone()
    torch.save({'python': random.getstate(), 'numpy': np.random.get_state(),
                'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state()},
               staging / f'rng-{accelerator.process_index}.pt')
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        state = {n: p.detach().cpu().contiguous() for n, p in unwrapped.named_parameters() if p.requires_grad}
        save_file(state, str(staging / 'trainable.safetensors'))
        save_adapters(unwrapped, staging / 'adapter')
        torch.save(optimizer.state_dict(), staging / 'optimizer.pt')
        (staging / 'state.json').write_text(json.dumps({'progress': progress, 'identity': identity}, indent=2))
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        (staging / 'COMPLETE').write_text('ok\n')
        os.rename(staging, destination)
    accelerator.wait_for_everyone()
    return destination


def load_checkpoint(accelerator, model, optimizer, path, identity):
    path = Path(path)
    if not (path / 'COMPLETE').is_file():
        raise RuntimeError(f'Incomplete checkpoint: {path}')
    metadata = json.loads((path / 'state.json').read_text())
    if metadata['identity'] != identity:
        raise ValueError('Resume requires the same Base, data, world size and training recipe')
    unwrapped = accelerator.unwrap_model(model)
    expected = {n for n, p in unwrapped.named_parameters() if p.requires_grad}
    state = load_file(str(path / 'trainable.safetensors'))
    if set(state) != expected:
        raise ValueError('Checkpoint trainable parameter names do not match the model')
    unwrapped.load_state_dict(state, strict=False)
    optimizer.load_state_dict(torch.load(path / 'optimizer.pt', map_location='cpu', weights_only=False))
    rng = torch.load(path / f'rng-{accelerator.process_index}.pt', map_location='cpu', weights_only=False)
    random.setstate(rng['python'])
    np.random.set_state(rng['numpy'])
    torch.set_rng_state(rng['torch'])
    torch.cuda.set_rng_state(rng['cuda'])
    return metadata['progress']


def write_rank_record(accelerator, output, record):
    record = {"rank": accelerator.process_index, "world_size": accelerator.num_processes, **record}
    path = Path(output) / f"audit-rank-{accelerator.process_index:05d}.jsonl"
    with path.open("a") as stream:
        stream.write(json.dumps(record) + "\n")


def verify_parameter_probes(accelerator, model, output, step):
    # Sample every trainable tensor; this detects rank drift without gathering
    # the full 4.6 GB of trainable weights. It is not a full-weight equality proof.
    probes = []
    for parameter in accelerator.unwrap_model(model).parameters():
        if parameter.requires_grad:
            flat = parameter.detach().reshape(-1)
            probes.append(flat[[0, flat.numel() // 2, flat.numel() - 1]].float())
    local = torch.cat(probes).contiguous()
    gathered = accelerator.gather(local).reshape(accelerator.num_processes, -1)
    if not torch.isfinite(gathered).all() or not torch.equal(gathered, gathered[:1].expand_as(gathered)):
        raise RuntimeError(f"Trainable parameter probes differ across ranks at step {step}")
    digest = hashlib.sha256(local.cpu().numpy().tobytes()).hexdigest()
    write_rank_record(accelerator, output, {"event": "parameter_probes", "step": step,
                      "trainable_tensors": len(probes), "probe_values": local.numel(),
                      "sha256": digest, "equal_across_ranks": True})
