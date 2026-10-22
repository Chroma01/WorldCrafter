"""Reference source9 -> frozen memory4 -> Stage-1 UCPE flow objective."""
import json
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from .flow_match import FlowMatchScheduler
from .noise import prepare_stage1_noise_input
from .random_chunk_inputs import build_helios_random_chunk_training_inputs, sample_sequential_history_mode
from .camera import build_ucpe_attention_kwargs_for_chunk


def make_scheduler():
    scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
    scheduler.set_timesteps(1000, training=True)
    return scheduler


def prepare_inputs(batch, repencoder, mode, device, *, audit_step=None):
    dtype = torch.bfloat16
    target = batch['input_latents'].to(device=device, dtype=dtype)
    if tuple(target.shape[1:]) != (16, 9, 48, 80):
        raise ValueError(f'Expected [B,16,9,48,80], got {tuple(target.shape)}')
    if mode == 'none':
        source = batch['lagernvs_source_latents'].to(device=device, dtype=dtype)
        with torch.no_grad():
            memory = repencoder(
                source_latents=source.permute(0, 2, 1, 3, 4).contiguous(),
                source_c2w_metric=batch['lagernvs_source_c2w_metric'].to(device=device, dtype=torch.float32),
                target_c2w_metric=batch['lagernvs_target_c2w_metric'].to(device=device, dtype=torch.float32),
                near_zero_baseline_m=float(batch['lagernvs_near_zero_baseline_m'][0]),
            ).detach().to(dtype)
        if tuple(memory.shape) != (target.shape[0], 16, 4, 48, 80) or not torch.isfinite(memory).all():
            raise RuntimeError('Invalid frozen RepEncoder memory4')
    else:
        memory = target.new_zeros(target.shape[0], 16, 4, 48, 80)
    history = torch.cat([memory, batch['helios_history_tail_latents'].to(device=device, dtype=dtype)], dim=2)
    clean = build_helios_random_chunk_training_inputs(
        input_latents=target, history_latents=history,
        x0_latents=batch['x0_latents'].to(device=device, dtype=dtype),
        num_latent_frames_per_chunk=9, memory_size=4, history_sizes=(2, 1),
        is_keep_x0=True, is_random_drop=False, forced_drop_mode=mode, dtype=dtype, device=device,
    )
    return clean


def flow_loss(model, unwrapped, repencoder, scheduler, batch, mode, device, step):
    clean = prepare_inputs(batch, repencoder, mode, device)
    args = SimpleNamespace(training_config=SimpleNamespace(**json.loads(
        Path(__file__).with_name('reference_noise.json').read_text())))
    noisy, _, timesteps, targets, short, mid, long, _ = prepare_stage1_noise_input(
        args=args, model_input=clean['input_latents'], noise_scheduler=scheduler,
        latents_history_short=clean['latents_history_short'],
        latents_history_mid=clean['latents_history_mid'], latents_history_long=clean['latents_history_long'],
        latent_window_size=9, is_keep_x0=True, saturation_audit_step=step if step <= 3 else None,
    )
    cameras = {key: batch[key].to(device=device, dtype=torch.float32) for key in ('pose', 'x_fov', 'xi')}
    attention_kwargs = build_ucpe_attention_kwargs_for_chunk(
        transformer=unwrapped, camera_control_panshot=cameras, height=384, width=640,
        num_latent_frames_per_chunk=9, chunk_index=0, vae_scale_factor_temporal=4,
        restart_each_chunk=False, translation_scale=1.0, pose_is_chunk_aligned=True,
    )
    pred = model(hidden_states=noisy[0], timestep=timesteps[0],
                 encoder_hidden_states=batch['context'].to(device=device, dtype=torch.bfloat16),
                 indices_hidden_states=clean['indices_hidden_states'],
                 indices_latents_history_short=clean['indices_latents_history_short'],
                 indices_latents_history_mid=clean['indices_latents_history_mid'],
                 indices_latents_history_long=clean['indices_latents_history_long'],
                 latents_history_short=short, latents_history_mid=mid, latents_history_long=long,
                 attention_kwargs=attention_kwargs, return_dict=False)[0]
    return F.mse_loss(pred.float(), targets[0].float()) * scheduler.training_weight(timesteps[0]).to(device)
