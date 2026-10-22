"""Flow-matching schedule, corruption and training objective."""

import torch
import math
import json
import random
import torch.nn.functional as F
from pathlib import Path
from types import SimpleNamespace
from .inputs import build_helios_random_chunk_training_inputs
from .inputs import sample_sequential_history_mode
from .camera import build_ucpe_attention_kwargs_for_chunk


class FlowMatchScheduler():

    def __init__(
        self,
        num_inference_steps=100,
        num_train_timesteps=1000,
        shift=3.0,
        sigma_max=1.0,
        sigma_min=0.003/1.002,
        inverse_timesteps=False,
        extra_one_step=False,
        reverse_sigmas=False,
        exponential_shift=False,
        exponential_shift_mu=None,
        shift_terminal=None,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.inverse_timesteps = inverse_timesteps
        self.extra_one_step = extra_one_step
        self.reverse_sigmas = reverse_sigmas
        self.exponential_shift = exponential_shift
        self.exponential_shift_mu = exponential_shift_mu
        self.shift_terminal = shift_terminal
        self.set_timesteps(num_inference_steps)


    def set_timesteps(self, num_inference_steps=100, denoising_strength=1.0, training=False, shift=None, dynamic_shift_len=None):
        if shift is not None:
            self.shift = shift
        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * denoising_strength
        if self.extra_one_step:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps + 1)[:-1]
        else:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps)
        if self.inverse_timesteps:
            self.sigmas = torch.flip(self.sigmas, dims=[0])
        if self.exponential_shift:
            mu = self.calculate_shift(dynamic_shift_len) if dynamic_shift_len is not None else self.exponential_shift_mu
            self.sigmas = math.exp(mu) / (math.exp(mu) + (1 / self.sigmas - 1))
        else:
            self.sigmas = self.shift * self.sigmas / (1 + (self.shift - 1) * self.sigmas)
        if self.shift_terminal is not None:
            one_minus_z = 1 - self.sigmas
            scale_factor = one_minus_z[-1] / (1 - self.shift_terminal)
            self.sigmas = 1 - (one_minus_z / scale_factor)
        if self.reverse_sigmas:
            self.sigmas = 1 - self.sigmas
        self.timesteps = self.sigmas * self.num_train_timesteps
        if training:
            x = self.timesteps
            y = torch.exp(-2 * ((x - num_inference_steps / 2) / num_inference_steps) ** 2)
            y_shifted = y - y.min()
            bsmntw_weighing = y_shifted * (num_inference_steps / y_shifted.sum())
            self.linear_timesteps_weights = bsmntw_weighing
            self.training = True
        else:
            self.training = False


    def step(self, model_output, timestep, sample, to_final=False, **kwargs):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        if to_final or timestep_id + 1 >= len(self.timesteps):
            sigma_ = 1 if (self.inverse_timesteps or self.reverse_sigmas) else 0
        else:
            sigma_ = self.sigmas[timestep_id + 1]
        prev_sample = sample + model_output * (sigma_ - sigma)
        return prev_sample


    def return_to_timestep(self, timestep, sample, sample_stablized):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        model_output = (sample - sample_stablized) / sigma
        return model_output


    def add_noise(self, original_samples, noise, timestep):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        sample = (1 - sigma) * original_samples + sigma * noise
        return sample


    def training_target(self, sample, noise, timestep):
        target = noise - sample
        return target


    def training_weight(self, timestep):
        timestep_id = torch.argmin((self.timesteps - timestep.to(self.timesteps.device)).abs())
        weights = self.linear_timesteps_weights[timestep_id]
        return weights


    def calculate_shift(
        self,
        image_seq_len,
        base_seq_len: int = 256,
        max_seq_len: int = 8192,
        base_shift: float = 0.5,
        max_shift: float = 0.9,
    ):
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        mu = image_seq_len * m + b
        return mu


def downsample_corrupt(model_input, downsample_min_corrupt_ratio, downsample_max_corrupt_ratio):
    corrupt_ratio = random.uniform(downsample_min_corrupt_ratio, downsample_max_corrupt_ratio)

    is_5d = model_input.ndim == 5

    if is_5d:
        B, C, T, H, W = model_input.shape
        model_input = model_input.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
    else:
        B, C, H, W = model_input.shape

    h0, w0 = model_input.shape[-2:]

    h1 = max(1, int(round(h0 * corrupt_ratio)))
    w1 = max(1, int(round(w0 * corrupt_ratio)))

    model_input = F.interpolate(model_input, size=(h1, w1), mode="bilinear", align_corners=False, antialias=True)

    model_input = F.interpolate(model_input, size=(h0, w0), mode="bilinear", align_corners=False, antialias=True)

    if is_5d:
        model_input = model_input.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4)

    return model_input

def get_corrupt_noise_sigma(model_input, batch_size, corrupt_ratio=1 / 3, num_frames=None, is_frame_independent=False):
    if is_frame_independent:
        noise_sigma_shape = (batch_size, 1, num_frames)
    else:
        noise_sigma_shape = (batch_size,)
    noise_sigma = (
        torch.rand(size=noise_sigma_shape, device=model_input.device, dtype=model_input.dtype) * corrupt_ratio
    )
    while len(noise_sigma.shape) < model_input.ndim:
        noise_sigma = noise_sigma.unsqueeze(-1)
    return noise_sigma

def corrupt_model_input(
    model_input,
    # choose mode
    corrupt_mode="noise",  # "noise" | "downsample" | "random"
    noise_mode_prob=0.9,  # when corrupt_mode="random", select the probability of noise (select downsample for the remaining probability).
    # for noise
    is_frame_independent=False,
    is_chunk_independent=False,
    noise_corrupt_ratio=1 / 3,
    noise_corrupt_clean_prob=0.1,
    # for downsample
    downsample_min_corrupt_ratio=0.9,
    downsample_max_corrupt_ratio=1.0,
):
    assert not (is_frame_independent and is_chunk_independent), (
        "is_frame_independent and is_chunk_independent cannot both be True"
    )
    assert corrupt_mode in ("noise", "downsample", "random"), (
        f"corrupt_mode must be 'noise', 'downsample', or 'random', got '{corrupt_mode}'"
    )

    # ==================== choose mode ====================
    if corrupt_mode == "random":
        mode = "noise" if random.random() < noise_mode_prob else "downsample"
    else:
        mode = corrupt_mode

    # ==================== downsample branch ====================
    if mode == "downsample":
        model_input = downsample_corrupt(
            model_input=model_input,
            downsample_min_corrupt_ratio=downsample_min_corrupt_ratio,
            downsample_max_corrupt_ratio=downsample_max_corrupt_ratio,
        )
        return model_input

    # ==================== noise branch ====================
    clean_random = random.random()
    if clean_random < noise_corrupt_clean_prob:
        return model_input

    noise_sigma = get_corrupt_noise_sigma(
        model_input=model_input,
        batch_size=model_input.shape[0],
        corrupt_ratio=noise_corrupt_ratio,
        num_frames=model_input.shape[2],
        is_frame_independent=is_frame_independent,
    )

    model_input = noise_sigma * torch.randn_like(model_input) + (1 - noise_sigma) * model_input

    return model_input

def corrupt_history_latents(
    latents_history_short,
    latents_history_mid,
    latents_history_long,
    latent_window_size,
    is_keep_x0=True,
    # choose mode
    corrupt_mode="noise",  # "noise" | "downsample" | "random"
    noise_mode_prob=0.9,  # when corrupt_mode="random", select the probability of noise (select downsample for the remaining probability).
    # for noise
    is_frame_independent=False,
    is_chunk_independent=False,
    corrupt_ratio_1x=1 / 3,
    corrupt_ratio_2x=1 / 3,
    corrupt_ratio_4x=1 / 3,
    noise_corrupt_clean_prob=0.1,
    # for downsample
    downsample_min_corrupt_ratio=0.9,
    downsample_max_corrupt_ratio=1.0,
):
    assert not (is_frame_independent and is_chunk_independent), (
        "is_frame_independent and is_chunk_independent cannot both be True"
    )
    assert corrupt_mode in ("noise", "downsample", "random"), (
        f"corrupt_mode must be 'noise', 'downsample', or 'random', got '{corrupt_mode}'"
    )

    clean_random = random.random()
    if clean_random < noise_corrupt_clean_prob:
        return latents_history_short, latents_history_mid, latents_history_long

    # ==================== choose mode ====================
    if corrupt_mode == "random":
        mode = "noise" if random.random() < noise_mode_prob else "downsample"
    else:
        mode = corrupt_mode

    # ==================== noise branch ====================
    if mode == "noise":
        batch_size = latents_history_short.shape[0]
        if not is_frame_independent and not is_chunk_independent:
            noise_sigma = get_corrupt_noise_sigma(
                model_input=latents_history_short, batch_size=batch_size, corrupt_ratio=corrupt_ratio_1x
            )

    len_4x = latents_history_long.shape[2]
    len_2x = latents_history_mid.shape[2]
    len_1x = latents_history_short.shape[2]

    hist_seq_len = len_4x + len_2x + len_1x
    hist_seq_len_copy = hist_seq_len

    ori_len_1x = len_1x
    if is_keep_x0:
        len_1x -= 1
        hist_seq_len -= 1
        begin_num = 1
    else:
        begin_num = 0

    max_windows = hist_seq_len // latent_window_size
    tail_num = hist_seq_len % latent_window_size

    assert hist_seq_len_copy == tail_num + max_windows * latent_window_size + begin_num

    tail_latents_history = None
    begin_latents_history = None

    def corrupt_history_segment(segment, corrupt_ratio):
        if segment.shape[2] == 0 or torch.count_nonzero(segment).item() == 0:
            return segment
        if mode == "downsample":
            return downsample_corrupt(
                model_input=segment,
                downsample_min_corrupt_ratio=downsample_min_corrupt_ratio,
                downsample_max_corrupt_ratio=downsample_max_corrupt_ratio,
            )

        noise_sigma = get_corrupt_noise_sigma(
            model_input=segment,
            batch_size=segment.shape[0],
            corrupt_ratio=corrupt_ratio,
            num_frames=segment.shape[2],
            is_frame_independent=is_frame_independent,
        )
        return noise_sigma * torch.randn_like(segment) + (1 - noise_sigma) * segment

    # Compact memory history is shorter than one full official history window.
    # In that case, corrupt each semantic segment directly and return before
    # the legacy tail/window split can reinterpret memory as an incomplete tail.
    if max_windows == 0:
        compact_short = latents_history_short
        if begin_num != 0:
            begin_latents_history = compact_short[:, :, :begin_num, :, :]
            compact_short = compact_short[:, :, begin_num:, :, :]
            begin_latents_history = corrupt_history_segment(
                begin_latents_history,
                corrupt_ratio_1x,
            )

        latents_history_long = corrupt_history_segment(latents_history_long, corrupt_ratio_4x)
        latents_history_mid = corrupt_history_segment(latents_history_mid, corrupt_ratio_2x)
        compact_short = corrupt_history_segment(compact_short, corrupt_ratio_1x)

        if begin_latents_history is not None:
            compact_short = torch.cat([begin_latents_history, compact_short], dim=2)

        return compact_short, latents_history_mid, latents_history_long

    if tail_num != 0:
        tail_latents_history = latents_history_long[:, :, :tail_num, :, :]
        latents_history_long = latents_history_long[:, :, tail_num:, :, :]
        if tail_latents_history.sum() != 0:
            if mode == "downsample":
                tail_latents_history = downsample_corrupt(
                    model_input=tail_latents_history,
                    downsample_min_corrupt_ratio=downsample_min_corrupt_ratio,
                    downsample_max_corrupt_ratio=downsample_max_corrupt_ratio,
                )
            else:
                noise_sigma = get_corrupt_noise_sigma(
                    model_input=latents_history_short,
                    batch_size=batch_size,
                    corrupt_ratio=corrupt_ratio_4x,
                    num_frames=tail_latents_history.shape[2],
                    is_frame_independent=is_frame_independent,
                )
                tail_latents_history = (
                    noise_sigma * torch.randn_like(tail_latents_history) + (1 - noise_sigma) * tail_latents_history
                )

    if begin_num != 0:
        begin_latents_history = latents_history_short[:, :, :begin_num, :, :]
        latents_history_short = latents_history_short[:, :, begin_num:, :, :]
        if begin_latents_history.sum() != 0:
            if mode == "downsample":
                begin_latents_history = downsample_corrupt(
                    model_input=begin_latents_history,
                    downsample_min_corrupt_ratio=downsample_min_corrupt_ratio,
                    downsample_max_corrupt_ratio=downsample_max_corrupt_ratio,
                )
            else:
                noise_sigma = get_corrupt_noise_sigma(
                    model_input=latents_history_short,
                    batch_size=batch_size,
                    corrupt_ratio=corrupt_ratio_1x,
                    num_frames=begin_latents_history.shape[2],
                    is_frame_independent=is_frame_independent,
                )
                begin_latents_history = (
                    noise_sigma * torch.randn_like(begin_latents_history) + (1 - noise_sigma) * begin_latents_history
                )

    mid_latents_history = torch.cat([latents_history_long, latents_history_mid, latents_history_short], dim=2)
    window_num = mid_latents_history.shape[2] // latent_window_size
    if mid_latents_history.shape[2] % latent_window_size != 0:
        assert window_num == 0, (
            f"mid length {mid_latents_history.shape[2]} not divisible by window size {latent_window_size}"
        )

    seq_begin = 0
    for idx in range(window_num):
        seq_end = seq_begin + latent_window_size
        if mid_latents_history[:, :, seq_begin:seq_end, :, :].sum() != 0:
            if idx == window_num - 1:
                len_2x_end = seq_begin + len_2x
                if mode == "downsample":
                    mid_latents_history[:, :, seq_begin:len_2x_end, :, :] = downsample_corrupt(
                        model_input=mid_latents_history[:, :, seq_begin:len_2x_end, :, :],
                        downsample_min_corrupt_ratio=downsample_min_corrupt_ratio,
                        downsample_max_corrupt_ratio=downsample_max_corrupt_ratio,
                    )
                else:
                    noise_sigma_4x = get_corrupt_noise_sigma(
                        model_input=latents_history_short,
                        batch_size=batch_size,
                        corrupt_ratio=corrupt_ratio_4x,
                        num_frames=len_2x,
                        is_frame_independent=is_frame_independent,
                    )
                    mid_latents_history[:, :, seq_begin:len_2x_end, :, :] = (
                        noise_sigma_4x * torch.randn_like(mid_latents_history[:, :, seq_begin:len_2x_end, :, :])
                        + (1 - noise_sigma_4x) * mid_latents_history[:, :, seq_begin:len_2x_end, :, :]
                    )

                remaining_frames = seq_end - len_2x_end
                if mode == "downsample":
                    mid_latents_history[:, :, len_2x_end:seq_end, :, :] = downsample_corrupt(
                        model_input=mid_latents_history[:, :, len_2x_end:seq_end, :, :],
                        downsample_min_corrupt_ratio=downsample_min_corrupt_ratio,
                        downsample_max_corrupt_ratio=downsample_max_corrupt_ratio,
                    )
                else:
                    noise_sigma_2x = get_corrupt_noise_sigma(
                        model_input=latents_history_short,
                        batch_size=batch_size,
                        corrupt_ratio=corrupt_ratio_2x,
                        num_frames=remaining_frames,
                        is_frame_independent=is_frame_independent,
                    )
                    mid_latents_history[:, :, len_2x_end:seq_end, :, :] = (
                        noise_sigma_2x * torch.randn_like(mid_latents_history[:, :, len_2x_end:seq_end, :, :])
                        + (1 - noise_sigma_2x) * mid_latents_history[:, :, len_2x_end:seq_end, :, :]
                    )
            else:
                if mode == "downsample":
                    mid_latents_history[:, :, seq_begin:seq_end, :, :] = downsample_corrupt(
                        model_input=mid_latents_history[:, :, seq_begin:seq_end, :, :],
                        downsample_min_corrupt_ratio=downsample_min_corrupt_ratio,
                        downsample_max_corrupt_ratio=downsample_max_corrupt_ratio,
                    )
                else:
                    noise_sigma = get_corrupt_noise_sigma(
                        model_input=latents_history_short,
                        batch_size=batch_size,
                        corrupt_ratio=corrupt_ratio_4x,
                        num_frames=latent_window_size,
                        is_frame_independent=is_frame_independent,
                    )
                    mid_latents_history[:, :, seq_begin:seq_end, :, :] = (
                        noise_sigma * torch.randn_like(mid_latents_history[:, :, seq_begin:seq_end, :, :])
                        + (1 - noise_sigma) * mid_latents_history[:, :, seq_begin:seq_end, :, :]
                    )
        seq_begin = seq_end

    recovers = []
    if tail_latents_history is not None:
        recovers.append(tail_latents_history)
    recovers.append(mid_latents_history[:, :, :-len_1x, :, :])
    if begin_latents_history is not None:
        recovers.append(begin_latents_history)
    recovers.append(mid_latents_history[:, :, -len_1x:, :, :])
    mid_latents_history = torch.cat(recovers, dim=2)

    # Split and update back to original tensors
    latents_4x_recovered, latents_2x_recovered, latents_history_short_recovered = mid_latents_history.split(
        [len_4x, len_2x, ori_len_1x], dim=2
    )

    return (
        latents_history_short_recovered,
        latents_2x_recovered,
        latents_4x_recovered,
    )

def add_saturation_to_history_latents(
    latents_history_short,
    latents_history_mid,
    latents_history_long,
    latent_window_size,
    is_keep_x0=False,
    saturation_ratio_min=0.7,
    saturation_ratio_max=2.0,
    saturation_clean_prob=0.2,
):
    # clean_random = random.random()
    # if clean_random < saturation_clean_prob:
    #     return latents_history_short, latents_history_mid, latents_history_long

    def get_saturation(x1, saturation_ratio_min, saturation_ratio_max):
        if random.random() < 0.5:
            sat_factor = random.uniform(saturation_ratio_min, 1.0 - 1e-3)
        else:
            sat_factor = random.uniform(1.0 + 1e-3, saturation_ratio_max)
        latent_mean = torch.mean(x1, dim=1, keepdim=True)
        x1_saturated = (x1 - latent_mean) * sat_factor + latent_mean
        return x1_saturated

    len_4x = latents_history_long.shape[2]
    len_2x = latents_history_mid.shape[2]
    len_1x = latents_history_short.shape[2]

    hist_seq_len = len_4x + len_2x + len_1x
    hist_seq_len_copy = hist_seq_len

    ori_len_1x = len_1x
    if is_keep_x0:
        len_1x -= 1
        hist_seq_len -= 1
        begin_num = 1
    else:
        begin_num = 0

    max_windows = hist_seq_len // latent_window_size
    tail_num = hist_seq_len % latent_window_size

    assert hist_seq_len_copy == tail_num + max_windows * latent_window_size + begin_num

    tail_latents_history = None
    begin_latents_history = None

    # Memory training uses a compact context shorter than one Helios history
    # window. Preserve the same semantic segmentation as compact corruption:
    # keep the x0 prefix clean and perturb memory, mid, and short separately.
    if max_windows == 0:
        def maybe_saturate(segment):
            if segment.shape[2] == 0 or torch.count_nonzero(segment).item() == 0:
                return segment
            if random.random() < saturation_clean_prob:
                return segment
            return get_saturation(
                segment,
                saturation_ratio_min=saturation_ratio_min,
                saturation_ratio_max=saturation_ratio_max,
            )

        compact_short = latents_history_short
        if begin_num != 0:
            begin_latents_history = compact_short[:, :, :begin_num, :, :]
            compact_short = compact_short[:, :, begin_num:, :, :]

        latents_history_long = maybe_saturate(latents_history_long)
        latents_history_mid = maybe_saturate(latents_history_mid)
        compact_short = maybe_saturate(compact_short)

        if begin_latents_history is not None:
            compact_short = torch.cat([begin_latents_history, compact_short], dim=2)
        return compact_short, latents_history_mid, latents_history_long

    if tail_num != 0:
        tail_latents_history = latents_history_long[:, :, :tail_num, :, :]
        latents_history_long = latents_history_long[:, :, tail_num:, :, :]
        if tail_latents_history.sum() != 0:
            if random.random() < saturation_clean_prob:
                tail_latents_history = tail_latents_history
            else:
                tail_latents_history = get_saturation(
                    tail_latents_history,
                    saturation_ratio_min=saturation_ratio_min,
                    saturation_ratio_max=saturation_ratio_max,
                )

    if begin_num != 0:
        begin_latents_history = latents_history_short[:, :, :begin_num, :, :]
        latents_history_short = latents_history_short[:, :, begin_num:, :, :]
        # if begin_latents_history.sum() != 0:
        #     begin_latents_history = get_saturation(
        #         begin_latents_history,
        #         saturation_ratio_min=saturation_ratio_min,
        #         saturation_ratio_max=saturation_ratio_max,
        #     )

    mid_latents_history = torch.cat([latents_history_long, latents_history_mid, latents_history_short], dim=2)
    window_num = mid_latents_history.shape[2] // latent_window_size
    assert mid_latents_history.shape[2] % latent_window_size == 0, (
        f"mid length {mid_latents_history.shape[2]} not divisible by window size {latent_window_size}"
    )

    seq_begin = 0
    for idx in range(window_num):
        seq_end = seq_begin + latent_window_size
        if mid_latents_history[:, :, seq_begin:seq_end, :, :].sum() != 0:
            if idx == window_num - 1:
                len_2x_end = seq_begin + len_2x
                if random.random() < saturation_clean_prob:
                    mid_latents_history[:, :, seq_begin:len_2x_end, :, :] = mid_latents_history[
                        :, :, seq_begin:len_2x_end, :, :
                    ]
                else:
                    mid_latents_history[:, :, seq_begin:len_2x_end, :, :] = get_saturation(
                        mid_latents_history[:, :, seq_begin:len_2x_end, :, :],
                        saturation_ratio_min=saturation_ratio_min,
                        saturation_ratio_max=saturation_ratio_max,
                    )

                if random.random() < saturation_clean_prob:
                    mid_latents_history[:, :, len_2x_end:seq_end, :, :] = mid_latents_history[
                        :, :, len_2x_end:seq_end, :, :
                    ]
                else:
                    mid_latents_history[:, :, len_2x_end:seq_end, :, :] = get_saturation(
                        mid_latents_history[:, :, len_2x_end:seq_end, :, :],
                        saturation_ratio_min=saturation_ratio_min,
                        saturation_ratio_max=saturation_ratio_max,
                    )
            else:
                if random.random() < saturation_clean_prob:
                    mid_latents_history[:, :, seq_begin:seq_end, :, :] = mid_latents_history[
                        :, :, seq_begin:seq_end, :, :
                    ]
                else:
                    mid_latents_history[:, :, seq_begin:seq_end, :, :] = get_saturation(
                        mid_latents_history[:, :, seq_begin:seq_end, :, :],
                        saturation_ratio_min=saturation_ratio_min,
                        saturation_ratio_max=saturation_ratio_max,
                    )

        seq_begin = seq_end

    recovers = []
    if tail_latents_history is not None:
        recovers.append(tail_latents_history)
    recovers.append(mid_latents_history[:, :, :-len_1x, :, :])
    if begin_latents_history is not None:
        recovers.append(begin_latents_history)
    recovers.append(mid_latents_history[:, :, -len_1x:, :, :])
    mid_latents_history = torch.cat(recovers, dim=2)

    # Split and update back to original tensors
    latents_4x_recovered, latents_2x_recovered, latents_history_short_recovered = mid_latents_history.split(
        [len_4x, len_2x, ori_len_1x], dim=2
    )

    return (
        latents_history_short_recovered,
        latents_2x_recovered,
        latents_4x_recovered,
    )

def prepare_stage1_noise_input(
    args,
    model_input,
    noise_scheduler,
    recycle_vars=None,
    latents_history_short=None,
    latents_history_mid=None,
    latents_history_long=None,
    latent_window_size=9,
    is_keep_x0=True,
    return_list=True,
    saturation_audit_step=None,
):
    if args.training_config.weighting_scheme != "ucpe_success" or args.training_config.use_error_recycling:
        raise ValueError("Only the reference Stage-1 UCPE objective is supported")
    # Sample noise that we'll add to the latents
    noise = torch.randn_like(model_input)
    bsz = model_input.shape[0]

    use_clean_input = False
    noise_w_error = noise
    model_input_w_error = model_input

    if args.training_config.weighting_scheme == "ucpe_success":
        timestep_id = torch.randint(
            0,
            noise_scheduler.num_train_timesteps,
            (1,),
            device=model_input.device,
        )
        timesteps = noise_scheduler.timesteps[timestep_id.cpu()].to(
            dtype=model_input.dtype,
            device=model_input.device,
        )
        resolved_timestep_id = torch.argmin((noise_scheduler.timesteps - timesteps.cpu()).abs())
        sigmas = noise_scheduler.sigmas[resolved_timestep_id].reshape(1)

    # Add noise according to flow matching.
    # zt = (1 - texp) * x + texp * z1
    while len(sigmas.shape) < model_input.ndim:
        sigmas = sigmas.unsqueeze(-1)

    sigmas = sigmas.to(model_input.device, dtype=model_input.dtype)


    if args.training_config.corrupt_history and latents_history_short is not None:
        latents_history_short, latents_history_mid, latents_history_long = corrupt_history_latents(
            latents_history_short,
            latents_history_mid,
            latents_history_long,
            latent_window_size,
            is_keep_x0=True,
            # choose mode
            corrupt_mode=args.training_config.corrupt_mode_history,
            noise_mode_prob=args.training_config.corrupt_mode_prob_history,
            # for noise
            is_frame_independent=args.training_config.is_frame_independent_corrupt_history,
            is_chunk_independent=args.training_config.is_chunk_independent_corrupt_history,
            corrupt_ratio_1x=args.training_config.noise_corrupt_ratio_history_short,
            corrupt_ratio_2x=args.training_config.noise_corrupt_ratio_history_mid,
            corrupt_ratio_4x=args.training_config.noise_corrupt_ratio_history_long,
            noise_corrupt_clean_prob=args.training_config.noise_corrupt_clean_prob_history,
            # for downsample
            downsample_min_corrupt_ratio=args.training_config.downsample_min_corrupt_ratio_history,
            downsample_max_corrupt_ratio=args.training_config.downsample_max_corrupt_ratio_history,
        )

    # Copied from the validated UCPE random-chunk training path. Saturation is
    # intentionally applied after history corruption and before model-input
    # corruption/noising; this keeps the original Helios flow loss unchanged.
    if args.training_config.is_add_saturation and latents_history_short is not None:
        saturation_inputs_before = None
        if saturation_audit_step is not None:
            saturation_inputs_before = tuple(
                tensor.detach().clone()
                for tensor in (latents_history_short, latents_history_mid, latents_history_long)
            )
        latents_history_short, latents_history_mid, latents_history_long = add_saturation_to_history_latents(
            latents_history_short,
            latents_history_mid,
            latents_history_long,
            latent_window_size,
            is_keep_x0=is_keep_x0,
            saturation_ratio_min=args.training_config.saturation_ratio_min,
            saturation_ratio_max=args.training_config.saturation_ratio_max,
            saturation_clean_prob=args.training_config.saturation_ratio_clean_prob,
        )
        if saturation_inputs_before is not None:
            saturation_inputs_after = (latents_history_short, latents_history_mid, latents_history_long)
            max_abs_diff = max(
                float((after.float() - before.float()).abs().max().item())
                for before, after in zip(saturation_inputs_before, saturation_inputs_after)
            )
            changed_tensor_count = sum(
                not torch.equal(before, after)
                for before, after in zip(saturation_inputs_before, saturation_inputs_after)
            )
            short_body_start = 1 if is_keep_x0 else 0
            segment_audit = {
                "short_body": (
                    saturation_inputs_before[0][:, :, short_body_start:],
                    saturation_inputs_after[0][:, :, short_body_start:],
                ),
                "mid": (saturation_inputs_before[1], saturation_inputs_after[1]),
                "memory": (saturation_inputs_before[2], saturation_inputs_after[2]),
            }
            x0_preserved = torch.equal(
                saturation_inputs_before[0][:, :, :1],
                saturation_inputs_after[0][:, :, :1],
            )
            zero_preserved = all(
                torch.count_nonzero(before).item() != 0 or torch.equal(after, before)
                for before, after in zip(saturation_inputs_before, saturation_inputs_after)
            )
            finite = all(torch.isfinite(tensor).all().item() for tensor in saturation_inputs_after)
            print(
                "[ucpe-saturation-audit] "
                + json.dumps(
                    {
                        "branch_executed": True,
                        "changed_tensor_count": changed_tensor_count,
                        "clean_prob": float(args.training_config.saturation_ratio_clean_prob),
                        "finite": finite,
                        "max_abs_diff": max_abs_diff,
                        **{
                            f"{name}_changed": not torch.equal(before, after)
                            for name, (before, after) in segment_audit.items()
                        },
                        **{
                            f"{name}_nonzero": torch.count_nonzero(before).item() > 0
                            for name, (before, _) in segment_audit.items()
                        },
                        "ratio_max": float(args.training_config.saturation_ratio_max),
                        "ratio_min": float(args.training_config.saturation_ratio_min),
                        "step": int(saturation_audit_step),
                        "x0_preserved": x0_preserved,
                        "zero_preserved": zero_preserved,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    if args.training_config.corrupt_model_input:
        model_input_w_error = corrupt_model_input(
            model_input_w_error,
            # choose mode
            corrupt_mode=args.training_config.corrupt_mode_model_input,
            noise_mode_prob=args.training_config.corrupt_mode_prob_model_input,
            # for noise
            is_frame_independent=args.training_config.is_frame_independent_corrupt_model_input,
            is_chunk_independent=args.training_config.is_chunk_independent_corrupt_model_input,
            noise_corrupt_ratio=args.training_config.noise_corrupt_ratio_model_input,
            noise_corrupt_clean_prob=args.training_config.noise_corrupt_clean_prob_model_input,
            # for downsample
            downsample_min_corrupt_ratio=args.training_config.downsample_min_corrupt_ratio_model_input,
            downsample_max_corrupt_ratio=args.training_config.downsample_max_corrupt_ratio_model_input,
        )

    # Get flow-matching target
    if args.training_config.weighting_scheme == "ucpe_success":
        noisy_model_input = noise_scheduler.add_noise(model_input_w_error, noise_w_error, timesteps)
    target = noise_w_error - model_input

    noisy_model_input_list = [noisy_model_input] if return_list else noisy_model_input
    sigmas_list = [sigmas] if return_list else sigmas
    timesteps_list = [timesteps] if return_list else timesteps
    targets_list = [target] if return_list else target

    return (
        noisy_model_input_list,
        sigmas_list,
        timesteps_list,
        targets_list,
        latents_history_short,
        latents_history_mid,
        latents_history_long,
        use_clean_input,
    )


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
