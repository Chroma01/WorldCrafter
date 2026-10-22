from __future__ import annotations

import torch

from .first_chunk_inputs import prepare_stage1_clean_input_from_latents


def sample_sequential_history_mode(
    *,
    t2v_ratio: float,
    v2v_ratio: float,
    generator: torch.Generator | None = None,
) -> str:
    """Sample the original sequential Helios t2v/v2v/full branch contract.

    ``v2v_ratio`` is conditional on t2v not being selected.  Thus the
    historical 0.4/0.4 configuration yields 40% t2v, 24% v2v and 36% full.
    Distributed training must call this on rank zero and broadcast the result.
    """

    if not 0.0 <= float(t2v_ratio) <= 1.0:
        raise ValueError(f"t2v_ratio must be in [0, 1], got {t2v_ratio}")
    if not 0.0 <= float(v2v_ratio) <= 1.0:
        raise ValueError(f"v2v_ratio must be in [0, 1], got {v2v_ratio}")
    if torch.rand((), generator=generator).item() <= float(t2v_ratio):
        return "t2v"
    if torch.rand((), generator=generator).item() <= float(v2v_ratio):
        return "v2v"
    return "none"


def build_helios_random_chunk_training_inputs(
    input_latents: torch.Tensor,
    history_latents: torch.Tensor,
    x0_latents: torch.Tensor | None = None,
    num_latent_frames_per_chunk: int = 9,
    memory_size: int = 4,
    history_sizes: tuple[int, int] = (2, 1),
    is_keep_x0: bool = True,
    is_random_drop: bool = False,
    random_drop_i2v_ratio: float = 0.0,
    random_drop_v2v_ratio: float = 0.0,
    random_drop_t2v_ratio: float = 0.0,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | str | None = None,
    forced_drop_mode: str | None = None,
):
    if input_latents.ndim != 5:
        raise ValueError(
            f"input_latents is expected to be [B, C, T, H, W], got shape={tuple(input_latents.shape)}"
        )
    if history_latents.ndim != 5:
        raise ValueError(
            f"history_latents is expected to be [B, C, T_hist, H, W], got shape={tuple(history_latents.shape)}"
        )
    if device is None:
        device = input_latents.device

    if input_latents.shape[2] != num_latent_frames_per_chunk:
        raise ValueError(
            f"input_latents is expected to have exactly {num_latent_frames_per_chunk} latent frames, "
            f"got {input_latents.shape[2]}"
        )
    expected_history_frames = int(memory_size) + sum(sorted(history_sizes, reverse=True))
    if history_latents.shape[2] != expected_history_frames:
        raise ValueError(
            f"history_latents is expected to have exactly {expected_history_frames} history latent frames, "
            f"got {history_latents.shape[2]}"
        )

    target_latents = input_latents.to(device=device, dtype=dtype)
    history_latents = history_latents.to(device=device, dtype=dtype)

    prepared_x0_latents = None
    if is_keep_x0:
        if x0_latents is None:
            prepared_x0_latents = torch.zeros(
                target_latents.shape[0],
                target_latents.shape[1],
                1,
                target_latents.shape[3],
                target_latents.shape[4],
                device=device,
                dtype=dtype,
            )
        else:
            if x0_latents.ndim == 4:
                x0_latents = x0_latents.unsqueeze(2)
            if x0_latents.ndim != 5:
                raise ValueError(
                    "x0_latents is expected to be [B, C, 1, H, W] or [B, C, H, W], "
                    f"got shape={tuple(x0_latents.shape)}"
                )
            prepared_x0_latents = x0_latents[:, :, :1].to(device=device, dtype=dtype)

    (
        target_latents,
        indices_hidden_states,
        indices_latents_history_short,
        indices_latents_history_mid,
        indices_latents_memory,
        latents_history_short,
        latents_history_mid,
        latents_memory,
        applied_drop_mode,
    ) = prepare_stage1_clean_input_from_latents(
        history_latents=history_latents,
        target_latents=target_latents,
        x0_latents=prepared_x0_latents,
        latent_window_size=num_latent_frames_per_chunk,
        memory_size=memory_size,
        history_sizes=list(history_sizes),
        is_random_drop=is_random_drop,
        random_drop_i2v_ratio=random_drop_i2v_ratio,
        random_drop_v2v_ratio=random_drop_v2v_ratio,
        random_drop_t2v_ratio=random_drop_t2v_ratio,
        is_keep_x0=is_keep_x0,
        dtype=dtype,
        device=device,
        forced_drop_mode=forced_drop_mode,
    )

    return {
        "input_latents": target_latents,
        "indices_hidden_states": indices_hidden_states.to(device=device),
        "indices_latents_history_short": indices_latents_history_short.to(device=device),
        "indices_latents_history_mid": indices_latents_history_mid.to(device=device),
        "indices_latents_history_long": indices_latents_memory.to(device=device),
        "latents_history_short": latents_history_short,
        "latents_history_mid": latents_history_mid,
        "latents_history_long": latents_memory,
        "applied_drop_mode": applied_drop_mode,
    }
