"""Clean target and history inputs for Stage-1 training."""
from __future__ import annotations

import random
import torch


def _zero_prefix_like(tensor: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(tensor, device=tensor.device, dtype=tensor.dtype)


def prepare_stage1_clean_input_from_latents(
    history_latents,
    target_latents,
    x0_latents=None,
    latent_window_size: int = 9,
    memory_size: int = 4,
    history_sizes: list = [2, 1],
    is_random_drop: bool = False,
    random_drop_i2v_ratio: float = 0,
    random_drop_v2v_ratio: float = 0,
    random_drop_t2v_ratio: float = 0,
    is_keep_x0: bool = True,
    dtype=torch.bfloat16,
    device="cpu",
    forced_drop_mode: str | None = None,
):
    if is_keep_x0:
        latents_prefix = x0_latents.to(device, dtype=dtype)
    else:
        assert x0_latents is None
    applied_drop_mode = "none"

    history_sizes = sorted(history_sizes, reverse=True)
    history_window_size = int(memory_size) + sum(history_sizes)
    total_window_size = history_window_size + latent_window_size
    assert total_window_size == history_latents.shape[2] + target_latents.shape[2], (
        f"total_window_size mismatch: expected {total_window_size}"
        f"(history={history_latents.shape[2]} + target={target_latents.shape[2]}), "
        f"but got {history_latents.shape[2] + target_latents.shape[2]}"
    )

    indices = torch.arange(0, sum([1, memory_size, *history_sizes, latent_window_size])).unsqueeze(0).expand(
        target_latents.shape[0], -1
    )
    (
        indices_prefix,
        indices_latents_memory,
        indices_latents_history_mid,
        indices_latents_history_1x,
        indices_hidden_states,
    ) = indices.split([1, memory_size, *history_sizes, latent_window_size], dim=1)
    indices_latents_history_short = torch.cat([indices_prefix, indices_latents_history_1x], dim=1)

    latents_memory, latents_history_mid, latents_history_1x = history_latents.split(
        [memory_size, *history_sizes], dim=2
    )

    if forced_drop_mode not in {None, "none", "t2v", "v2v"}:
        raise ValueError(
            "forced_drop_mode must be one of None/'none'/'t2v'/'v2v', "
            f"got {forced_drop_mode!r}"
        )

    # The LagerNVS joint-training path samples this mode once on rank zero and
    # broadcasts it to every rank.  Keeping the transformation here preserves
    # the successful Helios t2v/v2v/full semantics without allowing each
    # process to silently choose a different branch.
    if forced_drop_mode == "t2v":
        applied_drop_mode = "t2v"
        if is_keep_x0:
            latents_prefix = _zero_prefix_like(latents_prefix)
        latents_history_1x = _zero_prefix_like(latents_history_1x)
        latents_history_mid = _zero_prefix_like(latents_history_mid)
        latents_memory = _zero_prefix_like(latents_memory)
    elif forced_drop_mode == "v2v":
        applied_drop_mode = "v2v"
        # With the fixed [memory4, mid2, recent1] contract the historical v2v
        # branch keeps x0 + recent[-1] and removes memory4 + mid2.
        latents_memory = _zero_prefix_like(latents_memory)
        latents_history_mid = _zero_prefix_like(latents_history_mid)
    elif forced_drop_mode == "none":
        applied_drop_mode = "none"
    elif is_random_drop:
        if random_drop_t2v_ratio != 0 and torch.rand(1).item() <= random_drop_t2v_ratio:
            applied_drop_mode = "t2v"
            if is_keep_x0:
                latents_prefix = _zero_prefix_like(latents_prefix)
            latents_history_1x = _zero_prefix_like(latents_history_1x)
            latents_history_mid = _zero_prefix_like(latents_history_mid)
            latents_memory = _zero_prefix_like(latents_memory)
        else:
            len_memory = latents_memory.shape[2]
            len_2x = latents_history_mid.shape[2]
            len_1x = latents_history_1x.shape[2]
            hist_seq_len = len_memory + len_2x + len_1x

            total_drop = 0
            is_drop_triggered = False

            if random_drop_i2v_ratio != 0 and torch.rand(1).item() <= random_drop_i2v_ratio:
                total_drop = max(0, hist_seq_len - 1)
                is_drop_triggered = True
                applied_drop_mode = "i2v_drop"
            elif random_drop_v2v_ratio != 0 and torch.rand(1).item() <= random_drop_v2v_ratio:
                max_windows = hist_seq_len // latent_window_size
                tail_num = hist_seq_len % latent_window_size
                if max_windows > 0:
                    total_drop = tail_num
                    drop_windows = random.randint(0, max_windows - 1)
                    total_drop += drop_windows * latent_window_size
                else:
                    total_drop = max(0, hist_seq_len - 1)
                is_drop_triggered = True
                applied_drop_mode = "v2v"

            if is_drop_triggered and total_drop > 0:
                remaining_drop = total_drop
                if remaining_drop > 0 and len_memory > 0:
                    drop_memory = min(remaining_drop, len_memory)
                    latents_memory = torch.cat(
                        [
                            torch.zeros_like(latents_memory[:, :, :drop_memory]),
                            latents_memory[:, :, drop_memory:],
                        ],
                        dim=2,
                    )
                    remaining_drop -= drop_memory
                if remaining_drop > 0 and len_2x > 0:
                    drop_2x = min(remaining_drop, len_2x)
                    latents_history_mid = torch.cat(
                        [
                            torch.zeros_like(latents_history_mid[:, :, :drop_2x]),
                            latents_history_mid[:, :, drop_2x:],
                        ],
                        dim=2,
                    )
                    remaining_drop -= drop_2x
                if remaining_drop > 0 and len_1x > 0:
                    drop_1x = min(remaining_drop, len_1x)
                    latents_history_1x = torch.cat(
                        [
                            torch.zeros_like(latents_history_1x[:, :, :drop_1x]),
                            latents_history_1x[:, :, drop_1x:],
                        ],
                        dim=2,
                    )

    if is_keep_x0:
        latents_history_short = torch.cat([latents_prefix, latents_history_1x], dim=2)
    else:
        latents_history_short = latents_history_1x

    return (
        target_latents,
        indices_hidden_states,
        indices_latents_history_short,
        indices_latents_history_mid,
        indices_latents_memory,
        latents_history_short,
        latents_history_mid,
        latents_memory,
        applied_drop_mode,
    )


def build_helios_first_chunk_training_inputs(
    input_latents: torch.Tensor,
    first_frame_latents: torch.Tensor | None = None,
    num_latent_frames_per_chunk: int = 9,
    memory_size: int = 4,
    history_sizes: tuple[int, int] = (2, 1),
    is_keep_x0: bool = True,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | str | None = None,
):
    if input_latents.ndim != 5:
        raise ValueError(
            f"input_latents is expected to be [B, C, T, H, W], got shape={tuple(input_latents.shape)}"
        )

    if device is None:
        device = input_latents.device

    if input_latents.shape[2] < num_latent_frames_per_chunk:
        raise ValueError(
            f"input_latents only has {input_latents.shape[2]} latent frames, "
            f"but num_latent_frames_per_chunk={num_latent_frames_per_chunk}"
        )

    target_latents = input_latents[:, :, :num_latent_frames_per_chunk].to(device=device, dtype=dtype)

    history_window_size = int(memory_size) + sum(sorted(history_sizes, reverse=True))
    history_latents = torch.zeros(
        target_latents.shape[0],
        target_latents.shape[1],
        history_window_size,
        target_latents.shape[3],
        target_latents.shape[4],
        device=device,
        dtype=dtype,
    )

    x0_latents = None
    if is_keep_x0:
        if first_frame_latents is None:
            x0_latents = torch.zeros(
                target_latents.shape[0],
                target_latents.shape[1],
                1,
                target_latents.shape[3],
                target_latents.shape[4],
                device=device,
                dtype=dtype,
            )
        else:
            if first_frame_latents.ndim == 4:
                first_frame_latents = first_frame_latents.unsqueeze(2)
            if first_frame_latents.ndim != 5:
                raise ValueError(
                    "first_frame_latents is expected to be [B, C, 1, H, W] or [B, C, H, W], "
                    f"got shape={tuple(first_frame_latents.shape)}"
                )
            x0_latents = first_frame_latents[:, :, :1].to(device=device, dtype=dtype)

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
        x0_latents=x0_latents,
        latent_window_size=num_latent_frames_per_chunk,
        memory_size=memory_size,
        history_sizes=list(history_sizes),
        is_random_drop=False,
        random_drop_i2v_ratio=0.0,
        random_drop_v2v_ratio=0.0,
        random_drop_t2v_ratio=0.0,
        is_keep_x0=is_keep_x0,
        dtype=dtype,
        device=device,
    )

    if torch.count_nonzero(history_latents).item() != 0:
        raise ValueError("first-chunk history_latents should be all zeros before clean-input preparation")
    if first_frame_latents is None and is_keep_x0:
        prefix = latents_history_short[:, :, :1]
        if torch.count_nonzero(prefix).item() != 0:
            raise ValueError("first-chunk x0 prefix should be zero when first_frame_latents is absent")

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
