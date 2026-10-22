"""Explicit LagerNVS source/target sampling contract for Helios memory.

This module deliberately stops at clean standardized Wan latents and raw
video-global metric camera metadata.  LagerNVS camera normalization/ray
construction belongs to the GPU runtime immediately before its forward pass;
history corruption and saturation belong after the predicted memory4.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .lagernvs_native.latent_9x9_sampler import (
    Latent9x9Sample,
    sample_latent_9x9,
    validate_feature_for_9x9,
)
from .lagernvs_native.osp_latent_9x9_sampler import (
    DEFAULT_HISTORY_POSE_RETRIEVAL_PROBABILITY,
    DEFAULT_PREVIOUS_CHUNK_PROBABILITY,
    DEFAULT_TEMPORAL_MULTISCALE_PROBABILITY,
    OSPHistoryFutureSample,
    sample_osp_history_future_9x9,
    validate_osp_feature,
)


LAGERNVS_NUM_SOURCES = 9
LAGERNVS_TARGET_SLOTS = (2, 4, 6, 8)
LAGERNVS_RECENT_LOCAL_SLOT = 8


class LagerNVSSamplingUnavailable(RuntimeError):
    """A valid record cannot satisfy the requested native sampling draw."""


@dataclass(frozen=True)
class LagerNVSSamplingSelection:
    """One auditable nine-source/four-target LagerNVS selection."""

    domain: str
    mode: str
    target_chunk: int
    reference_index: tuple[int, int]
    source_indices: tuple[tuple[int, int], ...]
    target_indices: tuple[tuple[int, int], ...]
    native_source_indices: tuple[tuple[int, int], ...]
    native_target_indices: tuple[tuple[int, int], ...]
    source_processed_frames: tuple[int, ...]
    target_processed_frames: tuple[int, ...]
    source_raw_frames: tuple[int, ...]
    target_raw_frames: tuple[int, ...]
    target_known_mask: tuple[bool, ...]

    def to_jsonable(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in (
            "source_indices",
            "target_indices",
            "native_source_indices",
            "native_target_indices",
        ):
            payload[key] = [list(value) for value in payload[key]]
        payload["reference_index"] = list(payload["reference_index"])
        return payload


def _domain_from_source(dataset_source: str) -> str:
    value = str(dataset_source).lower()
    if "dl3dv" in value:
        return "dl3dv"
    if "osp" in value or "istock" in value:
        return "osp"
    raise ValueError(
        "LagerNVS sampling supports only explicit DL3DV or OSP sources, "
        f"got dataset_source={dataset_source!r}"
    )


def _required_valid_target_chunks(
    valid_target_chunks: Sequence[int] | None,
    *,
    chunks: int,
) -> tuple[int, ...]:
    if valid_target_chunks is None:
        raise KeyError(
            "LagerNVS sampling requires explicit valid_target_chunks; "
            "falling back to every chunk is forbidden"
        )
    values = tuple(sorted({int(value) for value in valid_target_chunks}))
    if not values:
        raise ValueError("valid_target_chunks must not be empty")
    invalid = [value for value in values if value < 0 or value >= chunks]
    if invalid:
        raise IndexError(
            f"valid_target_chunks contains indices outside [0,{chunks}): {invalid[:16]}"
        )
    return values


def _eligible_with_exact_recent(
    feature: Mapping[str, Any],
    valid_target_chunks: Sequence[int],
) -> tuple[int, ...]:
    valid = torch.as_tensor(feature["camera_valid_mask"], dtype=torch.bool).cpu()
    processed = torch.as_tensor(
        feature["camera_anchor_processed_frame_idx"], dtype=torch.int64
    ).cpu()
    eligible = []
    for target_chunk in valid_target_chunks:
        if target_chunk <= 0:
            continue
        recent_chunk = target_chunk - 1
        if not bool(valid[target_chunk].all()):
            continue
        if not bool(valid[recent_chunk, LAGERNVS_RECENT_LOCAL_SLOT]):
            continue
        recent_frame = int(processed[recent_chunk, LAGERNVS_RECENT_LOCAL_SLOT])
        target_start = int(processed[target_chunk].min())
        if recent_frame >= target_start:
            continue
        eligible.append(int(target_chunk))
    if not eligible:
        raise RuntimeError(
            "no valid target chunk has the required recent[-1] reference "
            "at (target_chunk-1, local_slot=8)"
        )
    return tuple(eligible)


def _grid_values(
    feature: Mapping[str, Any],
    name: str,
    indices: Sequence[tuple[int, int]],
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    if name not in feature:
        raise KeyError(f"feature lacks required LagerNVS field {name!r}")
    grid = torch.as_tensor(feature[name], dtype=dtype)
    return torch.stack([grid[int(chunk), int(local)] for chunk, local in indices], dim=0)


def _frame_values(
    feature: Mapping[str, Any],
    name: str,
    indices: Sequence[tuple[int, int]],
) -> tuple[int, ...]:
    grid = torch.as_tensor(feature[name], dtype=torch.int64).cpu()
    return tuple(int(grid[chunk, local]) for chunk, local in indices)


def _finish_selection(
    *,
    feature: Mapping[str, Any],
    domain: str,
    mode: str,
    target_chunk: int,
    source_indices: Sequence[tuple[int, int]],
    native_source_indices: Sequence[tuple[int, int]],
    native_target_indices: Sequence[tuple[int, int]],
) -> LagerNVSSamplingSelection:
    reference = (int(target_chunk) - 1, LAGERNVS_RECENT_LOCAL_SLOT)
    sources = tuple((int(chunk), int(local)) for chunk, local in source_indices)
    if len(sources) != LAGERNVS_NUM_SOURCES:
        raise RuntimeError(f"LagerNVS source count must be 9, got {len(sources)}")
    if sources[0] != reference:
        raise RuntimeError(
            f"source[0] must equal recent[-1]={reference}, got {sources[0]}"
        )
    if len(set(sources)) != LAGERNVS_NUM_SOURCES:
        raise RuntimeError(f"LagerNVS source indices are not unique: {sources}")

    targets = tuple((int(target_chunk), slot) for slot in LAGERNVS_TARGET_SLOTS)
    source_set = set(sources)
    known_mask = tuple(target in source_set for target in targets)
    return LagerNVSSamplingSelection(
        domain=domain,
        mode=str(mode),
        target_chunk=int(target_chunk),
        reference_index=reference,
        source_indices=sources,
        target_indices=targets,
        native_source_indices=tuple(native_source_indices),
        native_target_indices=tuple(native_target_indices),
        source_processed_frames=_frame_values(
            feature, "camera_anchor_processed_frame_idx", sources
        ),
        target_processed_frames=_frame_values(
            feature, "camera_anchor_processed_frame_idx", targets
        ),
        source_raw_frames=_frame_values(feature, "camera_anchor_frame_idx", sources),
        target_raw_frames=_frame_values(feature, "camera_anchor_frame_idx", targets),
        target_known_mask=known_mask,
    )


def _sample_dl3dv(
    feature: Mapping[str, Any],
    valid_target_chunks: Sequence[int],
    rng: np.random.Generator,
    *,
    delta_range: tuple[int, int],
    temporal_jitter_proportion: float,
    expansion_factor: float,
    overlap_probability: float,
    native_max_attempts: int,
    contract_max_attempts: int,
) -> LagerNVSSamplingSelection:
    chunks = validate_feature_for_9x9(feature)
    required = set(
        _eligible_with_exact_recent(
            feature,
            _required_valid_target_chunks(valid_target_chunks, chunks=chunks),
        )
    )
    for _ in range(int(contract_max_attempts)):
        try:
            native: Latent9x9Sample = sample_latent_9x9(
                feature,
                rng,
                delta_range=delta_range,
                temporal_jitter_proportion=temporal_jitter_proportion,
                expansion_factor=expansion_factor,
                overlap_probability=overlap_probability,
                target_sampling="full_chunk",
                max_attempts=native_max_attempts,
            )
        except RuntimeError as error:
            # Only the native proposal-exhaustion outcome is record-retryable.
            # Programming errors and malformed tensors must still fail loudly.
            if not str(error).startswith("failed to sample mode="):
                raise
            raise LagerNVSSamplingUnavailable(
                f"DL3DV native 9x9 proposal exhausted: {error}"
            ) from error
        native_targets = tuple(native.target_indices)
        target_chunks = {int(chunk) for chunk, _ in native_targets}
        if len(target_chunks) != 1:
            raise RuntimeError(
                f"full_chunk native target spans multiple chunks: {native_targets}"
            )
        target_chunk = next(iter(target_chunks))
        if target_chunk not in required:
            continue
        reference = (target_chunk - 1, LAGERNVS_RECENT_LOCAL_SLOT)
        native_sources = tuple(native.source_indices)
        if len(native_sources) != 9 or len(set(native_sources)) != 9:
            raise RuntimeError("native DL3DV sampler did not return nine unique sources")
        non_reference = [index for index in native_sources if index != reference]
        if len(non_reference) not in {8, 9}:
            raise RuntimeError(
                "removing recent[-1] from native DL3DV sources produced "
                f"{len(non_reference)} candidates"
            )
        # One permutation both selects eight (when recent was absent) and erases
        # all source[1:9] position semantics.  source[0] is never permuted.
        order = rng.permutation(len(non_reference)).tolist()
        selected_eight = tuple(non_reference[int(index)] for index in order[:8])
        return _finish_selection(
            feature=feature,
            domain="dl3dv",
            mode=native.mode,
            target_chunk=target_chunk,
            source_indices=(reference, *selected_eight),
            native_source_indices=native_sources,
            native_target_indices=native_targets,
        )
    raise LagerNVSSamplingUnavailable(
        "native DL3DV sampler did not produce an explicitly valid target with "
        f"recent[-1] after {contract_max_attempts} attempts"
    )


def _sample_osp(
    feature: Mapping[str, Any],
    valid_target_chunks: Sequence[int],
    rng: np.random.Generator,
    *,
    history_pose_retrieval_probability: float,
    previous_chunk_probability: float,
    temporal_multiscale_probability: float,
    max_source_parent_chunks: int,
    min_source_processed_gap: int,
    low_translation_threshold_m: float,
    rotation_weight: float,
    minimum_translation_scale_m: float,
) -> LagerNVSSamplingSelection:
    chunks = validate_osp_feature(feature)
    explicit = _required_valid_target_chunks(valid_target_chunks, chunks=chunks)
    eligible = _eligible_with_exact_recent(feature, explicit)
    # A valid motion chunk can still lack nine strictly historical camera-valid
    # anchors.  Try every eligible target chunk in a deterministic RNG order
    # before rejecting the record; this changes only the target chunk, never
    # the dataset domain or the LagerNVS/Helios memory contract.
    failures: list[tuple[int, str]] = []
    native: OSPHistoryFutureSample | None = None
    for position in rng.permutation(len(eligible)).tolist():
        target_chunk = int(eligible[int(position)])
        try:
            native = sample_osp_history_future_9x9(
                feature,
                (target_chunk,),
                rng,
                history_pose_retrieval_probability=history_pose_retrieval_probability,
                previous_chunk_probability=previous_chunk_probability,
                temporal_multiscale_probability=temporal_multiscale_probability,
                max_source_parent_chunks=max_source_parent_chunks,
                min_source_processed_gap=min_source_processed_gap,
                low_translation_threshold_m=low_translation_threshold_m,
                rotation_weight=rotation_weight,
                minimum_translation_scale_m=minimum_translation_scale_m,
                target_sampling="full_chunk",
                shuffle_non_reference_sources=False,
            )
        except RuntimeError as error:
            failures.append((target_chunk, str(error)))
            continue
        break
    if native is None:
        raise RuntimeError(
            "all eligible OSP target chunks failed strict 9x9 sampling: "
            + "; ".join(
                f"chunk={target_chunk}: {reason}"
                for target_chunk, reason in failures
            )
        )
    reference = (int(native.target_chunk) - 1, LAGERNVS_RECENT_LOCAL_SLOT)
    native_sources = tuple(native.source_indices)
    if native_sources[0] != reference:
        raise RuntimeError(
            "native OSP latest-history source is not Helios recent[-1]: "
            f"native={native_sources[0]} required={reference}"
        )
    remaining = native_sources[1:]
    order = rng.permutation(8).tolist()
    shuffled = tuple(remaining[int(index)] for index in order)
    return _finish_selection(
        feature=feature,
        domain="osp",
        mode=native.mode,
        target_chunk=int(native.target_chunk),
        source_indices=(reference, *shuffled),
        native_source_indices=native_sources,
        native_target_indices=tuple(native.target_indices),
    )


def sample_lagernvs_views(
    feature: Mapping[str, Any],
    *,
    dataset_source: str,
    valid_target_chunks: Sequence[int] | None,
    rng: np.random.Generator,
    dl3dv_delta_range: tuple[int, int] = (16, 48),
    dl3dv_temporal_jitter_proportion: float = 0.1,
    dl3dv_expansion_factor: float = 1.0,
    dl3dv_overlap_probability: float = 0.2,
    dl3dv_native_max_attempts: int = 512,
    dl3dv_contract_max_attempts: int = 64,
    osp_history_pose_retrieval_probability: float = DEFAULT_HISTORY_POSE_RETRIEVAL_PROBABILITY,
    osp_previous_chunk_probability: float = DEFAULT_PREVIOUS_CHUNK_PROBABILITY,
    osp_temporal_multiscale_probability: float = DEFAULT_TEMPORAL_MULTISCALE_PROBABILITY,
    osp_max_source_parent_chunks: int = 3,
    osp_min_source_processed_gap: int = 4,
    osp_low_translation_threshold_m: float = 0.3,
    osp_rotation_weight: float = 0.25,
    osp_minimum_translation_scale_m: float = 0.05,
) -> LagerNVSSamplingSelection:
    """Select exact LagerNVS inputs without any legacy-memory fallback."""
    if not isinstance(rng, np.random.Generator):
        raise TypeError(f"rng must be numpy.random.Generator, got {type(rng).__name__}")
    domain = _domain_from_source(dataset_source)
    if domain == "dl3dv":
        return _sample_dl3dv(
            feature,
            valid_target_chunks,
            rng,
            delta_range=tuple(int(value) for value in dl3dv_delta_range),
            temporal_jitter_proportion=float(dl3dv_temporal_jitter_proportion),
            expansion_factor=float(dl3dv_expansion_factor),
            overlap_probability=float(dl3dv_overlap_probability),
            native_max_attempts=int(dl3dv_native_max_attempts),
            contract_max_attempts=int(dl3dv_contract_max_attempts),
        )
    return _sample_osp(
        feature,
        valid_target_chunks,
        rng,
        history_pose_retrieval_probability=float(
            osp_history_pose_retrieval_probability
        ),
        previous_chunk_probability=float(osp_previous_chunk_probability),
        temporal_multiscale_probability=float(osp_temporal_multiscale_probability),
        max_source_parent_chunks=int(osp_max_source_parent_chunks),
        min_source_processed_gap=int(osp_min_source_processed_gap),
        low_translation_threshold_m=float(osp_low_translation_threshold_m),
        rotation_weight=float(osp_rotation_weight),
        minimum_translation_scale_m=float(osp_minimum_translation_scale_m),
    )


def gather_lagernvs_tensors(
    feature: Mapping[str, Any],
    selection: LagerNVSSamplingSelection,
) -> dict[str, torch.Tensor | str]:
    """Gather clean standardized latents and untouched global metric cameras."""
    latent_grid = torch.as_tensor(feature["vae_latent"])
    source_latents_vc = torch.stack(
        [latent_grid[chunk, :, local] for chunk, local in selection.source_indices],
        dim=0,
    ).contiguous()
    target_latents_vc = torch.stack(
        [latent_grid[chunk, :, local] for chunk, local in selection.target_indices],
        dim=0,
    ).contiguous()
    source_latents = source_latents_vc.permute(1, 0, 2, 3).contiguous()
    target_latents = target_latents_vc.permute(1, 0, 2, 3).contiguous()
    expected_source = (16, 9, 48, 80)
    expected_target = (16, 4, 48, 80)
    if tuple(source_latents.shape) != expected_source:
        raise ValueError(
            f"LagerNVS source_latents must be {expected_source}, got {tuple(source_latents.shape)}"
        )
    if tuple(target_latents.shape) != expected_target:
        raise ValueError(
            f"LagerNVS target_latents must be {expected_target}, got {tuple(target_latents.shape)}"
        )
    source_c2w = _grid_values(
        feature,
        "camera_pose_rel_video_c2w",
        selection.source_indices,
        dtype=torch.float32,
    ).contiguous()
    target_c2w = _grid_values(
        feature,
        "camera_pose_rel_video_c2w",
        selection.target_indices,
        dtype=torch.float32,
    ).contiguous()
    source_k = _grid_values(
        feature,
        "camera_K_norm",
        selection.source_indices,
        dtype=torch.float32,
    ).contiguous()
    target_k = _grid_values(
        feature,
        "camera_K_norm",
        selection.target_indices,
        dtype=torch.float32,
    ).contiguous()
    tensors = (source_latents, target_latents, source_c2w, target_c2w, source_k, target_k)
    if not all(torch.isfinite(value).all() for value in tensors):
        raise FloatingPointError("LagerNVS data contract contains NaN or Inf")
    return {
        "lagernvs_source_latents": source_latents,
        "lagernvs_target_latents_gt": target_latents,
        "lagernvs_source_c2w_metric": source_c2w,
        "lagernvs_target_c2w_metric": target_c2w,
        "lagernvs_source_K_norm_raw": source_k,
        "lagernvs_target_K_norm_raw": target_k,
        "lagernvs_source_indices": torch.as_tensor(selection.source_indices, dtype=torch.int64),
        "lagernvs_target_indices": torch.as_tensor(selection.target_indices, dtype=torch.int64),
        "lagernvs_native_source_indices": torch.as_tensor(
            selection.native_source_indices, dtype=torch.int64
        ),
        "lagernvs_native_target_indices": torch.as_tensor(
            selection.native_target_indices, dtype=torch.int64
        ),
        "lagernvs_source_processed_frames": torch.as_tensor(
            selection.source_processed_frames, dtype=torch.int64
        ),
        "lagernvs_target_processed_frames": torch.as_tensor(
            selection.target_processed_frames, dtype=torch.int64
        ),
        "lagernvs_source_raw_frames": torch.as_tensor(
            selection.source_raw_frames, dtype=torch.int64
        ),
        "lagernvs_target_raw_frames": torch.as_tensor(
            selection.target_raw_frames, dtype=torch.int64
        ),
        "lagernvs_target_known_mask": torch.as_tensor(
            selection.target_known_mask, dtype=torch.bool
        ),
        "lagernvs_target_slots": torch.as_tensor(LAGERNVS_TARGET_SLOTS, dtype=torch.int64),
        "lagernvs_reference_index": torch.as_tensor(
            selection.reference_index, dtype=torch.int64
        ),
        "lagernvs_near_zero_baseline_m": torch.tensor(
            0.05 if selection.domain == "osp" else 1e-6,
            dtype=torch.float32,
        ),
        "lagernvs_sampling_domain": selection.domain,
        "lagernvs_sampling_mode": selection.mode,
        "lagernvs_camera_contract": "raw_video_global_metric_c2w_runtime_normalized",
        "lagernvs_intrinsics_contract": "raw_metadata_audit_runtime_lager_nominal",
    }


__all__ = [
    "LAGERNVS_NUM_SOURCES",
    "LAGERNVS_RECENT_LOCAL_SLOT",
    "LAGERNVS_TARGET_SLOTS",
    "LagerNVSSamplingUnavailable",
    "LagerNVSSamplingSelection",
    "gather_lagernvs_tensors",
    "sample_lagernvs_views",
]
