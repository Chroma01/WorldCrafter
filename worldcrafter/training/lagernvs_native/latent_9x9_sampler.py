"""Official-style 9-source/9-target sampling for chunked Helios latents.

The offline DL3DV features store complete 33-RGB-frame Wan chunks as nine
latent slots and retain the true processed/raw RGB anchor index for every
slot.  Sampling must therefore happen in anchor-frame coordinates rather than
on a naively flattened latent index: adjacent chunks end/start at processed
frames 32 and 33, not 32 and 36.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping

import numpy as np
import torch


NUM_VIEWS = 9
SamplingMode = Literal[
    "interpolation",
    "extrapolation_left",
    "extrapolation_right",
    "known_mixed",
]
TargetSampling = Literal["full_chunk", "independent"]


@dataclass(frozen=True)
class LatentAnchor:
    chunk: int
    local: int
    processed_frame: int
    raw_frame: int

    @property
    def pair(self) -> tuple[int, int]:
        return self.chunk, self.local


@dataclass(frozen=True)
class Latent9x9Sample:
    source_indices: tuple[tuple[int, int], ...]
    target_indices: tuple[tuple[int, int], ...]
    source_processed_frames: tuple[int, ...]
    target_processed_frames: tuple[int, ...]
    source_raw_frames: tuple[int, ...]
    target_raw_frames: tuple[int, ...]
    target_known_mask: tuple[bool, ...]
    mode: SamplingMode
    base_region: str
    delta_processed_frames: int
    source_span_processed_frames: tuple[int, int]
    target_center_processed_frame: float
    num_known_targets: int

    def to_jsonable(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["source_indices"] = [list(value) for value in self.source_indices]
        payload["target_indices"] = [list(value) for value in self.target_indices]
        return payload


def _required_tensor(
    feature: Mapping[str, Any], name: str, shape_suffix: tuple[int, ...]
) -> torch.Tensor:
    if name not in feature:
        raise KeyError(f"feature lacks {name!r}")
    value = torch.as_tensor(feature[name])
    if value.ndim != len(shape_suffix) + 1 or tuple(value.shape[1:]) != shape_suffix:
        raise ValueError(
            f"{name} must have shape [N,{','.join(map(str, shape_suffix))}], "
            f"got {tuple(value.shape)}"
        )
    return value


def validate_feature_for_9x9(feature: Mapping[str, Any]) -> int:
    """Validate tensors used by the sampler and return the chunk count."""
    latents = _required_tensor(feature, "vae_latent", (16, 9, 48, 80))
    chunks = int(latents.shape[0])
    for name, suffix in (
        ("camera_pose_rel_video_c2w", (9, 4, 4)),
        ("camera_K_norm", (9, 4)),
        ("camera_valid_mask", (9,)),
        ("camera_anchor_processed_frame_idx", (9,)),
        ("camera_anchor_frame_idx", (9,)),
    ):
        value = _required_tensor(feature, name, suffix)
        if int(value.shape[0]) != chunks:
            raise ValueError(
                f"{name} has {value.shape[0]} chunks but vae_latent has {chunks}"
            )
    if chunks < 2:
        raise ValueError("9x9 sampling requires at least two complete chunks")
    return chunks


def _anchor_table(feature: Mapping[str, Any]) -> tuple[list[LatentAnchor], list[int]]:
    chunks = validate_feature_for_9x9(feature)
    valid = torch.as_tensor(feature["camera_valid_mask"], dtype=torch.bool).cpu()
    processed = torch.as_tensor(
        feature["camera_anchor_processed_frame_idx"], dtype=torch.int64
    ).cpu()
    raw = torch.as_tensor(feature["camera_anchor_frame_idx"], dtype=torch.int64).cpu()
    anchors = [
        LatentAnchor(
            chunk=chunk,
            local=local,
            processed_frame=int(processed[chunk, local]),
            raw_frame=int(raw[chunk, local]),
        )
        for chunk in range(chunks)
        for local in range(NUM_VIEWS)
        if bool(valid[chunk, local])
    ]
    anchors.sort(key=lambda value: (value.processed_frame, value.chunk, value.local))
    full_valid_chunks = [
        chunk for chunk in range(chunks) if bool(valid[chunk].all())
    ]
    if len(anchors) < 2 * NUM_VIEWS or not full_valid_chunks:
        raise ValueError("feature has too few valid anchors/full target chunks")
    return anchors, full_valid_chunks


def _nearest_distinct_anchors(
    anchors: list[LatentAnchor], desired: np.ndarray
) -> list[LatentAnchor]:
    coordinates = np.asarray([value.processed_frame for value in anchors], dtype=np.int64)
    chosen: list[LatentAnchor] = []
    used: set[tuple[int, int]] = set()
    for target in desired:
        order = np.argsort(np.abs(coordinates - int(target)), kind="stable")
        selected = next((anchors[int(index)] for index in order if anchors[int(index)].pair not in used), None)
        if selected is None:
            raise RuntimeError("could not choose distinct anchors")
        chosen.append(selected)
        used.add(selected.pair)
    return chosen


def _region_for_center(center: float, source_min: int, source_max: int) -> str:
    if center < source_min:
        return "extrapolation_left"
    if center > source_max:
        return "extrapolation_right"
    return "interpolation"


def _sample_known_count(rng: np.random.Generator) -> int:
    # This is the fixed 9x9 analogue of the released overlap draw: choose how
    # many of nine target slots are also exposed as sources.  K has mean 4.5
    # and is almost always 2..7.
    for _ in range(32):
        value = int(rng.hypergeometric(ngood=9, nbad=9, nsample=9))
        if value > 0:
            return value
    raise RuntimeError("failed to sample a positive known-target count")


def sample_latent_9x9(
    feature: Mapping[str, Any],
    rng: np.random.Generator,
    *,
    mode: SamplingMode | None = None,
    delta_range: tuple[int, int] = (16, 48),
    temporal_jitter_proportion: float = 0.1,
    expansion_factor: float = 1.0,
    overlap_probability: float = 0.2,
    target_sampling: TargetSampling = "full_chunk",
    max_attempts: int = 512,
) -> Latent9x9Sample:
    """Draw one 9-source/9-target training example.

    When ``mode`` is omitted, this uses the General checkpoint proportions:
    80% disjoint targets and 20% known/unknown mixtures.  Disjoint targets use
    the released expanded-linear geometry (about 80% interpolation and 20%
    left/right extrapolation for expansion factor one).
    """
    if target_sampling not in {"full_chunk", "independent"}:
        raise ValueError(f"unknown target_sampling={target_sampling!r}")
    if not 0.0 <= overlap_probability <= 1.0:
        raise ValueError("overlap_probability must be in [0,1]")
    delta_min, delta_max = (int(delta_range[0]), int(delta_range[1]))
    if delta_min <= 0 or delta_max < delta_min:
        raise ValueError(f"invalid delta_range {delta_range}")
    if temporal_jitter_proportion < 0 or expansion_factor < 0:
        raise ValueError("jitter and expansion must be non-negative")
    anchors, full_valid_chunks = _anchor_table(feature)
    coordinate_min = anchors[0].processed_frame
    coordinate_max = anchors[-1].processed_frame
    processed = torch.as_tensor(
        feature["camera_anchor_processed_frame_idx"], dtype=torch.int64
    ).cpu()
    raw = torch.as_tensor(feature["camera_anchor_frame_idx"], dtype=torch.int64).cpu()

    requested_mode: SamplingMode
    if mode is None:
        if float(rng.random()) < overlap_probability:
            requested_mode = "known_mixed"
        else:
            draw = float(rng.random())
            requested_mode = (
                "interpolation"
                if draw < 0.8
                else "extrapolation_left"
                if draw < 0.9
                else "extrapolation_right"
            )
    else:
        requested_mode = mode

    for _ in range(int(max_attempts)):
        delta = int(rng.integers(delta_min, delta_max + 1))
        span = delta * (NUM_VIEWS - 1)
        if coordinate_max - coordinate_min < span:
            continue
        start = int(rng.integers(coordinate_min, coordinate_max - span + 1))
        jitter = int(delta * temporal_jitter_proportion)
        offsets = (
            rng.integers(-jitter, jitter + 1, size=NUM_VIEWS)
            if jitter > 0
            else np.zeros(NUM_VIEWS, dtype=np.int64)
        )
        desired = start + np.arange(NUM_VIEWS, dtype=np.int64) * delta + offsets
        base_sources = _nearest_distinct_anchors(anchors, desired)
        source_pairs = {value.pair for value in base_sources}
        source_min = min(value.processed_frame for value in base_sources)
        source_max = max(value.processed_frame for value in base_sources)
        expanded_min = source_min - expansion_factor * delta
        expanded_max = source_max + expansion_factor * delta

        if target_sampling == "independent":
            # Match LagerNVS' released ExpandedLinearViewSelector semantics:
            # sources remain fixed, novel targets are sampled independently in
            # the expanded source range, and overlap replaces target entries
            # with exact source views.  Both source and target list orders are
            # independently randomized; no target-list position carries VAE
            # local-slot meaning.
            novel_candidates = [
                value
                for value in anchors
                if value.pair not in source_pairs
                and expanded_min <= value.processed_frame <= expanded_max
            ]
            if len(novel_candidates) < NUM_VIEWS:
                continue
            novel = [
                novel_candidates[int(index)]
                for index in rng.choice(
                    len(novel_candidates), size=NUM_VIEWS, replace=False
                )
            ]
            final_targets = list(novel)
            if requested_mode == "known_mixed":
                union = [*base_sources, *novel]
                final_targets = [
                    union[int(index)]
                    for index in rng.choice(len(union), size=NUM_VIEWS, replace=False)
                ]
                if not any(value.pair in source_pairs for value in final_targets):
                    continue

            final_sources = [
                base_sources[int(index)]
                for index in rng.permutation(NUM_VIEWS).tolist()
            ]
            final_targets = [
                final_targets[int(index)]
                for index in rng.permutation(NUM_VIEWS).tolist()
            ]
            known_mask = tuple(
                value.pair in source_pairs for value in final_targets
            )
            target_center = float(
                np.mean([value.processed_frame for value in final_targets])
            )
            base_region = _region_for_center(
                target_center, source_min, source_max
            )
            actual_mode: SamplingMode = (
                "known_mixed" if any(known_mask) else base_region
            )
            return Latent9x9Sample(
                source_indices=tuple(value.pair for value in final_sources),
                target_indices=tuple(value.pair for value in final_targets),
                source_processed_frames=tuple(
                    value.processed_frame for value in final_sources
                ),
                target_processed_frames=tuple(
                    value.processed_frame for value in final_targets
                ),
                source_raw_frames=tuple(value.raw_frame for value in final_sources),
                target_raw_frames=tuple(value.raw_frame for value in final_targets),
                target_known_mask=known_mask,
                mode=actual_mode,
                base_region=base_region,
                delta_processed_frames=delta,
                source_span_processed_frames=(source_min, source_max),
                target_center_processed_frame=target_center,
                num_known_targets=sum(known_mask),
            )

        candidates: list[tuple[int, float, str]] = []
        for chunk in full_valid_chunks:
            if any((chunk, local) in source_pairs for local in range(NUM_VIEWS)):
                continue
            target_coordinates = processed[chunk].tolist()
            center = float(np.mean(target_coordinates))
            if not expanded_min <= center <= expanded_max:
                continue
            region = _region_for_center(center, source_min, source_max)
            if requested_mode != "known_mixed" and region != requested_mode:
                continue
            candidates.append((chunk, center, region))
        if not candidates:
            continue

        target_chunk, target_center, base_region = candidates[
            int(rng.integers(0, len(candidates)))
        ]
        final_sources = list(base_sources)
        known_locals: set[int] = set()
        if requested_mode == "known_mixed":
            known_count = _sample_known_count(rng)
            known_locals = set(
                int(value)
                for value in rng.choice(NUM_VIEWS, size=known_count, replace=False)
            )
            replace_positions = rng.choice(
                NUM_VIEWS, size=known_count, replace=False
            ).tolist()
            for position, local in zip(
                replace_positions, sorted(known_locals), strict=True
            ):
                final_sources[int(position)] = LatentAnchor(
                    chunk=target_chunk,
                    local=local,
                    processed_frame=int(processed[target_chunk, local]),
                    raw_frame=int(raw[target_chunk, local]),
                )

        if len({value.pair for value in final_sources}) != NUM_VIEWS:
            continue
        order = rng.permutation(NUM_VIEWS).tolist()
        final_sources = [final_sources[int(index)] for index in order]
        target_indices = tuple((target_chunk, local) for local in range(NUM_VIEWS))
        target_processed = tuple(int(value) for value in processed[target_chunk])
        target_raw = tuple(int(value) for value in raw[target_chunk])
        known_mask = tuple(local in known_locals for local in range(NUM_VIEWS))
        return Latent9x9Sample(
            source_indices=tuple(value.pair for value in final_sources),
            target_indices=target_indices,
            source_processed_frames=tuple(
                value.processed_frame for value in final_sources
            ),
            target_processed_frames=target_processed,
            source_raw_frames=tuple(value.raw_frame for value in final_sources),
            target_raw_frames=target_raw,
            target_known_mask=known_mask,
            mode=requested_mode,
            base_region=base_region,
            delta_processed_frames=delta,
            source_span_processed_frames=(source_min, source_max),
            target_center_processed_frame=target_center,
            num_known_targets=len(known_locals),
        )
    raise RuntimeError(
        f"failed to sample mode={requested_mode!r} after {max_attempts} attempts"
    )


__all__ = [
    "Latent9x9Sample",
    "LatentAnchor",
    "NUM_VIEWS",
    "SamplingMode",
    "TargetSampling",
    "sample_latent_9x9",
    "validate_feature_for_9x9",
]
