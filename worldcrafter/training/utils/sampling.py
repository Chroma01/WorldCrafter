"""Causal source and target sampling for chunked video features."""
from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from typing import Any
from typing import Literal
from typing import Mapping
import numpy as np
import torch
import math
from typing import Sequence


NUM_VIEWS = 9
DL3DVSamplingMode = Literal[
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
    mode: DL3DVSamplingMode
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
    mode: DL3DVSamplingMode | None = None,
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

    requested_mode: DL3DVSamplingMode
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
            actual_mode: DL3DVSamplingMode = (
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


OSPSamplingMode = Literal[
    "history_pose_retrieval",
    "previous_chunk",
    "temporal_multiscale",
]

DEFAULT_HISTORY_POSE_RETRIEVAL_PROBABILITY = 0.55
DEFAULT_PREVIOUS_CHUNK_PROBABILITY = 0.25
DEFAULT_TEMPORAL_MULTISCALE_PROBABILITY = 0.20
_TEMPORAL_MULTISCALE_QUOTAS = (3, 3, 2)


@dataclass(frozen=True)
class OSPHistoryFutureSample:
    source_indices: tuple[tuple[int, int], ...]
    target_indices: tuple[tuple[int, int], ...]
    source_processed_frames: tuple[int, ...]
    target_processed_frames: tuple[int, ...]
    source_raw_frames: tuple[int, ...]
    target_raw_frames: tuple[int, ...]
    target_chunk: int
    mode: OSPSamplingMode
    target_chord_m: float
    target_rotation_deg: float
    low_translation_rotation: bool
    source_unique_parent_count: int

    def to_jsonable(self) -> dict[str, Any]:
        value = asdict(self)
        value["source_indices"] = [list(index) for index in self.source_indices]
        value["target_indices"] = [list(index) for index in self.target_indices]
        return value


def _required_grid(
    feature: Mapping[str, Any], name: str, shape_suffix: tuple[int, ...]
) -> torch.Tensor:
    if name not in feature:
        raise KeyError(f"OSP feature lacks {name!r}")
    value = torch.as_tensor(feature[name])
    if value.ndim != len(shape_suffix) + 1 or tuple(value.shape[1:]) != shape_suffix:
        raise ValueError(
            f"{name} must have shape [N,{','.join(map(str, shape_suffix))}], "
            f"got {tuple(value.shape)}"
        )
    return value


def validate_osp_feature(feature: Mapping[str, Any]) -> int:
    """Validate fields shared by sampling and the OSP latent dataset."""
    latents = _required_grid(feature, "vae_latent", (16, 9, 48, 80))
    chunks = int(latents.shape[0])
    for name, suffix in (
        ("camera_pose_rel_video_c2w", (9, 4, 4)),
        ("camera_K_norm", (9, 4)),
        ("camera_valid_mask", (9,)),
        ("camera_anchor_processed_frame_idx", (9,)),
        ("camera_anchor_frame_idx", (9,)),
    ):
        value = _required_grid(feature, name, suffix)
        if int(value.shape[0]) != chunks:
            raise ValueError(
                f"{name} has {value.shape[0]} chunks but vae_latent has {chunks}"
            )
    if chunks < 2:
        raise ValueError("causal OSP 9x9 sampling requires at least two chunks")
    return chunks


def _rotation_angles(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    relative = np.einsum("aij,bjk->abik", np.swapaxes(left, -1, -2), right)
    cosine = (np.trace(relative, axis1=-2, axis2=-1) - 1.0) * 0.5
    return np.arccos(np.clip(cosine, -1.0, 1.0))


def _target_motion(target_poses: np.ndarray) -> tuple[float, float]:
    chord = float(
        np.linalg.norm(target_poses[-1, :3, 3] - target_poses[0, :3, 3])
    )
    angles = _rotation_angles(
        target_poses[:1, :3, :3], target_poses[:, :3, :3]
    )
    rotation_deg = float(np.degrees(angles.max()))
    return chord, rotation_deg


def eligible_target_chunks(
    feature: Mapping[str, Any], valid_target_chunks: Sequence[int]
) -> tuple[int, ...]:
    """Return complete motion chunks with at least nine strictly older anchors."""
    chunks = validate_osp_feature(feature)
    valid = torch.as_tensor(feature["camera_valid_mask"], dtype=torch.bool).cpu()
    processed = torch.as_tensor(
        feature["camera_anchor_processed_frame_idx"], dtype=torch.int64
    ).cpu()
    eligible: list[int] = []
    for target_chunk in sorted({int(value) for value in valid_target_chunks}):
        if target_chunk <= 0 or target_chunk >= chunks:
            continue
        if not bool(valid[target_chunk].all()):
            continue
        target_start = int(processed[target_chunk].min())
        history_count = int(
            (valid & (processed < target_start)).to(dtype=torch.int64).sum()
        )
        if history_count >= NUM_VIEWS:
            eligible.append(target_chunk)
    return tuple(eligible)


def _pose_distance(
    candidate_poses: np.ndarray,
    target_poses: np.ndarray,
    *,
    low_translation_rotation: bool,
    rotation_weight: float,
    minimum_translation_scale_m: float,
) -> np.ndarray:
    translations = np.linalg.norm(
        candidate_poses[:, None, :3, 3] - target_poses[None, :, :3, 3], axis=-1
    )
    positive = translations[translations > 1e-8]
    translation_scale = (
        float(np.quantile(positive, 0.90))
        if positive.size
        else float(minimum_translation_scale_m)
    )
    translation_scale = max(translation_scale, float(minimum_translation_scale_m))
    translation_term = translations / translation_scale
    # In a low-translation target, tiny DA3 translation residuals must not
    # dominate actual changes in viewing direction.
    if low_translation_rotation:
        translation_term *= 0.0
    rotations = _rotation_angles(
        candidate_poses[:, :3, :3], target_poses[:, :3, :3]
    ) / math.pi
    return translation_term + float(rotation_weight) * rotations


def _choose_retrieval_sources(
    history: list[tuple[int, int]],
    latest: tuple[int, int],
    poses: np.ndarray,
    processed: np.ndarray,
    target_poses: np.ndarray,
    *,
    low_translation_rotation: bool,
    rotation_weight: float,
    minimum_translation_scale_m: float,
    max_source_parent_chunks: int,
    min_source_processed_gap: int,
) -> list[tuple[int, int]]:
    parent_to_indices: dict[int, list[tuple[int, int]]] = {}
    for index in history:
        parent_to_indices.setdefault(index[0], []).append(index)

    all_candidates = list(history)
    all_candidate_poses = np.stack([poses[index] for index in all_candidates])
    all_distances = _pose_distance(
        all_candidate_poses,
        target_poses,
        low_translation_rotation=low_translation_rotation,
        rotation_weight=rotation_weight,
        minimum_translation_scale_m=minimum_translation_scale_m,
    )
    distance_by_index = {
        index: all_distances[position]
        for position, index in enumerate(all_candidates)
    }

    latest_parent = latest[0]
    older_parents = [parent for parent in parent_to_indices if parent != latest_parent]
    older_parents.sort(
        key=lambda parent: min(
            float(distance_by_index[index].mean())
            for index in parent_to_indices[parent]
        )
    )
    parent_limit = max(1, int(max_source_parent_chunks))
    selected_parents = [latest_parent, *older_parents[: parent_limit - 1]]
    candidates = [
        index for index in history if index[0] in set(selected_parents)
    ]

    selected = [latest]
    selected_processed = [int(processed[latest])]
    # Seed one pose-relevant observation from every selected older parent so
    # long-term memory is not silently reduced to the immediately previous block.
    for parent in selected_parents[1:]:
        options = [index for index in parent_to_indices[parent] if index != latest]
        if not options:
            continue
        best = min(options, key=lambda index: float(distance_by_index[index].mean()))
        selected.append(best)
        selected_processed.append(int(processed[best]))

    coverage = np.min(
        np.stack([distance_by_index[index] for index in selected]), axis=0
    )
    while len(selected) < NUM_VIEWS:
        constrained = [
            index
            for index in candidates
            if index not in selected
            and all(
                abs(int(processed[index]) - other) >= min_source_processed_gap
                for other in selected_processed
            )
        ]
        options = constrained or [index for index in candidates if index not in selected]
        if not options:
            break
        best = min(
            options,
            key=lambda index: float(
                np.minimum(coverage, distance_by_index[index]).mean()
            ),
        )
        selected.append(best)
        selected_processed.append(int(processed[best]))
        coverage = np.minimum(coverage, distance_by_index[best])

    if len(selected) != NUM_VIEWS:
        # This should only be reachable for malformed sparse validity masks.
        remaining = [index for index in history if index not in selected]
        remaining.sort(key=lambda index: float(distance_by_index[index].mean()))
        selected.extend(remaining[: NUM_VIEWS - len(selected)])
    if len(selected) != NUM_VIEWS:
        raise RuntimeError("could not select nine strictly historical OSP sources")
    return selected


def sample_osp_sampling_mode(
    rng: np.random.Generator,
    *,
    history_pose_retrieval_probability: float = (
        DEFAULT_HISTORY_POSE_RETRIEVAL_PROBABILITY
    ),
    previous_chunk_probability: float = DEFAULT_PREVIOUS_CHUNK_PROBABILITY,
    temporal_multiscale_probability: float = (
        DEFAULT_TEMPORAL_MULTISCALE_PROBABILITY
    ),
) -> OSPSamplingMode:
    """Draw one of the three causal source policies.

    Keeping all three probabilities explicit prevents an accidental change in
    the OSP task mixture when one policy is added or removed from a config.
    """
    probabilities = (
        float(history_pose_retrieval_probability),
        float(previous_chunk_probability),
        float(temporal_multiscale_probability),
    )
    if any(not 0.0 <= value <= 1.0 for value in probabilities):
        raise ValueError("OSP sampling-mode probabilities must be in [0,1]")
    if not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError(
            "OSP sampling-mode probabilities must sum to one, got "
            f"{sum(probabilities):.12g}"
        )
    draw = float(rng.random())
    if draw < probabilities[0]:
        return "history_pose_retrieval"
    if draw < probabilities[0] + probabilities[1]:
        return "previous_chunk"
    return "temporal_multiscale"


def _choose_temporal_multiscale_sources(
    history: list[tuple[int, int]],
    latest: tuple[int, int],
    poses: np.ndarray,
    processed: np.ndarray,
    target_poses: np.ndarray,
    *,
    low_translation_rotation: bool,
    rotation_weight: float,
    minimum_translation_scale_m: float,
    max_source_parent_chunks: int,
    min_source_processed_gap: int,
) -> list[tuple[int, int]]:
    """Select latest history plus 3/3/2 near/middle/far observations.

    The three temporal strata are rank intervals over all strictly historical
    anchors, ordered newest to oldest.  Parent chunks are selected greedily to
    cover the strata before individual anchors are selected.  The parent limit
    is a preference rather than a correctness constraint: if the requested
    strata cannot be filled from those parents, the sampler admits the minimum
    additional candidates needed instead of dropping temporal coverage.

    Within every stratum, pose coverage uses the same metric-aware score as
    history retrieval.  For low-translation targets the translation term is
    disabled, so pure rotations are selected by viewing direction rather than
    by DA3 translation residuals.
    """
    remaining = [index for index in reversed(history) if index != latest]
    if len(remaining) < sum(_TEMPORAL_MULTISCALE_QUOTAS):
        raise RuntimeError("temporal multiscale needs eight historical anchors")
    split_positions = np.array_split(np.arange(len(remaining)), 3)
    strata = [
        [remaining[int(position)] for position in positions]
        for positions in split_positions
    ]
    if any(
        len(stratum) < quota
        for stratum, quota in zip(strata, _TEMPORAL_MULTISCALE_QUOTAS)
    ):
        raise RuntimeError("temporal strata cannot satisfy the 3/3/2 allocation")

    all_candidates = [index for stratum in strata for index in stratum]
    candidate_poses = np.stack([poses[index] for index in all_candidates])
    distances = _pose_distance(
        candidate_poses,
        target_poses,
        low_translation_rotation=low_translation_rotation,
        rotation_weight=rotation_weight,
        minimum_translation_scale_m=minimum_translation_scale_m,
    )
    distance_by_index = {
        index: distances[position]
        for position, index in enumerate(all_candidates)
    }

    # First select a compact set of parent chunks that can cover as many of
    # the requested temporal observations as possible.  The latest parent is
    # always retained because source[0] must be latest history.
    selected_parents = {latest[0]}
    parent_limit = max(1, int(max_source_parent_chunks))
    candidate_parents = sorted(
        {index[0] for index in all_candidates} - selected_parents
    )
    while len(selected_parents) < parent_limit:
        covered = [
            sum(index[0] in selected_parents for index in stratum)
            for stratum in strata
        ]
        unmet = [
            max(0, quota - count)
            for quota, count in zip(_TEMPORAL_MULTISCALE_QUOTAS, covered)
        ]
        if not any(unmet):
            break

        ranked: list[tuple[int, int, float, int]] = []
        for parent in candidate_parents:
            if parent in selected_parents:
                continue
            counts = [
                sum(index[0] == parent for index in stratum)
                for stratum in strata
            ]
            gain = sum(min(need, count) for need, count in zip(unmet, counts))
            strata_gain = sum(
                need > 0 and count > 0 for need, count in zip(unmet, counts)
            )
            parent_indices = [
                index for index in all_candidates if index[0] == parent
            ]
            pose_score = float(
                np.mean(
                    [float(distance_by_index[index].mean()) for index in parent_indices]
                )
            )
            # max gain, then max number of strata, then smaller pose distance;
            # the final parent id is a deterministic tie-breaker.
            ranked.append((gain, strata_gain, -pose_score, -parent))
        if not ranked:
            break
        best = max(ranked)
        if best[0] <= 0:
            break
        selected_parents.add(-best[3])

    selected = [latest]
    selected_processed = [int(processed[latest])]
    latest_distance = _pose_distance(
        np.stack([poses[latest]]),
        target_poses,
        low_translation_rotation=low_translation_rotation,
        rotation_weight=rotation_weight,
        minimum_translation_scale_m=minimum_translation_scale_m,
    )[0]
    coverage = latest_distance.copy()

    for stratum, quota in zip(strata, _TEMPORAL_MULTISCALE_QUOTAS):
        for _ in range(quota):
            remaining_pool = [index for index in stratum if index not in selected]
            preferred = [
                index
                for index in remaining_pool
                if index[0] in selected_parents
            ]
            # Exhaust candidates from the compact parent set first.  Filling
            # each temporal band remains the hard requirement, so only after
            # those candidates run out do we admit another parent.
            pool = preferred or remaining_pool
            constrained = [
                index
                for index in pool
                if all(
                    abs(int(processed[index]) - other)
                    >= min_source_processed_gap
                    for other in selected_processed
                )
            ]
            options = constrained or pool
            if not options:
                raise RuntimeError("could not fill a temporal multiscale stratum")
            best = min(
                options,
                key=lambda index: float(
                    np.minimum(coverage, distance_by_index[index]).mean()
                ),
            )
            selected.append(best)
            selected_processed.append(int(processed[best]))
            coverage = np.minimum(coverage, distance_by_index[best])

    if len(selected) != NUM_VIEWS:
        raise RuntimeError("temporal multiscale did not produce nine sources")
    return selected


def sample_osp_history_future_9x9(
    feature: Mapping[str, Any],
    valid_target_chunks: Sequence[int],
    rng: np.random.Generator,
    *,
    history_pose_retrieval_probability: float = (
        DEFAULT_HISTORY_POSE_RETRIEVAL_PROBABILITY
    ),
    previous_chunk_probability: float = DEFAULT_PREVIOUS_CHUNK_PROBABILITY,
    temporal_multiscale_probability: float = (
        DEFAULT_TEMPORAL_MULTISCALE_PROBABILITY
    ),
    max_source_parent_chunks: int = 3,
    min_source_processed_gap: int = 4,
    low_translation_threshold_m: float = 0.3,
    rotation_weight: float = 0.25,
    minimum_translation_scale_m: float = 0.05,
    target_sampling: Literal["full_chunk", "independent"] = "full_chunk",
    target_candidate_chunks: int = 3,
    shuffle_non_reference_sources: bool = False,
) -> OSPHistoryFutureSample:
    """Draw a strictly causal OSP sample while retaining rotation-only targets."""
    # Validate before reading the object so malformed configs fail fast.
    mode_probabilities = {
        "history_pose_retrieval_probability": history_pose_retrieval_probability,
        "previous_chunk_probability": previous_chunk_probability,
        "temporal_multiscale_probability": temporal_multiscale_probability,
    }
    if any(not 0.0 <= float(value) <= 1.0 for value in mode_probabilities.values()):
        raise ValueError("OSP sampling-mode probabilities must be in [0,1]")
    if not math.isclose(
        sum(float(value) for value in mode_probabilities.values()),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-8,
    ):
        raise ValueError("OSP sampling-mode probabilities must sum to one")
    if target_sampling not in {"full_chunk", "independent"}:
        raise ValueError(f"unknown target_sampling={target_sampling!r}")
    if target_candidate_chunks <= 0:
        raise ValueError("target_candidate_chunks must be positive")
    if max_source_parent_chunks <= 0 or min_source_processed_gap < 0:
        raise ValueError("source parent/gap parameters are invalid")
    eligible = eligible_target_chunks(feature, valid_target_chunks)
    if not eligible:
        raise RuntimeError("OSP feature has no eligible causal target motion chunk")

    valid = torch.as_tensor(feature["camera_valid_mask"], dtype=torch.bool).cpu().numpy()
    processed = torch.as_tensor(
        feature["camera_anchor_processed_frame_idx"], dtype=torch.int64
    ).cpu().numpy()
    raw = torch.as_tensor(
        feature["camera_anchor_frame_idx"], dtype=torch.int64
    ).cpu().numpy()
    poses = torch.as_tensor(
        feature["camera_pose_rel_video_c2w"], dtype=torch.float32
    ).cpu().numpy()

    possible_anchors = (
        eligible[:-1]
        if target_sampling == "independent" and len(eligible) > 1
        else eligible
    )
    target_chunk = possible_anchors[int(rng.integers(0, len(possible_anchors)))]
    target_start = int(processed[target_chunk].min())
    if target_sampling == "full_chunk":
        target_indices = tuple((target_chunk, local) for local in range(NUM_VIEWS))
    else:
        future_chunks = [
            chunk
            for chunk in eligible
            if int(processed[chunk].min()) >= target_start
        ][: int(target_candidate_chunks)]
        future_pool = [
            (chunk, local)
            for chunk in future_chunks
            for local in range(NUM_VIEWS)
            if bool(valid[chunk, local])
        ]
        if len(future_pool) < NUM_VIEWS:
            raise RuntimeError("OSP future pool has fewer than nine valid anchors")
        for _ in range(32):
            target_indices = tuple(
                future_pool[int(index)]
                for index in rng.choice(
                    len(future_pool), size=NUM_VIEWS, replace=False
                )
            )
            if len(future_chunks) == 1 or len({x[0] for x in target_indices}) > 1:
                break
        else:
            raise RuntimeError("failed to draw multi-chunk OSP targets")
    history = [
        (chunk, local)
        for chunk in range(poses.shape[0])
        for local in range(NUM_VIEWS)
        if valid[chunk, local] and int(processed[chunk, local]) < target_start
    ]
    history.sort(key=lambda index: (int(processed[index]), index[0], index[1]))
    if len(history) < NUM_VIEWS:
        raise RuntimeError("target has fewer than nine strictly historical anchors")
    latest = history[-1]
    target_poses = np.stack([poses[index] for index in target_indices])
    target_order = np.argsort(
        np.asarray([int(processed[index]) for index in target_indices])
    )
    target_chord_m, target_rotation_deg = _target_motion(target_poses[target_order])
    low_translation_rotation = target_chord_m < float(low_translation_threshold_m)

    mode = sample_osp_sampling_mode(
        rng,
        history_pose_retrieval_probability=history_pose_retrieval_probability,
        previous_chunk_probability=previous_chunk_probability,
        temporal_multiscale_probability=temporal_multiscale_probability,
    )
    if mode == "previous_chunk":
        # The newest nine valid anchors are normally the complete previous block.
        newest = list(reversed(history[-NUM_VIEWS:]))
        newest.remove(latest)
        sources = [latest, *newest]
    elif mode == "history_pose_retrieval":
        sources = _choose_retrieval_sources(
            history,
            latest,
            poses,
            processed,
            target_poses,
            low_translation_rotation=low_translation_rotation,
            rotation_weight=rotation_weight,
            minimum_translation_scale_m=minimum_translation_scale_m,
            max_source_parent_chunks=max_source_parent_chunks,
            min_source_processed_gap=min_source_processed_gap,
        )
    else:
        sources = _choose_temporal_multiscale_sources(
            history,
            latest,
            poses,
            processed,
            target_poses,
            low_translation_rotation=low_translation_rotation,
            rotation_weight=rotation_weight,
            minimum_translation_scale_m=minimum_translation_scale_m,
            max_source_parent_chunks=max_source_parent_chunks,
            min_source_processed_gap=min_source_processed_gap,
        )

    # ``source[0]`` is the deployment reference camera and must remain the
    # latest historical view.  The other source-list positions carry no
    # semantics in LagerNVS, so Stage-2 v3 can randomize them without changing
    # the causal source set or the reference coordinate system.
    if shuffle_non_reference_sources:
        remaining_sources = sources[1:]
        remaining = [
            remaining_sources[int(index)] for index in rng.permutation(8).tolist()
        ]
        sources = [sources[0], *remaining]

    if sources[0] != latest:
        raise RuntimeError("the first OSP source must be latest history")
    if len(set(sources)) != NUM_VIEWS:
        raise RuntimeError("OSP source indices are not unique")
    source_processed = tuple(int(processed[index]) for index in sources)
    target_processed = tuple(int(processed[index]) for index in target_indices)
    if max(source_processed) >= min(target_processed):
        raise RuntimeError("OSP source/target sample violates strict causality")

    return OSPHistoryFutureSample(
        source_indices=tuple(sources),
        target_indices=target_indices,
        source_processed_frames=source_processed,
        target_processed_frames=target_processed,
        source_raw_frames=tuple(int(raw[index]) for index in sources),
        target_raw_frames=tuple(int(raw[index]) for index in target_indices),
        target_chunk=target_chunk,
        mode=mode,
        target_chord_m=target_chord_m,
        target_rotation_deg=target_rotation_deg,
        low_translation_rotation=low_translation_rotation,
        source_unique_parent_count=len({chunk for chunk, _ in sources}),
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

__all__ = ['Latent9x9Sample', 'LatentAnchor', 'NUM_VIEWS', 'DL3DVSamplingMode', 'TargetSampling', 'sample_latent_9x9', 'validate_feature_for_9x9', 'DEFAULT_HISTORY_POSE_RETRIEVAL_PROBABILITY', 'DEFAULT_PREVIOUS_CHUNK_PROBABILITY', 'DEFAULT_TEMPORAL_MULTISCALE_PROBABILITY', 'OSPHistoryFutureSample', 'OSPSamplingMode', 'eligible_target_chunks', 'sample_osp_history_future_9x9', 'sample_osp_sampling_mode', 'validate_osp_feature', 'LAGERNVS_NUM_SOURCES', 'LAGERNVS_RECENT_LOCAL_SLOT', 'LAGERNVS_TARGET_SLOTS', 'LagerNVSSamplingUnavailable', 'LagerNVSSamplingSelection', 'gather_lagernvs_tensors', 'sample_lagernvs_views']
