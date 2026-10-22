"""Causal nine-source/nine-target sampling for offline OSP latents.

OSP feature objects store contiguous 33-RGB-frame Wan chunks.  Each chunk has
nine latent anchors and all camera poses are expressed in one video-relative
metric coordinate frame.  The compatibility path keeps one complete target
chunk.  The independent path flattens several causal future chunks and draws
nine unordered target latent-camera pairs, while only exposing older anchors.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import torch


NUM_VIEWS = 9
SamplingMode = Literal[
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
    mode: SamplingMode
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
) -> SamplingMode:
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


__all__ = [
    "DEFAULT_HISTORY_POSE_RETRIEVAL_PROBABILITY",
    "DEFAULT_PREVIOUS_CHUNK_PROBABILITY",
    "DEFAULT_TEMPORAL_MULTISCALE_PROBABILITY",
    "NUM_VIEWS",
    "OSPHistoryFutureSample",
    "SamplingMode",
    "eligible_target_chunks",
    "sample_osp_history_future_9x9",
    "sample_osp_sampling_mode",
    "validate_osp_feature",
]
