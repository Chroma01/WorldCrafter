from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Literal
import numpy as np
import torch
from torch.utils.data import Dataset
from .camera_geometry import compute_fov_from_fx_xi
from .lagernvs_sampling_contract import LagerNVSSamplingUnavailable, gather_lagernvs_tensors, sample_lagernvs_views
from .storage import list_pt_files, load_torch_pt, normalize_storage_path, path_exists, read_json, read_text, storage_basename, storage_join, storage_stem
IntrinsicsContract = Literal["native", "osp_v2_to_osp_v1_legacy"]
_DL3DV_RECORD_RETRY_SEED_TAG = 0x44525259
OSP_V2_TO_OSP_V1_LEGACY_K_SCALE = (504.0/640.0, 308.0/384.0, 504.0/640.0, 308.0/384.0)

class OSPFeatureDataset(Dataset):
    """
    Explicit LagerNVS-memory sample dataset for Helios + UCPE training.

    The loader returns only clean source/current/tail latents and untouched
    camera metadata.  It never constructs Helios memory4: the frozen
    RepEncoder runtime owns that operation on full-condition steps.
    """

    def __init__(
        self,
        feature_root: str | Path,
        split: str = "train",
        single_width: int = 640,
        default_xi: float = 0.0,
        max_chunk_delta_t: float | None = 5.0,
        intrinsics_strategy: Literal["scene_fov_center_principal_point"] = "scene_fov_center_principal_point",
        dataset_source_name: str | None = None,
        camera_json_root: str | Path | None = None,
        feature_files: list[str | Path] | None = None,
        valid_chunk_indices_json: str | Path | None = None,
        rand_feature_files: list[str | Path] | None = None,
        i2v_feature_files: list[str | Path] | None = None,
        memory_size: int = 4,
        history_sizes: tuple[int, int] = (2, 1),
        is_keep_x0: bool = True,
        first_chunk_i2v_ratio: float = 0.0,
        seed: int = 42,
        intrinsics_contract: IntrinsicsContract = "native",
        dl3dv_delta_range: tuple[int, int] = (16, 48),
        dl3dv_temporal_jitter_proportion: float = 0.1,
        dl3dv_expansion_factor: float = 1.0,
        dl3dv_overlap_probability: float = 0.2,
        dl3dv_native_max_attempts: int = 512,
        dl3dv_contract_max_attempts: int = 64,
        dl3dv_record_max_retries: int = 32,
        osp_history_pose_retrieval_probability: float = 0.55,
        osp_previous_chunk_probability: float = 0.25,
        osp_temporal_multiscale_probability: float = 0.20,
        osp_max_source_parent_chunks: int = 3,
        osp_min_source_processed_gap: int = 4,
        osp_low_translation_threshold_m: float = 0.3,
        osp_rotation_weight: float = 0.25,
        osp_minimum_translation_scale_m: float = 0.05,
    ):
        self.feature_root = normalize_storage_path(feature_root)
        self.split = split
        self.single_width = int(single_width)
        self.default_xi = float(default_xi)
        self.max_chunk_delta_t = None if max_chunk_delta_t is None else float(max_chunk_delta_t)
        self.intrinsics_strategy = intrinsics_strategy
        self.dataset_source_name = dataset_source_name or storage_basename(self.feature_root)
        self.camera_json_root = normalize_storage_path(camera_json_root) if camera_json_root is not None else None
        self.valid_chunk_indices_json = normalize_storage_path(valid_chunk_indices_json) if valid_chunk_indices_json is not None else None
        self.memory_size = int(memory_size)
        self.history_sizes = tuple(history_sizes)
        self.is_keep_x0 = bool(is_keep_x0)
        self.first_chunk_i2v_ratio = float(first_chunk_i2v_ratio)
        self.base_seed = int(seed)
        self._epoch = 0
        self.dl3dv_record_max_retries = int(dl3dv_record_max_retries)
        self.intrinsics_contract = intrinsics_contract
        self.lagernvs_sampling_kwargs = {
            "dl3dv_delta_range": tuple(int(value) for value in dl3dv_delta_range),
            "dl3dv_temporal_jitter_proportion": float(dl3dv_temporal_jitter_proportion),
            "dl3dv_expansion_factor": float(dl3dv_expansion_factor),
            "dl3dv_overlap_probability": float(dl3dv_overlap_probability),
            "dl3dv_native_max_attempts": int(dl3dv_native_max_attempts),
            "dl3dv_contract_max_attempts": int(dl3dv_contract_max_attempts),
            "osp_history_pose_retrieval_probability": float(
                osp_history_pose_retrieval_probability
            ),
            "osp_previous_chunk_probability": float(osp_previous_chunk_probability),
            "osp_temporal_multiscale_probability": float(
                osp_temporal_multiscale_probability
            ),
            "osp_max_source_parent_chunks": int(osp_max_source_parent_chunks),
            "osp_min_source_processed_gap": int(osp_min_source_processed_gap),
            "osp_low_translation_threshold_m": float(osp_low_translation_threshold_m),
            "osp_rotation_weight": float(osp_rotation_weight),
            "osp_minimum_translation_scale_m": float(osp_minimum_translation_scale_m),
        }

        if self.intrinsics_strategy != "scene_fov_center_principal_point":
            raise ValueError(f"Unsupported intrinsics_strategy: {self.intrinsics_strategy}")
        if self.dl3dv_record_max_retries < 0:
            raise ValueError("dl3dv_record_max_retries must be non-negative")
        if self.intrinsics_contract not in {"native", "osp_v2_to_osp_v1_legacy"}:
            raise ValueError(f"Unsupported intrinsics_contract: {self.intrinsics_contract}")
        if self.intrinsics_contract == "osp_v2_to_osp_v1_legacy" and self.dataset_source_name != "osp_istock_v2":
            raise ValueError(
                "osp_v2_to_osp_v1_legacy must only be applied to dataset_source_name=osp_istock_v2, "
                f"got {self.dataset_source_name}"
            )
        if self.memory_size != 4 or self.history_sizes != (2, 1):
            raise ValueError(
                "LagerNVS Helios contract is fixed to memory_size=4 and history_sizes=(2,1)"
            )
        if not self.is_keep_x0:
            raise ValueError("LagerNVS Helios contract requires is_keep_x0=true")
        if self.first_chunk_i2v_ratio != 0.0:
            raise ValueError(
                "LagerNVS memory sampling forbids first-chunk/i2v fallback; "
                "first_chunk_i2v_ratio must be zero"
            )
        if self.valid_chunk_indices_json is None:
            raise ValueError(
                "LagerNVS sampling requires an explicit valid_chunk_indices_json"
            )

        if feature_files is None:
            feature_files = list_pt_files(self.feature_root)
        else:
            feature_files = [normalize_storage_path(p) for p in feature_files]

        feature_basenames = [storage_basename(path) for path in feature_files]
        if len(feature_basenames) != len(set(feature_basenames)):
            raise ValueError(
                f"feature basenames collide under {self.feature_root}; "
                "strict pool mapping is ambiguous"
            )

        self.no_history_excluded_samples: list[dict] = []
        self.samples = self._build_samples(feature_files)
        if not self.samples:
            raise RuntimeError(f"No usable feature files found under {self.feature_root} for split={self.split}")
        self.sample_by_name = {storage_basename(sample["feature_path"]): sample for sample in self.samples}
        if len(self.sample_by_name) != len(self.samples):
            raise ValueError(
                f"feature basenames collide under {self.feature_root}; strict pool mapping is ambiguous"
            )

        self.rand_pool_samples = self._resolve_pool_samples(rand_feature_files)
        # The historical runner may still supply an i2v pool path, but this
        # contract never resolves or samples it.  first_chunk_i2v_ratio=0 above
        # makes that exclusion explicit rather than a runtime fallback.
        self.pool_sampling_enabled = self.rand_pool_samples is not None
        if self.pool_sampling_enabled:
            rand_sz = len(self.rand_pool_samples)
            if rand_sz == 0:
                raise RuntimeError(f"Random pool is empty for split={self.split}")
            print(
                f"[{self.__class__.__name__}] split={self.split} strict pool mapping enabled: "
                f"rand_pool={rand_sz} base_samples={len(self.samples)}"
            )
            self._effective_length = rand_sz
        else:
            self._effective_length = len(self.samples)

    def set_epoch(self, epoch: int):
        self._epoch = int(epoch)

    def _apply_intrinsics_contract(self, camera_k_norm: torch.Tensor) -> torch.Tensor:
        if camera_k_norm.ndim < 1 or camera_k_norm.shape[-1] != 4:
            raise ValueError(
                "camera_K_norm is expected to end with [fx, fy, cx, cy], "
                f"got shape={tuple(camera_k_norm.shape)}"
            )
        corrected = camera_k_norm.clone().contiguous()
        if self.intrinsics_contract == "osp_v2_to_osp_v1_legacy":
            # OSP-v1 divided the 504x308 DA3 intrinsics by the 640x384 video
            # size.  The successful checkpoint therefore learned that legacy
            # convention.  Map only newly added, correctly normalized OSP-v2
            # K into the same contract; leave OSP-v1 and DL3DV untouched.
            scale = corrected.new_tensor(OSP_V2_TO_OSP_V1_LEGACY_K_SCALE)
            corrected.mul_(scale)
        if not torch.isfinite(corrected).all():
            raise ValueError(f"Non-finite camera_K_norm after contract={self.intrinsics_contract}")
        if bool((corrected[..., :2] <= 0).any()):
            raise ValueError(f"Non-positive focal length after contract={self.intrinsics_contract}")
        return corrected

    def _intrinsics_scale_tensor(self, reference: torch.Tensor) -> torch.Tensor:
        if self.intrinsics_contract == "osp_v2_to_osp_v1_legacy":
            values = OSP_V2_TO_OSP_V1_LEGACY_K_SCALE
        else:
            values = (1.0, 1.0, 1.0, 1.0)
        return reference.new_tensor(values)

    @staticmethod
    def _read_valid_chunk_indices_json(path: str | Path) -> dict[str, list[int]]:
        obj = read_json(path)
        if isinstance(obj, dict) and "valid_chunk_indices" in obj:
            obj = obj["valid_chunk_indices"]
        if not isinstance(obj, dict):
            raise ValueError(f"Expected dict-like valid_chunk_indices json at {path}, got {type(obj).__name__}")
        return {str(k): [int(v) for v in vals] for k, vals in obj.items()}

    def _resolve_pool_samples(self, feature_files: list[str | Path] | None) -> list[dict] | None:
        if feature_files is None:
            return None
        normalized = [normalize_storage_path(path) for path in feature_files]
        if len(normalized) != len(set(normalized)):
            raise ValueError("LagerNVS random pool contains duplicate feature paths")
        resolved = []
        missing = []
        excluded_without_recent = []
        excluded_names = {
            storage_basename(sample["feature_path"])
            for sample in self.no_history_excluded_samples
        }
        for path in normalized:
            basename = storage_basename(path)
            sample = self.sample_by_name.get(basename)
            if sample is None:
                if basename in excluded_names:
                    excluded_without_recent.append(path)
                else:
                    missing.append(path)
            else:
                resolved.append(sample)
        if missing:
            raise KeyError(
                "LagerNVS random pool is not covered one-to-one by feature_root: "
                f"missing={len(missing)} first={missing[0]}"
            )
        if len(resolved) + len(excluded_without_recent) != len(normalized):
            raise RuntimeError(
                "strict LagerNVS pool resolution accounting is not one-to-one"
            )
        if excluded_without_recent:
            print(
                f"[{self.__class__.__name__}] excluded "
                f"{len(excluded_without_recent)} pool entries whose explicit "
                "valid-chunk map contains no target chunk > 0; recent[-1] is required"
            )
        return resolved

    def _valid_chunk_cache_path(self) -> Path | None:
        if self.max_chunk_delta_t is None or self.camera_json_root is None:
            return None
        cache_dir = Path(
            os.environ.get(
                "UCPE_RANDCHUNK_FILTER_CACHE_DIR",
                str(Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "worldcrafter" / "chunk_filter"),
            )
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        source_name = self.dataset_source_name.replace("/", "_")
        threshold_tag = str(self.max_chunk_delta_t).replace(".", "p")
        return cache_dir / f"{source_name}_randchunk_enddt_le{threshold_tag}.json"

    def _load_pose_chunks_from_json(self, feature_path: str | Path) -> torch.Tensor | None:
        if self.camera_json_root is None:
            return None
        json_path = storage_join(self.camera_json_root, f"{storage_stem(feature_path)}.json")
        if not path_exists(json_path):
            return None
        obj = read_json(json_path)
        pose_rel_chunk = obj["camera_pose_rel_chunk_c2w"]
        return torch.tensor(pose_rel_chunk, dtype=torch.float32)

    def _default_valid_chunk_indices(self, num_chunks: int) -> list[int]:
        if num_chunks <= 0:
            return []
        return list(range(num_chunks))

    @staticmethod
    def _pose34_to_pose44(pose_chunk: torch.Tensor) -> torch.Tensor:
        if pose_chunk.ndim != 3 or pose_chunk.shape[-2:] != (3, 4):
            raise ValueError(f"pose_chunk is expected to be [T, 3, 4], got shape={tuple(pose_chunk.shape)}")
        pose44 = torch.eye(4, dtype=pose_chunk.dtype, device=pose_chunk.device).unsqueeze(0).repeat(
            pose_chunk.shape[0], 1, 1
        )
        pose44[:, :3, :4] = pose_chunk
        return pose44

    def _build_reverse_clip_pose(self, pose_chunk: torch.Tensor) -> torch.Tensor:
        pose44 = self._pose34_to_pose44(pose_chunk)
        reversed_pose44 = torch.flip(pose44, dims=[0]).contiguous()
        rebased = torch.linalg.inv(reversed_pose44[0]).unsqueeze(0) @ reversed_pose44
        return rebased[:, :3, :4].contiguous()

    def _compute_valid_chunk_indices(self, pose_rel_chunk: torch.Tensor) -> list[int]:
        num_chunks = int(pose_rel_chunk.shape[0])
        valid_chunk_indices = self._default_valid_chunk_indices(num_chunks)
        if self.max_chunk_delta_t is None:
            return valid_chunk_indices
        filtered_indices = []
        for chunk_index in valid_chunk_indices:
            delta_t = float(torch.linalg.vector_norm(pose_rel_chunk[chunk_index, -1, :3, 3].float(), ord=2).item())
            if delta_t <= self.max_chunk_delta_t:
                filtered_indices.append(chunk_index)
        if not filtered_indices and num_chunks == 1:
            delta_t = float(torch.linalg.vector_norm(pose_rel_chunk[0, -1, :3, 3].float(), ord=2).item())
            if delta_t <= self.max_chunk_delta_t:
                return [0]
        return filtered_indices

    def _build_samples(self, feature_files: list[str]) -> list[dict]:
        cache_path = self._valid_chunk_cache_path()
        cached_valid_chunk_map = None
        has_explicit_valid_chunk_indices = self.valid_chunk_indices_json is not None
        if self.valid_chunk_indices_json is not None:
            if not path_exists(self.valid_chunk_indices_json):
                raise FileNotFoundError(f"valid_chunk_indices_json not found: {self.valid_chunk_indices_json}")
            cached_valid_chunk_map = self._read_valid_chunk_indices_json(self.valid_chunk_indices_json)
            print(
                f"[{self.__class__.__name__}] using precomputed valid chunks from "
                f"{self.valid_chunk_indices_json} for source={self.dataset_source_name}"
            )
        elif cache_path is not None and cache_path.exists():
            cached = json.loads(cache_path.read_text())
            cached_valid_chunk_map = cached.get("valid_chunk_indices", {})

        samples = []
        rebuilt_valid_chunk_map = {}
        filtered_count = 0

        for feature_path in feature_files:
            valid_chunk_indices = None
            if cached_valid_chunk_map is not None:
                feature_name = storage_basename(feature_path)
                if feature_name in cached_valid_chunk_map:
                    valid_chunk_indices = [int(v) for v in cached_valid_chunk_map[feature_name]]
                elif has_explicit_valid_chunk_indices:
                    raise KeyError(
                        f"valid_chunk_indices_json is missing {feature_name} for "
                        f"source={self.dataset_source_name}; strict LagerNVS sampling forbids fallback"
                    )
            if valid_chunk_indices is None and self.camera_json_root is not None and not has_explicit_valid_chunk_indices:
                pose_rel_chunk = self._load_pose_chunks_from_json(feature_path)
                if pose_rel_chunk is not None:
                    valid_chunk_indices = self._compute_valid_chunk_indices(pose_rel_chunk)
                    rebuilt_valid_chunk_map[storage_basename(feature_path)] = valid_chunk_indices

            if valid_chunk_indices is not None and not any(int(value) > 0 for value in valid_chunk_indices):
                filtered_count += 1
                self.no_history_excluded_samples.append(
                    {
                        "feature_path": feature_path,
                        "valid_chunk_indices": [
                            int(value) for value in valid_chunk_indices
                        ],
                    }
                )
                continue

            if valid_chunk_indices is None:
                raise KeyError(
                    f"No explicit valid chunk indices resolved for {feature_path}; "
                    "LagerNVS sampling does not fall back to all chunks"
                )

            samples.append(
                {
                    "feature_path": feature_path,
                    "valid_chunk_indices": valid_chunk_indices,
                }
            )

        if filtered_count > 0:
            print(
                f"[{self.__class__.__name__}] filtered {filtered_count} / {len(feature_files)} files with "
                f"no usable random chunks for split={self.split}"
            )

        if cache_path is not None and rebuilt_valid_chunk_map:
            cache_path.write_text(
                json.dumps(
                    {
                        "dataset_source": self.dataset_source_name,
                        "feature_root": str(self.feature_root),
                        "camera_json_root": str(self.camera_json_root),
                        "max_chunk_delta_t": self.max_chunk_delta_t,
                        "valid_chunk_indices": rebuilt_valid_chunk_map,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )

        if cache_path is not None and cached_valid_chunk_map is not None:
            print(
                f"[{self.__class__.__name__}] reused cached random-chunk filter from {cache_path}: "
                f"kept {len(samples)} / {len(feature_files)} files for split={self.split}"
            )

        return samples

    def _prepare_lagernvs_stage1_latents(
        self,
        vae_latent: torch.Tensor,
        target_chunk_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return x0, the untouched mid2+recent1 tail, and current chunk."""
        if vae_latent.ndim != 5 or tuple(vae_latent.shape[1:]) != (16, 9, 48, 80):
            raise ValueError(
                "vae_latent must be complete standardized Wan chunks "
                f"[N,16,9,48,80], got {tuple(vae_latent.shape)}"
            )
        target_chunk_index = int(target_chunk_index)
        if not 0 < target_chunk_index < vae_latent.shape[0]:
            raise IndexError(
                "LagerNVS current chunk must have a previous recent frame; "
                f"got target_chunk_index={target_chunk_index} chunks={vae_latent.shape[0]}"
            )
        x0_latent = vae_latent[0, :, :1].clone().contiguous()
        previous_chunk = vae_latent[target_chunk_index - 1]
        history_tail = previous_chunk[:, -3:].clone().contiguous()
        target_latent = vae_latent[target_chunk_index].contiguous()
        if tuple(history_tail.shape) != (16, 3, 48, 80):
            raise RuntimeError(
                f"Helios history tail must be [16,3,48,80], got {tuple(history_tail.shape)}"
            )
        return x0_latent, history_tail, target_latent

    def __len__(self):
        return self._effective_length

    def _getitem_once(self, idx: int):
        """Load one strict LagerNVS/Helios training example.

        Selection is deterministic for (base_seed, epoch, dataset index).
        The dataloader never creates, corrupts, or saturates memory4, and it
        never converts metric cameras into LagerNVS coordinates.
        """
        idx = int(idx)
        if not 0 <= idx < len(self):
            raise IndexError(f"dataset index {idx} outside [0,{len(self)})")
        pool = self.rand_pool_samples if self.pool_sampling_enabled else self.samples
        if pool is None or len(pool) != len(self):
            raise RuntimeError(
                "strict LagerNVS dataset index space does not match its resolved pool"
            )
        sample = pool[idx]
        if sample.get("valid_chunk_indices") is None:
            raise KeyError(
                f"{sample['feature_path']} lacks explicit valid chunk indices; "
                "legacy all-chunk fallback is forbidden"
            )

        rng = np.random.default_rng(
            np.random.SeedSequence(
                [self.base_seed, self._epoch, idx, 0x4C414745]
            )
        )
        feature_path = sample["feature_path"]
        data = load_torch_pt(feature_path, map_location="cpu", weights_only=False)
        required_fields = (
            "vae_latent",
            "prompt_embed",
            "camera_pose_rel_chunk_c2w",
            "camera_pose_rel_video_c2w",
            "camera_K_norm",
            "camera_valid_mask",
            "camera_anchor_processed_frame_idx",
            "camera_anchor_frame_idx",
        )
        missing = [name for name in required_fields if name not in data]
        if missing:
            raise KeyError(
                f"{feature_path} lacks strict LagerNVS fields: {missing}"
            )

        vae_latent_all = torch.as_tensor(data["vae_latent"])
        prompt_embed = torch.as_tensor(data["prompt_embed"])
        prompt_attention_mask = data.get("prompt_attention_mask")
        if prompt_attention_mask is None:
            prompt_attention_mask = torch.empty(0, dtype=torch.int64)
        else:
            prompt_attention_mask = torch.as_tensor(prompt_attention_mask)
        valid_chunk_indices = tuple(
            sorted({int(value) for value in sample["valid_chunk_indices"]})
        )
        if not valid_chunk_indices:
            raise RuntimeError(
                f"{feature_path} has an empty explicit valid-chunk set"
            )

        selection = sample_lagernvs_views(
            data,
            dataset_source=self.dataset_source_name,
            valid_target_chunks=valid_chunk_indices,
            rng=rng,
            **self.lagernvs_sampling_kwargs,
        )
        target_chunk_index = int(selection.target_chunk)
        x0_latent, history_tail_latent, target_latent = (
            self._prepare_lagernvs_stage1_latents(
                vae_latent_all,
                target_chunk_index,
            )
        )
        lagernvs = gather_lagernvs_tensors(data, selection)

        if not torch.equal(
            torch.as_tensor(lagernvs["lagernvs_source_latents"])[:, 0],
            history_tail_latent[:, -1],
        ):
            raise RuntimeError(
                "LagerNVS source[0] latent is not exactly Helios recent[-1]"
            )
        target_slots = torch.as_tensor(
            lagernvs["lagernvs_target_slots"], dtype=torch.int64
        )
        expected_targets = target_latent.index_select(1, target_slots)
        if not torch.equal(
            torch.as_tensor(lagernvs["lagernvs_target_latents_gt"]),
            expected_targets,
        ):
            raise RuntimeError(
                "LagerNVS supervision is not current chunk slots 2/4/6/8"
            )

        pose_chunk = torch.as_tensor(
            data["camera_pose_rel_chunk_c2w"], dtype=torch.float32
        )[target_chunk_index, :, :3, :4].contiguous()
        k_norm_raw = torch.as_tensor(
            data["camera_K_norm"], dtype=torch.float32
        )[target_chunk_index].contiguous()
        k_norm = self._apply_intrinsics_contract(k_norm_raw)

        if tuple(target_latent.shape) != (16, 9, 48, 80):
            raise ValueError(
                "input_latents must be [16,9,48,80], "
                f"got {tuple(target_latent.shape)}"
            )
        if tuple(history_tail_latent.shape) != (16, 3, 48, 80):
            raise ValueError(
                "helios_history_tail_latents must be [16,3,48,80], "
                f"got {tuple(history_tail_latent.shape)}"
            )
        if prompt_embed.ndim != 2 or prompt_embed.shape[-1] != 4096:
            raise ValueError(
                "prompt_embed must be [L,4096], "
                f"got {tuple(prompt_embed.shape)}"
            )
        if tuple(pose_chunk.shape) != (9, 3, 4):
            raise ValueError(
                f"UCPE pose must be [9,3,4], got {tuple(pose_chunk.shape)}"
            )
        if tuple(k_norm.shape) != (9, 4):
            raise ValueError(
                f"UCPE camera_K_norm must be [9,4], got {tuple(k_norm.shape)}"
            )

        # UCPE retains its existing chunk-relative camera convention and its
        # existing OSP-v2 intrinsics compatibility transform.  These tensors
        # are intentionally independent of the raw metric LagerNVS cameras.
        xi = self.default_xi
        width = int(target_latent.shape[-1]) * 8
        k0 = k_norm[0]
        fx = float(k0[0].item()) * float(width)
        x_fov = float(
            compute_fov_from_fx_xi(fx=fx, xi=xi, width=width).item()
        )
        if not 0.0 < x_fov < 180.0:
            raise ValueError(
                f"recovered x_fov must be in (0,180), got {x_fov} from {feature_path}"
            )

        # Preserve the pre-existing reverse-current-clip fields for UCPE
        # validation/training.  They are never used as LagerNVS memory or as a
        # fallback when LagerNVS sampling fails.
        previous_chunk_index = target_chunk_index - 1
        previous_target_latent = vae_latent_all[previous_chunk_index].contiguous()
        reverse_clip_input_latent = torch.flip(
            previous_target_latent, dims=[1]
        ).contiguous()
        previous_pose_chunk = torch.as_tensor(
            data["camera_pose_rel_chunk_c2w"], dtype=torch.float32
        )[previous_chunk_index, :, :3, :4].contiguous()
        reverse_clip_pose = self._build_reverse_clip_pose(previous_pose_chunk)
        reverse_clip_k_norm_raw = torch.flip(
            torch.as_tensor(data["camera_K_norm"], dtype=torch.float32)[
                previous_chunk_index
            ],
            dims=[0],
        ).contiguous()
        reverse_clip_k_norm = self._apply_intrinsics_contract(
            reverse_clip_k_norm_raw
        )
        reverse_fx = float(reverse_clip_k_norm[0, 0].item()) * float(width)
        reverse_x_fov = float(
            compute_fov_from_fx_xi(
                fx=reverse_fx,
                xi=xi,
                width=width,
            ).item()
        )
        if not 0.0 < reverse_x_fov < 180.0:
            raise ValueError(
                "recovered reverse_clip_x_fov must be in (0,180), "
                f"got {reverse_x_fov} from {feature_path}"
            )

        out = {
            "video_id": storage_stem(feature_path),
            "feature_path": str(feature_path),
            "dataset_source": self.dataset_source_name,
            "input_latents": target_latent.to(
                dtype=torch.bfloat16
            ).contiguous(),
            "helios_history_tail_latents": history_tail_latent.to(
                dtype=torch.bfloat16
            ).contiguous(),
            "x0_latents": x0_latent.to(dtype=torch.bfloat16).contiguous(),
            "context": prompt_embed,
            "prompt_attention_mask": prompt_attention_mask,
            "pose": pose_chunk,
            "x_fov": torch.tensor(x_fov, dtype=torch.float32),
            "xi": torch.tensor(xi, dtype=torch.float32),
            "camera_K_norm": k_norm,
            "camera_K_norm_first": k0.clone(),
            "camera_K_norm_raw_first": k_norm_raw[0].clone(),
            "camera_intrinsics_contract": self.intrinsics_contract,
            "camera_intrinsics_scale": self._intrinsics_scale_tensor(k_norm),
            "target_chunk_index": torch.tensor(
                target_chunk_index, dtype=torch.int64
            ),
            "is_first_chunk_i2v": torch.tensor(False, dtype=torch.bool),
            "num_valid_chunk_indices": torch.tensor(
                len(valid_chunk_indices), dtype=torch.int64
            ),
            "can_reverse_clip": torch.tensor(True, dtype=torch.bool),
            "reverse_source_chunk_index": torch.tensor(
                previous_chunk_index, dtype=torch.int64
            ),
            "reverse_clip_input_latents": reverse_clip_input_latent.to(
                dtype=torch.bfloat16
            ).contiguous(),
            "reverse_clip_pose": reverse_clip_pose,
            "reverse_clip_x_fov": torch.tensor(
                reverse_x_fov, dtype=torch.float32
            ),
            "reverse_clip_xi": torch.tensor(xi, dtype=torch.float32),
            "reverse_clip_camera_K_norm": reverse_clip_k_norm,
            "prompt_raw": data.get("prompt_raw", ""),
            "source_video_path": data.get("source_video_path", ""),
        }
        for name, value in lagernvs.items():
            if isinstance(value, torch.Tensor) and "latent" in name:
                value = value.to(dtype=torch.bfloat16).contiguous()
            out[name] = value
        if "history_latents" in out:
            raise AssertionError(
                "legacy history_latents must not exist in LagerNVS contract"
            )
        return out

    def __getitem__(self, idx):
        """Load one record, retrying only native DL3DV geometry exhaustion.

        Retry candidates are different DL3DV records drawn without replacement
        from a deterministic permutation keyed by epoch and scheduled index.
        OSP and non-sampling data errors retain their strict fail-fast behavior.
        """
        scheduled_idx = int(idx)
        if not 0 <= scheduled_idx < len(self):
            raise IndexError(
                f"dataset index {scheduled_idx} outside [0,{len(self)})"
            )
        candidate_indices = [scheduled_idx]
        if (
            self.dataset_source_name == "dl3dv"
            and self.dl3dv_record_max_retries > 0
            and len(self) > 1
        ):
            retry_rng = np.random.default_rng(
                np.random.SeedSequence(
                    [
                        self.base_seed,
                        self._epoch,
                        scheduled_idx,
                        _DL3DV_RECORD_RETRY_SEED_TAG,
                    ]
                )
            )
            replacements = [
                int(value)
                for value in retry_rng.permutation(len(self)).tolist()
                if int(value) != scheduled_idx
            ][: self.dl3dv_record_max_retries]
            candidate_indices.extend(replacements)

        failures: list[tuple[int, str]] = []
        last_error: LagerNVSSamplingUnavailable | None = None
        for retry_count, resolved_idx in enumerate(candidate_indices):
            try:
                out = self._getitem_once(resolved_idx)
            except LagerNVSSamplingUnavailable as error:
                if self.dataset_source_name != "dl3dv":
                    raise
                last_error = error
                failures.append((resolved_idx, str(error)))
                continue
            out["lagernvs_scheduled_record_index"] = torch.tensor(
                scheduled_idx, dtype=torch.int64
            )
            out["lagernvs_resolved_record_index"] = torch.tensor(
                resolved_idx, dtype=torch.int64
            )
            out["lagernvs_record_retry_count"] = torch.tensor(
                retry_count, dtype=torch.int64
            )
            return out

        if last_error is None:
            raise RuntimeError("DL3DV record retry reached an impossible empty state")
        attempted = ",".join(str(index) for index, _ in failures)
        raise RuntimeError(
            "DL3DV record retry exhausted without a valid native 9x9 sample: "
            f"scheduled_index={scheduled_idx} epoch={self._epoch} "
            f"attempted_indices=[{attempted}]"
        ) from last_error
