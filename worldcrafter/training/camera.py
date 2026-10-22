"""Camera conditioning for already chunk-aligned training anchors."""
from __future__ import annotations
from typing import Any
import torch
from einops import repeat, rearrange
from worldcrafter.ucpe import camera as ucpe_cc
from worldcrafter.ucpe.bridge import _build_stage_rope_coeffs

def _slice_pose_chunk(
    pose: torch.Tensor,
    chunk_index: int,
    window_num_frames: int,
    restart_each_chunk: bool = True,
):
    if restart_each_chunk:
        start = 0
        end = window_num_frames
    else:
        start = chunk_index * window_num_frames
        end = start + window_num_frames
    if pose.shape[1] < end:
        return None
    return pose[:, start:end]

def build_ucpe_attention_kwargs_for_chunk(
    transformer: torch.nn.Module,
    camera_control_panshot: dict[str, Any] | None,
    height: int,
    width: int,
    num_latent_frames_per_chunk: int,
    chunk_index: int,
    vae_scale_factor_temporal: int = 4,
    restart_each_chunk: bool = True,
    translation_scale: float = 1.0,
    pose_is_chunk_aligned: bool = False,
    token_grid_height: int | None = None,
    token_grid_width: int | None = None,
):
    if camera_control_panshot is None or getattr(transformer, "camera_condition", "none") == "none":
        return None

    pose = camera_control_panshot["pose"]
    x_fov = camera_control_panshot["x_fov"]
    xi = camera_control_panshot["xi"]
    method = transformer.camera_condition

    if pose.ndim == 3:
        pose = pose.unsqueeze(0)
    if x_fov.ndim == 0:
        x_fov = x_fov.unsqueeze(0)
    if xi.ndim == 0:
        xi = xi.unsqueeze(0)

    if pose_is_chunk_aligned:
        if pose.shape[1] < num_latent_frames_per_chunk:
            return None
        pose_chunk = pose[:, :num_latent_frames_per_chunk]
    else:
        window_num_frames = (num_latent_frames_per_chunk - 1) * vae_scale_factor_temporal + 1
        pose_chunk = _slice_pose_chunk(
            pose,
            chunk_index,
            window_num_frames,
            restart_each_chunk=restart_each_chunk,
        )
        if pose_chunk is None:
            return None
        pose_chunk = pose_chunk[:, ::vae_scale_factor_temporal]

    if pose_chunk.shape[-2:] == (4, 4):
        pose_chunk = pose_chunk[..., :3, :4]
    elif pose_chunk.shape[-2:] != (3, 4):
        raise ValueError(
            "pose_chunk is expected to be [B, T, 3, 4] or [B, T, 4, 4], "
            f"got shape={tuple(pose_chunk.shape)}"
        )

    if translation_scale != 1.0:
        pose_chunk = pose_chunk.clone()
        pose_chunk[..., 3] *= pose_chunk.new_tensor(float(translation_scale))

    c2w = torch.eye(4, device=pose_chunk.device, dtype=pose_chunk.dtype)
    c2w = repeat(c2w, "... -> B T ...", B=pose_chunk.shape[0], T=pose_chunk.shape[1]).clone()
    c2w[..., :3, :4] = pose_chunk

    if "gta" in method or "prope" in method:
        raise NotImplementedError("The Helios training bridge currently implements relray_absmap only.")

    if "relray" not in method:
        raise ValueError(f"Unsupported camera condition for Helios UCPE training bridge: {method}")

    attn = transformer.blocks[0].cam_self_attn
    grid_h = token_grid_height if token_grid_height is not None else attn.patches_y
    grid_w = token_grid_width if token_grid_width is not None else attn.patches_x

    d_cam = ucpe_cc.ucm_unproject_grid_fov(
        x_fov=x_fov,
        xi=xi,
        height=grid_h,
        width=grid_w,
        device=pose_chunk.device,
        dtype=pose_chunk.dtype,
    )
    raymats = ucpe_cc.world_to_ray_mats(d_cam, c2w)
    viewmats = rearrange(raymats, "B T H W ... -> B (T H W) ...")

    control_camera_dit_input = {"viewmats": viewmats}

    if token_grid_height is not None or token_grid_width is not None:
        coeffs_x, coeffs_y = _build_stage_rope_coeffs(
            cameras=pose_chunk.shape[1],
            patches_y=grid_h,
            patches_x=grid_w,
            head_dim=attn.head_dim,
            freq_base=attn.freq_base,
            freq_scale=attn.freq_scale,
            device=pose_chunk.device,
            dtype=pose_chunk.dtype,
        )
        control_camera_dit_input["coeffs_x"] = coeffs_x
        control_camera_dit_input["coeffs_y"] = coeffs_y

    if "absmap" in method:
        up_map, lat_map = ucpe_cc.compute_up_lat_map(
            R=pose_chunk[..., :3, :3],
            x_fov=x_fov,
            xi=xi,
            width=grid_w,
            height=grid_h,
            device=pose_chunk.device,
        )
        cam_emb = torch.cat([up_map, lat_map], dim=-1)
        cam_emb = rearrange(cam_emb, "B T H W C -> B (T H W) C")
        control_camera_dit_input["cam_emb"] = cam_emb

    expected_token_count = pose_chunk.shape[1] * grid_h * grid_w
    if control_camera_dit_input["viewmats"].shape[1] != expected_token_count:
        raise ValueError(
            "camera viewmats token count mismatch: "
            f"expected {expected_token_count}, got {control_camera_dit_input['viewmats'].shape[1]}"
        )
    if "cam_emb" in control_camera_dit_input and control_camera_dit_input["cam_emb"].shape[1] != expected_token_count:
        raise ValueError(
            "camera cam_emb token count mismatch: "
            f"expected {expected_token_count}, got {control_camera_dit_input['cam_emb'].shape[1]}"
        )

    return {"camera_control_ucpe_input": control_camera_dit_input}
