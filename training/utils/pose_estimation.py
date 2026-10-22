"""Depth Anything 3 camera estimation, isolated from the training environment."""
import atexit
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
ANCHOR_OFFSETS = tuple(range(0, 33, 4))

def write_json_atomic(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)
    os.replace(tmp_path, path)


def load_da3_model(model_dir: str, device: str):
    from depth_anything_3.api import DepthAnything3

    model = DepthAnything3.from_pretrained(model_dir).to(device)
    model.eval()
    return model


def run_da3_local_inference(
    model,
    image_paths: list[str],
    export_dir: Path | None,
    process_res: int,
    process_res_method: str,
    use_ray_pose: bool,
    ref_view_strategy: str,
):
    return model.inference(
        image=image_paths,
        export_dir=str(export_dir) if export_dir is not None else None,
        export_format="mini_npz",
        process_res=process_res,
        process_res_method=process_res_method,
        use_ray_pose=use_ray_pose,
        ref_view_strategy=ref_view_strategy,
    )


def run_da3_server(args):
    request_dir = Path(args.request_dir)
    request_dir.mkdir(parents=True, exist_ok=True)
    ready_path = request_dir / "server.ready"
    shutdown_path = request_dir / "server.shutdown"

    print(f"[INFO] starting persistent DA3 server on {args.da3_device}", flush=True)
    model = load_da3_model(args.da3_model_dir, args.da3_device)
    ready_path.write_text("ready\n")
    print(f"[INFO] DA3 server ready request_dir={request_dir}", flush=True)

    while True:
        if shutdown_path.exists():
            print("[INFO] DA3 server received shutdown signal", flush=True)
            break

        request_paths = sorted(request_dir.glob("*.request.json"))
        if not request_paths:
            time.sleep(0.1)
            continue

        for request_path in request_paths:
            job_id = request_path.name[: -len(".request.json")]
            done_path = request_dir / f"{job_id}.done.json"
            error_path = request_dir / f"{job_id}.error.json"
            try:
                with open(request_path, "r") as f:
                    request = json.load(f)

                frames_dir = Path(request["anchor_frames_dir"])
                image_paths = [str(p) for p in sorted(frames_dir.glob("*.png"))]
                if not image_paths:
                    raise ValueError(f"No PNG frames found in {frames_dir}")

                out_npz = Path(request["out_npz"])
                out_npz.parent.mkdir(parents=True, exist_ok=True)
                prediction = run_da3_local_inference(
                    model=model,
                    image_paths=image_paths,
                    export_dir=None,
                    process_res=request.get("process_res", args.process_res),
                    process_res_method=request.get("process_res_method", args.process_res_method),
                    use_ray_pose=request.get("use_ray_pose", args.use_ray_pose),
                    ref_view_strategy=request.get("ref_view_strategy", args.ref_view_strategy),
                )

                payload = {
                    "extrinsics": prediction.extrinsics,
                    "intrinsics": prediction.intrinsics,
                }
                if getattr(prediction, "depth", None) is not None:
                    payload["depth"] = prediction.depth
                if getattr(prediction, "conf", None) is not None:
                    payload["conf"] = prediction.conf
                if getattr(prediction, "scale_factor", None) is not None:
                    payload["scale_factor"] = np.asarray(prediction.scale_factor, dtype=np.float32)
                if getattr(prediction, "is_metric", None) is not None:
                    payload["is_metric"] = np.asarray(prediction.is_metric, dtype=np.int32)
                np.savez(out_npz, **payload)
                write_json_atomic(done_path, {"status": "ok", "out_npz": str(out_npz)})
            except Exception as exc:
                write_json_atomic(
                    error_path,
                    {
                        "status": "error",
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
            finally:
                if request_path.exists():
                    request_path.unlink()


class PersistentDA3Client:
    def __init__(self, args):
        self.args = args
        self.proc = None
        self.request_dir = Path(tempfile.mkdtemp(prefix="da3_server_", dir=args.tmp_dir))
        self.ready_path = self.request_dir / "server.ready"
        self.shutdown_path = self.request_dir / "server.shutdown"
        try:
            self._start()
        except BaseException:
            self.close()
            raise
        atexit.register(self.close)

    def _start(self):
        cmd = [
            self.args.da3_python,
            str(Path(__file__).resolve()),
            "--da3-server-mode",
            "--request-dir",
            str(self.request_dir),
            "--da3-model-dir",
            self.args.da3_model_dir,
            "--da3-device",
            self.args.da3_device,
            "--process-res",
            str(self.args.process_res),
            "--process-res-method",
            self.args.process_res_method,
            "--ref-view-strategy",
            self.args.ref_view_strategy,
            "--tmp-dir",
            self.args.tmp_dir,
        ]
        if self.args.use_ray_pose:
            cmd.append("--use-ray-pose")

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        self.proc = subprocess.Popen(cmd, env=env)

        deadline = time.time() + 300
        while time.time() < deadline:
            if self.ready_path.exists():
                return
            if self.proc.poll() is not None:
                raise RuntimeError(f"DA3 server exited early with code {self.proc.returncode}")
            time.sleep(0.2)
        raise TimeoutError("Timed out waiting for DA3 server readiness")

    def infer(self, anchor_frames_dir: Path, out_npz: Path):
        job_id = uuid.uuid4().hex
        request_path = self.request_dir / f"{job_id}.request.json"
        done_path = self.request_dir / f"{job_id}.done.json"
        error_path = self.request_dir / f"{job_id}.error.json"

        payload = {
            "anchor_frames_dir": str(anchor_frames_dir),
            "out_npz": str(out_npz),
            "process_res": self.args.process_res,
            "process_res_method": self.args.process_res_method,
            "ref_view_strategy": self.args.ref_view_strategy,
            "use_ray_pose": bool(self.args.use_ray_pose),
        }
        write_json_atomic(request_path, payload)

        deadline = time.time() + 1800
        while time.time() < deadline:
            if done_path.exists():
                done_path.unlink()
                return
            if error_path.exists():
                with open(error_path, "r") as f:
                    error = json.load(f)
                error_path.unlink()
                raise RuntimeError(error.get("traceback") or error.get("message") or "DA3 server request failed")
            if self.proc.poll() is not None:
                raise RuntimeError(f"DA3 server exited with code {self.proc.returncode}")
            time.sleep(0.1)
        raise TimeoutError("Timed out waiting for DA3 server response")

    def close(self):
        if self.proc is None:
            return
        if self.proc.poll() is None:
            self.shutdown_path.write_text("shutdown\n")
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=10)
        shutil.rmtree(self.request_dir, ignore_errors=True)
        self.proc = None


def save_frame_sequence(frames: list[torch.Tensor], out_dir: Path):
    import torchvision.transforms as transforms

    out_dir.mkdir(parents=True, exist_ok=True)
    to_pil = transforms.ToPILImage()
    for idx, frame in enumerate(frames):
        image = to_pil(frame).convert("RGB")
        image.save(out_dir / f"{idx:06d}.png")


def load_da3_result(npz_path: Path):
    data = np.load(npz_path)
    payload = {k: data[k] for k in data.files}
    data.close()
    return payload


def convert_extrinsics_to_c2w(extrinsics: np.ndarray) -> np.ndarray:
    if extrinsics.shape[-2:] == (3, 4):
        bottom = np.broadcast_to(
            np.array([0.0, 0.0, 0.0, 1.0], dtype=extrinsics.dtype),
            (extrinsics.shape[0], 1, 4),
        )
        extrinsics = np.concatenate([extrinsics, bottom], axis=1)
    return np.linalg.inv(extrinsics)


def intrinsics_to_fxfycxcy_norm(intrinsics: np.ndarray, width: int, height: int) -> np.ndarray:
    fx = intrinsics[:, 0, 0] / float(width)
    fy = intrinsics[:, 1, 1] / float(height)
    cx = intrinsics[:, 0, 2] / float(width)
    cy = intrinsics[:, 1, 2] / float(height)
    return np.stack([fx, fy, cx, cy], axis=-1)


def relative_from_reference(c2w: np.ndarray, ref_index: int) -> np.ndarray:
    ref_inv = np.linalg.inv(c2w[ref_index])
    return ref_inv[None, ...] @ c2w


def relative_per_chunk(c2w_chunked: np.ndarray) -> np.ndarray:
    ref_inv = np.linalg.inv(c2w_chunked[:, 0, :, :])[:, None, :, :]
    return ref_inv @ c2w_chunked


def read_processed_single_frame(sample: dict, raw_frame_idx: int) -> torch.Tensor:
    from training.utils.data import read_cut_crop_and_resize

    return read_cut_crop_and_resize(
        video_path=sample["video_path"],
        f_prime=1,
        h_prime=sample["bucket_height"],
        w_prime=sample["bucket_width"],
        stride=1,
        start_frame=raw_frame_idx,
        end_frame=raw_frame_idx + 1,
        crop=sample["crop"],
    )[0]


def estimate_camera_sparse_for_video(video, sample, args, da3_client: PersistentDA3Client) -> dict:
    frame_window_size = (args.latent_window_size - 1) * 4 + 1
    usable_frames = (video.shape[0] // frame_window_size) * frame_window_size
    num_chunks = usable_frames // frame_window_size
    if num_chunks < 1:
        raise ValueError(f"No full {frame_window_size}-frame chunk available for {sample['uttid']}")

    clip_anchor_processed_indices = []
    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * frame_window_size
        clip_anchor_processed_indices.extend(chunk_start + off for off in ANCHOR_OFFSETS)

    clip_anchor_processed_indices = np.asarray(clip_anchor_processed_indices, dtype=np.int32)
    clip_anchor_source_indices = (
        int(sample["selected_start_frame"]) + clip_anchor_processed_indices * int(sample["stride"])
    ).astype(np.int32)

    video_ref_raw_frame_idx = int(sample["cut_start_frame"])
    ordered_source_indices = []
    seen_indices = set()
    for raw_idx in [video_ref_raw_frame_idx, *clip_anchor_source_indices.tolist()]:
        if raw_idx not in seen_indices:
            ordered_source_indices.append(raw_idx)
            seen_indices.add(raw_idx)

    clip_frame_lookup = {
        int(int(sample["selected_start_frame"]) + proc_idx * int(sample["stride"])): video[int(proc_idx)]
        for proc_idx in clip_anchor_processed_indices.tolist()
    }

    da3_frames = []
    for raw_idx in ordered_source_indices:
        if raw_idx in clip_frame_lookup:
            frame = video[int((raw_idx - int(sample["selected_start_frame"])) // int(sample["stride"]))]
        else:
            frame = read_processed_single_frame(sample=sample, raw_frame_idx=raw_idx)
        frame = ((frame + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)
        da3_frames.append(frame)

    source_position_map = {raw_idx: pos for pos, raw_idx in enumerate(ordered_source_indices)}
    clip_anchor_sequence_positions = np.asarray(
        [source_position_map[int(raw_idx)] for raw_idx in clip_anchor_source_indices.tolist()],
        dtype=np.int32,
    )
    video_ref_sequence_index = source_position_map[video_ref_raw_frame_idx]

    with tempfile.TemporaryDirectory(prefix=f"da3_{sample['uttid']}_", dir=args.tmp_dir) as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        frames_dir = tmp_dir / "frames"
        out_npz = tmp_dir / "da3_result.npz"

        save_frame_sequence(da3_frames, frames_dir)
        print(
            (
                f"[INFO] running DA3 for {sample['uttid']} with {len(ordered_source_indices)} frames "
                f"(video_ref + {len(clip_anchor_source_indices)} clip anchors) on {args.da3_device}"
            ),
            flush=True,
        )

        da3_client.infer(anchor_frames_dir=frames_dir, out_npz=out_npz)
        print(f"[INFO] DA3 finished for {sample['uttid']}", flush=True)
        result = load_da3_result(out_npz)

    extrinsics = result["extrinsics"]
    intrinsics = result["intrinsics"]
    depth = result.get("depth")
    if extrinsics.shape[0] != len(ordered_source_indices):
        raise ValueError(
            f"DA3 returned {extrinsics.shape[0]} poses, expected {len(ordered_source_indices)} for {sample['uttid']}"
        )
    if depth is not None and depth.shape[0] != len(ordered_source_indices):
        raise ValueError(
            f"DA3 returned {depth.shape[0]} depth maps, expected {len(ordered_source_indices)} for {sample['uttid']}"
        )

    c2w_abs_all = convert_extrinsics_to_c2w(extrinsics)
    c2w_rel_video_all = relative_from_reference(c2w_abs_all, ref_index=video_ref_sequence_index)
    c2w_abs_clip = c2w_abs_all[clip_anchor_sequence_positions]
    c2w_rel_video_clip = c2w_rel_video_all[clip_anchor_sequence_positions]
    c2w_abs_chunk = c2w_abs_clip.reshape(num_chunks, len(ANCHOR_OFFSETS), 4, 4)
    c2w_rel_chunk = relative_per_chunk(c2w_abs_chunk)

    if depth is not None:
        da3_height = int(depth.shape[-2])
        da3_width = int(depth.shape[-1])
    else:
        raise ValueError("Depth Anything 3 must return depth dimensions to normalize intrinsics")

    k_norm_all = intrinsics_to_fxfycxcy_norm(
        intrinsics=intrinsics,
        width=da3_width,
        height=da3_height,
    )
    k_norm_clip = k_norm_all[clip_anchor_sequence_positions]
    chunk_anchor_source_indices = clip_anchor_source_indices.reshape(num_chunks, len(ANCHOR_OFFSETS))
    chunk_anchor_processed_indices = clip_anchor_processed_indices.reshape(num_chunks, len(ANCHOR_OFFSETS))

    camera_valid_mask = np.isfinite(c2w_rel_video_clip).all(axis=(1, 2)) & np.isfinite(k_norm_clip).all(axis=1)

    payload = {
        "camera_pose_rel_video_c2w": c2w_rel_video_clip.reshape(num_chunks, len(ANCHOR_OFFSETS), 4, 4).astype(np.float32),
        "camera_pose_rel_chunk_c2w": c2w_rel_chunk.astype(np.float32),
        "camera_K_norm": k_norm_clip.reshape(num_chunks, len(ANCHOR_OFFSETS), 4).astype(np.float32),
        "camera_valid_mask": camera_valid_mask.reshape(num_chunks, len(ANCHOR_OFFSETS)),
        "camera_anchor_frame_idx": chunk_anchor_source_indices,
        "camera_anchor_processed_frame_idx": chunk_anchor_processed_indices,
        "camera_num_video_raw_frames": np.int32(int(sample["cut_end_frame"]) - int(sample["cut_start_frame"])),
    }
    if depth is not None:
        payload["camera_ref_depth"] = depth[video_ref_sequence_index].astype(np.float32)
        payload["camera_anchor_depth"] = depth[clip_anchor_sequence_positions].reshape(
            num_chunks, len(ANCHOR_OFFSETS), depth.shape[-2], depth.shape[-1]
        ).astype(np.float32)
        payload["camera_depth_hw"] = np.asarray([depth.shape[-2], depth.shape[-1]], dtype=np.int32)
    if "conf" in result:
        conf = result["conf"]
        if conf.shape[0] == len(ordered_source_indices):
            conf_mean = conf.reshape(conf.shape[0], -1).mean(axis=1)
            payload["camera_pose_conf"] = conf_mean[clip_anchor_sequence_positions].reshape(
                num_chunks, len(ANCHOR_OFFSETS)
            ).astype(np.float32)
    if "scale_factor" in result:
        payload["camera_pose_scale_factor"] = float(np.asarray(result["scale_factor"]).item())
    if "is_metric" in result:
        payload["camera_pose_is_metric"] = int(np.asarray(result["is_metric"]).item())
    return payload


def camera_payload_to_jsonable(sample: dict, camera_payload: dict) -> dict:
    valid_mask = camera_payload["camera_valid_mask"]
    valid_ratio = float(valid_mask.mean()) if valid_mask.size else 0.0
    json_payload = {
        "sample_id": sample["uttid"],
        "source_video_path": sample["video_path"],
        "prompt_raw": sample["prompt"],
        "stride": int(sample["stride"]),
        "fps": float(sample["fps"]),
        "bucket_num_frame": int(sample["bucket_num_frame"]),
        "bucket_height": int(sample["bucket_height"]),
        "bucket_width": int(sample["bucket_width"]),
        "selected_start_frame": int(sample["selected_start_frame"]),
        "selected_end_frame": int(sample["selected_end_frame"]),
        "camera_num_video_raw_frames": int(camera_payload["camera_num_video_raw_frames"]),
        "camera_num_chunks": int(camera_payload["camera_pose_rel_video_c2w"].shape[0]),
        "camera_num_anchors_per_chunk": int(camera_payload["camera_pose_rel_video_c2w"].shape[1]),
        "camera_valid_ratio": valid_ratio,
        "camera_valid_mask": camera_payload["camera_valid_mask"].astype(bool).tolist(),
        "camera_anchor_frame_idx": camera_payload["camera_anchor_frame_idx"].astype(int).tolist(),
        "camera_anchor_processed_frame_idx": camera_payload["camera_anchor_processed_frame_idx"].astype(int).tolist(),
        "camera_K_norm": np.asarray(camera_payload["camera_K_norm"]).tolist(),
        "camera_pose_rel_video_c2w": np.asarray(camera_payload["camera_pose_rel_video_c2w"]).tolist(),
        "camera_pose_rel_chunk_c2w": np.asarray(camera_payload["camera_pose_rel_chunk_c2w"]).tolist(),
    }
    if "camera_pose_conf" in camera_payload:
        json_payload["camera_pose_conf"] = np.asarray(camera_payload["camera_pose_conf"]).tolist()
    if "camera_pose_scale_factor" in camera_payload:
        json_payload["camera_pose_scale_factor"] = float(camera_payload["camera_pose_scale_factor"])
    if "camera_pose_is_metric" in camera_payload:
        json_payload["camera_pose_is_metric"] = int(camera_payload["camera_pose_is_metric"])
    return json_payload

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--da3-server-mode", action="store_true", required=True)
    parser.add_argument("--request-dir", required=True)
    parser.add_argument("--da3-model-dir", required=True)
    parser.add_argument("--da3-device", default="cuda:0")
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument("--process-res-method", default="upper_bound_resize")
    parser.add_argument("--ref-view-strategy", default="middle")
    parser.add_argument("--use-ray-pose", action="store_true")
    parser.add_argument("--tmp-dir")
    run_da3_server(parser.parse_args())
