#!/usr/bin/env python3
import atexit
import argparse
import gc
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_FLAX", "0")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ANCHOR_OFFSETS = tuple(range(0, 33, 4))


def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def iter_json_array(path, chunk_size=1 << 20):
    decoder = json.JSONDecoder()
    with open(path, "r") as f:
        buffer = ""
        in_array = False
        eof = False

        while not eof:
            chunk = f.read(chunk_size)
            if chunk == "":
                eof = True
            else:
                buffer += chunk

            pos = 0
            if not in_array:
                while pos < len(buffer) and buffer[pos].isspace():
                    pos += 1
                if pos >= len(buffer):
                    buffer = ""
                    continue
                if buffer[pos] != "[":
                    raise ValueError(f"{path} is not a JSON array")
                pos += 1
                in_array = True

            while True:
                while pos < len(buffer) and (buffer[pos].isspace() or buffer[pos] == ","):
                    pos += 1
                if pos >= len(buffer):
                    break
                if buffer[pos] == "]":
                    return
                try:
                    item, end = decoder.raw_decode(buffer, pos)
                except json.JSONDecodeError:
                    break
                yield item
                pos = end

            buffer = buffer[pos:]

        raise ValueError(f"Unexpected EOF while parsing {path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Encode existing captions and video latents; estimate DA3 cameras. Captions are supplied by the user."
    )

    parser.add_argument("--da3-helper-mode", action="store_true")
    parser.add_argument("--da3-server-mode", action="store_true")
    parser.add_argument("--anchor-frames-dir", type=str, default=None)
    parser.add_argument("--out-npz", type=str, default=None)
    parser.add_argument("--request-dir", type=str, default=None)

    parser.add_argument(
        "--source-json",
        type=str,
        default="data/source.json",
    )
    parser.add_argument(
        "--output-latent-dir",
        type=str,
        default="data/features",
    )
    parser.add_argument(
        "--output-camera-json-dir",
        type=str,
        default="data/cameras",
    )
    parser.add_argument(
        "--pretrained-model-name-or-path",
        type=str,
        default="weights/WorldCrafter-Base",
    )
    parser.add_argument(
        "--da3-model-dir",
        type=str,
        default="weights/DA3NESTED-GIANT-LARGE-1.1",
    )
    parser.add_argument(
        "--da3-python",
        type=str,
        default=sys.executable,
    )
    parser.add_argument(
        "--local-model-root",
        type=str,
        default=None,
        help="If set, stage Helios/DA3 model directories to local disk before loading.",
    )
    parser.add_argument(
        "--skip-local-model-staging",
        action="store_true",
        help="Disable local-disk staging and load models directly from the source paths.",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--da3-device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=640)
    parser.add_argument("--single-height", type=int, default=384)
    parser.add_argument("--single-width", type=int, default=640)
    parser.add_argument("--min-bucket-frames", type=int, default=121)
    parser.add_argument("--latent-window-size", type=int, default=9)
    parser.add_argument(
        "--save-i2v-image-latents",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save clean I2V image and fake-image latents from the sampled first frame.",
    )
    parser.add_argument(
        "--bucket-flush-size",
        type=int,
        default=2,
        help="Flush a bucket once it accumulates this many samples. Defaults to batch size scale for earlier outputs.",
    )
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument("--process-res-method", type=str, default="upper_bound_resize")
    parser.add_argument("--ref-view-strategy", type=str, default="middle")
    parser.add_argument("--use-ray-pose", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--max-read-retries",
        type=int,
        default=0,
        help="Maximum same-bucket unique candidates to try when video reading fails.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--report-path", type=str, default=None)
    parser.add_argument("--tmp-dir", type=str, default=str(Path(tempfile.gettempdir()) / "worldcrafter-data"))
    return parser.parse_args()


def write_json_atomic(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)
    os.replace(tmp_path, path)


def stage_model_dir_to_local(src_dir: str, local_root: str | None, target_name: str) -> str:
    if not local_root:
        return src_dir

    src_path = Path(src_dir)
    dst_root = Path(local_root)
    dst_path = dst_root / target_name
    ready_flag = dst_path / ".stage_complete"

    if ready_flag.exists():
        return str(dst_path)

    dst_root.mkdir(parents=True, exist_ok=True)
    tmp_path = dst_root / f"{target_name}.tmp"
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    if dst_path.exists():
        shutil.rmtree(dst_path)

    print(f"[INFO] staging model {src_path} -> {dst_path}", flush=True)
    shutil.copytree(src_path, tmp_path)
    os.replace(tmp_path, dst_path)
    ready_flag.write_text("ok\n")
    print(f"[INFO] local model ready at {dst_path}", flush=True)
    return str(dst_path)


def prepare_model_paths(args, need_helios: bool, need_da3: bool):
    if args.skip_local_model_staging or not args.local_model_root:
        return

    local_root = args.local_model_root
    if need_helios:
        args.pretrained_model_name_or_path = stage_model_dir_to_local(
            args.pretrained_model_name_or_path,
            local_root,
            "helios_model",
        )
    if need_da3:
        args.da3_model_dir = stage_model_dir_to_local(
            args.da3_model_dir,
            local_root,
            "da3_model",
        )


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


def run_da3_helper(args):
    frames_dir = Path(args.anchor_frames_dir)
    image_paths = [str(p) for p in sorted(frames_dir.glob("*.png"))]
    if not image_paths:
        raise ValueError(f"No PNG frames found in {frames_dir}")

    model = load_da3_model(args.da3_model_dir, args.da3_device)
    prediction = run_da3_local_inference(
        model=model,
        image_paths=image_paths,
        export_dir=None,
        process_res=args.process_res,
        process_res_method=args.process_res_method,
        use_ray_pose=args.use_ray_pose,
        ref_view_strategy=args.ref_view_strategy,
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
    np.savez(args.out_npz, **payload)


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
        self._start()
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
        if self.args.skip_local_model_staging:
            cmd.append("--skip-local-model-staging")
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


def required_raw_frames(num_sampled_frames: int, stride: int) -> int:
    return (num_sampled_frames - 1) * stride + 1


def latent_output_path(output_dir: Path, sample: dict) -> Path:
    filename = f"{sample['uttid']}_{sample['bucket_num_frame']}_{sample['bucket_height']}_{sample['bucket_width']}.pt"
    return output_dir / filename


def camera_json_output_path(output_dir: Path, sample: dict) -> Path:
    filename = f"{sample['uttid']}_{sample['bucket_num_frame']}_{sample['bucket_height']}_{sample['bucket_width']}.json"
    return output_dir / filename


def build_output_signature(args, sample: dict) -> dict:
    return {
        "schema_version": "osp_camera_da3_v2_depth_da3k",
        "pretrained_model_name_or_path": str(args.pretrained_model_name_or_path),
        "da3_model_dir": str(args.da3_model_dir),
        "stride": int(sample["stride"]),
        "bucket_num_frame": int(sample["bucket_num_frame"]),
        "bucket_height": int(sample["bucket_height"]),
        "bucket_width": int(sample["bucket_width"]),
        "latent_window_size": int(args.latent_window_size),
        "save_i2v_image_latents": bool(getattr(args, "save_i2v_image_latents", True)),
        "process_res": int(args.process_res),
        "process_res_method": str(args.process_res_method),
        "ref_view_strategy": str(args.ref_view_strategy),
        "use_ray_pose": bool(args.use_ray_pose),
    }


def existing_outputs_match_current_config(output_path: Path, camera_json_path: Path, args, sample: dict) -> bool:
    if not (output_path.exists() and camera_json_path.exists()):
        return False

    expected_signature = build_output_signature(args, sample)
    try:
        with open(camera_json_path, "r") as f:
            camera_json_payload = json.load(f)
    except Exception:
        return False

    existing_signature = camera_json_payload.get("offload_config")
    return existing_signature == expected_signature


def load_models(model_path, device):
    from diffusers import AutoencoderKLWan
    from transformers import AutoTokenizer, UMT5EncoderModel

    print(f"[INFO] loading Helios tokenizer/text_encoder/VAE from {model_path} to {device}", flush=True)
    model_root = Path(model_path)
    config_path = model_root / "inference_config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
        model_path = str((model_root / config.get("shared_components", ".")).resolve())
    weight_dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    text_encoder = UMT5EncoderModel.from_pretrained(
        model_path,
        subfolder="text_encoder",
        torch_dtype=weight_dtype,
    ).to(device)
    vae = AutoencoderKLWan.from_pretrained(
        model_path,
        subfolder="vae",
        torch_dtype=torch.float32,
    ).to(device)

    vae.eval()
    text_encoder.eval()
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    latents_mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(device, weight_dtype)
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(device, weight_dtype)
    print("[INFO] Helios models loaded.", flush=True)
    return {
        "device": device,
        "tokenizer": tokenizer,
        "text_encoder": text_encoder,
        "vae": vae,
        "latents_mean": latents_mean,
        "latents_std": latents_std,
    }

def encode_i2v_first_image_latents(batch_videos, vae, latents_mean, latents_std, frame_window_size, device):
    first_frames = batch_videos[:, 0].to(dtype=vae.dtype, device=device)
    image_pixel_values = first_frames.unsqueeze(2)
    image_latents = vae.encode(image_pixel_values).latent_dist.sample()
    image_latents = (image_latents - latents_mean) * latents_std

    fake_pixel_values = image_pixel_values.repeat(1, 1, frame_window_size, 1, 1)
    fake_image_latents = vae.encode(fake_pixel_values).latent_dist.sample()
    fake_image_latents = (fake_image_latents - latents_mean) * latents_std
    fake_image_latents = fake_image_latents[:, :, -1:, :, :]
    return image_latents, fake_image_latents


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
    from worldcrafter.training.utils.data import read_cut_crop_and_resize

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
        # Fallback for legacy npz payloads that do not include depth.
        da3_height = int(sample["bucket_height"])
        da3_width = int(sample["bucket_width"])

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


def retry_candidates(primary_sample, sample_pool, max_retries):
    max_retries = max(1, int(max_retries))
    candidates = [primary_sample]
    seen = {primary_sample["uttid"]}
    alternates = []
    for sample in sample_pool or []:
        uttid = sample.get("uttid")
        if uttid in seen:
            continue
        seen.add(uttid)
        alternates.append(sample)
    random.shuffle(alternates)
    candidates.extend(alternates)
    return candidates[:max_retries]


def encode_and_save_bucket(samples, runtime, args, da3_client: PersistentDA3Client | None, sample_pool=None):
    import torchvision.transforms as transforms
    from worldcrafter.training.utils.data import read_cut_crop_and_resize
    from worldcrafter.training.utils.data import encode_prompt

    if not samples:
        return {"saved": 0, "saved_camera_json": 0, "skipped_existing": 0, "failed": 0}

    output_dir = Path(args.output_latent_dir)
    camera_json_dir = Path(args.output_camera_json_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    camera_json_dir.mkdir(parents=True, exist_ok=True)
    frame_window_size = (args.latent_window_size - 1) * 4 + 1
    to_pil = transforms.ToPILImage()

    valid_samples = []
    videos = []
    skipped_existing = 0
    failed = 0
    reserved_output_paths = set()

    if sample_pool is None:
        sample_pool = samples

    for sample in samples:
        output_path = latent_output_path(output_dir, sample)
        camera_json_path = camera_json_output_path(camera_json_dir, sample)
        if output_path in reserved_output_paths:
            skipped_existing += 1
            continue
        if existing_outputs_match_current_config(output_path, camera_json_path, args, sample):
            skipped_existing += 1
            continue

        last_exc = None
        attempted = 0
        candidates = retry_candidates(sample, sample_pool, args.max_read_retries)
        for chosen_sample in candidates:
            output_path = latent_output_path(output_dir, chosen_sample)
            camera_json_path = camera_json_output_path(camera_json_dir, chosen_sample)
            if output_path in reserved_output_paths:
                continue
            if existing_outputs_match_current_config(output_path, camera_json_path, args, chosen_sample):
                continue

            try:
                attempted += 1
                required_frames = required_raw_frames(chosen_sample["bucket_num_frame"], chosen_sample["stride"])
                start_frame = chosen_sample["cut_start_frame"]
                end_frame = start_frame + required_frames
                video = read_cut_crop_and_resize(
                    video_path=chosen_sample["video_path"],
                    f_prime=chosen_sample["bucket_num_frame"],
                    h_prime=chosen_sample["bucket_height"],
                    w_prime=chosen_sample["bucket_width"],
                    stride=chosen_sample["stride"],
                    start_frame=start_frame,
                    end_frame=end_frame,
                    crop=chosen_sample["crop"],
                )
                chosen_sample["selected_start_frame"] = start_frame
                chosen_sample["selected_end_frame"] = end_frame
                valid_samples.append(chosen_sample)
                videos.append(video)
                reserved_output_paths.add(output_path)
                print(
                    f"[INFO] prepared {chosen_sample['uttid']} frames={chosen_sample['bucket_num_frame']} stride={chosen_sample['stride']} size={chosen_sample['bucket_height']}x{chosen_sample['bucket_width']}",
                    flush=True,
                )
                break
            except Exception as exc:
                last_exc = exc
        else:
            print(
                f"skip {sample['uttid']}: failed to read/process video after {attempted} same-bucket unique candidate(s) ({last_exc})",
                flush=True,
            )
            failed += 1

    if not valid_samples:
        cleanup_cuda()
        return {"saved": 0, "saved_camera_json": 0, "skipped_existing": skipped_existing, "failed": failed}

    batch_videos = torch.stack(videos)
    device = runtime["device"]
    vae = runtime["vae"]
    tokenizer = runtime["tokenizer"]
    text_encoder = runtime["text_encoder"]
    latents_mean = runtime["latents_mean"]
    latents_std = runtime["latents_std"]

    with torch.inference_mode():
        print(f"[INFO] VAE encoding batch of {len(valid_samples)} samples on {device}", flush=True)
        pixel_values = batch_videos.permute(0, 2, 1, 3, 4).to(dtype=vae.dtype, device=device)
        num_chunk_to_encode = pixel_values.shape[2] // frame_window_size
        if num_chunk_to_encode < 1:
            cleanup_cuda()
            return {
                "saved": 0,
                "saved_camera_json": 0,
                "skipped_existing": skipped_existing,
                "failed": failed + len(valid_samples),
            }

        latent_list = []
        for section_idx in range(num_chunk_to_encode):
            start_idx = section_idx * frame_window_size
            end_idx = start_idx + frame_window_size
            cur_pixel_values = pixel_values[:, :, start_idx:end_idx, :, :]
            cur_latent = vae.encode(cur_pixel_values).latent_dist.sample()
            cur_latent = (cur_latent - latents_mean) * latents_std
            latent_list.append(cur_latent)
        vae_latents = torch.stack(latent_list, dim=1).cpu().detach()

        i2v_image_latents = None
        i2v_fake_image_latents = None
        if getattr(args, "save_i2v_image_latents", True):
            i2v_image_latents, i2v_fake_image_latents = encode_i2v_first_image_latents(
                batch_videos=batch_videos,
                vae=vae,
                latents_mean=latents_mean,
                latents_std=latents_std,
                frame_window_size=frame_window_size,
                device=device,
            )
            i2v_image_latents = i2v_image_latents.cpu().detach()
            i2v_fake_image_latents = i2v_fake_image_latents.cpu().detach()

        prompts = [sample["prompt"] for sample in valid_samples]
        prompt_embeds, _ = encode_prompt(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            prompt=prompts,
            device=device,
        )
        prompt_embeds = prompt_embeds.cpu().detach()
        images = [to_pil(((video[0] + 1) / 2 * 255).to(torch.uint8)) for video in batch_videos]
        print("[INFO] batch VAE/text encoding complete", flush=True)

    del pixel_values
    runtime["vae"] = runtime["vae"].to("cpu")
    runtime["text_encoder"] = runtime["text_encoder"].to("cpu")
    runtime["latents_mean"] = runtime["latents_mean"].to("cpu")
    runtime["latents_std"] = runtime["latents_std"].to("cpu")
    cleanup_cuda()

    saved = 0
    saved_camera_json = 0
    active_da3_client = da3_client
    owned_da3_client = None
    try:
        if active_da3_client is None:
            owned_da3_client = PersistentDA3Client(args)
            active_da3_client = owned_da3_client

        for sample_idx, (sample, video, cur_vae_latent, cur_prompt_embed, cur_first_frame_image) in enumerate(zip(
            valid_samples, videos, vae_latents, prompt_embeds, images
        )):
            output_path = latent_output_path(output_dir, sample)
            camera_json_path = camera_json_output_path(camera_json_dir, sample)
            try:
                camera_payload = estimate_camera_sparse_for_video(
                    video=video, sample=sample, args=args, da3_client=active_da3_client
                )
                output_signature = build_output_signature(args, sample)
                payload = {
                    "vae_latent": cur_vae_latent,
                    "prompt_embed": cur_prompt_embed,
                    "first_frames_image": cur_first_frame_image,
                    "prompt_raw": sample["prompt"],
                    "caption_record": sample["caption_record"],
                    "camera_pose_rel_video_c2w": torch.from_numpy(camera_payload["camera_pose_rel_video_c2w"]),
                    "camera_pose_rel_chunk_c2w": torch.from_numpy(camera_payload["camera_pose_rel_chunk_c2w"]),
                    "camera_K_norm": torch.from_numpy(camera_payload["camera_K_norm"]),
                    "camera_valid_mask": torch.from_numpy(camera_payload["camera_valid_mask"]),
                    "camera_anchor_frame_idx": torch.from_numpy(camera_payload["camera_anchor_frame_idx"]),
                    "camera_anchor_processed_frame_idx": torch.from_numpy(
                        camera_payload["camera_anchor_processed_frame_idx"]
                    ),
                    "camera_num_video_raw_frames": int(camera_payload["camera_num_video_raw_frames"]),
                    "source_video_path": sample["video_path"],
                    "selected_start_frame": int(sample["selected_start_frame"]),
                    "selected_end_frame": int(sample["selected_end_frame"]),
                    "fps": float(sample["fps"]),
                    "stride": int(sample["stride"]),
                    "offload_config": output_signature,
                }
                if getattr(args, "save_i2v_image_latents", True):
                    payload["i2v_image_latents"] = i2v_image_latents[sample_idx]
                    payload["i2v_fake_image_latents"] = i2v_fake_image_latents[sample_idx]
                if "camera_ref_depth" in camera_payload:
                    payload["camera_ref_depth"] = torch.from_numpy(camera_payload["camera_ref_depth"])
                if "camera_anchor_depth" in camera_payload:
                    payload["camera_anchor_depth"] = torch.from_numpy(camera_payload["camera_anchor_depth"])
                if "camera_depth_hw" in camera_payload:
                    payload["camera_depth_hw"] = torch.from_numpy(camera_payload["camera_depth_hw"])
                if "camera_pose_conf" in camera_payload:
                    payload["camera_pose_conf"] = torch.from_numpy(camera_payload["camera_pose_conf"])
                if "camera_pose_scale_factor" in camera_payload:
                    payload["camera_pose_scale_factor"] = float(camera_payload["camera_pose_scale_factor"])
                if "camera_pose_is_metric" in camera_payload:
                    payload["camera_pose_is_metric"] = int(camera_payload["camera_pose_is_metric"])
                temporary_pt = output_path.with_suffix(".pt.tmp")
                torch.save(payload, temporary_pt)
                os.replace(temporary_pt, output_path)
                camera_json_payload = camera_payload_to_jsonable(sample, camera_payload)
                camera_json_payload["offload_config"] = output_signature
                write_json_atomic(camera_json_path, camera_json_payload)
                saved += 1
                saved_camera_json += 1
                print(f"[INFO] saved {output_path}", flush=True)
                print(f"[INFO] saved {camera_json_path}", flush=True)
            except Exception as exc:
                print(f"skip {sample['uttid']}: failed DA3/camera packaging ({exc})", flush=True)
                failed += 1
    finally:
        if owned_da3_client is not None:
            owned_da3_client.close()
        runtime["vae"] = runtime["vae"].to(device)
        runtime["text_encoder"] = runtime["text_encoder"].to(device)
        runtime["latents_mean"] = runtime["latents_mean"].to(device)
        runtime["latents_std"] = runtime["latents_std"].to(device)
        cleanup_cuda()

    cleanup_cuda()
    return {"saved": saved, "saved_camera_json": saved_camera_json, "skipped_existing": skipped_existing, "failed": failed}


def default_report_path(args):
    if args.num_shards > 1:
        return str(
            Path(args.output_latent_dir)
            / f"offload_report_osp_camera.shard-{args.shard_index:02d}-of-{args.num_shards:02d}.json"
        )
    return str(Path(args.output_latent_dir) / "offload_report_osp_camera.json")


def build_sample_info_from_osp_item(
    item: dict,
    source_json: str,
    nominal_stride: int,
    resolution: int,
    single_height: int,
    single_width: int,
    min_bucket_frames: int,
):
    from worldcrafter.training.utils.data import find_nearest_length_bucket, find_nearest_resolution_bucket

    try:
        cut_start_frame, cut_end_frame = item["cut"]
        crop = item["crop"]
        fps = float(item["fps"])
        prompt = item["cap"][0]
        video_path = item["path"]
    except (KeyError, IndexError, TypeError, ValueError):
        return None

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Each source record needs a nonempty existing cap[0] string")

    if not isinstance(video_path, str) or not os.path.exists(video_path):
        return None

    num_frame = int(cut_end_frame) - int(cut_start_frame)
    if num_frame < min_bucket_frames:
        return None

    # Keep crop/cut semantics exactly aligned with get_short-latents_streaming.py.
    width = int(crop[1]) - int(crop[0])
    height = int(crop[3]) - int(crop[2])
    bucket_height, bucket_width = find_nearest_resolution_bucket(height, width, resolution=resolution)
    if (bucket_height, bucket_width) != (single_height, single_width):
        return None

    stride = int(nominal_stride)
    if stride <= 0:
        return None

    effective_num_frame = (num_frame + stride - 1) // stride
    bucket_num_frame = find_nearest_length_bucket(effective_num_frame, stride=stride)
    if bucket_num_frame < min_bucket_frames:
        return None

    required_frames = required_raw_frames(bucket_num_frame, stride)
    if required_frames > num_frame:
        return None

    stem = Path(video_path).stem
    path_hash = hashlib.sha1(video_path.encode("utf-8")).hexdigest()[:10]
    uttid = f"{stem}_{path_hash}_{int(cut_start_frame)}-{int(cut_end_frame)}"
    return {
        "uttid": uttid,
        "prompt": prompt,
        "caption_record": item,
        "video_path": video_path,
        "fps": fps,
        "stride": stride,
        "cut_start_frame": int(cut_start_frame),
        "cut_end_frame": int(cut_end_frame),
        "crop": crop,
        "bucket_num_frame": bucket_num_frame,
        "bucket_height": bucket_height,
        "bucket_width": bucket_width,
        "source_json": source_json,
        "motion_existing": item.get("motion"),
        "source_group": item.get("source_group"),
        "source_family": item.get("source_family"),
        "rule": item.get("rule"),
    }


def main():
    args = parse_args()
    if args.num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {args.num_shards}")
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError(f"shard_index must be in [0, {args.num_shards}), got {args.shard_index}")
    if args.stride != 1:
        raise ValueError(f"This OSP offload currently fixes stride=1, got {args.stride}")
    if args.bucket_flush_size < 1:
        raise ValueError(f"bucket_flush_size must be >= 1, got {args.bucket_flush_size}")

    tmp_dir = Path(args.tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if args.da3_helper_mode:
        prepare_model_paths(args, need_helios=False, need_da3=True)
        run_da3_helper(args)
        return
    if args.da3_server_mode:
        prepare_model_paths(args, need_helios=False, need_da3=True)
        run_da3_server(args)
        return

    prepare_model_paths(args, need_helios=True, need_da3=True)

    print(f"[INFO] source_json={args.source_json}", flush=True)
    print(f"[INFO] output_latent_dir={args.output_latent_dir}", flush=True)
    print(f"[INFO] output_camera_json_dir={args.output_camera_json_dir}", flush=True)
    print(f"[INFO] helios_device={args.device} da3_device={args.da3_device}", flush=True)
    print(f"[INFO] shard_index={args.shard_index} num_shards={args.num_shards}", flush=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    source_json = Path(args.source_json)

    latent_pools = defaultdict(list)
    output_dir = Path(args.output_latent_dir)
    camera_json_dir = Path(args.output_camera_json_dir)
    report = {
        "config": {
            "source_json": args.source_json,
            "output_latent_dir": args.output_latent_dir,
            "output_camera_json_dir": args.output_camera_json_dir,
            "device": args.device,
            "da3_device": args.da3_device,
            "batch_size": args.batch_size,
            "stride": args.stride,
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "single_height": args.single_height,
            "single_width": args.single_width,
            "min_bucket_frames": args.min_bucket_frames,
            "latent_window_size": args.latent_window_size,
            "save_i2v_image_latents": args.save_i2v_image_latents,
            "bucket_flush_size": args.bucket_flush_size,
            "max_read_retries": args.max_read_retries,
            "max_samples": args.max_samples,
            "tmp_dir": args.tmp_dir,
        },
        "totals": {
            "seen": 0,
            "accepted": 0,
            "saved_pt": 0,
            "saved_camera_json": 0,
            "skipped_existing_pt": 0,
            "failed_pt": 0,
        },
        "per_bucket": {},
    }

    runtime = None
    da3_client = None

    def ensure_runtime():
        nonlocal runtime, da3_client
        if runtime is None:
            runtime = load_models(args.pretrained_model_name_or_path, args.device)
        if da3_client is None and args.da3_device != args.device:
            da3_client = PersistentDA3Client(args)

    def flush_bucket(bucket_key, force=False):
        nonlocal runtime, da3_client
        bucket_samples = latent_pools[bucket_key]
        if not bucket_samples:
            return
        target_size = args.batch_size if not force else 1
        while len(bucket_samples) >= target_size and (force or len(bucket_samples) >= args.batch_size):
            ensure_runtime()
            batch_samples = bucket_samples[: args.batch_size]
            del bucket_samples[: args.batch_size]
            print(f"[INFO] processing bucket {bucket_key} batch={len(batch_samples)} pending={len(bucket_samples)}", flush=True)
            sample_pool = batch_samples + bucket_samples
            stats = encode_and_save_bucket(
                samples=batch_samples,
                runtime=runtime,
                args=args,
                da3_client=da3_client,
                sample_pool=sample_pool,
            )
            report["totals"]["saved_pt"] += stats["saved"]
            report["totals"]["saved_camera_json"] += stats["saved_camera_json"]
            report["totals"]["skipped_existing_pt"] += stats["skipped_existing"]
            report["totals"]["failed_pt"] += stats["failed"]
            if not force and len(bucket_samples) < args.batch_size:
                break

    for row_idx, item in enumerate(iter_json_array(source_json)):
        report["totals"]["seen"] += 1
        if row_idx % args.num_shards != args.shard_index:
            continue

        sample = build_sample_info_from_osp_item(
            item=item,
            source_json=source_json.name,
            nominal_stride=args.stride,
            resolution=args.resolution,
            single_height=args.single_height,
            single_width=args.single_width,
            min_bucket_frames=args.min_bucket_frames,
        )
        if sample is None:
            continue

        report["totals"]["accepted"] += 1
        output_path = latent_output_path(output_dir, sample)
        camera_json_path = camera_json_output_path(camera_json_dir, sample)
        if existing_outputs_match_current_config(output_path, camera_json_path, args, sample):
            report["totals"]["skipped_existing_pt"] += 1
            continue

        bucket_key = (
            sample["bucket_num_frame"],
            sample["bucket_height"],
            sample["bucket_width"],
            sample["stride"],
        )
        latent_pools[bucket_key].append(sample)
        report["per_bucket"].setdefault(str(bucket_key), 0)
        report["per_bucket"][str(bucket_key)] += 1

        if len(latent_pools[bucket_key]) >= args.bucket_flush_size:
            flush_bucket(bucket_key, force=False)

        if args.max_samples is not None and report["totals"]["accepted"] >= args.max_samples:
            break

    print(f"[INFO] scanned={report['totals']['seen']} accepted={report['totals']['accepted']} buckets={len(latent_pools)}", flush=True)

    for bucket_key in list(latent_pools.keys()):
        flush_bucket(bucket_key, force=True)

    report_path = Path(args.report_path or default_report_path(args))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, ensure_ascii=True, indent=2)
    print(f"wrote report to {report_path}", flush=True)

    if da3_client is not None:
        da3_client.close()
    if report["totals"]["accepted"] == 0 or report["totals"]["failed_pt"]:
        raise RuntimeError(f"Precomputation incomplete; inspect {report_path}")


if __name__ == "__main__":
    main()
