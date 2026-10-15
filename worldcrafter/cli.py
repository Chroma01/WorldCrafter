from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Sequence

from .caption import AUTO_PROMPTS, DEFAULT_CAPTION_MODEL


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "weights" / "WorldCrafter-Base"
DEFAULT_I2V_CASE = ROOT / "test" / "I2V" / "00_cat_robot_vacuum"
DEFAULT_T2V_CASE = ROOT / "test" / "T2V" / "00_red_balloon"
DEFAULT_IMAGE = DEFAULT_I2V_CASE / "image.png"
DEFAULT_I2V_CAMERA = DEFAULT_I2V_CASE / "camera.npy"
DEFAULT_T2V_CAMERA = DEFAULT_T2V_CASE / "camera.npy"
DEFAULT_I2V_PROMPT = DEFAULT_I2V_CASE / "prompt.txt"
DEFAULT_T2V_PROMPT = DEFAULT_T2V_CASE / "prompt.txt"
DEFAULT_NEGATIVE_PROMPT = ROOT / "test" / "negative_prompt.txt"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="WorldCrafter camera-controlled image-to-video and text-to-video inference"
    )
    parser.add_argument("--mode", choices=("i2v", "t2v"), default="i2v")
    parser.add_argument("--model-type", choices=("base", "fast"), default="base")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument(
        "--local-camera-path",
        type=Path,
        help="Optional precomputed chunk-local UCPE poses for fast; --camera-path always supplies global metric poses",
    )
    parser.add_argument("--image-path", type=Path)
    parser.add_argument(
        "--gravity-align", action=argparse.BooleanOptionalAction, default=True,
        help="Align I2V camera controls to image gravity (default: on); disable for calibrated trajectories",
    )
    camera = parser.add_mutually_exclusive_group()
    camera.add_argument("--camera-path", type=Path, help="Global c2w trajectory (.npy)")
    camera.add_argument("--actions", help='Camera actions, e.g. "forward1x2 yaw_left30x3 backward1"')
    camera.add_argument("--actions-file", type=Path, help="TXT file of camera actions")
    parser.add_argument("--orbit-radius", type=float, help="Metric radius for orbit actions (default: 2)")
    parser.add_argument(
        "--prompt", help="Prompt text, a .txt file, auto-first-person, or auto-third-person"
    )
    parser.add_argument(
        "--caption-model",
        help="Qwen3-VL model ID or local directory for automatic prompts",
    )
    parser.add_argument("--negative-prompt")
    parser.add_argument(
        "--negative-prompt-path", type=Path, default=DEFAULT_NEGATIVE_PROMPT
    )
    parser.add_argument("--output-path", type=Path)
    parser.add_argument("--chunk-output-dir", type=Path)
    parser.add_argument("--state-output-dir", type=Path)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--num-chunks", type=int)
    parser.add_argument("--stop-after-chunk", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--num-inference-steps", type=int)
    parser.add_argument("--guidance-scale", type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--image-noise-sigma-min", type=float, default=0.111)
    parser.add_argument("--image-noise-sigma-max", type=float, default=0.135)
    parser.add_argument("--camera-x-fov", type=float, default=100.0)
    parser.add_argument("--camera-xi", type=float, default=0.0)
    parser.add_argument("--memory-fov-h-deg", type=float, default=100.0)
    parser.add_argument("--memory-fov-v-deg", type=float, default=71.13349068444832)
    parser.add_argument("--memory-fov-samples-per-axis", type=int, default=10)
    parser.add_argument(
        "--attention-backend",
        choices=("native", "auto", "flash_hub", "_flash_3_hub"),
        default="native",
    )
    parser.add_argument(
        "--enable-compile",
        action="store_true",
        help="Enable torch.compile (off by default); Fast compiles both transformer block stacks",
    )
    return parser


def _read_text(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = path.read_text(encoding="utf-8").strip()
    if not value:
        raise ValueError(f"prompt file is empty: {path}")
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    caption_path = ROOT / "weights" / "Qwen3-VL-4B-Instruct"
    args.caption_model = args.caption_model or (
        str(caption_path) if caption_path.is_dir() else DEFAULT_CAPTION_MODEL
    )
    fast = args.model_type == "fast"
    args.model_path = args.model_path or ROOT / "weights" / (
        "WorldCrafter-Fast" if fast else "WorldCrafter-Base"
    )
    if args.num_inference_steps is None:
        args.num_inference_steps = 6 if fast else 50
    if args.guidance_scale is None:
        args.guidance_scale = 1.0 if fast else 5.0
    if fast and (args.num_inference_steps != 6 or args.guidance_scale != 1.0):
        raise ValueError(
            "Fast requires CFG=1 and six regular steps; the first T2V chunk uses twelve steps"
        )
    if fast and (args.resume_from or args.state_output_dir):
        raise ValueError("Fast does not support resume/state export")
    if args.num_chunks is not None and args.num_chunks <= 0:
        raise ValueError("--num-chunks must be positive")
    if args.stop_after_chunk is not None and args.stop_after_chunk < 0:
        raise ValueError("--stop-after-chunk must be non-negative")
    if args.resume_from is not None and args.chunk_output_dir is None:
        raise ValueError("--resume-from requires --chunk-output-dir")

    args.camera_events = None
    args.camera_options = {}
    if args.actions is not None or args.actions_file is not None:
        from .camera import count_chunks, parse_trajectory

        if args.local_camera_path is not None:
            raise ValueError("--local-camera-path cannot be combined with camera actions")
        text = args.actions if args.actions is not None else args.actions_file.read_text(encoding="utf-8-sig")
        args.camera_events, args.camera_options = parse_trajectory(text)
        if args.orbit_radius is not None:
            args.camera_options["orbit_radius"] = args.orbit_radius
        if args.num_chunks is not None:
            total = count_chunks(args.camera_events)
            if args.num_chunks > total:
                raise ValueError(f"Actions provide {total} chunks, but {args.num_chunks} were requested")
    elif args.camera_path is None:
        args.camera_path = (
            DEFAULT_I2V_CAMERA if args.mode == "i2v" else DEFAULT_T2V_CAMERA
        )
    if args.orbit_radius is not None and args.camera_events is None:
        raise ValueError("--orbit-radius requires --actions or --actions-file")
    prompt_file = None
    if args.prompt is None:
        prompt_file = DEFAULT_I2V_PROMPT if args.mode == "i2v" else DEFAULT_T2V_PROMPT
        args.prompt = _read_text(prompt_file)
    else:
        args.prompt = args.prompt.strip()
        if args.prompt.lower().endswith(".txt"):
            prompt_file = Path(args.prompt).expanduser()
            args.prompt = _read_text(prompt_file)
        elif not args.prompt:
            raise ValueError("prompt must not be empty")

    if args.negative_prompt is None:
        args.negative_prompt = _read_text(args.negative_prompt_path)
    if args.mode == "i2v":
        args.image_path = args.image_path or DEFAULT_IMAGE
    elif args.image_path is not None:
        raise ValueError("--image-path is only valid with --mode i2v")

    if args.prompt in AUTO_PROMPTS:
        if args.mode != "i2v":
            raise ValueError("Automatic prompts require --mode i2v and an input image")
        if not args.image_path.is_file():
            raise FileNotFoundError(args.image_path)

    if args.output_path is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        source = args.image_path if args.mode == "i2v" else prompt_file
        name = f"{args.mode}_{source.stem}" if source is not None else args.mode
        args.output_path = ROOT / "output" / run_id / f"{name}.mp4"
    return args


def prepare_camera(args: argparse.Namespace) -> None:
    args.gravity_calibration = None
    world_up = None
    if args.mode == "i2v" and args.gravity_align:
        if args.local_camera_path is not None:
            args.gravity_calibration = dict(
                enabled=False, status="fallback", world_up=None,
                reason="Preserving the supplied global/local camera pair",
            )
        else:
            from PIL import Image, ImageOps
            from .calibration import calibrate_image
            with Image.open(args.image_path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB").resize((args.width, args.height))
            print("[worldcrafter] calibrating image gravity")
            args.gravity_calibration = calibrate_image(image)
            world_up = args.gravity_calibration["world_up"]
    if args.camera_events is not None:
        from .camera import build_trajectory, save_trajectory

        camera, records = build_trajectory(args.camera_events, world_up=world_up, **args.camera_options)
        directory = args.output_path.parent / f"{args.output_path.stem}_trajectory"
        args.camera_path = save_trajectory(
            directory, camera, records, fps=args.fps,
            events=args.camera_events, options=args.camera_options,
        )
        print(f"[worldcrafter] saved {len(records)} camera chunks to {args.camera_path}")
    elif world_up is not None:
        import numpy as np
        from .gravity import align_trajectory
        original = args.camera_path
        camera = align_trajectory(np.load(original, allow_pickle=False), world_up)
        directory = args.output_path.parent / f"{args.output_path.stem}_trajectory"
        directory.mkdir(parents=True, exist_ok=True)
        args.camera_path = directory / "camera_gravity.npy"
        if args.camera_path.resolve() == original.resolve():
            raise ValueError("Aligned camera output would overwrite its input; choose a different output path")
        np.save(args.camera_path, camera)
        args.gravity_calibration["source_camera_path"] = str(original)
    if args.gravity_calibration is not None:
        import json
        directory = args.output_path.parent / f"{args.output_path.stem}_trajectory"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "calibration.json").write_text(
            json.dumps(args.gravity_calibration, indent=2) + "\n"
        )


__all__ = ["build_parser", "parse_args", "prepare_camera"]
