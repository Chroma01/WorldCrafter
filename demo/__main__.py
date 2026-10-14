"""Launch the I2V demo on one, two, four or eight GPUs."""

import argparse
import os
from pathlib import Path


def run_distributed(command):
    """Forward terminal signals and wait until torchrun has reaped its workers."""
    import signal
    import subprocess
    import time

    process = subprocess.Popen(command, start_new_session=True)
    stopping_at = None

    def forward_signal(signum, _frame):
        nonlocal stopping_at
        if stopping_at is None:
            stopping_at = time.monotonic()
        try:
            process.send_signal(signum)
        except ProcessLookupError:
            pass

    previous = {sig: signal.signal(sig, forward_signal)
                for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        while True:
            try:
                code = process.wait(timeout=1)
                return code if code >= 0 else 128 - code
            except subprocess.TimeoutExpired:
                if stopping_at is not None and time.monotonic() - stopping_at >= 60:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
                    return 137
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "weights/WorldCrafter-Fast",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "output/demo",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--devices", default="0", help="One, two, four or eight visible GPU indices")
    parser.add_argument("--backend", choices=("auto", "query", "context"), default="auto")
    parser.add_argument("--vae-mode", choices=("auto", "single", "pipeline4", "pipeline8"), default="auto")
    parser.add_argument("--performance-profile", choices=("auto", "native", "throughput"), default="auto")
    parser.add_argument("--kv-fusion", choices=("off", "rope"), default="rope")
    parser.add_argument("--no-compile", action="store_true", help="Disable DiT and VAE compilation for validation")
    args = parser.parse_args()
    devices = args.devices.split(",")
    if len(devices) not in (1, 2, 4, 8) or len(set(devices)) != len(devices) or not all(d.isdigit() for d in devices):
        parser.error("--devices must contain 1, 2, 4 or 8 distinct GPU indices")
    backend = args.backend
    if backend == "auto":
        backend = "context" if len(devices) >= 4 else ("query" if len(devices) == 2 else "single")
    if backend == "query" and len(devices) != 2:
        parser.error("Query parallelism requires two GPUs")
    if backend == "context" and len(devices) not in (2, 4, 8):
        parser.error("Context parallelism requires 2, 4 or 8 GPUs")
    vae = args.vae_mode
    if vae == "auto": vae = f"pipeline{len(devices)}" if len(devices) >= 4 else "single"
    if vae.startswith("pipeline") and (int(vae[-1]) != len(devices) or backend != "context"):
        parser.error("VAE pipeline size must match the GPU count and requires the context backend")
    optimized = args.performance_profile == "throughput" or (args.performance_profile == "auto" and len(devices) >= 4)
    if optimized and (len(devices) not in (4, 8) or backend != "context"):
        parser.error("The throughput profile requires four or eight context-parallel GPUs")
    if args.mock and backend == "context":
        parser.error("Context parallelism requires actual GPUs; use single-device mock for UI tests")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        available = visible.split(",")
        if any(int(device) >= len(available) for device in devices):
            parser.error("--devices exceeds the GPUs in CUDA_VISIBLE_DEVICES")
        devices = [available[int(device)] for device in devices]
    os.environ.update(
        WORLDCRAFTER_DEMO_MODEL=str(args.model_path.resolve()),
        WORLDCRAFTER_DEMO_DATA=str(args.output_dir.resolve()),
        WORLDCRAFTER_DEMO_MOCK=str(int(args.mock)),
        CUDA_VISIBLE_DEVICES=",".join(devices),
        WORLDCRAFTER_DEMO_GPUS=str(len(devices)),
        WORLDCRAFTER_DEMO_BACKEND=backend,
        WORLDCRAFTER_VAE_MODE=vae,
        WORLDCRAFTER_KV_FUSION=args.kv_fusion,
        WORLDCRAFTER_DEMO_COMPILE=str(int(not args.no_compile)),
        WORLDCRAFTER_DEMO_OPTIMIZED=str(int(optimized)),
    )
    if backend == "context":
        import sys
        import tempfile
        with tempfile.TemporaryDirectory(prefix="wc-control-") as directory:
            os.environ["WORLDCRAFTER_CONTROL_SOCKET"] = directory+"/control.sock"
            code = run_distributed([
                sys.executable, "-m", "torch.distributed.run", "--standalone",
                "--nproc-per-node", str(len(devices)), "--module", "demo.distributed_server",
                "--host", args.host, "--port", str(args.port),
            ])
        raise SystemExit(code)
    import uvicorn

    uvicorn.run("demo.server:app", host=args.host, port=args.port, ws="websockets")


if __name__ == "__main__":
    main()
