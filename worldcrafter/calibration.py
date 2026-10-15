"""Cached GeoCalib CPU worker, shared by CLI and interactive sessions."""
import atexit
import hashlib
import io
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import threading
import time

from PIL import Image, ImageOps

from .gravity import gravity_frame

VERSION = "geocalib-1dedb29f94fc11b896c5f572005978aeebccd67f-pinhole-v1"
_lock = threading.Lock()
_process = _connection = None


def _worker(connection):
    # Import and limit CPU threads in this process only, never in the GPU owner.
    try:
        # Receive before loading weights so the parent's send cannot block on
        # a slow download; its response timeout covers model initialization.
        data = connection.recv_bytes()
        import numpy as np
        import torch
        from geocalib import GeoCalib
        torch.set_num_threads(4)
        model = GeoCalib(weights="pinhole").eval().cpu()
        while True:
            start = time.monotonic()
            image = Image.open(io.BytesIO(data)).convert("RGB")
            tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255
            result = model.calibrate(tensor)
            up = result["gravity"].vec3d[0].detach().cpu().tolist()
            gravity_frame(up)
            report = dict(
                version=VERSION, world_up=up,
                elevation_deg=float(-torch.rad2deg(result["gravity"].pitch[0])),
                roll_deg=float(torch.rad2deg(result["gravity"].roll[0])),
                gravity_uncertainty_deg=float(torch.rad2deg(result["gravity_uncertainty"]).max()),
                estimated_horizontal_fov_deg=float(torch.rad2deg(result["camera"].hfov[0])),
                seconds=time.monotonic() - start,
            )
            connection.send(report)
            data = connection.recv_bytes()
    except EOFError:
        pass
    except Exception as exc:
        connection.send({"error": f"{type(exc).__name__}: {exc}"})
    finally:
        connection.close()


def close_calibrator():
    global _process, _connection
    if _connection is not None:
        _connection.close()
    if _process is not None:
        if _process.is_alive():
            _process.terminate()
        _process.join(timeout=5)
        if _process.is_alive():
            _process.kill()
            _process.join(timeout=5)
    _process = _connection = None


atexit.register(close_calibrator)


def calibrate_image(image):
    """Return calibration, or disable alignment when calibration is unavailable."""
    try:
        report = _calibrate_image(image)
        return dict(report, enabled=True, status="aligned")
    except Exception as exc:
        return dict(enabled=False, status="fallback", world_up=None, version=VERSION,
                    reason=f"{type(exc).__name__}: {exc}")


def _calibrate_image(image):
    """Calibrate the actual RGB input; cache includes pixels, size and version.

    Raises on failure; uncertainty is retained as calibration metadata.
    Focal estimates are diagnostic only and do not change generation FOV.
    """
    global _process, _connection
    if not isinstance(image, Image.Image):
        with Image.open(image) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
    image = image.convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    data = buffer.getvalue()
    image_hash = hashlib.sha256(data).hexdigest()
    key = hashlib.sha256(VERSION.encode() + data).hexdigest()
    root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "worldcrafter/gravity"
    path = root / f"{key}.json"
    with _lock:
        if path.is_file():
            report = json.loads(path.read_text())
            if report.get("version") != VERSION or report.get("image_sha256") != image_hash:
                raise RuntimeError("Invalid gravity calibration cache")
        else:
            if _process is None or not _process.is_alive():
                close_calibrator()
                ctx = mp.get_context("spawn")
                _connection, child = ctx.Pipe()
                _process = ctx.Process(target=_worker, args=(child,), daemon=True)
                _process.start()
                child.close()
            try:
                _connection.send_bytes(data)
                if not _connection.poll(120):
                    raise TimeoutError("GeoCalib calibration timed out")
                report = _connection.recv()
                if "error" in report:
                    raise RuntimeError(report["error"])
            except Exception as exc:
                close_calibrator()
                raise RuntimeError(
                    f"Gravity calibration failed: {exc}. Install the project dependencies, "
                    "retry, or explicitly disable gravity alignment."
                ) from exc
            report.update(image_sha256=image_hash, image_size=list(image.size))
            root.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(f".{os.getpid()}.tmp")
            temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            temporary.replace(path)
        gravity_frame(report["world_up"])
        uncertainty = float(report["gravity_uncertainty_deg"])
        if not math.isfinite(uncertainty) or uncertainty < 0:
            raise RuntimeError("Invalid gravity uncertainty estimate")
        return report
