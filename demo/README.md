# WorldCrafter Interactive Demo

Explore a scene from an image with keyboard camera controls, powered by
WorldCrafter-Fast. Compilation is enabled by default, so the first generation
takes longer.

## Run

Follow the [environment setup](../README.md#2-environment), which includes
FFmpeg, and download the [model weights](../README.md#3-model-weights).
Place the complete `WorldCrafter-Fast` folder under `weights/`, activate your
uv or conda environment, and run from the repository root:

```bash
python -m demo \
  --model-path weights/WorldCrafter-Fast \
  --output-dir output/demo
```

The demo has been validated on one and two NVIDIA H200 GPUs. Use `--devices 0`
for one GPU (the default), or `--devices 0,1` for two GPUs. Device indices refer
to the GPUs visible through `CUDA_VISIBLE_DEVICES` when it is set. Two-GPU mode
splits attention queries while keeping a complete model on each GPU; both modes
use the same precision, sampling settings, and controls.

## Start exploring

Open `http://localhost:8080` in your browser, then:

1. Select a scene preset to load its image and prompt, or upload your own image.
2. For an uploaded image, choose **First-person** or **Third-person** to generate
   a prompt with Qwen3-VL-4B-Instruct. You can edit the generated description
   before starting. See [model weights](../README.md#3-model-weights) for the
   Qwen download command.
3. Click **Explore**, then use the camera controls below. Generation begins
   when you enter an action.

Each action generates a 33-frame chunk. Pause, resume, restart, and video
download are available on the page. One session can generate at a time.
Video chunks, camera trajectories, and session metadata are saved to the
output directory. The interface defaults to English and can be switched to Chinese.

## Camera controls

| Input | Action |
| --- | --- |
| W / S | Move forward / backward on the horizontal plane |
| A / D | Move left / right on the horizontal plane |
| Q / E | Move up / down along the world vertical axis |
| Left / right arrows | Turn left / right in place |
| Up / down arrows | Look up / down in place |
| I / K | Orbit up / down around a point in front of the camera |
| J / L | Orbit left / right around a point in front of the camera |
| Space | Pause / resume generation |
| Esc or switching away from the window | Clear pending movement |

Each chunk uses one action. Before a chunk starts, a new key replaces the
pending action. Tap for one chunk or hold for continued movement. Generation
waits when there is no pending action. The active key stays highlighted while
its chunk is being generated.

Arrow keys rotate the camera in place. I/J/K/L move the camera around an orbit
center, with directions describing the camera's movement. Consecutive orbit
actions keep the same center. Moving, turning in place, or changing the radius
establishes a new center from the current pose.

The demo and [script camera actions](../test/README.md#camera-inputs) use the same
motion rules. Looking up or down does not change the height of W/S/A/D movement.

## Motion settings

Adjust the camera settings on the page:

| Setting | Range | Default |
| --- | --- | --- |
| Move distance (W/S/A/D) | 1–5 m per chunk | 2 m |
| Rise / descend distance (Q/E) | 1–5 m per chunk | 2 m |
| Rotation angle (arrow keys and I/J/K/L) | 10–45° per chunk | 30° |
| Orbit radius | 1–5 m | 2 m |

The orbit radius sets the distance from the camera to its orbit center.

## Remote access

The service listens on `127.0.0.1:8080` by default. When running on a remote
machine, forward the port from your local machine:

```bash
ssh -N -L 8080:127.0.0.1:8080 <host>
```

Then open `http://localhost:8080` locally. Use `--host` and `--port` to change
the server's listen address and port.

Run one server process per selected GPU group; multiple web workers would each
load a separate model. The demo has no built-in authentication, so use an
authenticated proxy when making it publicly accessible.
