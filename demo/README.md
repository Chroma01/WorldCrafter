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

## GPU modes

Use one, two, four or eight GPUs on the same machine. Select visible device
indices with `--devices`; one GPU is the default.

| Devices | Transformer backend (`--backend auto`) | VAE (`--vae-mode auto`) |
| --- | --- | --- |
| `0` | Single GPU | Single GPU |
| `0,1` | Query parallel | Single GPU |
| `0,1,2,3` | Context parallel | Single GPU |
| `0,1,2,3,4,5,6,7` | Context parallel | Eight-GPU pipeline |

Run from the repository root with the environment activated and weights ready:

```bash
# One GPU
python -m demo --devices 0

# Two GPUs
python -m demo --devices 0,1

# Four GPUs
python -m demo --devices 0,1,2,3

# Eight GPUs
python -m demo --devices 0,1,2,3,4,5,6,7
```

All commands accept `--model-path`, `--output-dir`, `--host` and `--port` as in
the single-GPU example. If `CUDA_VISIBLE_DEVICES` is set, `--devices` indexes
that visible list. For example, `CUDA_VISIBLE_DEVICES=4,5` with `--devices 0,1`
uses physical GPUs 4 and 5. The launcher starts and coordinates the workers;
run `python -m demo` once for each selected GPU group.

Query parallelism distributes attention queries across two complete model
replicas. Context parallelism distributes history and current tokens across
Transformer blocks, exchanging keys and values between workers. Each worker
keeps the model weights, so enough memory for a model replica is required on
each selected GPU. Context parallelism uses FlashAttention 3 and Triton kernels;
use the project's pinned dependencies and compatible GPUs with working NCCL
communication.

The eight-GPU VAE pipeline distributes decoding layers across the selected
GPUs and preserves causal state between chunks. Camera computations and VAE
decoding use FP32. Sampling steps, model routing, camera controls and the UI are
shared across modes; distributed floating-point operations can produce small
numerical differences.

### Backend options

- `--backend auto` selects the Transformer backend from the table.
- `--backend query` requires two GPUs.
- `--backend context` supports two, four or eight GPUs. To select it explicitly
  on two GPUs, use `python -m demo --devices 0,1 --backend context`.
- `--vae-mode auto` selects the VAE mode from the table. `single` keeps decoding
  on the primary GPU; `pipeline8` requires eight GPUs and the context backend.
- `--kv-fusion rope` enables fused RoPE and KV packing for the context backend;
  `off` provides an unfused comparison path.
- `--no-compile` disables Transformer compilation for troubleshooting. The
  eight-GPU VAE pipeline uses native FP32 operations independently of this flag.

Startup loads the models. The eight-GPU VAE also warms up and checks its output
against the original decoder before becoming ready. Transformer compilation
can add work on the first generation or when tensor shapes change. Wait for
`/api/health` to report `ready: true` and `error: null`; it also reports the
actual `gpu_count`, `gpu_mode` and `vae_decode` configuration. Initialization
errors are reported instead of silently selecting a different backend.

Workers use a private CPU control channel while waiting for user input and
NCCL during computation. Set `TORCHINDUCTOR_CACHE_DIR` and `TRITON_CACHE_DIR` to
local disk when configuring caches; context workers use separate rank
subdirectories. Keep generated sessions in a writable output directory.

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
