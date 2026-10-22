# Fine-tune WorldCrafter-Base

Continue the **released WorldCrafter-Base** model using precomputed video latents,
existing caption embeddings and Depth Anything 3 (DA3) camera estimates. This
Stage-1 flow-matching recipe fine-tunes Base; training or distilling WorldCrafter-Fast
is outside its scope. See [data preparation](data_preparation.md).

## What is trained

| Component | State |
| --- | --- |
| Existing DiT LoRA | Trainable; rank 128, alpha 128, dropout 0; 623,157,248 parameters |
| Existing UCPE camera adapter | Trainable; 525,388,800 parameters |
| DiT base weights | Frozen |
| RepEncoder | Frozen, evaluated only on full-memory steps |
| VAE and text encoder | Used offline; not loaded by the training loop |

Both adapters are initialized from Base, with a fresh AdamW optimizer. `--resume`
instead restores optimizer, adapter weights, progress and each rank's RNG state.
The exact tensor/parameter boundaries are checked when loading the model.

The recipe keeps the reference Stage-1 implementation: 640x384 pixels, 33 RGB
frames / 9 latent frames per target chunk, nine causal source views, four frozen
RepEncoder memory views at target slots `[2,4,6,8]`, and a mid2 + recent1 tail.
Source selection uses the native training sampler, not inference FOV retrieval.

History mode sampling is sequential: `t2v=0.4`, then `v2v=0.4` conditional on not
selecting t2v. The effective probabilities are **40% t2v, 24% v2v, 36% full**.
The mode is synchronized across ranks and held for an accumulated optimizer step.
Corruption and saturation retain the reference operation order and constants in
[`reference_noise.json`](../worldcrafter/training/reference_noise.json).

The reference optimizer is AdamW, LR `3e-5` (constant), betas `(0.9,0.999)`,
epsilon `1e-8`, weight decay `0.01`, no gradient clipping, bf16 compute with fp32
trainable parameters. The timestep distribution and weighted flow loss use the
reference shift-5 scheduler; dynamic shifting is disabled during training.

## Environment and models

From the repository root:

```bash
uv sync --project uvenv/training --frozen
source uvenv/training/.venv/bin/activate
```

For network-mounted repositories, set `UV_PROJECT_ENVIRONMENT` and `UV_CACHE_DIR`
to directories on a local SSD before installation, then activate that environment.

The environment uses native PyTorch attention and does not require
DeepSpeed. The launcher supports PyTorch DDP. Memory requirements must be measured
on the intended hardware; this recipe has roughly 1.15 billion trainable parameters.
The single-H20 integration run at the documented resolution peaked at approximately
59.12 GiB of allocated GPU memory with batch size one.

Place `WorldCrafter-Base` beside `WorldCrafter-Fast` under `weights/`, as in the
inference instructions. Base uses Fast's shared VAE, text encoder, tokenizer,
scheduler and RepEncoder; Fast's transformer and adapters are not used.

## Data

Download the video and camera-JSON release from
[Hugging Face](https://huggingface.co/datasets/Drexubery/WorldCrafter), then follow
the [data preparation guide](data_preparation.md) to encode complete `.pt` features
locally into `data/WorldCrafter/features/`. Precomputed `.pt` files are not distributed.
The supplied camera annotations can be reused without rerunning DA3. The downloaded
`metadata/valid_chunk_indices.json` identifies the 20,434-record OSP-v2 subset,
corresponding to 19,972 unique source videos.
Alternatively, prepare your own captioned videos and build a matching index.

```bash
python train.py --feature-root data/WorldCrafter/features \
  --valid-chunks data/WorldCrafter/metadata/valid_chunk_indices.json --check-data
```

Only local filesystem inputs are supported. Use trusted `.pt` files: the original
format includes Python/PIL metadata. Missing files, invalid camera values and
unavailable causal source views fail explicitly. Keep complete histories, even if
only some chunks are valid targets. Targets without a preceding chunk are excluded.

The published index is an all-training pool, with no held-out evaluation claim.
To create an independent evaluation set, split by original video identity, keeping
all cuts from one video in the same split. `--pool` accepts a text file with one
feature **basename** per line, restricted to the valid index.

### Dataset and DataLoader

[`OSPFeatureDataset`](../worldcrafter/training/dataset.py) reads the prepared
`.pt` files containing video latents, caption embeddings and camera annotations.
It selects eligible target chunks from the downloaded index and samples preceding
views for causal memory, while keeping each video's complete feature history.
Raw videos are encoded offline using the [data preparation tools](data_preparation.md).

The [training entry point](../worldcrafter/training/train.py) constructs a PyTorch
`DataLoader` with a `DistributedSampler` for rank-wise data allocation and epoch
shuffling. It uses batch size one per GPU, gradient accumulation, and
`num_workers=0` to preserve deterministic sampling on checkpoint resume.
The dataset and loader are included in the eight-H200 training/resume check.

## Launch and resume

A single-node example with the reference effective batch of 32:

```bash
NPROC_PER_NODE=8 bash scripts/training/finetune_base.sh \
  --gradient-accumulation-steps 4 --max-steps 500 \
  --output-dir outputs/base-osp-v2
```

The reference topology was 32 GPUs, batch 1 per GPU, accumulation 1. For that
topology run the launcher once per node with `NNODES=4 NPROC_PER_NODE=8`, a common
`MASTER_ADDR` and distinct `NODE_RANK=0..3`. Eight GPUs with accumulation 4 preserves
the effective batch size, but is a different execution topology and is not claimed
to be bitwise identical. LR is never automatically scaled with GPU count.

```bash
NPROC_PER_NODE=8 bash scripts/training/finetune_base.sh \
  --gradient-accumulation-steps 4 --max-steps 1000 \
  --output-dir outputs/base-osp-v2 --resume latest
```

Use the same world size, data and training recipe when resuming. Changing the
maximum step or checkpoint interval is allowed. Checkpoints contain trainable
parameters and optimizer/RNG states, not copies of the frozen backbone. A
`COMPLETE` marker is published only after every rank has saved its state.
Never manually rename an `.incomplete` directory to a complete checkpoint.

For a bounded integration check on one GPU:

```bash
python train.py --feature-root /path/to/small/features \
  --valid-chunks /path/to/small/valid.json --max-steps 3 \
  --checkpoint-every 3 --smoke-modes --output-dir outputs/base-smoke
```

`--smoke-modes` deliberately cycles t2v, v2v and full memory; it is restricted to
one GPU and at most six steps, and is not a production recipe. It preserves model
size and input resolution. The normal configuration uses stochastic mode sampling.

## Export and inference

```bash
python tools/export_finetuned_model.py \
  --base-model weights/WorldCrafter-Base \
  --checkpoint outputs/base-osp-v2/checkpoint-500 \
  --output weights/WorldCrafter-Base-finetuned
python inference.py --model-type base \
  --model-path weights/WorldCrafter-Base-finetuned --output-path outputs/finetuned.mp4
```

Export copies the frozen Base transformer and saves the updated adapters in the
existing inference format. By default the export references the original shared
components via a relative path. Add `--copy-shared` for a self-contained package
before moving or publishing it. Export refuses to overwrite an existing directory.
Allow space for an additional full transformer copy when exporting. Check the
container's storage quota as well as filesystem free space; the latter can exceed
the per-container quota.

## Verification scope

The integration check on one H20 executed all three history modes, saved a
three-step checkpoint, resumed it for step four, and generated a decodable
33-frame 640x384 video using the resulting adapters and the frozen Base
transformer. Preparing a real captioned video with DA3 and encoding a released
video while reusing its camera JSON were also checked.

A single-node check on eight H200 GPUs used 64 original OSP-v2 feature files,
with gradient accumulation of four (effective batch 32). It trained four steps
and resumed all eight ranks to step six, covering all three history modes with
finite losses. Per-step sample uniqueness, trainable-parameter probes across
ranks and resume, and complete rank-local checkpoint state were verified.
Parameter probes sample each trainable tensor; they do not prove full-weight
equality. These are pipeline checks;
the full dataset training run, multi-node DDP and fine-tuned model quality have
not been evaluated by these checks.
