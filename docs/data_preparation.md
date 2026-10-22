# Prepare features from captioned videos

Captions are supplied by the user. These tools **do not generate or rewrite
captions**. They preserve the existing caption cleaning/UMT5 encoding, chunked
Wan VAE encoding and Depth Anything 3 (DA3) camera-estimation pipeline used for
the OSP-v2 features.

The [WorldCrafter dataset](https://huggingface.co/datasets/Drexubery/WorldCrafter)
provides 19,972 unique source videos, 20,434 camera records, existing captions,
and the 77,642-entry valid-target chunk index. Access is granted automatically
after completing the academic research agreement on the dataset page. Log in
locally with `hf auth login` before downloading. Precomputed `.pt` features are
not distributed.

## Encode the release using its existing poses

Activate the [training environment](finetuning.md#environment-and-models), then
download, extract and encode the release:

```bash
python tools/data/download_dataset.py --root data/WorldCrafter --download --verify --extract
python tools/data/encode_released_dataset.py \
  --dataset-root data/WorldCrafter --model-path weights/WorldCrafter-Base \
  --output data/osp_v2/features
python train.py --feature-root data/osp_v2/features \
  --valid-chunks data/osp_v2/valid_chunk_indices.json --check-data
```

This path does not load or run DA3. It reuses the released poses, intrinsics and
anchor indices, and preserves the original feature filenames so the existing
valid index remains usable. It uses the supplied crop, selected frame interval
and stride; no caption generation is performed. The output omits depth maps and
unused image-conditioning latents, neither of which the reference Stage-1 trainer
requires. All chunks are retained for history sampling. VAE posterior sampling
uses a recorded per-record seed; regenerated tensors are not claimed to be
bitwise identical to historical `.pt` files. To parallelize, set `--num-shards N`
and a distinct `--shard-index I` for each worker.

For your own videos without supplied camera annotations, use the DA3 pipeline
below instead.

## Input manifest

Supply a JSON array in the historical format:

```json
[
  {
    "path": "/absolute/path/to/video.mp4",
    "cut": [0, 165],
    "crop": [0, 640, 0, 384],
    "fps": 24.0,
    "cap": ["An existing description of this video."]
  }
]
```

`cut` is `[start_frame, end_frame)` in the original decoded video. `crop` is
`[left, right, top, bottom]` in pixels. `cap[0]` must be a string. The reference
pipeline uses stride 1, the historical length buckets, and the 384x640 resolution
bucket. Other aspect-ratio buckets are rejected rather than silently converted to
a new dataset distribution. Within the chosen crop, center-crop and resize follow
the original implementation. Complete 33-frame chunks are encoded separately;
a trailing incomplete chunk is not used. Stored frame indices link every latent
anchor back to the processed and original frame sequence.

The code uses PyAV for portable decoding, followed by the reference float
normalization, torchvision crop/resize and feature computation. Decoder-level
pixel differences from historical FFmpeg builds are possible.

## Install two isolated environments

First install the [training environment](finetuning.md#environment-and-models).
DA3 runs in a separate Python process so its dependencies cannot alter training:

```bash
uv sync --project data_env --frozen
```

The DA3 environment pins the [official source](https://github.com/ByteDance-Seed/Depth-Anything-3)
to the revision recorded in `data_env/pyproject.toml` and `uv.lock`. Use the
`DA3NESTED-GIANT-LARGE-1.1` model matching the historical pipeline, downloaded to a
local directory from its official model repository. Record the model revision and
hashes alongside any newly prepared dataset.

## Video latents + caption embeddings + DA3 cameras

With the training environment activated:

```bash
python tools/data/prepare_features.py \
  --source-json data/source.json \
  --pretrained-model-name-or-path weights/WorldCrafter-Base \
  --da3-model-dir /path/to/DA3NESTED-GIANT-LARGE-1.1 \
  --da3-python data_env/.venv/bin/python \
  --output-latent-dir data/custom/features \
  --output-camera-json-dir data/custom/cameras \
  --tmp-dir /path/to/local-ssd/worldcrafter-precompute \
  --batch-size 1 --bucket-flush-size 1
```

Base's `shared_components` setting resolves the VAE/tokenizer/text encoder.
The main script and DA3 subprocess can share one GPU; encoding components are
moved off GPU before the DA3 phase. `--device` and `--da3-device` can target different
GPUs. `--num-shards N --shard-index I` supports independent workers; assign each a
unique temporary directory and report path. Completed outputs are resumed only
when their recorded preprocessing configuration matches.

The reference DA3 configuration is process resolution 504,
`upper_bound_resize`, reference view `middle`, and `use_ray_pose=false`.
OpenCV world-to-camera extrinsics are inverted to camera-to-world. Both video-relative
and chunk-relative poses are stored. Intrinsics are normalized using **DA3's actual
processed image dimensions**, not the 640x384 output video dimensions. Metric
scale metadata and depth are retained. The training dataset uses raw video-relative
metric poses for RepEncoder and chunk-relative poses for UCPE.

Main feature fields:

| Field | Meaning |
| --- | --- |
| `vae_latent` | `[chunks,16,9,48,80]`, standardized VAE latents |
| `prompt_embed` | `[512,4096]`, existing caption embedding with zero padding |
| `camera_pose_rel_video_c2w` | `[chunks,9,4,4]`, one video-relative metric frame |
| `camera_pose_rel_chunk_c2w` | `[chunks,9,4,4]`, per-chunk local camera frame |
| `camera_K_norm` | `[chunks,9,4]`, normalized fx,fy,cx,cy |
| `camera_valid_mask` | `[chunks,9]`, finite camera anchors |
| `camera_anchor_frame_idx` | Source video frame indices |
| `camera_anchor_processed_frame_idx` | Processed frame indices; chunk anchors are 0,4,...,32 |

A feature file also includes original caption metadata, image-conditioning latents,
depth and provenance. Do not remove historical chunks just because they are absent
from the valid-target index: they may be needed for causal memory sampling.

## Encode existing captions separately

For users who only need caption embeddings, provide a JSON object mapping sample
IDs to strings, then run:

```bash
python tools/data/encode_captions.py --model-path weights/WorldCrafter-Base \
  --captions data/captions.json --output data/caption_embeddings
```

This produces text-only `.pt` objects; training still needs complete video/camera
features. The combined preparation script already performs this same text encoding.

## Build a target index for custom data

```bash
python tools/data/build_valid_chunks.py --camera-root data/custom/cameras \
  --feature-root data/custom/features --output data/custom/valid_chunk_indices.json
python train.py --feature-root data/custom/features \
  --valid-chunks data/custom/valid_chunk_indices.json --check-data
```

The custom builder applies motion amplitude and trajectory smoothness checks. It
does not generate captions or apply the historical food/aerial topic exclusion.
For the released OSP-v2 subset, use its existing index rather than recreating a new
selection with this custom-data tool.

## Lossless feature distribution

```bash
python tools/data/pack_features.py --feature-root data/osp_v2/features \
  --valid-chunks data/osp_v2/valid_chunk_indices.json \
  --output /path/to/new-release-directory --shard-gib 2
```

The packer keeps complete `.pt` files and their dtypes unchanged, writes bounded
`.tar.gz` shards, and reads each shard back to verify every file's SHA-256. The
manifest records per-file and per-shard sizes and hashes. Extract verified shards
to a local feature directory before training. Packaging does not upload or publish
anything. Source videos, if distributed, remain separate from feature shards.
