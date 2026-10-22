# OSP-v2 index

This release index contains 20,434 feature basenames and 77,642 valid target chunks.
It is the strict motion + smooth trajectory + no-food/aerial subset dated 2026-06-14.
The reference memory sampler excludes chunk 0 targets: the index contains 19,696
files with 70,965 nonzero candidate targets, before checking complete camera anchors.
Keep the full feature history. This index does not contain video or feature binaries.

Download videos and camera JSON from the gated [WorldCrafter dataset](https://huggingface.co/datasets/Drexubery/WorldCrafter), then use `tools/data/encode_released_dataset.py` to produce local features. Precomputed feature binaries are not distributed. See [data preparation](../../docs/data_preparation.md). For an independent evaluation, split by original source video, including all its cuts, before training.
