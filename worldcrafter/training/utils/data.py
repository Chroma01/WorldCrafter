"""Feature storage, caption encoding and video preprocessing."""

import json
from pathlib import Path
import torch
from typing import Union
from typing import List
from typing import Optional
import ftfy
import html
import re
import random
import numpy as np
import torchvision


def normalize_storage_path(path):
    value = str(path)
    if "://" in value:
        raise ValueError("Download features locally first; training expects filesystem paths")
    return str(Path(value).expanduser())


def storage_basename(path):
    return Path(path).name


def storage_stem(path):
    return Path(path).stem


def storage_join(root, *parts):
    return str(Path(root).joinpath(*parts))


def path_exists(path):
    return Path(normalize_storage_path(path)).exists()


def read_text(path):
    return Path(normalize_storage_path(path)).read_text()


def read_json(path):
    return json.loads(read_text(path))


def list_pt_files(root):
    return sorted(str(p) for p in Path(normalize_storage_path(root)).glob('*.pt'))


def load_torch_pt(path, *, map_location='cpu', weights_only=False):
    # Historical feature objects include PIL images and caption metadata.
    # Only use feature files from a trusted source.
    return torch.load(normalize_storage_path(path), map_location=map_location, weights_only=weights_only)


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()

def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text

def prompt_clean(text):
    text = whitespace_clean(basic_clean(text))
    return text

def _get_t5_prompt_embeds(
    tokenizer,
    text_encoder,
    prompt: Union[str, List[str]] = None,
    num_videos_per_prompt: int = 1,
    max_sequence_length: int = 512,
    caption_dropout_p: float = 0.0,
    device: Optional[torch.device] = "cuda",
    dtype: Optional[torch.dtype] = torch.bfloat16,
):
    device = device
    dtype = dtype

    prompt = [prompt] if isinstance(prompt, str) else prompt
    prompt = [prompt_clean(u) for u in prompt]
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask

    prompt_embeds = text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    if random.random() < caption_dropout_p:
        prompt_embeds.fill_(0)
        mask.fill_(False)
    seq_lens = mask.gt(0).sum(dim=1).long()

    prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
    prompt_embeds = torch.stack(
        [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
    )

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

    return prompt_embeds, text_inputs.attention_mask

def encode_prompt(
    tokenizer,
    text_encoder,
    prompt: Union[str, List[str]],
    num_videos_per_prompt: int = 1,
    prompt_embeds: Optional[torch.Tensor] = None,
    max_sequence_length: int = 512,
    caption_dropout_p: float = 0.0,
    device: Optional[torch.device] = "cuda",
    dtype: Optional[torch.dtype] = torch.bfloat16,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt

    if prompt_embeds is None:
        prompt_embeds, prompt_attention_mask = _get_t5_prompt_embeds(
            tokenizer,
            text_encoder,
            prompt=prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            max_sequence_length=max_sequence_length,
            caption_dropout_p=caption_dropout_p,
            device=device,
            dtype=dtype,
        )

    return prompt_embeds, prompt_attention_mask


resolution_bucket_options = {
    640: [
        (768, 320),
        (768, 384),
        (640, 384),
        (768, 512),
        (576, 448),
        (512, 512),
        (448, 576),
        (512, 768),
        (384, 640),
        (384, 768),
        (320, 768),
    ],
}

length_bucket_options = {
    1: [
        501,
        481,
        461,
        441,
        421,
        401,
        381,
        361,
        341,
        321,
        301,
        281,
        261,
        241,
        221,
        193,
        181,
        161,
        141,
        121,
        101,
        81,
        61,
        41,
        21,
    ],
    2: [193, 177, 161, 156, 145, 133, 129, 121, 113, 109, 97, 85, 81, 73, 65, 61, 49, 37, 25],
}


def find_nearest_resolution_bucket(h, w, resolution=640):
    min_metric = float("inf")
    best_bucket = None
    for bucket_h, bucket_w in resolution_bucket_options[resolution]:
        metric = abs(h * bucket_w - w * bucket_h)
        if metric <= min_metric:
            min_metric = metric
            best_bucket = (bucket_h, bucket_w)
    return best_bucket


def find_nearest_length_bucket(length, stride=1):
    buckets = length_bucket_options[stride]
    min_bucket = min(buckets)
    if length < min_bucket:
        return length
    valid_buckets = [bucket for bucket in buckets if bucket <= length]
    return max(valid_buckets)


def read_cut_crop_and_resize(
    video_path, f_prime, h_prime, w_prime, stride=1, start_frame=None, end_frame=None, crop=None
):
    frame_indices = list(range(start_frame, end_frame, stride))
    assert len(frame_indices) == f_prime

    import av
    wanted = set(frame_indices)
    decoded = []
    with av.open(str(video_path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index in wanted:
                decoded.append(frame.to_ndarray(format="rgb24"))
            if index >= frame_indices[-1]:
                break
    if len(decoded) != f_prime:
        raise ValueError(f"Decoded {len(decoded)} of {f_prime} requested frames: {video_path}")
    frames = torch.from_numpy(np.stack(decoded)).float()

    frames = (frames / 127.5) - 1
    video = frames.permute(0, 3, 1, 2)

    s_x, e_x, s_y, e_y = crop
    video = video[:, :, s_y:e_y, s_x:e_x]

    frames, channels, h, w = video.shape
    aspect_ratio_original = h / w
    aspect_ratio_target = h_prime / w_prime

    if aspect_ratio_original >= aspect_ratio_target:
        new_h = int(w * aspect_ratio_target)
        top = (h - new_h) // 2
        bottom = top + new_h
        left = 0
        right = w
    else:
        new_w = int(h / aspect_ratio_target)
        left = (w - new_w) // 2
        right = left + new_w
        top = 0
        bottom = h

    # Crop the video
    cropped_video = video[:, :, top:bottom, left:right]
    # Resize the cropped video
    resized_video = torchvision.transforms.functional.resize(cropped_video, (h_prime, w_prime))
    return resized_video
