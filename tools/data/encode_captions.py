#!/usr/bin/env python3
"""Encode supplied captions with the shared WorldCrafter text encoder; no caption generation."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from transformers import AutoTokenizer, UMT5EncoderModel
from worldcrafter.training.text import encode_prompt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--captions', type=Path, required=True, help='JSON object mapping sample IDs to existing caption strings')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    captions = json.loads(args.captions.read_text())
    if not isinstance(captions, dict) or not captions or any(not isinstance(v, str) for v in captions.values()):
        raise ValueError('Expected a nonempty ID -> caption string mapping')
    if args.batch_size < 1 or any(Path(k).name != k or k in ('.', '..') for k in captions):
        raise ValueError('Invalid batch size or sample ID')
    config = args.model_path / 'inference_config.json'
    shared = args.model_path
    if config.exists():
        shared = (shared / json.loads(config.read_text()).get('shared_components', '.')).resolve()
    tokenizer = AutoTokenizer.from_pretrained(shared / 'tokenizer')
    encoder = UMT5EncoderModel.from_pretrained(shared / 'text_encoder', torch_dtype=torch.bfloat16).to(args.device)
    encoder.requires_grad_(False).eval()
    args.output.mkdir(parents=True, exist_ok=False)
    items = list(captions.items())
    with torch.inference_mode():
        for start in range(0, len(items), args.batch_size):
            part = items[start:start + args.batch_size]
            embeddings, masks = encode_prompt(tokenizer, encoder, [text for _, text in part], device=args.device)
            for (key, text), embedding, mask in zip(part, embeddings, masks):
                torch.save({'prompt_raw': text, 'prompt_embed': embedding.cpu(), 'prompt_attention_mask': mask.cpu()},
                           args.output / (key + '.pt'))


if __name__ == '__main__':
    main()
