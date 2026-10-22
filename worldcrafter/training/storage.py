"""Local feature IO. Download/extract published shards before training."""
import json
from pathlib import Path
import torch


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
