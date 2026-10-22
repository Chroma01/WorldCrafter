"""Optional bounded-run diagnostics that do not consume training RNG."""
import hashlib
import json
from pathlib import Path

import torch


def write_rank_record(accelerator, output, record):
    record = {"rank": accelerator.process_index, "world_size": accelerator.num_processes, **record}
    path = Path(output) / f"audit-rank-{accelerator.process_index:05d}.jsonl"
    with path.open("a") as stream:
        stream.write(json.dumps(record) + "\n")


def verify_parameter_probes(accelerator, model, output, step):
    # Sample every trainable tensor; this detects rank drift without gathering
    # the full 4.6 GB of trainable weights. It is not a full-weight equality proof.
    probes = []
    for parameter in accelerator.unwrap_model(model).parameters():
        if parameter.requires_grad:
            flat = parameter.detach().reshape(-1)
            probes.append(flat[[0, flat.numel() // 2, flat.numel() - 1]].float())
    local = torch.cat(probes).contiguous()
    gathered = accelerator.gather(local).reshape(accelerator.num_processes, -1)
    if not torch.isfinite(gathered).all() or not torch.equal(gathered, gathered[:1].expand_as(gathered)):
        raise RuntimeError(f"Trainable parameter probes differ across ranks at step {step}")
    digest = hashlib.sha256(local.cpu().numpy().tobytes()).hexdigest()
    write_rank_record(accelerator, output, {"event": "parameter_probes", "step": step,
                      "trainable_tensors": len(probes), "probe_values": local.numel(),
                      "sha256": digest, "equal_across_ranks": True})
