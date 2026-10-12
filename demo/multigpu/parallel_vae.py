"""Eight-device Wan VAE decode, promoted from the measured layer pipeline.

The serving owner drives independent GPU streams. DiT peers remain on their
control sockets during decoding; no tensors or caches cross process boundaries.
Each stage retains its own complete causal history within one 33-frame chunk.
"""
import copy
import time

import torch
from torch import nn
from diffusers.models.autoencoders.vae import DecoderOutput


# Reference H200 partition; validate its behavior and timings on the target GPU.
STAGE_NAMES = (
    ("input", "middle"),
    ("up0_res0", "up0_res1", "up0_res2", "up0_sample",
     "up1_res0", "up1_res1", "up1_res2", "up1_sample"),
    ("up2_res0", "up2_res1"),
    ("up2_res2", "up2_sample"),
    ("up3_res0",),
    ("up3_res1",),
    ("up3_res2",),
    ("output",),
)
LATENT_SHAPE = (1, 16, 9, 48, 80)


class CachedConv(nn.Module):
    def __init__(self, conv, norm=None, activation=None):
        super().__init__()
        self.conv, self.norm, self.activation = conv, norm, activation

    def forward(self, x, feat_cache, feat_idx):
        if self.norm is not None:
            x = self.activation(self.norm(x))
        idx = feat_idx[0]
        cache = x[:, :, -2:].clone()
        if cache.shape[2] < 2 and feat_cache[idx] is not None:
            cache = torch.cat([feat_cache[idx][:, :, -1:].to(cache.device), cache], dim=2)
        x = self.conv(x, feat_cache[idx])
        feat_cache[idx] = cache
        feat_idx[0] += 1
        return x


def operations(decoder):
    ops = [("input", CachedConv(decoder.conv_in)), ("middle", decoder.mid_block)]
    for i, block in enumerate(decoder.up_blocks):
        if type(block).__name__ != "WanUpBlock":
            raise ValueError("pipeline8 requires the validated WanUpBlock decoder")
        ops.extend((f"up{i}_res{j}", layer) for j, layer in enumerate(block.resnets))
        if block.upsamplers is not None:
            ops.append((f"up{i}_sample", block.upsamplers[0]))
    ops.append(("output", CachedConv(decoder.conv_out, decoder.norm_out, decoder.nonlinearity)))
    return ops


class Stage(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x, cache):
        idx = [0]
        for layer in self.layers:
            x = layer(x, feat_cache=cache, feat_idx=idx)
        return x


class EightGPUDecoder:
    def __init__(self, vae):
        if torch.cuda.device_count() != 8:
            raise ValueError("The VAE owner must see all eight GPUs")
        if vae.config.patch_size is not None or vae.use_tiling or vae.dtype != torch.float32:
            raise ValueError("pipeline8 requires untiled FP32 Wan VAE weights without patching")
        ops = operations(vae.decoder)
        if [name for name, _ in ops] != [name for names in STAGE_NAMES for name in names]:
            raise ValueError("VAE decoder layout differs from the validated partition")
        self.post = vae.post_quant_conv
        self.stages = []
        offset = 0
        for device, names in enumerate(STAGE_NAMES):
            # Copy only actual layers; the original decoder.forward may already
            # be compiled and is deliberately not copied into a pipeline stage.
            layers = [copy.deepcopy(layer).cpu() for _, layer in ops[offset:offset + len(names)]]
            stage = Stage(layers).to(f"cuda:{device}").eval()
            self.stages.append(stage)
            offset += len(names)
        self.last_decode_s = None
        self.info = dict(mode="pipeline8", gpu_count=8, warmed=False,
                         stages=[list(names) for names in STAGE_NAMES],
                         transport="owner_cuda_peer_copy", causal_cache="per_stage_per_chunk",
                         compute_dtype="float32", compile_enabled=False)

    @staticmethod
    def synchronize():
        for device in range(8):
            torch.cuda.synchronize(device)

    def _run(self, z):
        caches = [[None] * 64 for _ in self.stages]
        slots = [[None] * 9 for _ in self.stages]
        # Keep source activations alive until all cross-device copies finish.
        with torch.cuda.device(0), torch.inference_mode(), torch.autocast("cuda", enabled=False):
            try:
                x = self.post(z)
                for tick in range(16):
                    for rank in range(7, -1, -1):
                        t = tick - rank
                        if not 0 <= t < 9:
                            continue
                        torch.cuda.set_device(rank)
                        if rank == 0:
                            part = x[:, :, t:t + 1]
                        else:
                            part, event = slots[rank - 1][t]
                            torch.cuda.current_stream(rank).wait_event(event)
                            part = part.to(f"cuda:{rank}", non_blocking=True)
                        part = self.stages[rank](part, caches[rank])
                        event = torch.cuda.Event()
                        event.record()
                        slots[rank][t] = (part, event)
                torch.cuda.set_device(7)
                result = torch.cat([item[0] for item in slots[-1]], dim=2).clamp(-1, 1).to("cuda:0")
                return result
            finally:
                self.synchronize()

    def warmup(self, reference_decode=None):
        print("VAE_PIPELINE8_WARMUP", flush=True)
        started = time.perf_counter()
        z = torch.zeros(LATENT_SHAPE, dtype=torch.float32, device="cuda:0")
        self._run(z)
        self._run(z)
        if reference_decode is not None:
            # Check the actual serving decoder with nonzero activations. This
            # independent generator does not alter session RNG.
            generator = torch.Generator(device='cuda:0').manual_seed(42)
            latent = torch.randn(LATENT_SHAPE, generator=generator, device='cuda:0') * .25
            with torch.inference_mode():
                expected = reference_decode(latent, return_dict=False)[0]
                actual = self._run(latent)
                error = (actual-expected).abs()
                self.info['numerical_validation'] = dict(
                    max_abs_error=float(error.max()), mean_abs_error=float(error.mean()),
                    rtol=1e-4, atol=1e-4, finite=bool(torch.isfinite(actual).all()))
                torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
        self.info.update(warmed=True, warmup_seconds=time.perf_counter() - started)
        print("VAE_PIPELINE8_READY", self.info, flush=True)

    def decode(self, z, return_dict=True):
        if tuple(z.shape) != LATENT_SHAPE or z.device != torch.device("cuda:0") or z.dtype != torch.float32:
            raise ValueError(f"pipeline8 expects FP32 CUDA:0 latent {LATENT_SHAPE}; got {z.shape}, {z.device}, {z.dtype}")
        self.synchronize()
        started = time.perf_counter()
        result = self._run(z)
        self.last_decode_s = time.perf_counter() - started
        return DecoderOutput(sample=result) if return_dict else (result,)
