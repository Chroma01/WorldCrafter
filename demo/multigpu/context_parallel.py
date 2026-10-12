"""Balanced token ownership with complete current-step K/V attention.

Adapted from the eight_gpu_20261008T085026Z handoff. Camera projections and
attention retain WorldCrafter's FP32 compute contract, including over transport.
"""
from functools import partial
import types

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as fc
import torch.nn.functional as F


def initialize_attention():
    # Loading the model with native SDPA does not initialize Diffusers' hub
    # registry. Explicitly resolve FA3 before installing the parallel processor.
    from diffusers.models.attention_dispatch import (
        AttentionBackendName, _check_attention_backend_requirements,
        _maybe_download_kernel_for_backend,
    )
    backend = AttentionBackendName._FLASH_3_HUB
    _check_attention_backend_requirements(backend)
    _maybe_download_kernel_for_backend(backend)


class BalancedLayout:
    def __init__(self, rank, world_size):
        if world_size not in (1, 2, 4, 8) or not 0 <= rank < world_size:
            raise ValueError("Expected a valid rank in a 1/2/4/8-rank group")
        self.rank, self.world_size = rank, world_size

    def configure(self, total, current):
        if not 0 < current <= total:
            raise ValueError("Invalid history/current token lengths")
        self.total, self.current, self.history = total, current, total-current
        n = self.world_size
        self.hp, self.cp = (self.history+n-1)//n, (current+n-1)//n
        self.local = self.hp+self.cp
        self.history_parts = [self.partition(self.history, r) for r in range(n)]
        self.current_parts = [self.partition(current, r) for r in range(n)]
        self.hs, self.hn = self.history_parts[self.rank]
        self.cs, self.cn = self.current_parts[self.rank]

    def partition(self, length, rank):
        size, remainder = divmod(length, self.world_size)
        return rank*size+min(rank, remainder), size+int(rank < remainder)

    @staticmethod
    def pad(x, count):
        if x.shape[1] == count:
            return x
        return torch.cat((x, x.new_zeros((x.shape[0], count-x.shape[1], *x.shape[2:]))), 1)

    def split(self, x):
        return torch.cat((self.pad(x[:, self.hs:self.hs+self.hn], self.hp),
                          self.pad(x[:, self.history+self.cs:self.history+self.cs+self.cn], self.cp)), 1)

    def restore(self, x):
        history = [x[:, r*self.local:r*self.local+count]
                   for r, (_, count) in enumerate(self.history_parts)]
        current = [x[:, r*self.local+self.hp:r*self.local+self.hp+count]
                   for r, (_, count) in enumerate(self.current_parts)]
        return torch.cat(history+current, 1)

    def restore_current(self, x):
        return torch.cat([x[:, r*self.cp:r*self.cp+count]
                          for r, (_, count) in enumerate(self.current_parts)], 1)

    def gather(self, x):
        return self.restore(fc.wait_tensor(fc.all_gather_tensor(x.contiguous(), 1, dist.group.WORLD)))


class KVAttention:
    _attention_backend = '_flash_3_hub'
    _parallel_config = None

    def __init__(self, layout, fusion):
        if fusion not in ('off', 'rope'):
            raise ValueError("K/V fusion must be off or rope")
        self.layout, self.fusion = layout, fusion

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, rotary_emb=None, original_context_length=None):
        from worldcrafter.diffusers import transformer
        from .kv8_fusion import pack_kv, restore_kv
        if encoder_hidden_states is not None or attention_mask is not None or attn.is_amplify_history:
            raise ValueError("Balanced K/V attention requires unmasked Fast self-attention")
        q, k, v = transformer._get_qkv_projections(attn, hidden_states, None)
        q, k = attn.norm_q(q), attn.norm_k(k)
        q, k, v = [x.unflatten(2, (attn.heads, -1)) for x in (q, k, v)]
        if self.fusion == 'off':
            q = transformer.apply_rotary_emb_transposed(q, rotary_emb)
            k = transformer.apply_rotary_emb_transposed(k, rotary_emb)
        q, send = pack_kv(q, k, v, rotary_emb if self.fusion == 'rope' else None)
        received = torch.ops._c10d_functional.all_gather_into_tensor(
            send, self.layout.world_size, dist.group.WORLD.group_name)
        k, v = restore_kv(fc.wait_tensor(received), self.layout.history,
                          self.layout.current, self.layout.world_size)
        out = transformer.dispatch_attention_fn(q, k, v, dropout_p=0., is_causal=False,
                                                backend=self._attention_backend).flatten(2, 3)
        return attn.to_out[1](attn.to_out[0](out.to(q.dtype)))


@torch.compiler.disable
def local_geometry(camera, control, layout):
    """Slice this forward's geometry; never reuse geometry from another pose."""
    prope = camera.prope_attn
    prope._precompute_and_cache_apply_fns(
        viewmats=control['viewmats'], Ks=control.get('K'),
        coeffs_x=control.get('coeffs_x'), coeffs_y=control.get('coeffs_y'))
    functions = []
    for original in (prope.apply_fn_q, prope.apply_fn_kv, prope.apply_fn_o):
        parts = []
        for function, size in original.keywords['func_size_pairs']:
            kw = dict(function.keywords)
            if 'matrix' in kw:
                matrix = kw['matrix']
                if matrix.shape[1] != layout.current:
                    matrix = matrix.repeat_interleave(layout.current//matrix.shape[1], dim=1)
                kw['matrix'] = matrix[:, layout.cs:layout.cs+layout.cn].contiguous()
            else:
                coeffs = []
                for value in kw['coeffs']:
                    if value.shape[2] != layout.current:
                        value = value.repeat(1, 1, layout.current//value.shape[2], 1)
                    coeffs.append(value[:, :, layout.cs:layout.cs+layout.cn].contiguous())
                kw['coeffs'] = tuple(coeffs)
            parts.append((partial(function.func, **kw), size))
        functions.append(partial(original.func, func_size_pairs=parts))
    return functions


def camera_prepare(camera, x, control, layout):
    # Keep FP32 camera operations. The handoff's BF16 autocast is intentionally
    # not inherited: it would alter the current public model's precision.
    with torch.autocast('cuda', enabled=False):
        b, t, _ = x.shape
        if hasattr(camera, 'cam_encoder') and 'cam_emb' in control:
            y = camera.cam_encoder(control['cam_emb'])
            y = y.repeat_interleave(layout.current//y.shape[1], dim=1)
            x = x+y[:, layout.cs:layout.cs+layout.cn]
        def project(layer):
            return layer(x.float()).view(b, t, camera.num_heads, camera.head_dim).transpose(1, 2)
        aq, akv, ao = local_geometry(camera, control, layout)
        q = aq(project(camera.q_proj))
        k, v = akv(project(camera.k_proj)), akv(project(camera.v_proj))
        assert q.dtype == k.dtype == v.dtype == torch.float32
        kv = layout.pad(torch.stack((k, v), dim=3).permute(0, 2, 1, 3, 4), layout.cp)
        pending = torch.ops._c10d_functional.all_gather_into_tensor(
            kv.permute(1, 0, 2, 3, 4).contiguous(), layout.world_size, dist.group.WORLD.group_name)
        return q, pending, ao


def camera_finish(camera, prepared, layout):
    with torch.autocast('cuda', enabled=False):
        q, pending, ao = prepared
        kv = layout.restore_current(fc.wait_tensor(pending).permute(1, 0, 2, 3, 4))
        k, v = kv.permute(0, 2, 1, 3, 4).unbind(3)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        return camera.out_proj(ao(out).transpose(1, 2).flatten(2))


def balanced_block(block, hidden_states, encoder_hidden_states, temb, rotary_emb,
                   original_context_length=None, camera_control_ucpe_input=None):
    layout = block.context_layout
    if temb.ndim == 4:
        values = (block.scale_shift_table.unsqueeze(0)+layout.split(temb).float()).chunk(6, dim=2)
        shift, scale, g, cshift, cscale, cgate = [v.squeeze(2) for v in values]
    else:
        shift, scale, g, cshift, cscale, cgate = (block.scale_shift_table+temb.float()).chunk(6, dim=1)
    normalized = (block.norm1(hidden_states.float())*(1+scale)+shift).type_as(hidden_states)
    prepared = None
    if camera_control_ucpe_input is not None:
        prepared = camera_prepare(block.cam_self_attn, normalized[:, layout.hp:layout.hp+layout.cn],
                                  camera_control_ucpe_input, layout)
    attn = block.attn1(normalized, None, None, layout.split(rotary_emb), original_context_length)
    if prepared is not None:
        camera = camera_finish(block.cam_self_attn, prepared, layout)
        attn = torch.cat((attn[:, :layout.hp],
                          attn[:, layout.hp:layout.hp+layout.cn]+camera.to(attn.dtype),
                          attn[:, layout.hp+layout.cn:]), dim=1)
    hidden_states = (hidden_states.float()+attn*g).type_as(hidden_states)
    if block.guidance_cross_attn:
        current = hidden_states[:, layout.hp:layout.hp+layout.cn]
        normalized = block.norm2(current.float()).type_as(current)
        current = current+block.attn2(normalized, encoder_hidden_states, None, None, original_context_length)
        hidden_states = torch.cat((hidden_states[:, :layout.hp], current,
                                   hidden_states[:, layout.hp+layout.cn:]), dim=1)
    else:
        normalized = block.norm2(hidden_states.float()).type_as(hidden_states)
        hidden_states = hidden_states+block.attn2(normalized, encoder_hidden_states, None, None, original_context_length)
    normalized = (block.norm3(hidden_states.float())*(1+cscale)+cshift).type_as(hidden_states)
    return (hidden_states.float()+block.ffn(normalized).float()*cgate).type_as(hidden_states)


def install(pipe, fusion='rope', compile_blocks=True):
    initialize_attention()
    torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 256)
    rank, size = dist.get_rank(), dist.get_world_size()
    if size not in (2, 4, 8):
        raise ValueError("Context parallel serving requires 2, 4 or 8 ranks")
    records = []
    for index, model in enumerate((pipe.stage_transformers[0], pipe.stage_transformers[2])):
        if model.config.is_amplify_history:
            raise ValueError("History amplification is not supported by this backend")
        layout = BalancedLayout(rank, size)
        def split(module, args, layout=layout, index=index):
            layout.configure(args[0].shape[1], args[4])
            if layout.cn == 0:
                raise ValueError("Each rank must own current tokens")
            record = dict(branch=index, rank=rank, history=layout.history, current=layout.current,
                          local_history=layout.hn, local_current=layout.cn)
            if record not in records:
                records.append(record)
            return (layout.split(args[0]),)+args[1:]
        def collect(module, args, output, layout=layout):
            return layout.gather(output)
        model.blocks[0].register_forward_pre_hook(split)
        model.blocks[-1].register_forward_hook(collect)
        for block in model.blocks:
            if block.cam_self_attn.adaptation_method != 'parallel':
                raise ValueError("Expected parallel UCPE residual")
            block.context_layout = layout
            block.attn1.set_processor(KVAttention(layout, fusion))
            block.forward = types.MethodType(balanced_block, block)
            if compile_blocks:
                block.compile(mode='default', dynamic=False)
    return dict(strategy=f'cp_balanced_kv{size}', world_size=size, fusion=fusion,
                camera_compute_dtype='float32', camera_transport_dtype='float32',
                padding='excluded_from_keys_and_output', observed_layouts=records)
