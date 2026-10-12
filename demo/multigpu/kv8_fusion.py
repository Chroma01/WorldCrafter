"""KV8 communication layouts, without changing attention or Q/K normalization.

The send kernel optionally applies WorldCrafter' interleaved RoPE while packing K/V
straight into the token-major collective buffer. The receive kernel removes
padding and restores global history/current order in one pass. BF16 arithmetic
boundaries, token coverage and the full attention backend remain unchanged.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _pack(Q, K, V, R, QOUT, SEND, N: tl.constexpr,
          B: tl.constexpr, S: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
          Q0: tl.constexpr, Q1: tl.constexpr, Q2: tl.constexpr, Q3: tl.constexpr,
          K0: tl.constexpr, K1: tl.constexpr, K2: tl.constexpr, K3: tl.constexpr,
          V0: tl.constexpr, V1: tl.constexpr, V2: tl.constexpr, V3: tl.constexpr,
          R0: tl.constexpr, R1: tl.constexpr, R2: tl.constexpr,
          ROPE: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < N
    width: tl.constexpr = D // 2 if ROPE else D
    b, s, h, lane = i // (S * H * width), i // (H * width) % S, i // width % H, i % width
    d = lane*2 if ROPE else lane
    k_offset = b*K0 + s*K1 + h*K2
    target = ((s*B+b)*2*H+h)*D+d
    if ROPE:
        q_offset = b*Q0 + s*Q1 + h*Q2
        qe = tl.load(Q + q_offset + d*Q3, valid, 0)
        qo = tl.load(Q + q_offset + (d+1)*Q3, valid, 0)
        ke = tl.load(K + k_offset + d*K3, valid, 0)
        ko = tl.load(K + k_offset + (d+1)*K3, valid, 0)
        cosine = tl.load(R + b*R0 + s*R1 + d*R2, valid, 0)
        sine = tl.load(R + b*R0 + s*R1 + (D+d+1)*R2, valid, 0)
        # Match the deployed Flash_RoPE kernel's pair arithmetic and FMA
        # contraction, rather than the unfused PyTorch reference's rounding.
        q_even = qe*cosine - qo*sine
        q_odd = qe*sine + qo*cosine
        k_even = ke*cosine - ko*sine
        k_odd = ke*sine + ko*cosine
        q_target = ((b*S+s)*H+h)*D+d
        tl.store(QOUT + q_target, q_even, valid)
        tl.store(QOUT + q_target+1, q_odd, valid)
        tl.store(SEND + target, k_even, valid)
        tl.store(SEND + target+1, k_odd, valid)
        ve = tl.load(V + b*V0 + s*V1 + h*V2 + d*V3, valid, 0)
        vo = tl.load(V + b*V0 + s*V1 + h*V2 + (d+1)*V3, valid, 0)
        tl.store(SEND + target + H*D, ve, valid)
        tl.store(SEND + target + H*D+1, vo, valid)
    else:
        k = tl.load(K + k_offset + d*K3, valid, 0)
        v = tl.load(V + b*V0 + s*V1 + h*V2 + d*V3, valid, 0)
        tl.store(SEND + target, k, valid)
        tl.store(SEND + target + H*D, v, valid)



@triton.jit
def _restore(RECV, OUT, N: tl.constexpr, B: tl.constexpr,
             HISTORY: tl.constexpr, CURRENT: tl.constexpr, WORLD: tl.constexpr,
             H: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < N
    total = HISTORY + CURRENT
    b, token, feature = i // (total*2*H*D), i // (2*H*D) % total, i % (2*H*D)
    is_history = token < HISTORY
    position = tl.where(is_history, token, token-HISTORY)
    length = tl.where(is_history, HISTORY, CURRENT)
    small, remainder = length // WORLD, length % WORLD
    boundary = (small+1)*remainder
    owner = tl.where(position < boundary, position // (small+1),
                     remainder + (position-boundary) // tl.maximum(small, 1))
    local_position = position - owner*small - tl.minimum(owner, remainder)
    hp, cp = triton.cdiv(HISTORY, WORLD), triton.cdiv(CURRENT, WORLD)
    source_token = owner*(hp+cp) + tl.where(is_history, 0, hp) + local_position
    value = tl.load(RECV + (source_token*B+b)*2*H*D + feature, valid, 0)
    tl.store(OUT+i, value, valid)


def pack_kv(q, k, v, rotary_emb=None):
    """Return local Q and a contiguous token-major K/V send buffer."""
    b, s, h, d = q.shape
    assert q.is_cuda and q.dtype == k.dtype == v.dtype
    assert q.shape == k.shape == v.shape
    rotate = rotary_emb is not None
    if rotate:
        # WorldCrafter uses FP32 rotary coefficients and rounds only at the output.
        # Refuse other precision contracts instead of silently changing them.
        assert rotary_emb.dtype == torch.float32 and d % 2 == 0
        assert rotary_emb.shape == (b, s, 2*d)
    result = torch.empty_like(q, memory_format=torch.contiguous_format) if rotate else q
    send = torch.empty((s, b, 2, h, d), dtype=k.dtype, device=k.device)
    rope_strides = rotary_emb.stride() if rotate else (0, 0, 0)
    elements = q.numel() // 2 if rotate else q.numel()
    _pack[(triton.cdiv(elements, 256),)](
        q, k, v, rotary_emb if rotate else q, result, send, elements, b, s, h, d,
        *q.stride(), *k.stride(), *v.stride(), *rope_strides, rotate, 256,
        enable_fp_fusion=True,
    )
    return result, send


def restore_kv(received, history, current, world):
    """Restore [B, global_tokens, 2, heads, dim] with the original K/V strides."""
    _, b, _, h, d = received.shape
    out = torch.empty((b, history+current, 2, h, d), dtype=received.dtype,
                      device=received.device)
    _restore[(triton.cdiv(out.numel(), 1024),)](
        received, out, out.numel(), b, history, current, world, h, d, 1024)
    return out.unbind(2)
