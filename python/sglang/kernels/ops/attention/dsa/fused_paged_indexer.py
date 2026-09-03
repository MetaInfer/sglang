"""Portable Triton paged FP8 scoring for the DSA indexer.

DeepGEMM's paged-MQA kernels only support SM90/SM100. This implementation
keeps FP8 decoding, the per-head dot/ReLU/weighting, and reduction in registers
so KPool decode can run on Ampere without materializing a gathered cache.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _e4m3_to_f32(b):
    e = (b >> 3) & 0xF
    m = b & 0x7
    mag = (((e + 120) << 23) | (m << 20)).to(tl.float32, bitcast=True)
    mag = tl.where(e == 0, m.to(tl.float32) * 1.953125e-3, mag)
    return tl.where((b >> 7) & 1 == 1, -mag, mag)


@triton.autotune(
    configs=[
        triton.Config({"BN": 32}, num_warps=4, num_stages=2),
        triton.Config({"BN": 64}, num_warps=4, num_stages=2),
        triton.Config({"BN": 64}, num_warps=8, num_stages=2),
        triton.Config({"BN": 64}, num_warps=4, num_stages=3),
        triton.Config({"BN": 128}, num_warps=8, num_stages=2),
    ],
    key=["H"],
)
@triton.jit
def _fused_paged_indexer_kernel(
    CACHE_I32,
    Q4,
    W,
    PT,
    OUT,
    s_q4_b,
    s_q4_j,
    s_q4_h,
    s_w_b,
    s_pt_b,
    s_out_b,
    PADDED_EFF,
    H: tl.constexpr,
    BN: tl.constexpr,
):
    # int64 row/page bases are required for very long contexts.
    pid_b = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1)
    offs_n = pid_n * BN + tl.arange(0, BN)
    n_mask = offs_n < PADDED_EFF

    page_idx = offs_n // 64
    tok = offs_n % 64
    pids = tl.load(PT + pid_b * s_pt_b + page_idx, mask=n_mask, other=0)
    pids = tl.maximum(pids, 0).to(tl.int64)

    # Each cache page contains 64x128 FP8 bytes followed by 64 FP32 scales.
    cache_ptrs = CACHE_I32 + (pids * 2112 + tok.to(tl.int64) * 32)
    kv_words = tl.load(
        cache_ptrs[:, None] + tl.arange(0, 32)[None, :],
        mask=n_mask[:, None],
        other=0,
    )
    kv0 = _e4m3_to_f32(kv_words & 0xFF).to(tl.bfloat16)
    kv1 = _e4m3_to_f32((kv_words >> 8) & 0xFF).to(tl.bfloat16)
    kv2 = _e4m3_to_f32((kv_words >> 16) & 0xFF).to(tl.bfloat16)
    kv3 = _e4m3_to_f32((kv_words >> 24) & 0xFF).to(tl.bfloat16)

    offs_h = tl.arange(0, H)
    offs_k = tl.arange(0, 32)
    q_base = Q4 + pid_b * s_q4_b
    q0 = tl.load(q_base + 0 * s_q4_j + offs_h[:, None] * s_q4_h + offs_k[None, :])
    q1 = tl.load(q_base + 1 * s_q4_j + offs_h[:, None] * s_q4_h + offs_k[None, :])
    q2 = tl.load(q_base + 2 * s_q4_j + offs_h[:, None] * s_q4_h + offs_k[None, :])
    q3 = tl.load(q_base + 3 * s_q4_j + offs_h[:, None] * s_q4_h + offs_k[None, :])

    acc = tl.dot(kv0, tl.trans(q0))
    acc += tl.dot(kv1, tl.trans(q1))
    acc += tl.dot(kv2, tl.trans(q2))
    acc += tl.dot(kv3, tl.trans(q3))
    acc = tl.maximum(acc, 0.0)
    head_weights = tl.load(W + pid_b * s_w_b + offs_h).to(tl.float32)
    reduced = tl.sum(acc * head_weights[None, :], axis=1)

    scale_bits = tl.load(
        CACHE_I32 + (pids * 2112 + 2048 + tok.to(tl.int64)),
        mask=n_mask,
        other=0,
    )
    reduced *= scale_bits.to(tl.float32, bitcast=True)
    tl.store(OUT + pid_b * s_out_b + offs_n, reduced, mask=n_mask)


def fused_paged_mqa_logits(
    q_fp8,
    kvcache_fp8,
    weight,
    seq_lens,
    page_table,
    schedule_metadata,
    max_seq_len,
    clean_logits=True,
):
    """Match DeepGEMM's paged-MQA logits interface without its SM90 gate."""
    del schedule_metadata
    assert clean_logits is False
    batch_size, _, num_heads, head_dim = q_fp8.shape
    assert head_dim == 128 and kvcache_fp8.shape[1:] == (64, 1, 132)
    max_pages = page_table.shape[1]
    padded_eff = min(max_pages * 64, max_seq_len)

    cache_i32 = kvcache_fp8.view(-1).view(torch.int32)
    q4 = (
        q_fp8[:, 0]
        .to(torch.bfloat16)
        .view(batch_size, num_heads, 32, 4)
        .permute(0, 3, 1, 2)
        .contiguous()
    )
    weights_f32 = weight if weight.dtype == torch.float32 else weight.float()
    logits = torch.empty(
        (batch_size, max_seq_len), dtype=torch.float32, device=q_fp8.device
    )
    grid = lambda meta: (batch_size, triton.cdiv(padded_eff, meta["BN"]))
    _fused_paged_indexer_kernel[grid](
        cache_i32,
        q4,
        weights_f32,
        page_table,
        logits,
        q4.stride(0),
        q4.stride(1),
        q4.stride(2),
        weights_f32.stride(0),
        page_table.stride(0),
        logits.stride(0),
        padded_eff,
        H=num_heads,
    )
    positions = torch.arange(max_seq_len, device=q_fp8.device)
    invalid = positions[None, :] >= seq_lens.reshape(batch_size)[:, None]
    logits.masked_fill_(invalid, float("-inf"))
    return logits


@triton.autotune(
    configs=[
        triton.Config({"BN": 32}, num_warps=4, num_stages=2),
        triton.Config({"BN": 64}, num_warps=4, num_stages=2),
        triton.Config({"BN": 64}, num_warps=8, num_stages=2),
        triton.Config({"BN": 128}, num_warps=8, num_stages=2),
    ],
    key=["H"],
)
@triton.jit
def _fused_ragged_indexer_kernel(
    K_I32,
    K_SCALE,
    Q4,
    W,
    KS,
    KE,
    OUT,
    s_q4_b,
    s_q4_j,
    s_q4_h,
    s_w_b,
    s_out_b,
    K_ROWS,
    H: tl.constexpr,
    BN: tl.constexpr,
):
    pid_q = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1)
    offs_n = pid_n * BN + tl.arange(0, BN)
    start = tl.load(KS + pid_q)
    end = tl.load(KE + pid_q)
    valid = (offs_n < K_ROWS) & (offs_n >= start) & (offs_n < end)

    k_ptrs = K_I32 + offs_n.to(tl.int64)[:, None] * 32 + tl.arange(0, 32)[None, :]
    kv_words = tl.load(k_ptrs, mask=valid[:, None], other=0)
    kv0 = _e4m3_to_f32(kv_words & 0xFF).to(tl.bfloat16)
    kv1 = _e4m3_to_f32((kv_words >> 8) & 0xFF).to(tl.bfloat16)
    kv2 = _e4m3_to_f32((kv_words >> 16) & 0xFF).to(tl.bfloat16)
    kv3 = _e4m3_to_f32((kv_words >> 24) & 0xFF).to(tl.bfloat16)

    offs_h = tl.arange(0, H)
    offs_k = tl.arange(0, 32)
    q_base = Q4 + pid_q * s_q4_b
    q0 = tl.load(q_base + 0 * s_q4_j + offs_h[:, None] * s_q4_h + offs_k[None, :])
    q1 = tl.load(q_base + 1 * s_q4_j + offs_h[:, None] * s_q4_h + offs_k[None, :])
    q2 = tl.load(q_base + 2 * s_q4_j + offs_h[:, None] * s_q4_h + offs_k[None, :])
    q3 = tl.load(q_base + 3 * s_q4_j + offs_h[:, None] * s_q4_h + offs_k[None, :])

    acc = tl.dot(kv0, tl.trans(q0))
    acc += tl.dot(kv1, tl.trans(q1))
    acc += tl.dot(kv2, tl.trans(q2))
    acc += tl.dot(kv3, tl.trans(q3))
    acc = tl.maximum(acc, 0.0)
    head_weights = tl.load(W + pid_q * s_w_b + offs_h).to(tl.float32)
    reduced = tl.sum(acc * head_weights[None, :], axis=1)
    reduced *= tl.load(K_SCALE + offs_n, mask=valid, other=0.0)
    tl.store(
        OUT + pid_q * s_out_b + offs_n,
        tl.where(valid, reduced, -float("inf")),
        mask=offs_n < K_ROWS,
    )


def fused_ragged_mqa_logits(
    q_fp8,
    k_fp8,
    k_scale,
    weight,
    starts,
    ends,
    *,
    clean_logits=True,
):
    """Portable contiguous/ragged equivalent of DeepGEMM fp8_mqa_logits."""
    assert clean_logits is True
    num_q, num_heads, head_dim = q_fp8.shape
    assert head_dim == 128 and k_fp8.shape[1] == 128
    k_rows = k_fp8.shape[0]
    assert k_scale.numel() == k_rows
    assert weight.shape == (num_q, num_heads)
    assert starts.shape == ends.shape == (num_q,)

    q4 = (
        q_fp8.to(torch.bfloat16)
        .view(num_q, num_heads, 32, 4)
        .permute(0, 3, 1, 2)
        .contiguous()
    )
    k_i32 = k_fp8.contiguous().view(torch.uint8).view(-1, 32).view(torch.int32)
    weights_f32 = weight if weight.dtype == torch.float32 else weight.float()
    scales_f32 = k_scale.reshape(-1).float()
    logits = torch.empty((num_q, k_rows), dtype=torch.float32, device=q_fp8.device)
    grid = lambda meta: (num_q, triton.cdiv(k_rows, meta["BN"]))
    _fused_ragged_indexer_kernel[grid](
        k_i32,
        scales_f32,
        q4,
        weights_f32,
        starts,
        ends,
        logits,
        q4.stride(0),
        q4.stride(1),
        q4.stride(2),
        weights_f32.stride(0),
        logits.stride(0),
        k_rows,
        H=num_heads,
    )
    return logits
