"""Portable fused paged FP8 scoring for the DeepSeek-V4 C4 indexer."""

import torch
import triton
import triton.language as tl


@triton.jit
def _e4m3_to_f32(value):
    exponent = (value >> 3) & 0xF
    mantissa = value & 0x7
    magnitude = (((exponent + 120) << 23) | (mantissa << 20)).to(
        tl.float32, bitcast=True
    )
    magnitude = tl.where(
        exponent == 0, mantissa.to(tl.float32) * 1.953125e-3, magnitude
    )
    return tl.where((value >> 7) & 1 == 1, -magnitude, magnitude)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_N": 128}, num_warps=8, num_stages=2),
    ],
    key=["NUM_HEADS"],
)
@triton.jit
def _fused_paged_indexer_kernel(
    cache_i32,
    q_lanes,
    weights,
    page_table,
    output,
    q_batch_stride,
    q_lane_stride,
    q_head_stride,
    weight_batch_stride,
    page_table_batch_stride,
    output_batch_stride,
    padded_length,
    NUM_HEADS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    batch_idx = tl.program_id(0).to(tl.int64)
    block_idx = tl.program_id(1)
    token_offsets = block_idx * BLOCK_N + tl.arange(0, BLOCK_N)
    token_mask = token_offsets < padded_length

    page_offsets = token_offsets // 64
    page_ids = tl.load(
        page_table + batch_idx * page_table_batch_stride + page_offsets,
        mask=token_mask,
        other=0,
    )
    page_ids = tl.maximum(page_ids, 0).to(tl.int64)
    tokens_in_page = token_offsets % 64

    # Each page is 8192 E4M3 value bytes followed by 64 FP32 scales.
    packed_ptrs = cache_i32 + (page_ids * 2112 + tokens_in_page.to(tl.int64) * 32)
    packed = tl.load(
        packed_ptrs[:, None] + tl.arange(0, 32)[None, :],
        mask=token_mask[:, None],
        other=0,
    )
    kv0 = _e4m3_to_f32(packed & 0xFF).to(tl.bfloat16)
    kv1 = _e4m3_to_f32((packed >> 8) & 0xFF).to(tl.bfloat16)
    kv2 = _e4m3_to_f32((packed >> 16) & 0xFF).to(tl.bfloat16)
    kv3 = _e4m3_to_f32((packed >> 24) & 0xFF).to(tl.bfloat16)

    head_offsets = tl.arange(0, NUM_HEADS)
    dim_offsets = tl.arange(0, 32)
    q_base = q_lanes + batch_idx * q_batch_stride
    q0 = tl.load(q_base + head_offsets[:, None] * q_head_stride + dim_offsets[None, :])
    q1 = tl.load(
        q_base
        + q_lane_stride
        + head_offsets[:, None] * q_head_stride
        + dim_offsets[None, :]
    )
    q2 = tl.load(
        q_base
        + 2 * q_lane_stride
        + head_offsets[:, None] * q_head_stride
        + dim_offsets[None, :]
    )
    q3 = tl.load(
        q_base
        + 3 * q_lane_stride
        + head_offsets[:, None] * q_head_stride
        + dim_offsets[None, :]
    )

    logits = tl.dot(kv0, tl.trans(q0))
    logits += tl.dot(kv1, tl.trans(q1))
    logits += tl.dot(kv2, tl.trans(q2))
    logits += tl.dot(kv3, tl.trans(q3))
    logits = tl.maximum(logits, 0.0)

    row_weights = tl.load(weights + batch_idx * weight_batch_stride + head_offsets).to(
        tl.float32
    )
    scores = tl.sum(logits * row_weights[None, :], axis=1)
    scale_bits = tl.load(
        cache_i32 + page_ids * 2112 + 2048 + tokens_in_page.to(tl.int64),
        mask=token_mask,
        other=0,
    )
    scores *= scale_bits.to(tl.float32, bitcast=True)
    tl.store(
        output + batch_idx * output_batch_stride + token_offsets,
        scores,
        mask=token_mask,
    )


def fused_paged_mqa_logits(
    q_fp8,
    kvcache_fp8,
    weight,
    seq_lens,
    page_table,
    deep_gemm_metadata,
    max_seq_len,
    clean_logits=True,
):
    """Match the DeepGEMM paged-MQA signature without using FP8 instructions."""
    _ = deep_gemm_metadata
    assert clean_logits is False
    batch_size, next_n, num_heads, head_dim = q_fp8.shape
    assert next_n == 1
    assert head_dim == 128
    assert kvcache_fp8.shape[1:] == (64, 1, 132)

    max_paged_length = page_table.shape[1] * 64
    padded_length = min(max_paged_length, max_seq_len)
    cache_i32 = kvcache_fp8.view(-1).view(torch.int32)
    q_lanes = (
        q_fp8[:, 0]
        .to(torch.bfloat16)
        .view(batch_size, num_heads, 32, 4)
        .permute(0, 3, 1, 2)
        .contiguous()
    )
    weights_fp32 = weight if weight.dtype == torch.float32 else weight.float()
    logits = torch.empty(
        (batch_size, max_seq_len), dtype=torch.float32, device=q_fp8.device
    )

    grid = lambda meta: (
        batch_size,
        triton.cdiv(padded_length, meta["BLOCK_N"]),
    )
    _fused_paged_indexer_kernel[grid](
        cache_i32,
        q_lanes,
        weights_fp32,
        page_table,
        logits,
        q_lanes.stride(0),
        q_lanes.stride(1),
        q_lanes.stride(2),
        weights_fp32.stride(0),
        page_table.stride(0),
        logits.stride(0),
        padded_length,
        NUM_HEADS=num_heads,
    )

    positions = torch.arange(max_seq_len, device=q_fp8.device)
    invalid = positions[None, :] >= seq_lens.reshape(batch_size)[:, None]
    logits.masked_fill_(invalid, float("-inf"))
    return logits
