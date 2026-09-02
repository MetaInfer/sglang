import pytest
import torch
from sglang.kernels.ops.attention.dsv4.fused_paged_indexer import (
    fused_paged_mqa_logits,
)
from sglang.srt.layers.attention.dsv4.indexer import (
    FP8_DTYPE,
    fp8_paged_mqa_logits_torch_sm120,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0),
    reason="requires an SM80 CUDA GPU",
)


def test_fused_paged_indexer_matches_torch_reference():
    torch.manual_seed(17)
    device = "cuda"
    batch_size, num_heads, head_dim = 2, 64, 128
    num_pages, page_size, max_seq_len = 6, 64, 192

    values = (
        torch.randn(
            num_pages,
            page_size,
            head_dim,
            device=device,
            dtype=torch.bfloat16,
        )
        * 0.25
    ).to(FP8_DTYPE)
    scales = torch.rand(num_pages, page_size, device=device, dtype=torch.float32).add_(
        0.5
    )
    packed = torch.empty(
        num_pages, page_size * (head_dim + 4), device=device, dtype=torch.uint8
    )
    packed[:, : page_size * head_dim].copy_(
        values.view(torch.uint8).reshape(num_pages, -1)
    )
    packed[:, page_size * head_dim :].copy_(
        scales.view(torch.uint8).reshape(num_pages, -1)
    )
    kvcache = packed.view(FP8_DTYPE).view(num_pages, page_size, 1, head_dim + 4)

    q = (
        torch.randn(
            batch_size,
            1,
            num_heads,
            head_dim,
            device=device,
            dtype=torch.bfloat16,
        )
        * 0.25
    ).to(FP8_DTYPE)
    weights = torch.randn(batch_size, num_heads, device=device, dtype=torch.float32)
    seq_lens = torch.tensor([145, 97], device=device, dtype=torch.int32)
    page_table = torch.tensor([[2, 0, 1], [5, 3, 4]], device=device, dtype=torch.int32)

    expected = fp8_paged_mqa_logits_torch_sm120(
        q,
        kvcache,
        weights,
        seq_lens,
        page_table,
        None,
        max_seq_len,
        clean_logits=False,
    )
    actual = fused_paged_mqa_logits(
        q,
        kvcache,
        weights,
        seq_lens,
        page_table,
        None,
        max_seq_len,
        clean_logits=False,
    )
    torch.cuda.synchronize()

    for row, valid_length in enumerate(seq_lens.tolist()):
        torch.testing.assert_close(
            actual[row, :valid_length],
            expected[row, :valid_length],
            atol=2e-2,
            rtol=2e-3,
        )
        assert torch.isneginf(actual[row, valid_length:]).all()
