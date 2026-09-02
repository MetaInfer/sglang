import pytest
import torch
from sglang.kernels.ops.attention.dsv4.dequant_k_cache import (
    DIM_ROPE,
    NOPE_ROPE_BYTES,
    PADDED_SCALE_PER_TOKEN,
    dequantize_k_cache_paged,
    dequantize_k_cache_paged_ref,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0),
    reason="requires an SM80 CUDA GPU",
)


def test_paged_e4m3_dequant_matches_torch_reference():
    torch.manual_seed(23)
    page_size, num_pages, num_tokens = 256, 4, 333
    raw_bytes = page_size * (NOPE_ROPE_BYTES + PADDED_SCALE_PER_TOKEN)
    bytes_per_page = (
        (raw_bytes + NOPE_ROPE_BYTES - 1) // NOPE_ROPE_BYTES
    ) * NOPE_ROPE_BYTES
    packed = torch.randint(
        0,
        256,
        (num_pages, bytes_per_page),
        dtype=torch.uint8,
        device="cuda",
    )
    locations = torch.randint(
        0,
        num_pages * page_size,
        (num_tokens,),
        dtype=torch.int32,
        device="cuda",
    )

    actual = dequantize_k_cache_paged(packed, locations, page_size)
    expected = dequantize_k_cache_paged_ref(packed, locations, page_size)
    torch.cuda.synchronize()

    assert actual.shape[-1] == NOPE_ROPE_BYTES - DIM_ROPE
    torch.testing.assert_close(actual, expected, atol=0, rtol=0, equal_nan=True)
