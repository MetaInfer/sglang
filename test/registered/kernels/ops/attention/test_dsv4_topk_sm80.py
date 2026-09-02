import pytest
import torch
from sglang.kernels.ops.attention.dsv4.topk import topk_transform_512
from sglang.srt.utils.common import is_sm80_supported

pytestmark = pytest.mark.skipif(
    not is_sm80_supported(), reason="DSV4 Top-K A800 regression requires CUDA SM80"
)


def test_topk_v1_matches_torch_on_sm80():
    torch.manual_seed(11)
    batch_size = 2
    max_seq_len = 1024
    topk = 512
    page_size = 64
    seq_lens = torch.tensor([1024, 777], dtype=torch.int32, device="cuda")
    scores = torch.randn(batch_size, max_seq_len, dtype=torch.float32, device="cuda")

    num_pages = max_seq_len // page_size
    page_table = (
        torch.arange(num_pages, dtype=torch.int32, device="cuda")
        .unsqueeze(0)
        .expand(batch_size, -1)
        .contiguous()
    )
    output = torch.empty(batch_size, topk, dtype=torch.int32, device="cuda")
    topk_transform_512(scores, seq_lens, page_table, output, page_size)
    torch.cuda.synchronize()

    for row, length in enumerate(seq_lens.tolist()):
        expected = torch.topk(scores[row, :length], topk).indices.to(torch.int32)
        torch.testing.assert_close(
            output[row].sort().values,
            expected.sort().values,
            rtol=0,
            atol=0,
        )
