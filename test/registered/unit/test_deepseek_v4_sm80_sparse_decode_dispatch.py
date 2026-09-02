from types import SimpleNamespace
from unittest.mock import patch

from sglang.kernels.ops.attention import flash_mla_sm120
from sglang.srt.environ import envs
from sglang.srt.layers.attention import deepseek_v4_backend


def test_sm80_uses_sparse_decode_fallback():
    sm80 = SimpleNamespace(is_sm120=False, device_sm=80)
    with (
        patch.object(deepseek_v4_backend, "get_platform", return_value=sm80),
        patch.object(deepseek_v4_backend, "_is_cuda", True),
    ):
        assert deepseek_v4_backend._use_sparse_decode_fallback()


def test_sm90_keeps_aot_sparse_decode():
    sm90 = SimpleNamespace(is_sm120=False, device_sm=90)
    with (
        patch.object(deepseek_v4_backend, "get_platform", return_value=sm90),
        patch.object(deepseek_v4_backend, "_is_cuda", True),
    ):
        assert not deepseek_v4_backend._use_sparse_decode_fallback()


def test_sm80_routes_unsupported_flashinfer_backend_to_triton():
    with (
        envs.SGLANG_SM120_FLASHMLA_BACKEND.override("flashinfer"),
        patch.object(flash_mla_sm120.torch.cuda, "is_available", return_value=True),
        patch.object(
            flash_mla_sm120.torch.cuda,
            "get_device_capability",
            return_value=(8, 0),
        ),
    ):
        assert flash_mla_sm120._resolve_sparse_decode_backend() == "triton"


def test_sm90_keeps_flashinfer_backend():
    with (
        envs.SGLANG_SM120_FLASHMLA_BACKEND.override("flashinfer"),
        patch.object(flash_mla_sm120.torch.cuda, "is_available", return_value=True),
        patch.object(
            flash_mla_sm120.torch.cuda,
            "get_device_capability",
            return_value=(9, 0),
        ),
    ):
        assert flash_mla_sm120._resolve_sparse_decode_backend() == "flashinfer"


def test_flashmla_cuda_backend_forwards_sparse_decode_arguments():
    scheduler = object()
    expected_result = (object(), object())
    fake_flash_mla = SimpleNamespace(
        get_mla_metadata=lambda: (scheduler, None),
        flash_mla_with_kvcache=lambda **kwargs: (kwargs, expected_result),
    )
    q = SimpleNamespace(shape=(1, 1, 64, 512))

    with (
        patch.object(flash_mla_sm120, "_sparse_decode_backend", "flashmla_cuda"),
        patch.dict("sys.modules", {"flash_mla": fake_flash_mla}),
    ):
        forwarded, result = flash_mla_sm120.flash_mla_with_kvcache_sm120(
            q=q,
            k_cache="k_cache",
            indices="indices",
            topk_length="topk_length",
            attn_sink="attn_sink",
            head_dim_v=512,
            softmax_scale=0.125,
            extra_k_cache="extra_k_cache",
            extra_indices_in_kvcache="extra_indices",
            extra_topk_length="extra_topk_length",
        )

    assert result is expected_result
    assert forwarded["tile_scheduler_metadata"] is scheduler
    assert forwarded["is_fp8_kvcache"] is True
    assert forwarded["indices"] == "indices"
    assert forwarded["topk_length"] == "topk_length"
    assert forwarded["extra_indices_in_kvcache"] == "extra_indices"
