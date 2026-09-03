import sys
import unittest
from types import ModuleType
from unittest.mock import MagicMock, patch

from sglang.srt.layers.attention.dsa import dsa_indexer_kpool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestKPoolMqaBackend(CustomTestCase):
    def test_paged_selectors_preserve_sm90_tilelang_and_route_sm80_to_triton(self):
        q = MagicMock()
        q.device = object()
        q.shape = (1, 1, 16, 128)

        with (
            patch.object(dsa_indexer_kpool, "is_cuda", return_value=True),
            patch("torch.cuda.get_device_capability", return_value=(9, 0)),
        ):
            self.assertTrue(
                dsa_indexer_kpool.IndexerKPool._should_use_tilelang_paged_mqa_logits(q)
            )
            self.assertFalse(
                dsa_indexer_kpool.IndexerKPool._should_use_triton_paged_mqa_logits(q)
            )

        with (
            patch.object(dsa_indexer_kpool, "is_cuda", return_value=True),
            patch("torch.cuda.get_device_capability", return_value=(8, 0)),
        ):
            self.assertFalse(
                dsa_indexer_kpool.IndexerKPool._should_use_tilelang_paged_mqa_logits(q)
            )
            self.assertTrue(
                dsa_indexer_kpool.IndexerKPool._should_use_triton_paged_mqa_logits(q)
            )

    def test_rocm_keeps_aiter_ragged_mqa_logits(self):
        marker = object()
        aiter_impl = MagicMock(return_value=marker)
        module = ModuleType("aiter.ops.triton.fp8_mqa_logits")
        module.fp8_mqa_logits = aiter_impl
        args = tuple(object() for _ in range(6))

        with (
            patch.object(dsa_indexer_kpool, "is_hip", return_value=True),
            patch.dict(sys.modules, {"aiter.ops.triton.fp8_mqa_logits": module}),
        ):
            result = dsa_indexer_kpool.IndexerKPool._fp8_mqa_logits(
                *args, clean_logits=False
            )

        self.assertIs(result, marker)
        aiter_impl.assert_called_once_with(*args, clean_logits=False)

    def test_sm90_keeps_deep_gemm_ragged_mqa_logits(self):
        marker = object()
        deep_gemm = MagicMock()
        deep_gemm.fp8_mqa_logits.return_value = marker
        q_fp8 = MagicMock()
        q_fp8.device = object()
        k_fp8, k_scale, weights, starts, ends = (object() for _ in range(5))

        with (
            patch.object(dsa_indexer_kpool, "is_hip", return_value=False),
            patch.object(dsa_indexer_kpool, "is_cuda", return_value=True),
            patch("torch.cuda.get_device_capability", return_value=(9, 0)),
            patch.object(dsa_indexer_kpool, "deep_gemm", deep_gemm, create=True),
        ):
            result = dsa_indexer_kpool.IndexerKPool._fp8_mqa_logits(
                q_fp8,
                k_fp8,
                k_scale,
                weights,
                starts,
                ends,
                clean_logits=True,
            )

        self.assertIs(result, marker)
        deep_gemm.fp8_mqa_logits.assert_called_once_with(
            q_fp8,
            (k_fp8, k_scale),
            weights,
            starts,
            ends,
            clean_logits=True,
        )

    def test_sm80_uses_portable_ragged_mqa_logits(self):
        marker = object()
        q_fp8 = MagicMock()
        q_fp8.device = object()
        k_fp8, k_scale, weights, starts, ends = (object() for _ in range(5))

        with (
            patch.object(dsa_indexer_kpool, "is_hip", return_value=False),
            patch.object(dsa_indexer_kpool, "is_cuda", return_value=True),
            patch("torch.cuda.get_device_capability", return_value=(8, 0)),
            patch(
                "sglang.kernels.ops.attention.dsa.fused_paged_indexer."
                "fused_ragged_mqa_logits",
                return_value=marker,
            ) as portable,
        ):
            result = dsa_indexer_kpool.IndexerKPool._fp8_mqa_logits(
                q_fp8,
                k_fp8,
                k_scale,
                weights,
                starts,
                ends,
                clean_logits=True,
            )

        self.assertIs(result, marker)
        portable.assert_called_once_with(
            q_fp8,
            k_fp8,
            k_scale,
            weights,
            starts,
            ends,
            clean_logits=True,
        )


if __name__ == "__main__":
    unittest.main()
