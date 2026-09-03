import sys
import unittest
from types import ModuleType
from unittest.mock import MagicMock, patch

import torch
from sglang.srt.layers.attention import dsa_backend
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDSAFlashMLASM80Backend(unittest.TestCase):
    @staticmethod
    def _backend(capability=(8, 0)):
        backend = object.__new__(dsa_backend.DeepseekSparseAttnBackend)
        backend.device_capability = capability
        backend.device_sm_major = capability[0]
        return backend

    @staticmethod
    def _inputs(rows=2, q_dim=512, kv_dim=512):
        return (
            torch.empty((rows, 8, q_dim), dtype=torch.bfloat16),
            torch.empty((16, 1, kv_dim), dtype=torch.bfloat16),
            torch.zeros((rows, 8), dtype=torch.int32),
            torch.full((rows,), 8, dtype=torch.int32),
        )

    def _run(self, backend, q, kv, indices, lengths, *, decode=True, v_dim=512):
        return backend._forward_flashmla_sparse(
            q_all=q,
            kv_cache=kv,
            v_head_dim=v_dim,
            page_table_1=indices,
            sm_scale=0.0625,
            topk_length=lengths,
            use_sm80_decode_extension=decode,
        )

    def _assert_triton_fallback(self, **overrides):
        q, kv, indices, lengths = self._inputs()
        backend = self._backend(overrides.pop("capability", (8, 0)))
        q = overrides.pop("q", q)
        kv = overrides.pop("kv", kv)
        indices = overrides.pop("indices", indices)
        lengths = overrides.pop("lengths", lengths)
        marker = object()

        with (
            patch.object(
                dsa_backend.envs.SGLANG_DSA_SM80_FLASHMLA,
                "get",
                return_value=overrides.pop("enabled", True),
            ),
            patch(
                "sglang.kernels.ops.attention.dsa.triton_sparse_mla."
                "triton_sparse_mla_fwd",
                return_value=marker,
            ) as triton_call,
        ):
            output = self._run(
                backend,
                q,
                kv,
                indices,
                lengths,
                decode=overrides.pop("decode", True),
                v_dim=overrides.pop("v_dim", 512),
            )

        self.assertFalse(overrides, overrides)
        self.assertIs(output, marker)
        triton_call.assert_called_once()

    def test_decode_routes_exact_glm_shape_to_external_flashmla(self):
        backend = self._backend()
        q, kv, indices, lengths = self._inputs()
        expected = torch.empty_like(q)
        flashmla_call = MagicMock(return_value=expected)
        flashmla_module = ModuleType("flash_mla")
        flashmla_module.flash_mla_sparse_decode_bf16_sm80 = flashmla_call

        with (
            patch.object(
                dsa_backend.envs.SGLANG_DSA_SM80_FLASHMLA,
                "get",
                return_value=True,
            ),
            patch.dict(sys.modules, {"flash_mla": flashmla_module}),
        ):
            output = self._run(backend, q, kv, indices, lengths)

        self.assertIs(output._base, expected)
        self.assertEqual(output.shape, (1, 2, 8, 512))
        flashmla_call.assert_called_once_with(
            q=q,
            kv=kv,
            indices=indices,
            sm_scale=0.0625,
            topk_length=lengths,
        )

    def test_prefill_never_routes_to_external_extension(self):
        self._assert_triton_fallback(decode=False)

    def test_non_sm80_capability_falls_back(self):
        self._assert_triton_fallback(capability=(8, 9))

    def test_disabled_flag_falls_back(self):
        self._assert_triton_fallback(enabled=False)

    def test_non_bf16_or_non_512_shapes_fall_back(self):
        q, kv, _, _ = self._inputs()
        self._assert_triton_fallback(q=q.float())
        self._assert_triton_fallback(kv=kv.float())
        q256, kv256, _, _ = self._inputs(q_dim=256, kv_dim=256)
        self._assert_triton_fallback(q=q256, kv=kv256, v_dim=256)

    def test_indices_contract_mismatch_falls_back(self):
        self._assert_triton_fallback(indices=torch.zeros((2, 8), dtype=torch.int64))
        self._assert_triton_fallback(indices=torch.zeros((1, 8), dtype=torch.int32))

    def test_topk_length_contract_mismatch_falls_back(self):
        self._assert_triton_fallback(lengths=None)
        self._assert_triton_fallback(lengths=torch.full((2,), 8, dtype=torch.int64))
        self._assert_triton_fallback(lengths=torch.full((1,), 8, dtype=torch.int32))

    def test_missing_extension_falls_back(self):
        backend = self._backend()
        q, kv, indices, lengths = self._inputs()
        marker = object()

        with (
            patch.object(
                dsa_backend.envs.SGLANG_DSA_SM80_FLASHMLA,
                "get",
                return_value=True,
            ),
            patch.dict(sys.modules, {"flash_mla": ModuleType("flash_mla")}),
            patch(
                "sglang.kernels.ops.attention.dsa.triton_sparse_mla."
                "triton_sparse_mla_fwd",
                return_value=marker,
            ),
        ):
            output = self._run(backend, q, kv, indices, lengths)

        self.assertIs(output, marker)

    def test_external_runtime_error_is_not_swallowed(self):
        backend = self._backend()
        q, kv, indices, lengths = self._inputs()
        flashmla_module = ModuleType("flash_mla")
        flashmla_module.flash_mla_sparse_decode_bf16_sm80 = MagicMock(
            side_effect=RuntimeError("kernel launch failed")
        )

        with (
            patch.object(
                dsa_backend.envs.SGLANG_DSA_SM80_FLASHMLA,
                "get",
                return_value=True,
            ),
            patch.dict(sys.modules, {"flash_mla": flashmla_module}),
            self.assertRaisesRegex(RuntimeError, "kernel launch failed"),
        ):
            self._run(backend, q, kv, indices, lengths)


if __name__ == "__main__":
    unittest.main()
