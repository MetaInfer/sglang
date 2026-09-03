import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import get_scalar_type
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.marlin import (
    MarlinMoeQuantInfo,
    fused_experts_none_to_marlin,
)
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestFp8MarlinMoeWiring(CustomTestCase):
    def test_fp8_scalar_type_is_not_integer_int8(self):
        from sgl_kernel.scalar_type import scalar_types

        self.assertEqual(
            get_scalar_type(num_bits=8, has_zp=False, is_fp8=True),
            scalar_types.float8_e4m3fn,
        )

    def test_fp8_method_builds_marlin_quant_info_from_prepared_fields(self):
        method = object.__new__(Fp8MoEMethod)
        method.moe_runner_config = MoeRunnerConfig(num_experts=16)

        w13 = torch.empty(2, 3, dtype=torch.int32)
        w2 = torch.empty(2, 3, dtype=torch.int32)
        s13 = torch.empty(2, 4, dtype=torch.bfloat16)
        s2 = torch.empty(2, 4, dtype=torch.bfloat16)
        expert_map = torch.tensor([0, 3], dtype=torch.int32)
        layer = SimpleNamespace(
            w13_weight=w13,
            w2_weight=w2,
            w13_weight_scale=s13,
            w2_weight_scale=s2,
            dispatcher=SimpleNamespace(local_expert_mapping=expert_map),
        )

        info = method.get_marlin_quant_info(layer)

        self.assertIs(info.w13_qweight, w13)
        self.assertIs(info.w2_qweight, w2)
        self.assertIs(info.w13_scales, s13)
        self.assertIs(info.w2_scales, s2)
        self.assertIs(info.expert_map, expert_map)
        self.assertEqual(info.global_num_experts, 16)
        self.assertEqual(info.weight_bits, 8)
        self.assertTrue(info.is_fp8)

    def test_post_load_prepares_fp8_experts_for_marlin(self):
        method = object.__new__(Fp8MoEMethod)
        method.block_quant = True
        method.is_fp4_expert = False
        method.weight_block_size = [128, 128]
        method._owns_moe_runner = True
        method.moe_runner_config = MoeRunnerConfig(params_dtype=torch.bfloat16)
        layer = SimpleNamespace(
            w13_weight=torch.empty(1, dtype=torch.float8_e4m3fn),
            dispatcher=SimpleNamespace(set_quant_config=MagicMock()),
        )

        with (
            patch.object(method, "process_weights_after_loading_block_quant") as base,
            patch(
                "sglang.srt.layers.quantization.fp8.get_moe_runner_backend",
                return_value=MoeRunnerBackend.MARLIN,
            ),
            patch(
                "sglang.srt.layers.quantization.fp8.prepare_moe_fp8_layer_for_marlin"
            ) as prepare,
        ):
            method.process_weights_after_loading(layer)

        base.assert_called_once_with(layer)
        prepare.assert_called_once_with(layer, size_k_first=False)
        self.assertEqual(layer.orig_dtype, torch.bfloat16)
        self.assertEqual(layer.weight_block_size, [128, 128])

    def test_marlin_runner_forwards_fp8_marker_and_prepared_fields(self):
        hidden_states = torch.ones(1, 4, dtype=torch.bfloat16)
        w13 = torch.empty(2, 3, dtype=torch.int32)
        w2 = torch.empty(2, 3, dtype=torch.int32)
        s13 = torch.empty(2, 4, dtype=torch.bfloat16)
        s2 = torch.empty(2, 4, dtype=torch.bfloat16)
        quant_info = MarlinMoeQuantInfo(
            w13_qweight=w13,
            w2_qweight=w2,
            w13_scales=s13,
            w2_scales=s2,
            w13_g_idx_sort_indices=None,
            w2_g_idx_sort_indices=None,
            weight_bits=8,
            is_fp8=True,
        )
        dispatch_output = SimpleNamespace(
            hidden_states=hidden_states,
            topk_output=SimpleNamespace(
                router_logits=torch.zeros(1, 2),
                topk_weights=torch.ones(1, 1),
                topk_ids=torch.zeros(1, 1, dtype=torch.int32),
            ),
        )
        workspace = torch.empty(1, dtype=torch.int32)

        with (
            patch(
                "sglang.srt.layers.quantization.marlin_utils.marlin_make_workspace",
                return_value=workspace,
            ),
            patch(
                "sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe."
                "fused_marlin_moe",
                return_value=torch.zeros_like(hidden_states),
            ) as fused,
        ):
            output = fused_experts_none_to_marlin(
                dispatch_output,
                quant_info,
                MoeRunnerConfig(activation="silu", is_gated=True),
            )

        self.assertEqual(output.hidden_states.dtype, hidden_states.dtype)
        kwargs = fused.call_args.kwargs
        self.assertIs(kwargs["w1"], w13)
        self.assertIs(kwargs["w2"], w2)
        self.assertIs(kwargs["w1_scale"], s13)
        self.assertIs(kwargs["w2_scale"], s2)
        self.assertIs(kwargs["workspace"], workspace)
        self.assertEqual(kwargs["num_bits"], 8)
        self.assertTrue(kwargs["is_fp8"])


if __name__ == "__main__":
    unittest.main()
