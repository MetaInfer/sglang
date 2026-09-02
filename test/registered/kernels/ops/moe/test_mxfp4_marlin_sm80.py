from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (
    fused_marlin_moe,
)
from sglang.srt.layers.quantization.marlin_utils_fp4 import (
    prepare_moe_mxfp4_layer_for_marlin,
)
from sglang.srt.layers.quantization.mxfp4_marlin_moe import (
    is_mxfp4_marlin_supported,
)
from sglang.srt.utils.common import is_sm80_supported

pytestmark = pytest.mark.skipif(
    not is_sm80_supported(), reason="MXFP4 Marlin A800 regression requires CUDA SM80"
)

_E2M1_LUT = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)


def _dequant_mxfp4(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Dequantize checkpoint MXFP4: low nibble is the even K element."""
    packed_u8 = packed.view(torch.uint8)
    codes = torch.stack((packed_u8 & 0x0F, packed_u8 >> 4), dim=-1).long()
    magnitudes = _E2M1_LUT.to(packed.device)[codes & 0x7]
    values = torch.where(codes >= 8, -magnitudes, magnitudes).flatten(-2)
    scale_values = scales.view(torch.float8_e8m0fnu).float()
    return values * scale_values.repeat_interleave(32, dim=-1)


def test_mxfp4_marlin_gated_moe_matches_dequant_reference_on_sm80():
    assert is_mxfp4_marlin_supported()
    torch.manual_seed(7)
    device = "cuda"
    num_tokens = 11
    num_experts = 4
    topk = 2
    hidden_size = 256
    intermediate_size = 128

    def packed(*shape: int) -> torch.Tensor:
        return torch.randint(0, 256, shape, dtype=torch.uint8, device=device)

    def e8m0_scales(*shape: int) -> torch.Tensor:
        # Keep the reference well conditioned: 2**-5 through 2**-3.
        raw = torch.randint(122, 125, shape, dtype=torch.uint8, device=device)
        return raw.view(torch.float8_e8m0fnu)

    raw_w13 = packed(num_experts, 2 * intermediate_size, hidden_size // 2)
    raw_w2 = packed(num_experts, hidden_size, intermediate_size // 2)
    raw_w13_scale = e8m0_scales(num_experts, 2 * intermediate_size, hidden_size // 32)
    raw_w2_scale = e8m0_scales(num_experts, hidden_size, intermediate_size // 32)

    w13_ref = _dequant_mxfp4(raw_w13, raw_w13_scale)
    w2_ref = _dequant_mxfp4(raw_w2, raw_w2_scale)

    layer = torch.nn.Module()
    layer.orig_dtype = torch.bfloat16
    layer.moe_runner_config = SimpleNamespace(is_gated=True)
    layer.w13_weight = torch.nn.Parameter(raw_w13, requires_grad=False)
    layer.w2_weight = torch.nn.Parameter(raw_w2, requires_grad=False)
    layer.w13_weight_scale_inv = torch.nn.Parameter(raw_w13_scale, requires_grad=False)
    layer.w2_weight_scale_inv = torch.nn.Parameter(raw_w2_scale, requires_grad=False)
    prepare_moe_mxfp4_layer_for_marlin(layer)

    hidden_states = (
        torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16) / 20
    )
    router_logits = torch.randn(num_tokens, num_experts, device=device)
    routing = torch.softmax(router_logits, dim=-1, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(routing, topk, dim=-1)

    output = fused_marlin_moe(
        hidden_states=hidden_states,
        w1=layer.w13_weight,
        w2=layer.w2_weight,
        w1_scale=layer.w13_weight_scale,
        w2_scale=layer.w2_weight_scale,
        gating_output=router_logits,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        workspace=layer.workspace,
        num_bits=4,
        is_k_full=True,
        routed_scaling_factor=1.0,
        activation="silu",
        is_gated=True,
    )

    output_ref = torch.zeros_like(hidden_states, dtype=torch.float32)
    hidden_float = hidden_states.float()
    for token_idx in range(num_tokens):
        for route_idx in range(topk):
            expert_id = int(topk_ids[token_idx, route_idx])
            intermediate = hidden_float[token_idx] @ w13_ref[expert_id].T
            gate, up = intermediate.chunk(2)
            activated = F.silu(gate) * up
            routed = activated @ w2_ref[expert_id].T
            output_ref[token_idx] += routed * topk_weights[token_idx, route_idx]

    torch.cuda.synchronize()
    abs_error = (output.float() - output_ref).abs()
    print(
        "MXFP4 Marlin SM80 error: "
        f"max_abs={abs_error.max().item():.6f}, "
        f"mean_abs={abs_error.mean().item():.6f}"
    )
    torch.testing.assert_close(output.float(), output_ref, rtol=0.08, atol=0.35)
