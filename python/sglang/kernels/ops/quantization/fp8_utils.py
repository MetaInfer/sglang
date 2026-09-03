from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import (
    get_jit_cuda_arch,
    is_hip_runtime,
    is_musa_runtime,
)


@triton.jit
def e4m3fn_to_uint8(x, USE_NATIVE_FP8: tl.constexpr):
    """Encode float32 values as E4M3FN bytes without requiring SM89 FP8 IR."""
    if USE_NATIVE_FP8:
        return x.to(tl.float8e4nv).to(tl.uint8, bitcast=True)

    bits = x.to(tl.int32, bitcast=True)
    sign = (bits >> 24) & 0x80
    abs_bits = bits & 0x7FFFFFFF
    exponent = ((abs_bits >> 23) & 0xFF) - 127
    mantissa = abs_bits & 0x7FFFFF

    mantissa3 = mantissa >> 20
    remainder = mantissa & 0xFFFFF
    round_up = (remainder > 0x80000) | ((remainder == 0x80000) & ((mantissa3 & 1) != 0))
    mantissa3 += round_up.to(tl.int32)
    exponent8 = exponent + 7 + (mantissa3 >> 3)
    normal = (exponent8 << 3) | (mantissa3 & 0x7)

    # E4M3FN subnormals are integer multiples of 2^-9. Round ties to even,
    # including the transition to the smallest normal value (byte 0x08).
    abs_x = tl.abs(x)
    subnormal_scaled = abs_x * 512.0
    subnormal = subnormal_scaled.to(tl.int32)
    subnormal_fraction = subnormal_scaled - subnormal.to(tl.float32)
    subnormal_round_up = (subnormal_fraction > 0.5) | (
        (subnormal_fraction == 0.5) & ((subnormal & 1) != 0)
    )
    subnormal += subnormal_round_up.to(tl.int32)

    magnitude = tl.where(abs_x < 0.015625, subnormal, normal)
    magnitude = tl.minimum(magnitude, 0x7E)
    return (sign | magnitude).to(tl.uint8)


def cuda_capability_uses_fp8_e4b15(cuda_capability: Tuple[int, int]) -> bool:
    """Triton names E4M3 as fp8e4b15 on CUDA architectures before SM89."""
    return cuda_capability < (8, 9)


def use_fp8_e4b15_for_e4m3fn(
    device: Optional[int] = None,
    cuda_capability: Optional[Tuple[int, int]] = None,
) -> bool:
    if cuda_capability is None:
        if is_hip_runtime() or is_musa_runtime() or not torch.cuda.is_available():
            return False
        if device is None:
            arch = get_jit_cuda_arch()
            cuda_capability = (arch.major, arch.minor)
        else:
            cuda_capability = torch.cuda.get_device_capability(device)

    return cuda_capability_uses_fp8_e4b15(cuda_capability)


def fp8_dtype_to_triton(
    fp8_dtype: torch.dtype,
    *,
    device: Optional[int] = None,
    cuda_capability: Optional[Tuple[int, int]] = None,
):
    if fp8_dtype == torch.float8_e4m3fn:
        if use_fp8_e4b15_for_e4m3fn(device, cuda_capability):
            return tl.float8e4b15
        return tl.float8e4nv
    if fp8_dtype == torch.float8_e4m3fnuz:
        return tl.float8e4b8
    if fp8_dtype == torch.float8_e5m2:
        return tl.float8e5
    raise ValueError(f"Unsupported FP8 dtype: {fp8_dtype}")
