from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.model_hook import (
    _configure_deepseek_v4_cuda_capability_envs,
)
from sglang.srt.environ import envs


def test_sm80_disables_cluster_topk_fp8_wo_and_deepgemm_mhc():
    sm80 = SimpleNamespace(is_cuda=True, device_sm=80)
    with (
        patch("sglang.srt.arg_groups.model_hook.get_platform", return_value=sm80),
        patch(
            "sglang.srt.layers.deep_gemm_wrapper.configurer.ENABLE_JIT_DEEPGEMM",
            False,
        ),
        envs.SGLANG_OPT_USE_TOPK_V2.override(True),
        envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.override(True),
        envs.SGLANG_OPT_FP8_WO_A_GEMM.override(True),
    ):
        _configure_deepseek_v4_cuda_capability_envs()
        assert not envs.SGLANG_OPT_USE_TOPK_V2.get()
        assert not envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.get()
        assert not envs.SGLANG_OPT_FP8_WO_A_GEMM.get()


def test_sm90_keeps_supported_paths_when_deepgemm_is_available():
    sm90 = SimpleNamespace(is_cuda=True, device_sm=90)
    with (
        patch("sglang.srt.arg_groups.model_hook.get_platform", return_value=sm90),
        patch(
            "sglang.srt.layers.deep_gemm_wrapper.configurer.ENABLE_JIT_DEEPGEMM",
            True,
        ),
        envs.SGLANG_OPT_USE_TOPK_V2.override(True),
        envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.override(True),
        envs.SGLANG_OPT_FP8_WO_A_GEMM.override(True),
    ):
        _configure_deepseek_v4_cuda_capability_envs()
        assert envs.SGLANG_OPT_USE_TOPK_V2.get()
        assert envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.get()
        assert envs.SGLANG_OPT_FP8_WO_A_GEMM.get()


def test_cuda_without_sm_keeps_architecture_gates():
    cuda_without_sm = SimpleNamespace(is_cuda=True, device_sm=None)
    with (
        patch(
            "sglang.srt.arg_groups.model_hook.get_platform",
            return_value=cuda_without_sm,
        ),
        patch(
            "sglang.srt.layers.deep_gemm_wrapper.configurer.ENABLE_JIT_DEEPGEMM",
            True,
        ),
        envs.SGLANG_OPT_USE_TOPK_V2.override(True),
        envs.SGLANG_OPT_FP8_WO_A_GEMM.override(True),
    ):
        _configure_deepseek_v4_cuda_capability_envs()
        assert envs.SGLANG_OPT_USE_TOPK_V2.get()
        assert envs.SGLANG_OPT_FP8_WO_A_GEMM.get()


def test_non_cuda_keeps_capability_envs():
    cpu = SimpleNamespace(is_cuda=False, device_sm=None)
    with (
        patch("sglang.srt.arg_groups.model_hook.get_platform", return_value=cpu),
        envs.SGLANG_OPT_USE_TOPK_V2.override(True),
        envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.override(True),
        envs.SGLANG_OPT_FP8_WO_A_GEMM.override(True),
    ):
        _configure_deepseek_v4_cuda_capability_envs()
        assert envs.SGLANG_OPT_USE_TOPK_V2.get()
        assert envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.get()
        assert envs.SGLANG_OPT_FP8_WO_A_GEMM.get()
