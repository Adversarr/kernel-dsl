import torch

from group_gemm_kernel import (
    DEFAULT_BLOCK_K,
    DEFAULT_BLOCK_M,
    DEFAULT_BLOCK_N,
    DEFAULT_NUM_STAGES,
    DEFAULT_THREADS,
    DEFAULT_ENABLE_SWIZZLE,
    autotune_group_gemm,
    make_demo_inputs,
    make_group_gemm_jit_kernel,
    pack_group_gemm_inputs,
    run_group_gemm,
    unpack_group_gemm_output,
    group_gemm_reference,
)


def assert_group_close(
    actual_group: list[torch.Tensor],
    expected_group: list[torch.Tensor],
    *,
    atol: float = 2e-2,
    rtol: float = 2e-2,
) -> None:
    assert len(actual_group) == len(expected_group)
    for index, (actual, expected) in enumerate(zip(actual_group, expected_group)):
        torch.testing.assert_close(
            actual,
            expected,
            atol=atol,
            rtol=rtol,
            msg=lambda message: f"group index {index}: {message}",
        )


def test_group_gemm_fixed_config_matches_pytorch() -> None:
    group_lhs, group_rhs = make_demo_inputs(
        shapes=[
            (137, 193, 211),
            (288, 160, 224),
            (96, 320, 128),
            (511, 255, 384),
        ],
        seed=17,
    )

    actual_group = run_group_gemm(
        group_lhs,
        group_rhs,
        block_M=DEFAULT_BLOCK_M,
        block_N=DEFAULT_BLOCK_N,
        block_K=DEFAULT_BLOCK_K,
        threads=DEFAULT_THREADS,
        num_stages=DEFAULT_NUM_STAGES,
        enable_swizzle=DEFAULT_ENABLE_SWIZZLE,
    )
    expected_group = group_gemm_reference(group_lhs, group_rhs)
    assert_group_close(actual_group, expected_group)


def test_group_gemm_compiled_kernel_matches_pytorch() -> None:
    group_lhs, group_rhs = make_demo_inputs(
        shapes=[
            (256, 384, 512),
            (384, 192, 512),
            (128, 640, 512),
            (512, 96, 512),
        ],
        seed=23,
    )
    packed_lhs, packed_rhs, group_sizes = pack_group_gemm_inputs(group_lhs, group_rhs)
    kernel = make_group_gemm_jit_kernel(
        packed_lhs,
        packed_rhs,
        block_M=128,
        block_N=128,
        block_K=32,
        threads=128,
        num_stages=3,
        enable_swizzle=False,
    )

    packed_out = kernel(packed_lhs, packed_rhs, group_sizes)
    actual_group = unpack_group_gemm_output(packed_out, group_sizes)
    expected_group = group_gemm_reference(group_lhs, group_rhs)
    assert_group_close(actual_group, expected_group)


def test_group_gemm_autotune_smoke() -> None:
    group_lhs, group_rhs = make_demo_inputs(
        shapes=[
            (256, 256, 512),
            (192, 384, 512),
            (320, 128, 512),
            (128, 512, 512),
        ],
        seed=31,
    )
    packed_lhs, packed_rhs, group_sizes = pack_group_gemm_inputs(group_lhs, group_rhs)

    result = autotune_group_gemm(
        packed_lhs,
        packed_rhs,
        group_sizes,
        warmup=1,
        rep=1,
    )
    packed_out = result.kernel(packed_lhs, packed_rhs, group_sizes)
    actual_group = unpack_group_gemm_output(packed_out, group_sizes)
    expected_group = group_gemm_reference(group_lhs, group_rhs)
    assert_group_close(actual_group, expected_group)
