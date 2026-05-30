import torch

from gemm_residual_rmsnorm_gemm_swiglu_kernel import (
    gemm_residual_rmsnorm_gemm_swiglu,
    gemm_residual_rmsnorm_gemm_swiglu_reference,
    gemm_residual_rmsnorm_gemm_swiglu_split_weights,
    gemm_residual_rmsnorm_gemm_swiglu_split_weights_reference,
    gemm_residual_rmsnorm_gemm_swiglu_transposed_reference,
    gemm_residual_rmsnorm_gemm_swiglu_transposed_stage2,
    make_inputs,
    make_inputs_split_weights,
    make_inputs_transposed_stage2,
)


def assert_case(
    rows: int,
    hidden: int,
    inner0: int | None = None,
    mlp: int | None = None,
    block_M: int = 64,
    block_N: int = 64,
    block_K: int = 32,
    threads: int = 128,
    num_stages: int = 3,
) -> None:
    dtype = torch.float16
    inputs = make_inputs(
        rows=rows,
        hidden=hidden,
        inner0=inner0,
        mlp=mlp,
        dtype=dtype,
        seed=rows + hidden,
    )
    actual = gemm_residual_rmsnorm_gemm_swiglu(
        *inputs,
        block_M=block_M,
        block_N=block_N,
        block_K=block_K,
        threads=threads,
        num_stages=num_stages,
    )
    expected = gemm_residual_rmsnorm_gemm_swiglu_reference(*inputs)
    torch.testing.assert_close(actual, expected, rtol=4e-2, atol=4e-2)


def assert_transposed_case(
    rows: int,
    hidden: int,
    inner0: int | None = None,
    mlp: int | None = None,
    block_M: int = 64,
    block_N: int = 64,
    block_K: int = 32,
    threads: int = 128,
    num_stages: int = 3,
) -> None:
    dtype = torch.float16
    inputs = make_inputs_transposed_stage2(
        rows=rows,
        hidden=hidden,
        inner0=inner0,
        mlp=mlp,
        dtype=dtype,
        seed=rows + hidden,
    )
    actual = gemm_residual_rmsnorm_gemm_swiglu_transposed_stage2(
        *inputs,
        block_M=block_M,
        block_N=block_N,
        block_K=block_K,
        threads=threads,
        num_stages=num_stages,
    )
    expected = gemm_residual_rmsnorm_gemm_swiglu_transposed_reference(*inputs)
    torch.testing.assert_close(actual, expected, rtol=4e-2, atol=4e-2)


def assert_split_weights_case(
    rows: int,
    hidden: int,
    inner0: int | None = None,
    mlp: int | None = None,
    block_M: int = 64,
    block_N: int = 64,
    block_K: int = 32,
    threads: int = 128,
    num_stages: int = 3,
) -> None:
    dtype = torch.float16
    inputs = make_inputs_split_weights(
        rows=rows,
        hidden=hidden,
        inner0=inner0,
        mlp=mlp,
        dtype=dtype,
        seed=rows + hidden,
    )
    actual = gemm_residual_rmsnorm_gemm_swiglu_split_weights(
        *inputs,
        block_M=block_M,
        block_N=block_N,
        block_K=block_K,
        threads=threads,
        num_stages=num_stages,
    )
    expected = gemm_residual_rmsnorm_gemm_swiglu_split_weights_reference(*inputs)
    torch.testing.assert_close(actual, expected, rtol=4e-2, atol=4e-2)


def main() -> None:
    assert_case(rows=64, hidden=128, inner0=96, mlp=192)
    assert_case(rows=128, hidden=256, inner0=256, mlp=512, block_M=128, block_N=64)
    assert_transposed_case(rows=64, hidden=128, inner0=96, mlp=192)
    assert_transposed_case(rows=128, hidden=256, inner0=256, mlp=512, block_M=128, block_N=64)
    assert_split_weights_case(rows=64, hidden=128, inner0=96, mlp=192)
    assert_split_weights_case(rows=128, hidden=256, inner0=256, mlp=512, block_M=128, block_N=64)
    print("gemm_residual_rmsnorm_gemm_swiglu: correctness checks passed")


if __name__ == "__main__":
    main()
