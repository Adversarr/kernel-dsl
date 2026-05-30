import torch

from fused_gemm_relu_kernel import fused_gemm_relu, fused_gemm_relu_reference


def assert_case(
    rows: int,
    cols: int,
    inner: int,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
    threads: int = 128,
    num_stages: int = 3,
) -> None:
    dtype = torch.float16
    lhs = torch.randn(rows, inner, device="cuda", dtype=dtype)
    rhs = torch.randn(inner, cols, device="cuda", dtype=dtype)

    actual = fused_gemm_relu(
        lhs,
        rhs,
        block_M=block_M,
        block_N=block_N,
        block_K=block_K,
        threads=threads,
        num_stages=num_stages,
    )
    expected = fused_gemm_relu_reference(lhs, rhs)

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def main() -> None:
    assert_case(rows=128, cols=128, inner=128)
    assert_case(rows=192, cols=96, inner=64, block_M=64, block_N=64, block_K=32)
    print("fused_gemm_relu: correctness checks passed")


if __name__ == "__main__":
    main()
