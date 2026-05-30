import torch

from fused_softmax_kernel import fused_softmax, fused_softmax_reference


def assert_case(rows: int, cols: int, block_N: int = 128, threads: int = 128) -> None:
    x = torch.randn(rows, cols, device="cuda", dtype=torch.float16)

    actual = fused_softmax(
        x,
        block_N=block_N,
        threads=threads,
    )
    expected = fused_softmax_reference(x)

    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=2e-3)


def main() -> None:
    assert_case(rows=16, cols=33, block_N=64)
    assert_case(rows=128, cols=257, block_N=128)
    print("fused_softmax: correctness checks passed")


if __name__ == "__main__":
    main()
