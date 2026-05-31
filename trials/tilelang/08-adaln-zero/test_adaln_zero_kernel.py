import torch

from adaln_zero_kernel import (
    adaln_zero,
    adaln_zero_reference,
    make_inputs,
)


def assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    rtol: float = 1e-2,
    atol: float = 1e-2,
) -> None:
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


def run_case(
    *,
    N: int,
    D: int,
    eps: float = 1e-5,
) -> None:
    x, scale, shift, gate = make_inputs(N=N, D=D, seed=N + D)
    actual = adaln_zero(x, scale, shift, gate, eps=eps)
    expected = adaln_zero_reference(x, scale, shift, gate, eps=eps)
    assert_close(actual, expected)


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this test.")

    run_case(N=64, D=128)
    run_case(N=128, D=256)
    run_case(N=256, D=512)
    run_case(N=512, D=1024)
    run_case(N=1024, D=2048)
    run_case(N=2048, D=4096)
    run_case(N=4096, D=8192)
    print("All AdaLN-Zero correctness checks passed.")


if __name__ == "__main__":
    main()
