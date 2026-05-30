import torch

from flash_attention2_kernel import (
    flash_attention2,
    flash_attention2_reference,
    make_inputs,
)


def assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    rtol: float = 3e-2,
    atol: float = 3e-2,
) -> None:
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


def run_case(
    *,
    batch: int,
    heads: int,
    seq_len: int,
    head_dim: int,
    causal: bool,
) -> None:
    q, k, v = make_inputs(
        batch=batch,
        heads=heads,
        seq_len=seq_len,
        head_dim=head_dim,
        seed=seq_len + head_dim + batch + heads,
    )
    actual = flash_attention2(q, k, v, causal=causal)
    expected = flash_attention2_reference(q, k, v, causal=causal)
    assert_close(actual, expected)


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this test.")

    run_case(batch=1, heads=4, seq_len=64, head_dim=32, causal=False)
    run_case(batch=2, heads=4, seq_len=128, head_dim=64, causal=False)
    run_case(batch=1, heads=8, seq_len=128, head_dim=64, causal=True)
    print("All FlashAttention-2 correctness checks passed.")


if __name__ == "__main__":
    main()
