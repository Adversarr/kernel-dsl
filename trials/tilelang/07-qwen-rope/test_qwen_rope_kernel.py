import torch

from qwen_rope_kernel import (
    DEFAULT_ELEMENTS_PER_THREAD,
    DEFAULT_THREADS,
    QwenRopeFunction,
    make_inputs,
    qwen_rope,
    qwen_rope_reference,
)


def assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    rtol: float = 2e-2,
    atol: float = 2e-2,
) -> None:
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


def run_forward_case(
    *,
    batch_size: int,
    seq_len: int,
    n_q_heads: int,
    n_k_heads: int,
    head_dim: int,
    dtype: torch.dtype,
) -> None:
    q, k, freqs_cis = make_inputs(
        batch_size=batch_size,
        seq_len=seq_len,
        n_q_heads=n_q_heads,
        n_k_heads=n_k_heads,
        head_dim=head_dim,
        dtype=dtype,
        seed=batch_size + seq_len + n_q_heads + n_k_heads + head_dim,
    )
    actual_q, actual_k = qwen_rope(q, k, freqs_cis)
    expected_q, expected_k = qwen_rope_reference(q, k, freqs_cis)
    assert_close(actual_q, expected_q)
    assert_close(actual_k, expected_k)


def run_backward_case(
    *,
    batch_size: int,
    seq_len: int,
    n_q_heads: int,
    n_k_heads: int,
    head_dim: int,
    dtype: torch.dtype,
) -> None:
    q, k, freqs_cis = make_inputs(
        batch_size=batch_size,
        seq_len=seq_len,
        n_q_heads=n_q_heads,
        n_k_heads=n_k_heads,
        head_dim=head_dim,
        dtype=dtype,
        seed=7,
    )
    q_test = q.detach().clone().requires_grad_(True)
    k_test = k.detach().clone().requires_grad_(True)
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)

    q_out, k_out = QwenRopeFunction.apply(
        q_test,
        k_test,
        freqs_cis,
        DEFAULT_THREADS,
        DEFAULT_ELEMENTS_PER_THREAD,
    )
    ref_q_out, ref_k_out = qwen_rope_reference(q_ref, k_ref, freqs_cis)

    loss = q_out.float().square().mean() + k_out.float().square().mean()
    ref_loss = ref_q_out.float().square().mean() + ref_k_out.float().square().mean()
    loss.backward()
    ref_loss.backward()

    assert_close(q_test.grad, q_ref.grad, rtol=3e-2, atol=3e-2)
    assert_close(k_test.grad, k_ref.grad, rtol=3e-2, atol=3e-2)


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this test.")

    run_forward_case(batch_size=1, seq_len=128, n_q_heads=8, n_k_heads=2, head_dim=64, dtype=torch.float16)
    run_forward_case(batch_size=2, seq_len=256, n_q_heads=16, n_k_heads=4, head_dim=128, dtype=torch.float16)
    run_forward_case(batch_size=1, seq_len=64, n_q_heads=4, n_k_heads=4, head_dim=32, dtype=torch.float32)
    run_backward_case(batch_size=1, seq_len=128, n_q_heads=8, n_k_heads=2, head_dim=64, dtype=torch.float16)
    print("All Qwen rotate-half RoPE correctness checks passed.")


if __name__ == "__main__":
    main()
