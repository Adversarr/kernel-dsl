import torch

from layer_rms_norm_kernel import (
    DEFAULT_EPS,
    layer_rms_norm,
    layer_rms_norm_reference,
    make_inputs,
)


def assert_case(
    rows: int,
    cols: int,
    eps: float = DEFAULT_EPS,
    threads: int = 128,
    elements_per_thread: int = 8,
) -> None:
    x, weight = make_inputs(rows=rows, cols=cols, seed=rows + cols)

    actual = layer_rms_norm(
        x,
        weight,
        eps=eps,
        threads=threads,
        elements_per_thread=elements_per_thread,
    )
    expected = layer_rms_norm_reference(x, weight, eps=eps)

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def main() -> None:
    assert_case(rows=128, cols=256)
    assert_case(rows=384, cols=768, threads=64, elements_per_thread=4)
    assert_case(rows=512, cols=4096, threads=256, elements_per_thread=8)
    print("layer_rms_norm: correctness checks passed")


if __name__ == "__main__":
    main()
