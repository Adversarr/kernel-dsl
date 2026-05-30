import torch

from vector_add_kernel import vector_add, vector_add_reference


def assert_case(length: int, threads: int = 128, elements_per_thread: int = 8) -> None:
    dtype = torch.float32
    lhs = torch.randn(length, device="cuda", dtype=dtype)
    rhs = torch.randn(length, device="cuda", dtype=dtype)

    actual = vector_add(
        lhs,
        rhs,
        threads=threads,
        elements_per_thread=elements_per_thread,
    )
    expected = vector_add_reference(lhs, rhs)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def main() -> None:
    assert_case(length=257)
    assert_case(length=4096)
    print("vector_add: correctness checks passed")


if __name__ == "__main__":
    main()
