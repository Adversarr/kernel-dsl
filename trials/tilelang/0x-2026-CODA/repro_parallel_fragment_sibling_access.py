"""Reproduce TileLang layout-inference failure for sibling fragment reads.

Expected behavior:
    Reading neighboring accumulator lanes produced by a FullRow GEMM should be
    legal inside one T.Parallel loop.

Observed with tilelang 0.1.10:
    Layout inference rejects the two reads because the same fragment buffer is
    accessed as both acc[i, j * 2] and acc[i, j * 2 + 1] in the same
    T.Parallel loop.

Run:
    uv run python trials/tilelang/0x-2026-CODA/repro_parallel_fragment_sibling_access.py
"""

import tilelang as tl
import tilelang.language as T
import torch


@tl.jit(out_idx=[-1])
def sibling_fragment_access_repro(
    rows: int = 16,
    hidden: int = 32,
    out_cols: int = 64,
    block_m: int = 16,
    block_n: int = 64,
    block_k: int = 32,
    threads: int = 128,
    dtype=T.float16,
    accum_dtype=T.float32,
):
    paired_cols = out_cols // 2

    @T.prim_func
    def main(
        a: T.Tensor((rows, hidden), dtype),
        b: T.Tensor((hidden, out_cols), dtype),
        out: T.Tensor((rows, paired_cols), dtype),
    ):
        with T.Kernel(1, 1, threads=threads):
            a_shared = T.alloc_shared((block_m, block_k), dtype)
            b_shared = T.alloc_shared((block_k, block_n), dtype)
            acc = T.alloc_fragment((block_m, block_n), accum_dtype)

            T.clear(acc)
            T.copy(a[0, 0], a_shared)
            T.copy(b[0, 0], b_shared)
            T.gemm(a_shared, b_shared, acc, policy=T.GemmWarpPolicy.FullRow)

            for i, j in T.Parallel(block_m, paired_cols):
                # This pair of sibling reads is rejected by layout inference:
                #   acc: (i, j * 2 + 1) and (i, j * 2)
                gate = acc[i, j * 2]
                up = acc[i, j * 2 + 1]
                out[i, j] = T.cast(gate + up, dtype)

    return main


def main() -> None:
    print("tilelang", getattr(tl, "__version__", "unknown"))
    print("torch", torch.__version__)
    if torch.cuda.is_available():
        print("device", torch.cuda.get_device_name())
        print("capability", torch.cuda.get_device_capability())

    kernel = sibling_fragment_access_repro()
    a = torch.randn(16, 32, device="cuda", dtype=torch.float16)
    b = torch.randn(32, 64, device="cuda", dtype=torch.float16)

    # The failure happens during JIT compile/lowering before the kernel runs.
    kernel(a, b)


if __name__ == "__main__":
    main()
