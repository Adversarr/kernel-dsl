import argparse

import tilelang.language as T
import torch

from gemm_residual_rmsnorm_gemm_swiglu_kernel import (
    gemm_residual_rmsnorm_stage1_with_config,
    gemm_rmsnorm_swiglu_stage2_with_config,
    make_inputs,
    rms_partial_reduce_with_config,
    torch_dtype_to_tilelang_dtype,
)


def benchmark_callable(fn, *args, warmup: int = 10, iters: int = 50) -> float:
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep selected CODA TileLang stage configs.")
    parser.add_argument("--rows", type=int, default=16 * 1024)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--inner0", type=int, default=None)
    parser.add_argument("--mlp", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    inner0 = args.hidden if args.inner0 is None else args.inner0
    mlp = args.hidden * 4 if args.mlp is None else args.mlp
    x, weight0, residual, gamma, weight1 = make_inputs(
        rows=args.rows,
        hidden=args.hidden,
        inner0=inner0,
        mlp=mlp,
        dtype=torch.float16,
    )
    dtype = torch_dtype_to_tilelang_dtype(torch.float16)

    configs = [
        (128, 128, 32, 128, 1),
        (128, 128, 32, 128, 2),
        (128, 128, 32, 128, 3),
        (128, 128, 64, 128, 2),
        (128, 128, 64, 128, 3),
        (128, 64, 32, 128, 3),
        (64, 128, 32, 128, 3),
        (128, 128, 32, 256, 3),
    ]
    for block_m, block_n, block_k, threads, num_stages in configs:
        partial_cols = (args.hidden + block_n - 1) // block_n
        stage1 = gemm_residual_rmsnorm_stage1_with_config(
            args.rows,
            args.hidden,
            inner0,
            partial_cols,
            block_M=block_m,
            block_N=block_n,
            block_K=block_k,
            threads=threads,
            num_stages=num_stages,
            dtype=dtype,
            accum_dtype=T.float32,
        )
        weighted_residual, partial_squares = stage1(x, weight0, residual, gamma)
        reduce = rms_partial_reduce_with_config(
            args.rows,
            partial_cols,
            threads=threads,
            accum_dtype=T.float32,
        )
        inv_rms = reduce(partial_squares)
        stage2 = gemm_rmsnorm_swiglu_stage2_with_config(
            args.rows,
            mlp,
            args.hidden,
            block_M=block_m,
            block_N=block_n,
            block_K=block_k,
            threads=threads,
            num_stages=num_stages,
            dtype=dtype,
            accum_dtype=T.float32,
        )

        stage1_ms = benchmark_callable(stage1, x, weight0, residual, gamma, warmup=args.warmup, iters=args.iters)
        reduce_ms = benchmark_callable(reduce, partial_squares, warmup=args.warmup, iters=args.iters)
        stage2_ms = benchmark_callable(stage2, weighted_residual, weight1, inv_rms, warmup=args.warmup, iters=args.iters)
        print(
            f"block_M={block_m} block_N={block_n} block_K={block_k} threads={threads} "
            f"num_stages={num_stages} stage1_ms={stage1_ms:.4f} reduce_ms={reduce_ms:.4f} "
            f"stage2_ms={stage2_ms:.4f} sum_ms={stage1_ms + reduce_ms + stage2_ms:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
