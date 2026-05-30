import argparse

import torch

from fused_gemm_relu_kernel import (
    DEFAULT_BLOCK_K,
    DEFAULT_BLOCK_M,
    DEFAULT_BLOCK_N,
    DEFAULT_ENABLE_SWIZZLE,
    DEFAULT_NUM_STAGES,
    DEFAULT_THREADS,
    autotune_fused_gemm_relu,
    fused_gemm_relu_reference,
    fused_gemm_relu_with_config,
    make_inputs,
)


def benchmark_callable(fn, *args, warmup: int = 20, iters: int = 100) -> float:
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


def benchmark(
    rows: int,
    cols: int,
    inner: int,
    block_M: int,
    block_N: int,
    block_K: int,
    threads: int,
    num_stages: int,
    enable_swizzle: bool,
    use_autotune: bool,
    autotune_warmup: int,
    autotune_rep: int,
) -> None:
    lhs, rhs = make_inputs(rows=rows, cols=cols, inner=inner)

    if use_autotune:
        result = autotune_fused_gemm_relu(
            rows=rows,
            cols=cols,
            inner=inner,
            warmup=autotune_warmup,
            rep=autotune_rep,
        )
        kernel = result.kernel
        best_config = result.config
    else:
        kernel = fused_gemm_relu_with_config(
            rows,
            cols,
            inner,
            block_M=block_M,
            block_N=block_N,
            block_K=block_K,
            threads=threads,
            num_stages=num_stages,
            enable_swizzle=enable_swizzle,
        )
        best_config = {
            "block_M": block_M,
            "block_N": block_N,
            "block_K": block_K,
            "threads": threads,
            "num_stages": num_stages,
            "enable_swizzle": enable_swizzle,
        }
    kernel(lhs, rhs)

    compiled_reference = torch.compile(fused_gemm_relu_reference, fullgraph=True)
    compiled_reference(lhs, rhs)

    tilelang_latency_ms = benchmark_callable(kernel, lhs, rhs)
    torch_compile_latency_ms = benchmark_callable(compiled_reference, lhs, rhs)
    gemm_flops = 2 * rows * cols * inner
    tilelang_tflops = gemm_flops / (tilelang_latency_ms * 1e9)
    torch_compile_tflops = gemm_flops / (torch_compile_latency_ms * 1e9)
    speedup = torch_compile_latency_ms / tilelang_latency_ms

    print(f"rows={rows}")
    print(f"cols={cols}")
    print(f"inner={inner}")
    print(f"use_autotune={use_autotune}")
    print(f"best_config={best_config}")
    print(f"tilelang_latency_ms={tilelang_latency_ms:.4f}")
    print(f"tilelang_tflops={tilelang_tflops:.2f}")
    print(f"torch_compile_latency_ms={torch_compile_latency_ms:.4f}")
    print(f"torch_compile_tflops={torch_compile_tflops:.2f}")
    print(f"tilelang_vs_torch_compile_speedup={speedup:.2f}x")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile the TileLang fused GEMM+ReLU kernel.")
    parser.add_argument("--rows", type=int, default=2048)
    parser.add_argument("--cols", type=int, default=2048)
    parser.add_argument("--inner", type=int, default=2048)
    parser.add_argument("--block-m", type=int, default=DEFAULT_BLOCK_M)
    parser.add_argument("--block-n", type=int, default=DEFAULT_BLOCK_N)
    parser.add_argument("--block-k", type=int, default=DEFAULT_BLOCK_K)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--num-stages", type=int, default=DEFAULT_NUM_STAGES)
    parser.add_argument("--enable-swizzle", action="store_true", default=DEFAULT_ENABLE_SWIZZLE)
    parser.add_argument("--no-autotune", action="store_true", help="Use the fixed configuration instead of autotuning.")
    parser.add_argument("--autotune-warmup", type=int, default=3)
    parser.add_argument("--autotune-rep", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    benchmark(
        rows=args.rows,
        cols=args.cols,
        inner=args.inner,
        block_M=args.block_m,
        block_N=args.block_n,
        block_K=args.block_k,
        threads=args.threads,
        num_stages=args.num_stages,
        enable_swizzle=args.enable_swizzle,
        use_autotune=not args.no_autotune,
        autotune_warmup=args.autotune_warmup,
        autotune_rep=args.autotune_rep,
    )
