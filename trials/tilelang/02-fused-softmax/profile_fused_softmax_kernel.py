import argparse

import torch

from fused_softmax_kernel import (
    DEFAULT_BLOCK_N,
    DEFAULT_THREADS,
    fused_softmax,
    fused_softmax_reference,
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


def benchmark(rows: int, cols: int, block_N: int, threads: int) -> None:
    x = make_inputs(rows=rows, cols=cols)

    kernel = fused_softmax.compile(
        M=rows,
        N=cols,
        block_N=block_N,
        threads=threads,
    )
    kernel(x)

    compiled_reference = torch.compile(fused_softmax_reference, fullgraph=True)
    compiled_reference(x)

    tilelang_latency_ms = benchmark_callable(kernel, x)
    torch_compile_latency_ms = benchmark_callable(compiled_reference, x)
    bytes_moved = 2 * rows * cols * x.element_size()
    tilelang_bandwidth_gb_s = bytes_moved / (tilelang_latency_ms * 1e6)
    torch_compile_bandwidth_gb_s = bytes_moved / (torch_compile_latency_ms * 1e6)
    speedup = torch_compile_latency_ms / tilelang_latency_ms

    print(f"rows={rows}")
    print(f"cols={cols}")
    print(f"block_N={block_N}")
    print(f"threads={threads}")
    print(f"tilelang_latency_ms={tilelang_latency_ms:.4f}")
    print(f"tilelang_effective_bandwidth_gb_s={tilelang_bandwidth_gb_s:.2f}")
    print(f"torch_compile_latency_ms={torch_compile_latency_ms:.4f}")
    print(f"torch_compile_effective_bandwidth_gb_s={torch_compile_bandwidth_gb_s:.2f}")
    print(f"tilelang_vs_torch_compile_speedup={speedup:.2f}x")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile the TileLang fused softmax kernel.")
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--cols", type=int, default=4096)
    parser.add_argument("--block-n", type=int, default=DEFAULT_BLOCK_N)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    benchmark(
        rows=args.rows,
        cols=args.cols,
        block_N=args.block_n,
        threads=args.threads,
    )
