import argparse

import torch

from vector_add_kernel import (
    DEFAULT_ELEMENTS_PER_THREAD,
    DEFAULT_THREADS,
    make_inputs,
    vector_add,
    vector_add_reference,
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
    length: int,
    threads: int,
    elements_per_thread: int,
) -> None:
    lhs, rhs = make_inputs(length=length, dtype=torch.float32)

    kernel = vector_add.compile(
        N=length,
        threads=threads,
        elements_per_thread=elements_per_thread,
    )
    kernel(lhs, rhs)

    compiled_reference = torch.compile(vector_add_reference, fullgraph=True)
    compiled_reference(lhs, rhs)

    tilelang_latency_ms = benchmark_callable(kernel, lhs, rhs)
    torch_compile_latency_ms = benchmark_callable(compiled_reference, lhs, rhs)
    bytes_moved = 3 * length * lhs.element_size()
    tilelang_bandwidth_gb_s = bytes_moved / (tilelang_latency_ms * 1e6)
    torch_compile_bandwidth_gb_s = bytes_moved / (torch_compile_latency_ms * 1e6)
    speedup = torch_compile_latency_ms / tilelang_latency_ms

    print(f"length={length}")
    print(f"threads={threads}")
    print(f"elements_per_thread={elements_per_thread}")
    print(f"tilelang_latency_ms={tilelang_latency_ms:.4f}")
    print(f"tilelang_bandwidth_gb_s={tilelang_bandwidth_gb_s:.2f}")
    print(f"torch_compile_latency_ms={torch_compile_latency_ms:.4f}")
    print(f"torch_compile_bandwidth_gb_s={torch_compile_bandwidth_gb_s:.2f}")
    print(f"tilelang_vs_torch_compile_speedup={speedup:.2f}x")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile the TileLang vector add kernel.")
    parser.add_argument("--length", type=int, default=1 << 25)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument(
        "--elements-per-thread",
        type=int,
        default=DEFAULT_ELEMENTS_PER_THREAD,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    benchmark(
        length=args.length,
        threads=args.threads,
        elements_per_thread=args.elements_per_thread,
    )
