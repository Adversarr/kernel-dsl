import argparse

import torch

from layer_rms_norm_kernel import (
    DEFAULT_ELEMENTS_PER_THREAD,
    DEFAULT_EPS,
    DEFAULT_THREADS,
    autotune_layer_rms_norm,
    layer_rms_norm_reference,
    layer_rms_norm_with_config,
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


def make_compiled_reference(eps: float):
    def reference(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return layer_rms_norm_reference(x, weight, eps=eps)

    return torch.compile(reference, fullgraph=True)


def benchmark(
    rows: int,
    cols: int,
    eps: float,
    threads: int,
    elements_per_thread: int,
    use_autotune: bool,
    autotune_warmup: int,
    autotune_rep: int,
) -> None:
    x, weight = make_inputs(rows=rows, cols=cols)

    if use_autotune:
        result = autotune_layer_rms_norm(
            rows=rows,
            cols=cols,
            eps=eps,
            warmup=autotune_warmup,
            rep=autotune_rep,
        )
        kernel = result.kernel
        best_config = result.config
    else:
        kernel = layer_rms_norm_with_config(
            rows,
            cols,
            eps=eps,
            threads=threads,
            elements_per_thread=elements_per_thread,
        )
        best_config = {
            "threads": threads,
            "elements_per_thread": elements_per_thread,
        }

    kernel(x, weight)

    compiled_reference = make_compiled_reference(eps=eps)
    compiled_reference(x, weight)

    tilelang_latency_ms = benchmark_callable(kernel, x, weight)
    torch_compile_latency_ms = benchmark_callable(compiled_reference, x, weight)

    bytes_per_element = x.element_size()
    total_bytes = bytes_per_element * (2 * rows * cols + cols)
    tilelang_bandwidth_gbps = total_bytes / (tilelang_latency_ms * 1e6)
    torch_compile_bandwidth_gbps = total_bytes / (torch_compile_latency_ms * 1e6)
    speedup = torch_compile_latency_ms / tilelang_latency_ms

    print(f"rows={rows}")
    print(f"cols={cols}")
    print(f"eps={eps}")
    print(f"use_autotune={use_autotune}")
    print(f"best_config={best_config}")
    print(f"tilelang_latency_ms={tilelang_latency_ms:.4f}")
    print(f"tilelang_bandwidth_gbps={tilelang_bandwidth_gbps:.2f}")
    print(f"torch_compile_latency_ms={torch_compile_latency_ms:.4f}")
    print(f"torch_compile_bandwidth_gbps={torch_compile_bandwidth_gbps:.2f}")
    print(f"tilelang_vs_torch_compile_speedup={speedup:.2f}x")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile the TileLang layer RMSNorm kernel.")
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--cols", type=int, default=4096)
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument(
        "--elements-per-thread",
        type=int,
        default=DEFAULT_ELEMENTS_PER_THREAD,
    )
    parser.add_argument(
        "--no-autotune",
        action="store_true",
        help="Use the fixed configuration instead of autotuning.",
    )
    parser.add_argument("--autotune-warmup", type=int, default=3)
    parser.add_argument("--autotune-rep", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    benchmark(
        rows=args.rows,
        cols=args.cols,
        eps=args.eps,
        threads=args.threads,
        elements_per_thread=args.elements_per_thread,
        use_autotune=not args.no_autotune,
        autotune_warmup=args.autotune_warmup,
        autotune_rep=args.autotune_rep,
    )
