import argparse
from functools import partial

import torch

from adaln_zero_kernel import (
    DEFAULT_BLOCK_M,
    DEFAULT_EPS,
    DEFAULT_THREADS,
    adaln_zero,
    adaln_zero_reference,
    autotune_adaln_zero,
    get_autotune_configs,
    make_inputs,
)


def benchmark_callable(fn, warmup: int = 20, iters: int = 100) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def benchmark(
    N: int,
    D: int,
    eps: float,
    block_M: int,
    threads: int,
    use_autotune: bool,
) -> None:
    x, scale, shift, gate = make_inputs(N=N, D=D)
    gate_scale = gate * scale
    gate_shift = gate * shift

    if use_autotune:
        result = autotune_adaln_zero(N=N, D=D, eps=eps)
        best_config = result.config
    else:
        best_config = {"block_M": block_M, "threads": threads}

    bm = best_config["block_M"]
    th = best_config["threads"]

    tilelang_fn = partial(adaln_zero, x, scale, shift, gate, eps=eps, block_M=bm, threads=th)
    tilelang_fn()
    torch.cuda.synchronize()

    compiled_ref = torch.compile(
        partial(adaln_zero_reference, eps=eps), fullgraph=True
    )
    compiled_ref(x, scale, shift, gate)
    torch.cuda.synchronize()

    gs = partial(adaln_zero, x, scale, shift, gate, eps=eps, block_M=bm, threads=th)
    tilelang_latency_ms = benchmark_callable(gs)
    torch_compile_latency_ms = benchmark_callable(partial(compiled_ref, x, scale, shift, gate))

    elements = N * D
    tilelang_bandwidth = (elements * x.element_size() * 2) / (tilelang_latency_ms * 1e6)
    torch_compile_bandwidth = (elements * x.element_size() * 2) / (torch_compile_latency_ms * 1e6)
    speedup = torch_compile_latency_ms / tilelang_latency_ms

    print(f"N={N}  D={D}")
    print(f"eps={eps}  best_config={best_config}")
    print(f"tilelang_latency_ms={tilelang_latency_ms:.4f}")
    print(f"tilelang_bandwidth_GB_s={tilelang_bandwidth:.2f}")
    print(f"torch_compile_latency_ms={torch_compile_latency_ms:.4f}")
    print(f"torch_compile_bandwidth_GB_s={torch_compile_bandwidth:.2f}")
    print(f"tilelang_vs_torch_compile_speedup={speedup:.2f}x")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile the TileLang AdaLN-Zero kernel.")
    parser.add_argument("--N", type=int, default=4096)
    parser.add_argument("--D", type=int, default=8192)
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS)
    parser.add_argument("--block-m", type=int, default=DEFAULT_BLOCK_M)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--no-autotune", action="store_true", help="Use fixed config instead of autotuning.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    benchmark(
        N=args.N,
        D=args.D,
        eps=args.eps,
        block_M=args.block_m,
        threads=args.threads,
        use_autotune=not args.no_autotune,
    )
