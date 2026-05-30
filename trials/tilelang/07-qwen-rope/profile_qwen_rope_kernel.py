import argparse
import time

import torch

from qwen_rope_kernel import (
    DEFAULT_ELEMENTS_PER_THREAD,
    DEFAULT_ENABLE_SWIZZLE,
    DEFAULT_THREADS,
    autotune_qwen_rope,
    make_inputs,
    make_qwen_rope_jit_kernel,
    prepare_inputs,
    qwen_rope_reference,
    qwen_rope_reference_prepared,
)
from qwen_rope_triton import qwen_rope_forward


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


def make_compiled_reference(rotation_sign: float):
    def reference(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        return qwen_rope_reference_prepared(q, k, cos, sin, rotation_sign=rotation_sign)

    return torch.compile(reference, fullgraph=True)


def approximate_rope_flops(
    batch_size: int,
    seq_len: int,
    n_q_heads: int,
    n_k_heads: int,
    head_dim: int,
) -> int:
    head_dim_half = head_dim // 2
    return 6 * batch_size * seq_len * head_dim_half * (n_q_heads + n_k_heads)


def benchmark(
    batch_size: int,
    seq_len: int,
    n_q_heads: int,
    n_k_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    threads: int,
    elements_per_thread: int,
    enable_swizzle: bool,
    use_autotune: bool,
    rotation_sign: float,
    autotune_warmup: int,
    autotune_rep: int,
    benchmark_warmup: int,
    benchmark_iters: int,
) -> None:
    q, k, freqs_cis = make_inputs(
        batch_size=batch_size,
        seq_len=seq_len,
        n_q_heads=n_q_heads,
        n_k_heads=n_k_heads,
        head_dim=head_dim,
        dtype=dtype,
    )
    q_work, k_work, cos, sin, _ = prepare_inputs(q, k, freqs_cis)

    if use_autotune:
        autotune_start = time.perf_counter()
        result = autotune_qwen_rope(
            q,
            k,
            freqs_cis,
            rotation_sign=rotation_sign,
            warmup=autotune_warmup,
            rep=autotune_rep,
        )
        autotune_seconds = time.perf_counter() - autotune_start
        kernel = result.kernel
        best_config = result.config
    else:
        autotune_seconds = 0.0
        kernel = make_qwen_rope_jit_kernel(
            q_work,
            k_work,
            cos,
            threads=threads,
            elements_per_thread=elements_per_thread,
            enable_swizzle=enable_swizzle,
            rotation_sign=rotation_sign,
        )
        best_config = {
            "threads": threads,
            "elements_per_thread": elements_per_thread,
            "enable_swizzle": enable_swizzle,
        }

    kernel(q_work, k_work, cos, sin)

    compiled_reference = make_compiled_reference(rotation_sign=rotation_sign)
    compiled_reference(q_work, k_work, cos, sin)

    triton_q = q_work.clone()
    triton_k = k_work.clone()
    qwen_rope_forward(triton_q, triton_k, freqs_cis, rotation_sign=rotation_sign)

    tilelang_latency_ms = benchmark_callable(
        kernel,
        q_work,
        k_work,
        cos,
        sin,
        warmup=benchmark_warmup,
        iters=benchmark_iters,
    )
    torch_compile_latency_ms = benchmark_callable(
        compiled_reference,
        q_work,
        k_work,
        cos,
        sin,
        warmup=benchmark_warmup,
        iters=benchmark_iters,
    )
    triton_latency_ms = benchmark_callable(
        qwen_rope_forward,
        triton_q,
        triton_k,
        freqs_cis,
        warmup=benchmark_warmup,
        iters=benchmark_iters,
    )

    flops = approximate_rope_flops(batch_size, seq_len, n_q_heads, n_k_heads, head_dim)
    tilelang_gflops = flops / (tilelang_latency_ms * 1e6)
    torch_compile_gflops = flops / (torch_compile_latency_ms * 1e6)
    triton_gflops = flops / (triton_latency_ms * 1e6)

    ref_q, ref_k = qwen_rope_reference(q, k, freqs_cis, rotation_sign=rotation_sign)
    actual_q, actual_k = kernel(q_work, k_work, cos, sin)
    if actual_q.dtype != ref_q.dtype:
        actual_q = actual_q.to(ref_q.dtype)
    if actual_k.dtype != ref_k.dtype:
        actual_k = actual_k.to(ref_k.dtype)
    torch.testing.assert_close(actual_q, ref_q, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(actual_k, ref_k, rtol=2e-2, atol=2e-2)

    print(f"batch_size={batch_size}")
    print(f"seq_len={seq_len}")
    print(f"n_q_heads={n_q_heads}")
    print(f"n_k_heads={n_k_heads}")
    print(f"head_dim={head_dim}")
    print(f"dtype={dtype}")
    print(f"use_autotune={use_autotune}")
    print(f"best_config={best_config}")
    print(f"autotune_seconds={autotune_seconds:.2f}")
    print(f"tilelang_latency_ms={tilelang_latency_ms:.4f}")
    print(f"tilelang_gflops={tilelang_gflops:.2f}")
    print(f"torch_compile_latency_ms={torch_compile_latency_ms:.4f}")
    print(f"torch_compile_gflops={torch_compile_gflops:.2f}")
    print(f"tilelang_vs_torch_compile_speedup={torch_compile_latency_ms / tilelang_latency_ms:.2f}x")
    print(f"triton_latency_ms={triton_latency_ms:.4f}")
    print(f"triton_gflops={triton_gflops:.2f}")
    print(f"tilelang_vs_triton_ratio={triton_latency_ms / tilelang_latency_ms:.2f}x")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile the TileLang Qwen rotate-half RoPE kernel.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--q-heads", type=int, default=32)
    parser.add_argument("--k-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", type=str, default="float16", choices=("float16", "float32", "bfloat16"))
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--elements-per-thread", type=int, default=DEFAULT_ELEMENTS_PER_THREAD)
    parser.add_argument("--enable-swizzle", action="store_true", default=DEFAULT_ENABLE_SWIZZLE)
    parser.add_argument("--rotation-sign", type=float, default=1.0)
    parser.add_argument("--no-autotune", action="store_true", help="Use the fixed configuration instead of autotuning.")
    parser.add_argument("--autotune-warmup", type=int, default=3)
    parser.add_argument("--autotune-rep", type=int, default=20)
    parser.add_argument("--benchmark-warmup", type=int, default=20)
    parser.add_argument("--benchmark-iters", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    benchmark(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        n_q_heads=args.q_heads,
        n_k_heads=args.k_heads,
        head_dim=args.head_dim,
        dtype=getattr(torch, args.dtype),
        threads=args.threads,
        elements_per_thread=args.elements_per_thread,
        enable_swizzle=args.enable_swizzle,
        use_autotune=not args.no_autotune,
        rotation_sign=args.rotation_sign,
        autotune_warmup=args.autotune_warmup,
        autotune_rep=args.autotune_rep,
        benchmark_warmup=args.benchmark_warmup,
        benchmark_iters=args.benchmark_iters,
    )
