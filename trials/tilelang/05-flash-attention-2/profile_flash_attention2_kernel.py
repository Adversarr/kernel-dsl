import argparse
import time

import torch
import torch.nn.functional as F

from flash_attention2_kernel import (
    DEFAULT_BLOCK_M,
    DEFAULT_BLOCK_N,
    DEFAULT_CAUSAL,
    DEFAULT_ENABLE_SWIZZLE,
    DEFAULT_NUM_STAGES,
    DEFAULT_THREADS,
    autotune_flash_attention2,
    flash_attention2_reference,
    flash_attention2_with_config,
    make_inputs,
    resolve_softmax_scale,
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


def make_compiled_reference(causal: bool, softmax_scale: float):
    def reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return flash_attention2_reference(
            q,
            k,
            v,
            causal=causal,
            softmax_scale=softmax_scale,
        )

    return torch.compile(reference, fullgraph=True)


def make_sdpa_flash_reference(causal: bool, softmax_scale: float):
    def reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        q_bhsd = q.permute(0, 2, 1, 3)
        k_bhsd = k.permute(0, 2, 1, 3)
        v_bhsd = v.permute(0, 2, 1, 3)
        with torch.nn.attention.sdpa_kernel(torch.backends.cuda.SDPBackend.FLASH_ATTENTION):
            out = F.scaled_dot_product_attention(
                q_bhsd,
                k_bhsd,
                v_bhsd,
                dropout_p=0.0,
                is_causal=causal,
                scale=softmax_scale,
            )
        return out.permute(0, 2, 1, 3)

    return reference


def benchmark(
    batch: int,
    heads: int,
    seq_len: int,
    head_dim: int,
    causal: bool,
    block_M: int,
    block_N: int,
    threads: int,
    num_stages: int,
    enable_swizzle: bool,
    use_autotune: bool,
    autotune_warmup: int,
    autotune_rep: int,
) -> None:
    q, k, v = make_inputs(
        batch=batch,
        heads=heads,
        seq_len=seq_len,
        head_dim=head_dim,
    )
    softmax_scale = resolve_softmax_scale(head_dim, None)

    if use_autotune:
        autotune_start = time.perf_counter()
        result = autotune_flash_attention2(
            batch=batch,
            heads=heads,
            seq_len=seq_len,
            head_dim=head_dim,
            causal=causal,
            warmup=autotune_warmup,
            rep=autotune_rep,
        )
        autotune_seconds = time.perf_counter() - autotune_start
        kernel = result.kernel
        best_config = result.config
    else:
        autotune_seconds = 0.0
        kernel = flash_attention2_with_config(
            batch,
            heads,
            seq_len,
            head_dim,
            causal=causal,
            softmax_scale=softmax_scale,
            block_M=block_M,
            block_N=block_N,
            threads=threads,
            num_stages=num_stages,
            enable_swizzle=enable_swizzle,
        )
        best_config = {
            "block_M": block_M,
            "block_N": block_N,
            "threads": threads,
            "num_stages": num_stages,
            "enable_swizzle": enable_swizzle,
        }

    kernel(q, k, v)

    compiled_reference = make_compiled_reference(causal=causal, softmax_scale=softmax_scale)
    compiled_reference(q, k, v)
    sdpa_flash_reference = make_sdpa_flash_reference(causal=causal, softmax_scale=softmax_scale)
    sdpa_flash_reference(q, k, v)

    tilelang_latency_ms = benchmark_callable(kernel, q, k, v)
    torch_compile_latency_ms = benchmark_callable(compiled_reference, q, k, v)
    sdpa_flash_latency_ms = benchmark_callable(sdpa_flash_reference, q, k, v)

    attention_flops = 4 * batch * heads * seq_len * seq_len * head_dim
    tilelang_tflops = attention_flops / (tilelang_latency_ms * 1e9)
    torch_compile_tflops = attention_flops / (torch_compile_latency_ms * 1e9)
    sdpa_flash_tflops = attention_flops / (sdpa_flash_latency_ms * 1e9)
    speedup = torch_compile_latency_ms / tilelang_latency_ms
    vs_sdpa_flash = sdpa_flash_latency_ms / tilelang_latency_ms

    print(f"batch={batch}")
    print(f"heads={heads}")
    print(f"seq_len={seq_len}")
    print(f"head_dim={head_dim}")
    print(f"causal={causal}")
    print(f"use_autotune={use_autotune}")
    print(f"best_config={best_config}")
    print(f"autotune_seconds={autotune_seconds:.2f}")
    print(f"tilelang_latency_ms={tilelang_latency_ms:.4f}")
    print(f"tilelang_tflops={tilelang_tflops:.2f}")
    print(f"torch_compile_latency_ms={torch_compile_latency_ms:.4f}")
    print(f"torch_compile_tflops={torch_compile_tflops:.2f}")
    print(f"tilelang_vs_torch_compile_speedup={speedup:.2f}x")
    print(f"sdpa_flash_latency_ms={sdpa_flash_latency_ms:.4f}")
    print(f"sdpa_flash_tflops={sdpa_flash_tflops:.2f}")
    print(f"tilelang_vs_sdpa_flash_speedup={vs_sdpa_flash:.2f}x")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile the TileLang FlashAttention-2 forward kernel.")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--causal", action="store_true", default=DEFAULT_CAUSAL)
    parser.add_argument("--block-m", type=int, default=DEFAULT_BLOCK_M)
    parser.add_argument("--block-n", type=int, default=DEFAULT_BLOCK_N)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--num-stages", type=int, default=DEFAULT_NUM_STAGES)
    parser.add_argument("--enable-swizzle", action="store_true", default=DEFAULT_ENABLE_SWIZZLE)
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
        batch=args.batch,
        heads=args.heads,
        seq_len=args.seq_len,
        head_dim=args.head_dim,
        causal=args.causal,
        block_M=args.block_m,
        block_N=args.block_n,
        threads=args.threads,
        num_stages=args.num_stages,
        enable_swizzle=args.enable_swizzle,
        use_autotune=not args.no_autotune,
        autotune_warmup=args.autotune_warmup,
        autotune_rep=args.autotune_rep,
    )
