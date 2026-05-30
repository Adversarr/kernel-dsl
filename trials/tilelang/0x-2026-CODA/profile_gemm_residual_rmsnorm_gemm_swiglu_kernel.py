import argparse

import tilelang.language as T
import torch

from gemm_residual_rmsnorm_gemm_swiglu_kernel import (
    DEFAULT_BLOCK_K,
    DEFAULT_BLOCK_M,
    DEFAULT_BLOCK_N,
    DEFAULT_ENABLE_SWIZZLE,
    DEFAULT_NUM_STAGES,
    DEFAULT_THREADS,
    _build_kernels,
    _build_kernels_transposed_stage2,
    gemm_residual_rmsnorm_stage1_with_config,
    gemm_rmsnorm_swiglu_stage2_split_weights_with_config,
    rms_partial_reduce_with_config,
    torch_dtype_to_tilelang_dtype,
    gemm_residual_rmsnorm_gemm_swiglu,
    gemm_residual_rmsnorm_gemm_swiglu_split_weights,
    gemm_residual_rmsnorm_gemm_swiglu_transposed_stage2,
    naive_torch_compile_reference,
    naive_torch_compile_split_weights_reference,
    naive_torch_compile_transposed_reference,
    make_inputs,
    make_inputs_split_weights,
    make_inputs_transposed_stage2,
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
    hidden: int,
    inner0: int,
    mlp: int,
    block_M: int,
    block_N: int,
    block_K: int,
    threads: int,
    num_stages: int,
    stage1_num_stages: int | None,
    enable_swizzle: bool,
    warmup: int,
    iters: int,
    dtype: torch.dtype,
    breakdown: bool,
    transposed_stage2: bool,
    split_weights: bool,
) -> None:
    if split_weights and transposed_stage2:
        raise ValueError("--split-weights and --transposed-stage2 are mutually exclusive")
    if split_weights:
        inputs = make_inputs_split_weights(rows=rows, hidden=hidden, inner0=inner0, mlp=mlp, dtype=dtype)
    elif transposed_stage2:
        inputs = make_inputs_transposed_stage2(rows=rows, hidden=hidden, inner0=inner0, mlp=mlp, dtype=dtype)
    else:
        inputs = make_inputs(rows=rows, hidden=hidden, inner0=inner0, mlp=mlp, dtype=dtype)

    def tilelang_fn(*fn_inputs):
        if split_weights:
            fn = gemm_residual_rmsnorm_gemm_swiglu_split_weights
        elif transposed_stage2:
            fn = gemm_residual_rmsnorm_gemm_swiglu_transposed_stage2
        else:
            fn = gemm_residual_rmsnorm_gemm_swiglu
        return fn(
            *fn_inputs,
            block_M=block_M,
            block_N=block_N,
            block_K=block_K,
            threads=threads,
            num_stages=num_stages,
            stage1_num_stages=stage1_num_stages,
            enable_swizzle=enable_swizzle,
        )

    tilelang_fn(*inputs)
    if split_weights:
        reference = naive_torch_compile_split_weights_reference
    elif transposed_stage2:
        reference = naive_torch_compile_transposed_reference
    else:
        reference = naive_torch_compile_reference
    compiled_reference = torch.compile(reference, fullgraph=True)
    compiled_reference(*inputs)

    tilelang_latency_ms = benchmark_callable(tilelang_fn, *inputs, warmup=warmup, iters=iters)
    torch_compile_latency_ms = benchmark_callable(
        compiled_reference,
        *inputs,
        warmup=warmup,
        iters=iters,
    )
    gemm_flops = 2 * rows * inner0 * hidden + 2 * rows * hidden * (mlp * 2)
    tilelang_tflops = gemm_flops / (tilelang_latency_ms * 1e9)
    torch_compile_tflops = gemm_flops / (torch_compile_latency_ms * 1e9)
    speedup = torch_compile_latency_ms / tilelang_latency_ms

    print(f"rows={rows}")
    print(f"hidden={hidden}")
    print(f"inner0={inner0}")
    print(f"mlp={mlp}")
    print(
        "config="
        f"block_M={block_M}, block_N={block_N}, block_K={block_K}, "
        f"threads={threads}, num_stages={num_stages}, stage1_num_stages={stage1_num_stages}, "
        f"enable_swizzle={enable_swizzle}"
    )
    print(f"transposed_stage2={transposed_stage2}")
    print(f"split_weights={split_weights}")
    print(f"tilelang_latency_ms={tilelang_latency_ms:.4f}")
    print(f"tilelang_effective_tflops={tilelang_tflops:.2f}")
    print(f"torch_compile_latency_ms={torch_compile_latency_ms:.4f}")
    print(f"torch_compile_effective_tflops={torch_compile_tflops:.2f}")
    print(f"tilelang_vs_torch_compile_speedup={speedup:.2f}x")

    if breakdown:
        x, weight0, residual, gamma, *stage2_weights = inputs
        partial_cols = (hidden + block_N - 1) // block_N
        if split_weights:
            stage1_stages = num_stages if stage1_num_stages is None else stage1_num_stages
            stage1 = gemm_residual_rmsnorm_stage1_with_config(
                rows,
                hidden,
                inner0,
                partial_cols,
                block_M=block_M,
                block_N=block_N,
                block_K=block_K,
                threads=threads,
                num_stages=stage1_stages,
                enable_swizzle=enable_swizzle,
                dtype=torch_dtype_to_tilelang_dtype(dtype),
                accum_dtype=T.float32,
            )
            reduce = rms_partial_reduce_with_config(
                rows,
                partial_cols,
                threads=threads,
                accum_dtype=T.float32,
            )
            stage2 = gemm_rmsnorm_swiglu_stage2_split_weights_with_config(
                rows,
                mlp,
                hidden,
                block_M=block_M,
                block_N=block_N,
                block_K=block_K,
                threads=threads,
                num_stages=num_stages,
                enable_swizzle=enable_swizzle,
                dtype=torch_dtype_to_tilelang_dtype(dtype),
                accum_dtype=T.float32,
            )
        else:
            build_kernels = _build_kernels_transposed_stage2 if transposed_stage2 else _build_kernels
            stage1, reduce, stage2 = build_kernels(
                rows=rows,
                hidden=hidden,
                inner0=inner0,
                mlp=mlp,
                block_M=block_M,
                block_N=block_N,
                block_K=block_K,
                threads=threads,
                num_stages=num_stages,
                stage1_num_stages=stage1_num_stages,
                enable_swizzle=enable_swizzle,
                dtype=torch_dtype_to_tilelang_dtype(dtype),
                accum_dtype=T.float32,
            )
        weighted_residual, partial_squares = stage1(x, weight0, residual, gamma)
        inv_rms = reduce(partial_squares)
        stage1_ms = benchmark_callable(stage1, x, weight0, residual, gamma, warmup=warmup, iters=iters)
        reduce_ms = benchmark_callable(reduce, partial_squares, warmup=warmup, iters=iters)
        stage2_ms = benchmark_callable(stage2, weighted_residual, *stage2_weights, inv_rms, warmup=warmup, iters=iters)
        print(f"stage1_latency_ms={stage1_ms:.4f}")
        print(f"reduce_latency_ms={reduce_ms:.4f}")
        print(f"stage2_latency_ms={stage2_ms:.4f}")
        print(f"stage_sum_latency_ms={stage1_ms + reduce_ms + stage2_ms:.4f}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile CODA GEMM-Residual-RMSNorm-GEMM-SwiGLU TileLang kernels."
    )
    parser.add_argument("--rows", type=int, default=16 * 1024)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--inner0", type=int, default=None)
    parser.add_argument("--mlp", type=int, default=None)
    parser.add_argument("--block-m", type=int, default=DEFAULT_BLOCK_M)
    parser.add_argument("--block-n", type=int, default=DEFAULT_BLOCK_N)
    parser.add_argument("--block-k", type=int, default=DEFAULT_BLOCK_K)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--num-stages", type=int, default=DEFAULT_NUM_STAGES)
    parser.add_argument("--stage1-num-stages", type=int, default=None)
    parser.add_argument("--enable-swizzle", action="store_true", default=DEFAULT_ENABLE_SWIZZLE)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--breakdown", action="store_true")
    parser.add_argument("--transposed-stage2", action="store_true")
    parser.add_argument("--split-weights", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    inner0 = args.hidden if args.inner0 is None else args.inner0
    mlp = args.hidden * 4 if args.mlp is None else args.mlp
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    benchmark(
        rows=args.rows,
        hidden=args.hidden,
        inner0=inner0,
        mlp=mlp,
        block_M=args.block_m,
        block_N=args.block_n,
        block_K=args.block_k,
        threads=args.threads,
        num_stages=args.num_stages,
        stage1_num_stages=args.stage1_num_stages,
        enable_swizzle=args.enable_swizzle,
        warmup=args.warmup,
        iters=args.iters,
        dtype=dtype,
        breakdown=args.breakdown,
        transposed_stage2=args.transposed_stage2,
        split_weights=args.split_weights,
    )
