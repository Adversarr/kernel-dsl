import argparse
import ast
import types
from pathlib import Path

import torch

from group_gemm_kernel import (
    DEFAULT_BLOCK_K,
    DEFAULT_BLOCK_M,
    DEFAULT_BLOCK_N,
    DEFAULT_ENABLE_SWIZZLE,
    DEFAULT_NUM_STAGES,
    DEFAULT_THREADS,
    autotune_group_gemm,
    effective_group_gemm_flops,
    make_demo_inputs,
    make_group_gemm_jit_kernel,
    pack_group_gemm_inputs,
    packed_group_gemm_reference,
)


def parse_shapes(spec: str) -> list[tuple[int, int, int]]:
    shapes = []
    for raw_shape in spec.split(","):
        rows, cols, inner = (int(value) for value in raw_shape.lower().split("x"))
        shapes.append((rows, cols, inner))
    return shapes


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


def load_triton_group_gemm_namespace() -> types.SimpleNamespace:
    triton_path = Path(__file__).with_name("group_gemm_triton.py")
    parsed = ast.parse(triton_path.read_text(), filename=str(triton_path))

    allowed_assignments = {"DEVICE", "tma_configs"}
    filtered_body = []
    for node in parsed.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef)):
            filtered_body.append(node)
            continue
        if isinstance(node, ast.Assign):
            target_names = {
                target.id
                for target in node.targets
                if isinstance(target, ast.Name)
            }
            if target_names and target_names.issubset(allowed_assignments):
                filtered_body.append(node)

    module = types.ModuleType("group_gemm_triton_baseline")
    exec(
        compile(ast.Module(body=filtered_body, type_ignores=[]), str(triton_path), "exec"),
        module.__dict__,
    )
    return types.SimpleNamespace(
        group_gemm_fn=module.group_gemm_fn,
        triton_perf_fn=module.triton_perf_fn,
    )


def prepare_triton_perf_inputs(
    group_lhs: list[torch.Tensor],
    group_rhs: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = group_lhs[0].device
    group_size = len(group_lhs)

    lhs_addrs = []
    rhs_addrs = []
    out_addrs = []
    group_sizes = []
    group_lds = []
    for lhs, rhs in zip(group_lhs, group_rhs):
        rows, inner = lhs.shape
        _, cols = rhs.shape
        out = torch.empty((rows, cols), device=device, dtype=lhs.dtype)
        lhs_addrs.append(lhs.data_ptr())
        rhs_addrs.append(rhs.data_ptr())
        out_addrs.append(out.data_ptr())
        group_sizes.extend([rows, cols, inner])
        group_lds.extend([lhs.stride(0), rhs.stride(0), out.stride(0)])

    return (
        torch.tensor(lhs_addrs, device=device),
        torch.tensor(rhs_addrs, device=device),
        torch.tensor(out_addrs, device=device),
        torch.tensor(group_sizes, device=device, dtype=torch.int32),
        torch.tensor(group_lds, device=device, dtype=torch.int32),
    )


def benchmark(
    shapes: list[tuple[int, int, int]],
    block_M: int,
    block_N: int,
    block_K: int,
    threads: int,
    num_stages: int,
    enable_swizzle: bool,
    use_autotune: bool,
    autotune_warmup: int,
    autotune_rep: int,
    benchmark_warmup: int,
    benchmark_iters: int,
) -> None:
    group_lhs, group_rhs = make_demo_inputs(shapes=shapes)
    packed_lhs, packed_rhs, group_sizes = pack_group_gemm_inputs(group_lhs, group_rhs)

    if use_autotune:
        result = autotune_group_gemm(
            packed_lhs,
            packed_rhs,
            group_sizes,
            warmup=autotune_warmup,
            rep=autotune_rep,
        )
        kernel = result.kernel
        best_config = result.config
    else:
        kernel = make_group_gemm_jit_kernel(
            packed_lhs,
            packed_rhs,
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

    kernel(packed_lhs, packed_rhs, group_sizes)

    compiled_reference = torch.compile(packed_group_gemm_reference, fullgraph=True)
    compiled_reference(packed_lhs, packed_rhs)

    triton_ns = load_triton_group_gemm_namespace()
    triton_inputs = prepare_triton_perf_inputs(group_lhs, group_rhs)
    triton_ns.group_gemm_fn(group_lhs, group_rhs)

    tilelang_latency_ms = benchmark_callable(
        kernel,
        packed_lhs,
        packed_rhs,
        group_sizes,
        warmup=benchmark_warmup,
        iters=benchmark_iters,
    )
    torch_compile_latency_ms = benchmark_callable(
        compiled_reference,
        packed_lhs,
        packed_rhs,
        warmup=benchmark_warmup,
        iters=benchmark_iters,
    )
    triton_latency_ms = benchmark_callable(
        triton_ns.triton_perf_fn,
        *triton_inputs,
        len(group_lhs),
        warmup=benchmark_warmup,
        iters=benchmark_iters,
    )

    flops = effective_group_gemm_flops(group_sizes)
    tilelang_tflops = flops / (tilelang_latency_ms * 1e9)
    torch_compile_tflops = flops / (torch_compile_latency_ms * 1e9)
    triton_tflops = flops / (triton_latency_ms * 1e9)

    print(f"shapes={shapes}")
    print(f"use_autotune={use_autotune}")
    print(f"best_config={best_config}")
    print(f"tilelang_latency_ms={tilelang_latency_ms:.4f}")
    print(f"tilelang_tflops={tilelang_tflops:.2f}")
    print(f"torch_compile_latency_ms={torch_compile_latency_ms:.4f}")
    print(f"torch_compile_tflops={torch_compile_tflops:.2f}")
    print(f"tilelang_vs_torch_compile_speedup={torch_compile_latency_ms / tilelang_latency_ms:.2f}x")
    print(f"triton_latency_ms={triton_latency_ms:.4f}")
    print(f"triton_tflops={triton_tflops:.2f}")
    print(f"tilelang_vs_triton_ratio={triton_latency_ms / tilelang_latency_ms:.2f}x")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile the TileLang grouped GEMM kernel.")
    parser.add_argument(
        "--shapes",
        type=str,
        default="2048x2048x2048,1536x1536x2048,1024x1024x2048,512x512x2048",
        help="Comma-separated shapes formatted as MxNxK.",
    )
    parser.add_argument("--block-m", type=int, default=DEFAULT_BLOCK_M)
    parser.add_argument("--block-n", type=int, default=DEFAULT_BLOCK_N)
    parser.add_argument("--block-k", type=int, default=DEFAULT_BLOCK_K)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--num-stages", type=int, default=DEFAULT_NUM_STAGES)
    parser.add_argument("--enable-swizzle", action="store_true", default=DEFAULT_ENABLE_SWIZZLE)
    parser.add_argument("--no-autotune", action="store_true", help="Use the fixed configuration instead of autotuning.")
    parser.add_argument("--autotune-warmup", type=int, default=3)
    parser.add_argument("--autotune-rep", type=int, default=20)
    parser.add_argument("--benchmark-warmup", type=int, default=20)
    parser.add_argument("--benchmark-iters", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    benchmark(
        shapes=parse_shapes(args.shapes),
        block_M=args.block_m,
        block_N=args.block_n,
        block_K=args.block_k,
        threads=args.threads,
        num_stages=args.num_stages,
        enable_swizzle=args.enable_swizzle,
        use_autotune=not args.no_autotune,
        autotune_warmup=args.autotune_warmup,
        autotune_rep=args.autotune_rep,
        benchmark_warmup=args.benchmark_warmup,
        benchmark_iters=args.benchmark_iters,
    )
