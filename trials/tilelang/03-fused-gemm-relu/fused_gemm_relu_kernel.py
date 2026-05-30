import argparse
import itertools

import tilelang as tl
import tilelang.language as T
import torch
from tilelang.autotuner import AutoTuner


DEFAULT_BLOCK_M = 128
DEFAULT_BLOCK_N = 128
DEFAULT_BLOCK_K = 32
DEFAULT_THREADS = 128
DEFAULT_NUM_STAGES = 3
DEFAULT_ENABLE_SWIZZLE = False


def torch_dtype_to_tilelang_dtype(dtype: torch.dtype):
    mapping = {
        torch.float16: T.float16,
        torch.float32: T.float32,
        torch.bfloat16: T.bfloat16,
    }
    try:
        return mapping[dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported torch dtype: {dtype}") from exc


@tl.jit
def fused_gemm_relu(
    lhs,
    rhs,
    block_M: int = DEFAULT_BLOCK_M,
    block_N: int = DEFAULT_BLOCK_N,
    block_K: int = DEFAULT_BLOCK_K,
    threads: int = DEFAULT_THREADS,
    num_stages: int = DEFAULT_NUM_STAGES,
    dtype=T.float16,
    accum_dtype=T.float32,
):
    M, N, K = T.const("M, N, K")

    lhs: T.Tensor((M, K), dtype)
    rhs: T.Tensor((K, N), dtype)
    out = T.empty((M, N), dtype)

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (
        bx,
        by,
    ):
        lhs_shared = T.alloc_shared((block_M, block_K), dtype)
        rhs_shared = T.alloc_shared((block_K, block_N), dtype)
        acc = T.alloc_fragment((block_M, block_N), accum_dtype)

        T.clear(acc)
        for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
            T.copy(lhs[by * block_M, ko * block_K], lhs_shared)
            T.copy(rhs[ko * block_K, bx * block_N], rhs_shared)
            T.gemm(lhs_shared, rhs_shared, acc)

        for i, j in T.Parallel(block_M, block_N):
            acc[i, j] = T.max(acc[i, j], T.cast(0.0, accum_dtype))

        T.copy(acc, out[by * block_M, bx * block_N])

    return out


def get_autotune_configs() -> list[dict[str, int | bool]]:
    iter_params = dict(
        block_M=[64, 128],
        block_N=[128, 256],
        block_K=[32, 64],
        num_stages=[2, 3],
        threads=[128, 256],
        enable_swizzle=[False, True],
    )
    return [
        dict(zip(iter_params.keys(), values))
        for values in itertools.product(*iter_params.values())
    ]


def make_fused_gemm_relu_prim_func(
    rows: int,
    cols: int,
    inner: int,
    block_M: int,
    block_N: int,
    block_K: int,
    threads: int,
    num_stages: int,
    enable_swizzle: bool,
    dtype=T.float16,
    accum_dtype=T.float32,
):
    @T.prim_func
    def main(
        lhs: T.Tensor((rows, inner), dtype),
        rhs: T.Tensor((inner, cols), dtype),
        out: T.Tensor((rows, cols), dtype),
    ):
        with T.Kernel(T.ceildiv(cols, block_N), T.ceildiv(rows, block_M), threads=threads) as (
            bx,
            by,
        ):
            lhs_shared = T.alloc_shared((block_M, block_K), dtype)
            rhs_shared = T.alloc_shared((block_K, block_N), dtype)
            acc = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.use_swizzle(panel_size=10, enable=enable_swizzle)
            T.clear(acc)
            for ko in T.Pipelined(T.ceildiv(inner, block_K), num_stages=num_stages):
                T.copy(lhs[by * block_M, ko * block_K], lhs_shared)
                T.copy(rhs[ko * block_K, bx * block_N], rhs_shared)
                T.gemm(lhs_shared, rhs_shared, acc)

            for i, j in T.Parallel(block_M, block_N):
                acc[i, j] = T.max(acc[i, j], T.cast(0.0, accum_dtype))

            T.copy(acc, out[by * block_M, bx * block_N])

    return main


def autotune_fused_gemm_relu(
    rows: int,
    cols: int,
    inner: int,
    profile_backend: str = "event",
    warmup: int = 3,
    rep: int = 20,
):
    def kernel(
        block_M=None,
        block_N=None,
        block_K=None,
        num_stages=None,
        threads=None,
        enable_swizzle=None,
    ):
        return make_fused_gemm_relu_prim_func(
            rows=rows,
            cols=cols,
            inner=inner,
            block_M=block_M,
            block_N=block_N,
            block_K=block_K,
            threads=threads,
            num_stages=num_stages,
            enable_swizzle=enable_swizzle,
        )

    autotuner = (
        AutoTuner.from_kernel(kernel=kernel, configs=get_autotune_configs())
        .set_compile_args(
            out_idx=[-1],
            target="auto",
        )
        .set_profile_args(
            supply_type=tl.TensorSupplyType.Normal,
            ref_prog=fused_gemm_relu_reference,
            skip_check=False,
            rtol=2e-2,
            atol=2e-2,
            backend=profile_backend,
        )
    )
    return autotuner.run(warmup=warmup, rep=rep)


@tl.jit(out_idx=[-1])
def fused_gemm_relu_with_config(
    rows: int,
    cols: int,
    inner: int,
    block_M: int = DEFAULT_BLOCK_M,
    block_N: int = DEFAULT_BLOCK_N,
    block_K: int = DEFAULT_BLOCK_K,
    threads: int = DEFAULT_THREADS,
    num_stages: int = DEFAULT_NUM_STAGES,
    enable_swizzle: bool = DEFAULT_ENABLE_SWIZZLE,
    dtype=T.float16,
    accum_dtype=T.float32,
):
    return make_fused_gemm_relu_prim_func(
        rows=rows,
        cols=cols,
        inner=inner,
        block_M=block_M,
        block_N=block_N,
        block_K=block_K,
        threads=threads,
        num_stages=num_stages,
        enable_swizzle=enable_swizzle,
        dtype=dtype,
        accum_dtype=accum_dtype,
    )


def fused_gemm_relu_reference(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    return torch.relu(lhs @ rhs)


def make_inputs(
    rows: int,
    cols: int,
    inner: int,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    lhs = torch.randn(rows, inner, device=device, dtype=dtype, generator=generator)
    rhs = torch.randn(inner, cols, device=device, dtype=dtype, generator=generator)
    return lhs, rhs


def run_demo(
    rows: int = 1024,
    cols: int = 1024,
    inner: int = 1024,
    block_M: int = DEFAULT_BLOCK_M,
    block_N: int = DEFAULT_BLOCK_N,
    block_K: int = DEFAULT_BLOCK_K,
    threads: int = DEFAULT_THREADS,
    num_stages: int = DEFAULT_NUM_STAGES,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    lhs, rhs = make_inputs(rows=rows, cols=cols, inner=inner, dtype=dtype)
    return fused_gemm_relu(
        lhs,
        rhs,
        block_M=block_M,
        block_N=block_N,
        block_K=block_K,
        threads=threads,
        num_stages=num_stages,
        dtype=torch_dtype_to_tilelang_dtype(dtype),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the TileLang fused GEMM+ReLU kernel.")
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--cols", type=int, default=1024)
    parser.add_argument("--inner", type=int, default=1024)
    parser.add_argument("--block-m", type=int, default=DEFAULT_BLOCK_M)
    parser.add_argument("--block-n", type=int, default=DEFAULT_BLOCK_N)
    parser.add_argument("--block-k", type=int, default=DEFAULT_BLOCK_K)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--num-stages", type=int, default=DEFAULT_NUM_STAGES)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    output = run_demo(
        rows=args.rows,
        cols=args.cols,
        inner=args.inner,
        block_M=args.block_m,
        block_N=args.block_n,
        block_K=args.block_k,
        threads=args.threads,
        num_stages=args.num_stages,
    )
    print(output[:4, :4])
