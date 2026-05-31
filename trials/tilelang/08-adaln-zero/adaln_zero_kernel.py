import argparse
import itertools

import tilelang as tl
import tilelang.language as T
import torch
from tilelang.autotuner import AutoTuner

DEFAULT_BLOCK_M = 2
DEFAULT_THREADS = 256
DEFAULT_EPS = 1e-5


def _torch_dtype_to_tilelang(dtype: torch.dtype):
    mapping = {
        torch.float16: T.float16,
        torch.float32: T.float32,
        torch.bfloat16: T.bfloat16,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported torch dtype: {dtype}")
    return mapping[dtype]


def adaln_zero_reference(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    gate: torch.Tensor,
    eps: float = DEFAULT_EPS,
) -> torch.Tensor:
    gate_scale = gate * scale
    gate_shift = gate * shift
    ln = torch.nn.functional.layer_norm(x, (x.shape[-1],), eps=eps)
    return gate_scale * ln + gate_shift


def _make_adaln_zero_prim_func(
    N: int,
    D: int,
    eps: float = DEFAULT_EPS,
    block_M: int = DEFAULT_BLOCK_M,
    threads: int = DEFAULT_THREADS,
    dtype=T.float16,
    accum_dtype=T.float32,
):
    @T.prim_func
    def main(
        X: T.Tensor((N, D), dtype),
        GateScale: T.Tensor((D,), dtype),
        GateShift: T.Tensor((D,), dtype),
        Y: T.Tensor((N, D), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_M), threads=threads) as bx:
            X_smem = T.alloc_shared((block_M, D), dtype)
            GS_smem = T.alloc_shared((D,), dtype)
            GH_smem = T.alloc_shared((D,), dtype)
            X_local = T.alloc_fragment((block_M, D), accum_dtype)
            X_sq_local = T.alloc_fragment((block_M, D), accum_dtype)
            sum_row = T.alloc_fragment((block_M,), accum_dtype)
            sumsq_row = T.alloc_fragment((block_M,), accum_dtype)
            mean_row = T.alloc_fragment((block_M,), accum_dtype)
            rstd_row = T.alloc_fragment((block_M,), accum_dtype)

            T.copy(X[bx * block_M, 0], X_smem)
            T.copy(GateScale, GS_smem)
            T.copy(GateShift, GH_smem)

            for i, j in T.Parallel(block_M, D):
                X_local[i, j] = T.Cast(accum_dtype, X_smem[i, j])

            for i, j in T.Parallel(block_M, D):
                X_sq_local[i, j] = X_local[i, j] * X_local[i, j]

            T.reduce_sum(X_local, sum_row, dim=1)
            T.reduce_sum(X_sq_local, sumsq_row, dim=1)

            inv_D = T.float32(1.0) / T.Cast(accum_dtype, D)
            for i in T.Parallel(block_M):
                mean_row[i] = sum_row[i] * inv_D
                var = sumsq_row[i] * inv_D - mean_row[i] * mean_row[i]
                rstd_row[i] = T.rsqrt(var + T.Cast(accum_dtype, eps))

            for i, j in T.Parallel(block_M, D):
                norm = (X_local[i, j] - mean_row[i]) * rstd_row[i]
                X_smem[i, j] = T.Cast(
                    dtype,
                    norm * T.Cast(accum_dtype, GS_smem[j]) + T.Cast(accum_dtype, GH_smem[j]),
                )

            T.copy(X_smem, Y[bx * block_M, 0])

    return main


@tl.jit(out_idx=[-1], pass_configs={tl.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def _adaln_zero_with_config(
    N: int,
    D: int,
    eps: float = DEFAULT_EPS,
    block_M: int = DEFAULT_BLOCK_M,
    threads: int = DEFAULT_THREADS,
    dtype=T.float16,
    accum_dtype=T.float32,
):
    return _make_adaln_zero_prim_func(
        N=N,
        D=D,
        eps=eps,
        block_M=block_M,
        threads=threads,
        dtype=dtype,
        accum_dtype=accum_dtype,
    )


def adaln_zero(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    gate: torch.Tensor,
    eps: float = DEFAULT_EPS,
    block_M: int = DEFAULT_BLOCK_M,
    threads: int = DEFAULT_THREADS,
) -> torch.Tensor:
    gate_scale = gate * scale
    gate_shift = gate * shift
    N, D = x.shape
    kernel = _adaln_zero_with_config(
        N=N,
        D=D,
        eps=eps,
        block_M=block_M,
        threads=threads,
        dtype=_torch_dtype_to_tilelang(x.dtype),
    )
    return kernel(x, gate_scale, gate_shift)


def get_autotune_configs(N: int) -> list[dict[str, int]]:
    iter_params = dict(
        block_M=[v for v in [1, 2, 4] if N % v == 0],
        threads=[128, 256, 512],
    )
    return [
        dict(zip(iter_params.keys(), values))
        for values in itertools.product(*iter_params.values())
    ]


def autotune_adaln_zero(
    N: int,
    D: int,
    eps: float = DEFAULT_EPS,
    profile_backend: str = "event",
    warmup: int = 3,
    rep: int = 20,
):
    def kernel(block_M=None, threads=None):
        return _make_adaln_zero_prim_func(
            N=N,
            D=D,
            eps=eps,
            block_M=block_M,
            threads=threads,
        )

    x, scale, shift, gate = make_inputs(N, D)
    gate_scale = gate * scale
    gate_shift = gate * shift

    autotuner = (
        AutoTuner.from_kernel(kernel=kernel, configs=get_autotune_configs(N))
        .set_compile_args(
            out_idx=[-1],
            target="auto",
        )
        .set_profile_args(
            supply_type=tl.TensorSupplyType.Normal,
            ref_prog=lambda x, gs, gh: gs * torch.nn.functional.layer_norm(x, (x.shape[-1],), eps=eps) + gh,
            skip_check=False,
            rtol=1e-2,
            atol=1e-2,
            backend=profile_backend,
        )
    )
    return autotuner.run(warmup=warmup, rep=rep)


def make_inputs(
    N: int,
    D: int,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    x = torch.randn(N, D, device=device, dtype=dtype, generator=generator)
    scale = torch.randn(D, device=device, dtype=dtype, generator=generator)
    shift = torch.randn(D, device=device, dtype=dtype, generator=generator)
    gate = torch.randn(D, device=device, dtype=dtype, generator=generator)
    return x, scale, shift, gate


def run_demo(
    N: int = 4096,
    D: int = 8192,
    eps: float = DEFAULT_EPS,
    block_M: int = DEFAULT_BLOCK_M,
    threads: int = DEFAULT_THREADS,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    x, scale, shift, gate = make_inputs(N=N, D=D, dtype=dtype)
    return adaln_zero(
        x,
        scale,
        shift,
        gate,
        eps=eps,
        block_M=block_M,
        threads=threads,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the TileLang AdaLN-Zero kernel.")
    parser.add_argument("--N", type=int, default=4096)
    parser.add_argument("--D", type=int, default=8192)
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS)
    parser.add_argument("--block-m", type=int, default=DEFAULT_BLOCK_M)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    output = run_demo(
        N=args.N,
        D=args.D,
        eps=args.eps,
        block_M=args.block_m,
        threads=args.threads,
    )
    print(output[0, :4])
