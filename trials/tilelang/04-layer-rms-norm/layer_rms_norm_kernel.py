import argparse
import itertools

import tilelang as tl
import tilelang.language as T
import torch
from tilelang.autotuner import AutoTuner


DEFAULT_THREADS = 128
DEFAULT_ELEMENTS_PER_THREAD = 8
DEFAULT_EPS = 1e-6


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
def layer_rms_norm(
    x,
    weight,
    eps: float = DEFAULT_EPS,
    threads: int = DEFAULT_THREADS,
    elements_per_thread: int = DEFAULT_ELEMENTS_PER_THREAD,
    dtype=T.float16,
    accum_dtype=T.float32,
):
    rows, cols = T.const("rows, cols")

    x: T.Tensor((rows, cols), dtype)
    weight: T.Tensor((cols,), dtype)
    out = T.empty((rows, cols), dtype)

    tile_cols = threads * elements_per_thread

    with T.Kernel(rows, threads=threads) as row:
        row_square_sum = T.alloc_fragment((1,), accum_dtype)
        tile_square_sum = T.alloc_fragment((1,), accum_dtype)
        tile_values = T.alloc_fragment((1, tile_cols), accum_dtype)
        inv_rms = T.alloc_fragment((1,), accum_dtype)

        T.clear(row_square_sum)

        for block_idx in T.serial(T.ceildiv(cols, tile_cols)):
            for tid, lane in T.Parallel(threads, elements_per_thread):
                idx = tid * elements_per_thread + lane
                col = block_idx * tile_cols + idx
                value = T.if_then_else(
                    col < cols,
                    T.cast(x[row, col], accum_dtype),
                    T.cast(0.0, accum_dtype),
                )
                tile_values[0, idx] = value * value

            T.reduce_sum(tile_values, tile_square_sum, dim=1, clear=True)
            row_square_sum[0] = row_square_sum[0] + tile_square_sum[0]

        inv_rms[0] = T.rsqrt(
            row_square_sum[0] / T.cast(cols, accum_dtype) + T.cast(eps, accum_dtype)
        )

        for block_idx in T.serial(T.ceildiv(cols, tile_cols)):
            for tid, lane in T.Parallel(threads, elements_per_thread):
                idx = tid * elements_per_thread + lane
                col = block_idx * tile_cols + idx
                if col < cols:
                    value = T.cast(x[row, col], accum_dtype)
                    gamma = T.cast(weight[col], accum_dtype)
                    out[row, col] = T.cast(value * inv_rms[0] * gamma, dtype)

    return out


def get_autotune_configs() -> list[dict[str, int]]:
    iter_params = dict(
        threads=[64, 128, 256],
        elements_per_thread=[2, 4, 8, 16],
    )
    return [
        dict(zip(iter_params.keys(), values))
        for values in itertools.product(*iter_params.values())
    ]


def make_layer_rms_norm_prim_func(
    rows: int,
    cols: int,
    eps: float,
    threads: int,
    elements_per_thread: int,
    dtype=T.float16,
    accum_dtype=T.float32,
):
    tile_cols = threads * elements_per_thread

    @T.prim_func
    def main(
        x: T.Tensor((rows, cols), dtype),
        weight: T.Tensor((cols,), dtype),
        out: T.Tensor((rows, cols), dtype),
    ):
        with T.Kernel(rows, threads=threads) as row:
            row_square_sum = T.alloc_fragment((1,), accum_dtype)
            tile_square_sum = T.alloc_fragment((1,), accum_dtype)
            tile_values = T.alloc_fragment((1, tile_cols), accum_dtype)
            inv_rms = T.alloc_fragment((1,), accum_dtype)

            T.clear(row_square_sum)

            for block_idx in T.serial(T.ceildiv(cols, tile_cols)):
                for tid, lane in T.Parallel(threads, elements_per_thread):
                    idx = tid * elements_per_thread + lane
                    col = block_idx * tile_cols + idx
                    value = T.if_then_else(
                        col < cols,
                        T.cast(x[row, col], accum_dtype),
                        T.cast(0.0, accum_dtype),
                    )
                    tile_values[0, idx] = value * value

                T.reduce_sum(tile_values, tile_square_sum, dim=1, clear=True)
                row_square_sum[0] = row_square_sum[0] + tile_square_sum[0]

            inv_rms[0] = T.rsqrt(
                row_square_sum[0] / T.cast(cols, accum_dtype) + T.cast(eps, accum_dtype)
            )

            for block_idx in T.serial(T.ceildiv(cols, tile_cols)):
                for tid, lane in T.Parallel(threads, elements_per_thread):
                    idx = tid * elements_per_thread + lane
                    col = block_idx * tile_cols + idx
                    if col < cols:
                        value = T.cast(x[row, col], accum_dtype)
                        gamma = T.cast(weight[col], accum_dtype)
                        out[row, col] = T.cast(value * inv_rms[0] * gamma, dtype)

    return main


def autotune_layer_rms_norm(
    rows: int,
    cols: int,
    eps: float = DEFAULT_EPS,
    profile_backend: str = "event",
    warmup: int = 3,
    rep: int = 20,
):
    def kernel(threads=None, elements_per_thread=None):
        return make_layer_rms_norm_prim_func(
            rows=rows,
            cols=cols,
            eps=eps,
            threads=threads,
            elements_per_thread=elements_per_thread,
        )

    autotuner = (
        AutoTuner.from_kernel(kernel=kernel, configs=get_autotune_configs())
        .set_compile_args(
            out_idx=[-1],
            target="auto",
        )
        .set_profile_args(
            supply_type=tl.TensorSupplyType.Normal,
            ref_prog=lambda x, weight: layer_rms_norm_reference(x, weight, eps=eps),
            skip_check=False,
            rtol=2e-2,
            atol=2e-2,
            backend=profile_backend,
        )
    )
    return autotuner.run(warmup=warmup, rep=rep)


@tl.jit(out_idx=[-1])
def layer_rms_norm_with_config(
    rows: int,
    cols: int,
    eps: float = DEFAULT_EPS,
    threads: int = DEFAULT_THREADS,
    elements_per_thread: int = DEFAULT_ELEMENTS_PER_THREAD,
    dtype=T.float16,
    accum_dtype=T.float32,
):
    return make_layer_rms_norm_prim_func(
        rows=rows,
        cols=cols,
        eps=eps,
        threads=threads,
        elements_per_thread=elements_per_thread,
        dtype=dtype,
        accum_dtype=accum_dtype,
    )


def layer_rms_norm_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = DEFAULT_EPS,
) -> torch.Tensor:
    variance = torch.mean(x.float() * x.float(), dim=-1, keepdim=True)
    normalized = x.float() * torch.rsqrt(variance + eps)
    return (normalized * weight.float()).to(dtype=x.dtype)


def make_inputs(
    rows: int,
    cols: int,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    x = torch.randn(rows, cols, device=device, dtype=dtype, generator=generator)
    weight = torch.randn(cols, device=device, dtype=dtype, generator=generator)
    return x, weight


def run_demo(
    rows: int = 4096,
    cols: int = 4096,
    eps: float = DEFAULT_EPS,
    threads: int = DEFAULT_THREADS,
    elements_per_thread: int = DEFAULT_ELEMENTS_PER_THREAD,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    x, weight = make_inputs(rows=rows, cols=cols, dtype=dtype)
    return layer_rms_norm(
        x,
        weight,
        eps=eps,
        threads=threads,
        elements_per_thread=elements_per_thread,
        dtype=torch_dtype_to_tilelang_dtype(dtype),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the TileLang layer RMSNorm kernel.")
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--cols", type=int, default=4096)
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument(
        "--elements-per-thread",
        type=int,
        default=DEFAULT_ELEMENTS_PER_THREAD,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    output = run_demo(
        rows=args.rows,
        cols=args.cols,
        eps=args.eps,
        threads=args.threads,
        elements_per_thread=args.elements_per_thread,
    )
    print(output[:2, :8])
