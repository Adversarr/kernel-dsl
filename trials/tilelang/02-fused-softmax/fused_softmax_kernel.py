import argparse

import tilelang
import tilelang.language as T
import torch


DEFAULT_THREADS = 128
DEFAULT_BLOCK_N = 128
LOG2E = 1.4426950408889634


@tilelang.jit
def fused_softmax(
    x,
    block_N: int = DEFAULT_BLOCK_N,
    threads: int = DEFAULT_THREADS,
    dtype=T.float16,
    accum_dtype=T.float32,
):
    M, N = T.const("M, N")

    x: T.Tensor((M, N), dtype)
    y = T.empty((M, N), dtype)

    with T.Kernel(M, threads=threads) as row:
        row_max = T.alloc_fragment((1,), accum_dtype)
        row_max_prev = T.alloc_fragment((1,), accum_dtype)
        row_sum = T.alloc_fragment((1,), accum_dtype)
        row_scale = T.alloc_fragment((1,), accum_dtype)
        tile_values = T.alloc_fragment((1, block_N), accum_dtype)
        tile_max = T.alloc_fragment((1,), accum_dtype)
        tile_reduction = T.alloc_fragment((1,), accum_dtype)

        T.fill(row_max, -T.infinity(accum_dtype))
        T.clear(row_sum)

        for block_idx in T.serial(T.ceildiv(N, block_N)):
            row_max_prev[0] = row_max[0]

            for col_in_block in T.Parallel(block_N):
                col = block_idx * block_N + col_in_block
                tile_values[0, col_in_block] = T.if_then_else(
                    col < N,
                    T.cast(x[row, col], accum_dtype),
                    -T.infinity(accum_dtype),
                )

            T.reduce_max(tile_values, tile_max, dim=1, clear=True)
            row_max[0] = T.max(row_max_prev[0], tile_max[0])
            row_scale[0] = T.exp2((row_max_prev[0] - row_max[0]) * LOG2E)
            row_sum[0] = row_sum[0] * row_scale[0]

            for col_in_block in T.Parallel(block_N):
                col = block_idx * block_N + col_in_block
                tile_values[0, col_in_block] = T.if_then_else(
                    col < N,
                    T.exp2((T.cast(x[row, col], accum_dtype) - row_max[0]) * LOG2E),
                    T.cast(0.0, accum_dtype),
                )

            T.reduce_sum(tile_values, tile_reduction, dim=1, clear=True)
            row_sum[0] = row_sum[0] + tile_reduction[0]

        for block_idx in T.serial(T.ceildiv(N, block_N)):
            for col_in_block in T.Parallel(block_N):
                col = block_idx * block_N + col_in_block
                if col < N:
                    numerator = T.exp2((T.cast(x[row, col], accum_dtype) - row_max[0]) * LOG2E)
                    y[row, col] = T.cast(numerator / row_sum[0], dtype)

    return y


def fused_softmax_reference(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, dim=-1)


def make_inputs(
    rows: int,
    cols: int,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
    seed: int = 0,
) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return torch.randn(rows, cols, device=device, dtype=dtype, generator=generator)


def run_demo(
    rows: int = 1024,
    cols: int = 4096,
    block_N: int = DEFAULT_BLOCK_N,
    threads: int = DEFAULT_THREADS,
) -> torch.Tensor:
    x = make_inputs(rows=rows, cols=cols)
    return fused_softmax(x, block_N=block_N, threads=threads)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the TileLang fused softmax kernel.")
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--cols", type=int, default=4096)
    parser.add_argument("--block-n", type=int, default=DEFAULT_BLOCK_N)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    output = run_demo(
        rows=args.rows,
        cols=args.cols,
        block_N=args.block_n,
        threads=args.threads,
    )
    print(output[:2, :8])
