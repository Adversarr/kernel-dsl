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


def _validate_swiglu_weight(weight1: torch.Tensor) -> None:
    if weight1.shape[1] % 2 != 0:
        raise ValueError(f"weight1 output dimension must be even, got {weight1.shape[1]}")


def _validate_swiglu_weight_t(weight1_t: torch.Tensor) -> None:
    if weight1_t.shape[0] % 2 != 0:
        raise ValueError(f"transposed weight1 output dimension must be even, got {weight1_t.shape[0]}")


@tl.jit(out_idx=[-2, -1])
def gemm_residual_rmsnorm_stage1_with_config(
    rows: int,
    hidden: int,
    inner0: int,
    partial_cols: int,
    block_M: int = DEFAULT_BLOCK_M,
    block_N: int = DEFAULT_BLOCK_N,
    block_K: int = DEFAULT_BLOCK_K,
    threads: int = DEFAULT_THREADS,
    num_stages: int = DEFAULT_NUM_STAGES,
    enable_swizzle: bool = DEFAULT_ENABLE_SWIZZLE,
    dtype=T.float16,
    accum_dtype=T.float32,
):
    return make_gemm_residual_rmsnorm_stage1_prim_func(
        rows=rows,
        hidden=hidden,
        inner0=inner0,
        partial_cols=partial_cols,
        block_M=block_M,
        block_N=block_N,
        block_K=block_K,
        threads=threads,
        num_stages=num_stages,
        enable_swizzle=enable_swizzle,
        dtype=dtype,
        accum_dtype=accum_dtype,
    )


def make_gemm_residual_rmsnorm_stage1_prim_func(
    rows: int,
    hidden: int,
    inner0: int,
    partial_cols: int,
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
        x: T.Tensor((rows, inner0), dtype),
        weight0: T.Tensor((inner0, hidden), dtype),
        residual: T.Tensor((rows, hidden), dtype),
        gamma: T.Tensor((hidden,), dtype),
        weighted_residual: T.Tensor((rows, hidden), dtype),
        partial_squares: T.Tensor((partial_cols, rows), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(hidden, block_N), T.ceildiv(rows, block_M), threads=threads) as (
            bx,
            by,
        ):
            x_shared = T.alloc_shared((block_M, block_K), dtype)
            w_shared = T.alloc_shared((block_K, block_N), dtype)
            acc = T.alloc_fragment((block_M, block_N), accum_dtype)
            squares = T.alloc_fragment((block_M, block_N), accum_dtype)
            row_sums = T.alloc_fragment((block_M,), accum_dtype)

            T.use_swizzle(panel_size=10, enable=enable_swizzle)
            T.clear(acc)
            for ko in T.Pipelined(T.ceildiv(inner0, block_K), num_stages=num_stages):
                T.copy(x[by * block_M, ko * block_K], x_shared)
                T.copy(weight0[ko * block_K, bx * block_N], w_shared)
                T.gemm(x_shared, w_shared, acc)

            for i, j in T.Parallel(block_M, block_N):
                row = by * block_M + i
                col = bx * block_N + j
                if row < rows and col < hidden:
                    value = acc[i, j] + T.cast(residual[row, col], accum_dtype)
                    weighted_residual[row, col] = T.cast(value * T.cast(gamma[col], accum_dtype), dtype)
                    squares[i, j] = value * value
                else:
                    squares[i, j] = T.cast(0.0, accum_dtype)

            T.reduce_sum(squares, row_sums, dim=1, clear=True)
            for i in T.Parallel(block_M):
                row_sums[i] = row_sums[i] / T.cast(hidden, accum_dtype)
            T.copy(row_sums, partial_squares[bx, by * block_M])

    return main


@tl.jit(out_idx=[-1])
def rms_partial_reduce_with_config(
    rows: int,
    partial_cols: int,
    eps: float = DEFAULT_EPS,
    threads: int = DEFAULT_THREADS,
    elements_per_thread: int = 8,
    accum_dtype=T.float32,
):
    tile_cols = threads * elements_per_thread

    @T.prim_func
    def main(
        partial_squares: T.Tensor((partial_cols, rows), accum_dtype),
        inv_rms: T.Tensor((rows,), accum_dtype),
    ):
        with T.Kernel(rows, threads=threads) as row:
            row_sum = T.alloc_fragment((1,), accum_dtype)
            tile_values = T.alloc_fragment((1, tile_cols), accum_dtype)
            tile_sum = T.alloc_fragment((1,), accum_dtype)

            T.clear(row_sum)
            for block_idx in T.serial(T.ceildiv(partial_cols, tile_cols)):
                for tid, lane in T.Parallel(threads, elements_per_thread):
                    idx = tid * elements_per_thread + lane
                    col = block_idx * tile_cols + idx
                    tile_values[0, idx] = T.if_then_else(
                        col < partial_cols,
                        partial_squares[col, row],
                        T.cast(0.0, accum_dtype),
                    )
                T.reduce_sum(tile_values, tile_sum, dim=1, clear=True)
                row_sum[0] = row_sum[0] + tile_sum[0]

            inv_rms[row] = T.rsqrt(row_sum[0] + T.cast(eps, accum_dtype))

    return main


@tl.jit(out_idx=[-1])
def gemm_rmsnorm_swiglu_stage2_with_config(
    rows: int,
    mlp: int,
    hidden: int,
    block_M: int = DEFAULT_BLOCK_M,
    block_N: int = DEFAULT_BLOCK_N,
    block_K: int = DEFAULT_BLOCK_K,
    threads: int = DEFAULT_THREADS,
    num_stages: int = DEFAULT_NUM_STAGES,
    enable_swizzle: bool = DEFAULT_ENABLE_SWIZZLE,
    dtype=T.float16,
    accum_dtype=T.float32,
):
    return make_gemm_rmsnorm_swiglu_stage2_prim_func(
        rows=rows,
        mlp=mlp,
        hidden=hidden,
        block_M=block_M,
        block_N=block_N,
        block_K=block_K,
        threads=threads,
        num_stages=num_stages,
        enable_swizzle=enable_swizzle,
        dtype=dtype,
        accum_dtype=accum_dtype,
    )


def make_gemm_rmsnorm_swiglu_stage2_prim_func(
    rows: int,
    mlp: int,
    hidden: int,
    block_M: int,
    block_N: int,
    block_K: int,
    threads: int,
    num_stages: int,
    enable_swizzle: bool,
    dtype=T.float16,
    accum_dtype=T.float32,
):
    paired_cols = mlp * 2
    paired_outputs_per_tile = block_N // 2

    @T.prim_func
    def main(
        weighted_residual: T.Tensor((rows, hidden), dtype),
        weight1: T.Tensor((hidden, paired_cols), dtype),
        inv_rms: T.Tensor((rows,), accum_dtype),
        out: T.Tensor((rows, mlp), dtype),
    ):
        with T.Kernel(T.ceildiv(paired_cols, block_N), T.ceildiv(rows, block_M), threads=threads) as (
            bx,
            by,
        ):
            a_shared = T.alloc_shared((block_M, block_K), dtype)
            w_shared = T.alloc_shared((block_K, block_N), dtype)
            acc = T.alloc_fragment((block_M, block_N), accum_dtype)
            gate_frag = T.alloc_fragment((block_M, paired_outputs_per_tile), accum_dtype)
            up_frag = T.alloc_fragment((block_M, paired_outputs_per_tile), accum_dtype)

            T.use_swizzle(panel_size=10, enable=enable_swizzle)
            T.clear(acc)
            for ko in T.Pipelined(T.ceildiv(hidden, block_K), num_stages=num_stages):
                T.copy(weighted_residual[by * block_M, ko * block_K], a_shared)
                T.copy(weight1[ko * block_K, bx * block_N], w_shared)
                T.gemm(a_shared, w_shared, acc)

            for i, j in T.Parallel(block_M, paired_outputs_per_tile):
                gate_frag[i, j] = acc[i, j * 2]
            for i, j in T.Parallel(block_M, paired_outputs_per_tile):
                up_frag[i, j] = acc[i, j * 2 + 1]
            for i, j in T.Parallel(block_M, paired_outputs_per_tile):
                row = by * block_M + i
                paired_col = bx * block_N + j * 2
                col = paired_col // 2
                if row < rows and col < mlp:
                    scale = inv_rms[row]
                    gate = gate_frag[i, j] * scale
                    up = up_frag[i, j] * scale
                    out[row, col] = T.cast(
                        (gate * (T.cast(1.0, accum_dtype) / (T.cast(1.0, accum_dtype) + T.exp2(-gate * T.cast(1.4426950408889634, accum_dtype)))))
                        * up,
                        dtype,
                    )

    return main


@tl.jit(out_idx=[-1])
def gemm_rmsnorm_swiglu_stage2_transposed_with_config(
    rows: int,
    mlp: int,
    hidden: int,
    block_M: int = DEFAULT_BLOCK_M,
    block_N: int = DEFAULT_BLOCK_N,
    block_K: int = DEFAULT_BLOCK_K,
    threads: int = DEFAULT_THREADS,
    num_stages: int = DEFAULT_NUM_STAGES,
    enable_swizzle: bool = DEFAULT_ENABLE_SWIZZLE,
    dtype=T.float16,
    accum_dtype=T.float32,
):
    return make_gemm_rmsnorm_swiglu_stage2_transposed_prim_func(
        rows=rows,
        mlp=mlp,
        hidden=hidden,
        block_M=block_M,
        block_N=block_N,
        block_K=block_K,
        threads=threads,
        num_stages=num_stages,
        enable_swizzle=enable_swizzle,
        dtype=dtype,
        accum_dtype=accum_dtype,
    )


def make_gemm_rmsnorm_swiglu_stage2_transposed_prim_func(
    rows: int,
    mlp: int,
    hidden: int,
    block_M: int,
    block_N: int,
    block_K: int,
    threads: int,
    num_stages: int,
    enable_swizzle: bool,
    dtype=T.float16,
    accum_dtype=T.float32,
):
    paired_cols = mlp * 2
    paired_outputs_per_tile = block_N // 2

    @T.prim_func
    def main(
        weighted_residual: T.Tensor((rows, hidden), dtype),
        weight1_t: T.Tensor((paired_cols, hidden), dtype),
        inv_rms: T.Tensor((rows,), accum_dtype),
        out: T.Tensor((rows, mlp), dtype),
    ):
        with T.Kernel(T.ceildiv(paired_cols, block_N), T.ceildiv(rows, block_M), threads=threads) as (
            bx,
            by,
        ):
            a_shared = T.alloc_shared((block_M, block_K), dtype)
            w_shared = T.alloc_shared((block_N, block_K), dtype)
            acc = T.alloc_fragment((block_M, block_N), accum_dtype)
            gate_frag = T.alloc_fragment((block_M, paired_outputs_per_tile), accum_dtype)
            up_frag = T.alloc_fragment((block_M, paired_outputs_per_tile), accum_dtype)

            T.use_swizzle(panel_size=10, enable=enable_swizzle)
            T.clear(acc)
            for ko in T.Pipelined(T.ceildiv(hidden, block_K), num_stages=num_stages):
                T.copy(weighted_residual[by * block_M, ko * block_K], a_shared)
                T.copy(weight1_t[bx * block_N, ko * block_K], w_shared)
                T.gemm(a_shared, w_shared, acc, transpose_B=True)

            for i, j in T.Parallel(block_M, paired_outputs_per_tile):
                gate_frag[i, j] = acc[i, j * 2]
            for i, j in T.Parallel(block_M, paired_outputs_per_tile):
                up_frag[i, j] = acc[i, j * 2 + 1]
            for i, j in T.Parallel(block_M, paired_outputs_per_tile):
                row = by * block_M + i
                paired_col = bx * block_N + j * 2
                col = paired_col // 2
                if row < rows and col < mlp:
                    scale = inv_rms[row]
                    gate = gate_frag[i, j] * scale
                    up = up_frag[i, j] * scale
                    out[row, col] = T.cast(
                        (gate * (T.cast(1.0, accum_dtype) / (T.cast(1.0, accum_dtype) + T.exp2(-gate * T.cast(1.4426950408889634, accum_dtype)))))
                        * up,
                        dtype,
                    )

    return main


@tl.jit(out_idx=[-1])
def gemm_rmsnorm_swiglu_stage2_split_weights_with_config(
    rows: int,
    mlp: int,
    hidden: int,
    block_M: int = DEFAULT_BLOCK_M,
    block_N: int = DEFAULT_BLOCK_N,
    block_K: int = DEFAULT_BLOCK_K,
    threads: int = DEFAULT_THREADS,
    num_stages: int = DEFAULT_NUM_STAGES,
    enable_swizzle: bool = DEFAULT_ENABLE_SWIZZLE,
    dtype=T.float16,
    accum_dtype=T.float32,
):
    @T.prim_func
    def main(
        weighted_residual: T.Tensor((rows, hidden), dtype),
        weight_gate: T.Tensor((hidden, mlp), dtype),
        weight_up: T.Tensor((hidden, mlp), dtype),
        inv_rms: T.Tensor((rows,), accum_dtype),
        out: T.Tensor((rows, mlp), dtype),
    ):
        with T.Kernel(T.ceildiv(mlp, block_N), T.ceildiv(rows, block_M), threads=threads) as (
            bx,
            by,
        ):
            a_shared = T.alloc_shared((block_M, block_K), dtype)
            gate_shared = T.alloc_shared((block_K, block_N), dtype)
            up_shared = T.alloc_shared((block_K, block_N), dtype)
            gate_acc = T.alloc_fragment((block_M, block_N), accum_dtype)
            up_acc = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.use_swizzle(panel_size=10, enable=enable_swizzle)
            T.clear(gate_acc)
            T.clear(up_acc)
            for ko in T.Pipelined(T.ceildiv(hidden, block_K), num_stages=num_stages):
                T.copy(weighted_residual[by * block_M, ko * block_K], a_shared)
                T.copy(weight_gate[ko * block_K, bx * block_N], gate_shared)
                T.copy(weight_up[ko * block_K, bx * block_N], up_shared)
                T.gemm(a_shared, gate_shared, gate_acc, policy=T.GemmWarpPolicy.FullRow)
                T.gemm(a_shared, up_shared, up_acc, policy=T.GemmWarpPolicy.FullRow)

            for i, j in T.Parallel(block_M, block_N):
                row = by * block_M + i
                col = bx * block_N + j
                if row < rows and col < mlp:
                    scale = inv_rms[row]
                    gate = gate_acc[i, j] * scale
                    up = up_acc[i, j] * scale
                    out[row, col] = T.cast(
                        (gate * (T.cast(1.0, accum_dtype) / (T.cast(1.0, accum_dtype) + T.exp2(-gate * T.cast(1.4426950408889634, accum_dtype)))))
                        * up,
                        dtype,
                    )

    return main


def get_autotune_configs() -> list[dict[str, int | bool]]:
    iter_params = dict(
        block_M=[64, 128],
        block_N=[64, 128],
        block_K=[32, 64],
        num_stages=[2, 3],
        threads=[128, 256],
        enable_swizzle=[False, True],
    )
    return [
        dict(zip(iter_params.keys(), values))
        for values in itertools.product(*iter_params.values())
    ]


def _build_kernels(
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
    dtype,
    accum_dtype,
):
    stage1_stages = num_stages if stage1_num_stages is None else stage1_num_stages
    partial_cols = (hidden + block_N - 1) // block_N
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
        dtype=dtype,
        accum_dtype=accum_dtype,
    )
    reduce = rms_partial_reduce_with_config(
        rows,
        partial_cols,
        eps=DEFAULT_EPS,
        threads=threads,
        accum_dtype=accum_dtype,
    )
    stage2 = gemm_rmsnorm_swiglu_stage2_with_config(
        rows,
        mlp,
        hidden,
        block_M=block_M,
        block_N=block_N,
        block_K=block_K,
        threads=threads,
        num_stages=num_stages,
        enable_swizzle=enable_swizzle,
        dtype=dtype,
        accum_dtype=accum_dtype,
    )
    return stage1, reduce, stage2


def _build_kernels_transposed_stage2(
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
    dtype,
    accum_dtype,
):
    stage1_stages = num_stages if stage1_num_stages is None else stage1_num_stages
    partial_cols = (hidden + block_N - 1) // block_N
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
        dtype=dtype,
        accum_dtype=accum_dtype,
    )
    reduce = rms_partial_reduce_with_config(
        rows,
        partial_cols,
        eps=DEFAULT_EPS,
        threads=threads,
        accum_dtype=accum_dtype,
    )
    stage2 = gemm_rmsnorm_swiglu_stage2_transposed_with_config(
        rows,
        mlp,
        hidden,
        block_M=block_M,
        block_N=block_N,
        block_K=block_K,
        threads=threads,
        num_stages=num_stages,
        enable_swizzle=enable_swizzle,
        dtype=dtype,
        accum_dtype=accum_dtype,
    )
    return stage1, reduce, stage2


def gemm_residual_rmsnorm_gemm_swiglu(
    x: torch.Tensor,
    weight0: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    weight1: torch.Tensor,
    *,
    eps: float = DEFAULT_EPS,
    block_M: int = DEFAULT_BLOCK_M,
    block_N: int = DEFAULT_BLOCK_N,
    block_K: int = DEFAULT_BLOCK_K,
    threads: int = DEFAULT_THREADS,
    num_stages: int = DEFAULT_NUM_STAGES,
    stage1_num_stages: int | None = None,
    enable_swizzle: bool = DEFAULT_ENABLE_SWIZZLE,
) -> torch.Tensor:
    if eps != DEFAULT_EPS:
        raise ValueError("Use with_config kernels directly to specialize a non-default eps.")
    if x.shape[0] != residual.shape[0]:
        raise ValueError("x and residual must have the same row count")
    if weight0.shape[1] != residual.shape[1] or gamma.shape[0] != residual.shape[1]:
        raise ValueError("weight0, residual, and gamma hidden dimensions must match")
    if weight1.shape[0] != residual.shape[1]:
        raise ValueError("weight1 input dimension must match hidden")
    _validate_swiglu_weight(weight1)

    rows, inner0 = x.shape
    hidden = residual.shape[1]
    mlp = weight1.shape[1] // 2
    partial_cols = (hidden + block_N - 1) // block_N
    dtype = torch_dtype_to_tilelang_dtype(x.dtype)
    accum_dtype = T.float32

    stage1, reduce, stage2 = _build_kernels(
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
        dtype=dtype,
        accum_dtype=accum_dtype,
    )

    weighted_residual, partial_squares = stage1(x, weight0, residual, gamma)
    inv_rms = reduce(partial_squares)
    return stage2(weighted_residual, weight1, inv_rms)


def gemm_residual_rmsnorm_gemm_swiglu_transposed_stage2(
    x: torch.Tensor,
    weight0: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    weight1_t: torch.Tensor,
    *,
    eps: float = DEFAULT_EPS,
    block_M: int = DEFAULT_BLOCK_M,
    block_N: int = DEFAULT_BLOCK_N,
    block_K: int = DEFAULT_BLOCK_K,
    threads: int = DEFAULT_THREADS,
    num_stages: int = DEFAULT_NUM_STAGES,
    stage1_num_stages: int | None = None,
    enable_swizzle: bool = DEFAULT_ENABLE_SWIZZLE,
) -> torch.Tensor:
    if eps != DEFAULT_EPS:
        raise ValueError("Use with_config kernels directly to specialize a non-default eps.")
    if x.shape[0] != residual.shape[0]:
        raise ValueError("x and residual must have the same row count")
    if weight0.shape[1] != residual.shape[1] or gamma.shape[0] != residual.shape[1]:
        raise ValueError("weight0, residual, and gamma hidden dimensions must match")
    if weight1_t.shape[1] != residual.shape[1]:
        raise ValueError("transposed weight1 input dimension must match hidden")
    _validate_swiglu_weight_t(weight1_t)

    rows, inner0 = x.shape
    hidden = residual.shape[1]
    mlp = weight1_t.shape[0] // 2
    dtype = torch_dtype_to_tilelang_dtype(x.dtype)
    accum_dtype = T.float32

    stage1, reduce, stage2 = _build_kernels_transposed_stage2(
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
        dtype=dtype,
        accum_dtype=accum_dtype,
    )

    weighted_residual, partial_squares = stage1(x, weight0, residual, gamma)
    inv_rms = reduce(partial_squares)
    return stage2(weighted_residual, weight1_t, inv_rms)


def gemm_residual_rmsnorm_gemm_swiglu_reference(
    x: torch.Tensor,
    weight0: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    weight1: torch.Tensor,
    *,
    eps: float = DEFAULT_EPS,
) -> torch.Tensor:
    residual_updated = x.float() @ weight0.float() + residual.float()
    weighted = residual_updated * gamma.float()
    inv_rms = torch.rsqrt(torch.mean(residual_updated * residual_updated, dim=-1, keepdim=True) + eps)
    paired = (weighted @ weight1.float()) * inv_rms
    gate = paired[:, 0::2]
    up = paired[:, 1::2]
    return (torch.nn.functional.silu(gate) * up).to(dtype=x.dtype)


def gemm_residual_rmsnorm_gemm_swiglu_transposed_reference(
    x: torch.Tensor,
    weight0: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    weight1_t: torch.Tensor,
    *,
    eps: float = DEFAULT_EPS,
) -> torch.Tensor:
    return gemm_residual_rmsnorm_gemm_swiglu_reference(
        x,
        weight0,
        residual,
        gamma,
        weight1_t.t().contiguous(),
        eps=eps,
    )


def gemm_residual_rmsnorm_gemm_swiglu_split_weights(
    x: torch.Tensor,
    weight0: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    weight_gate: torch.Tensor,
    weight_up: torch.Tensor,
    *,
    eps: float = DEFAULT_EPS,
    block_M: int = DEFAULT_BLOCK_M,
    block_N: int = DEFAULT_BLOCK_N,
    block_K: int = DEFAULT_BLOCK_K,
    threads: int = DEFAULT_THREADS,
    num_stages: int = DEFAULT_NUM_STAGES,
    stage1_num_stages: int | None = None,
    enable_swizzle: bool = DEFAULT_ENABLE_SWIZZLE,
) -> torch.Tensor:
    if eps != DEFAULT_EPS:
        raise ValueError("Use with_config kernels directly to specialize a non-default eps.")
    if x.shape[0] != residual.shape[0]:
        raise ValueError("x and residual must have the same row count")
    if weight0.shape[1] != residual.shape[1] or gamma.shape[0] != residual.shape[1]:
        raise ValueError("weight0, residual, and gamma hidden dimensions must match")
    if weight_gate.shape != weight_up.shape:
        raise ValueError("gate and up weights must have the same shape")
    if weight_gate.shape[0] != residual.shape[1]:
        raise ValueError("split stage2 weights input dimension must match hidden")

    rows, inner0 = x.shape
    hidden, mlp = weight_gate.shape
    dtype = torch_dtype_to_tilelang_dtype(x.dtype)
    accum_dtype = T.float32
    stage1_stages = num_stages if stage1_num_stages is None else stage1_num_stages
    partial_cols = (hidden + block_N - 1) // block_N

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
        dtype=dtype,
        accum_dtype=accum_dtype,
    )
    reduce = rms_partial_reduce_with_config(
        rows,
        partial_cols,
        eps=DEFAULT_EPS,
        threads=threads,
        accum_dtype=accum_dtype,
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
        dtype=dtype,
        accum_dtype=accum_dtype,
    )
    weighted_residual, partial_squares = stage1(x, weight0, residual, gamma)
    inv_rms = reduce(partial_squares)
    return stage2(weighted_residual, weight_gate, weight_up, inv_rms)


def gemm_residual_rmsnorm_gemm_swiglu_split_weights_reference(
    x: torch.Tensor,
    weight0: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    weight_gate: torch.Tensor,
    weight_up: torch.Tensor,
    *,
    eps: float = DEFAULT_EPS,
) -> torch.Tensor:
    residual_updated = x.float() @ weight0.float() + residual.float()
    weighted = residual_updated * gamma.float()
    inv_rms = torch.rsqrt(torch.mean(residual_updated * residual_updated, dim=-1, keepdim=True) + eps)
    gate = (weighted @ weight_gate.float()) * inv_rms
    up = (weighted @ weight_up.float()) * inv_rms
    return (torch.nn.functional.silu(gate) * up).to(dtype=x.dtype)


def naive_torch_compile_reference(
    x: torch.Tensor,
    weight0: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    weight1: torch.Tensor,
) -> torch.Tensor:
    residual_updated = x @ weight0 + residual
    inv_rms = torch.rsqrt(
        torch.mean(residual_updated.float() * residual_updated.float(), dim=-1, keepdim=True)
        + DEFAULT_EPS
    )
    normalized = residual_updated * inv_rms.to(dtype=residual_updated.dtype) * gamma
    paired = normalized @ weight1
    gate = paired[:, 0::2]
    up = paired[:, 1::2]
    return torch.nn.functional.silu(gate) * up


def naive_torch_compile_split_weights_reference(
    x: torch.Tensor,
    weight0: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    weight_gate: torch.Tensor,
    weight_up: torch.Tensor,
) -> torch.Tensor:
    residual_updated = x @ weight0 + residual
    inv_rms = torch.rsqrt(
        torch.mean(residual_updated.float() * residual_updated.float(), dim=-1, keepdim=True)
        + DEFAULT_EPS
    )
    normalized = residual_updated * inv_rms.to(dtype=residual_updated.dtype) * gamma
    gate = normalized @ weight_gate
    up = normalized @ weight_up
    return torch.nn.functional.silu(gate) * up


def naive_torch_compile_transposed_reference(
    x: torch.Tensor,
    weight0: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    weight1_t: torch.Tensor,
) -> torch.Tensor:
    residual_updated = x @ weight0 + residual
    inv_rms = torch.rsqrt(
        torch.mean(residual_updated.float() * residual_updated.float(), dim=-1, keepdim=True)
        + DEFAULT_EPS
    )
    normalized = residual_updated * inv_rms.to(dtype=residual_updated.dtype) * gamma
    paired = normalized @ weight1_t.t()
    gate = paired[:, 0::2]
    up = paired[:, 1::2]
    return torch.nn.functional.silu(gate) * up


def make_inputs(
    rows: int,
    hidden: int,
    inner0: int | None = None,
    mlp: int | None = None,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    inner0 = hidden if inner0 is None else inner0
    mlp = hidden * 4 if mlp is None else mlp
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    x = torch.randn(rows, inner0, device=device, dtype=dtype, generator=generator)
    weight0 = torch.randn(inner0, hidden, device=device, dtype=dtype, generator=generator) / (inner0**0.5)
    residual = torch.randn(rows, hidden, device=device, dtype=dtype, generator=generator)
    gamma = torch.randn(hidden, device=device, dtype=dtype, generator=generator)
    weight1 = torch.randn(hidden, mlp * 2, device=device, dtype=dtype, generator=generator) / (hidden**0.5)
    return x, weight0, residual, gamma, weight1


def make_inputs_transposed_stage2(
    rows: int,
    hidden: int,
    inner0: int | None = None,
    mlp: int | None = None,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x, weight0, residual, gamma, weight1 = make_inputs(
        rows=rows,
        hidden=hidden,
        inner0=inner0,
        mlp=mlp,
        dtype=dtype,
        device=device,
        seed=seed,
    )
    return x, weight0, residual, gamma, weight1.t().contiguous()


def make_inputs_split_weights(
    rows: int,
    hidden: int,
    inner0: int | None = None,
    mlp: int | None = None,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x, weight0, residual, gamma, weight1 = make_inputs(
        rows=rows,
        hidden=hidden,
        inner0=inner0,
        mlp=mlp,
        dtype=dtype,
        device=device,
        seed=seed,
    )
    return x, weight0, residual, gamma, weight1[:, 0::2].contiguous(), weight1[:, 1::2].contiguous()


def autotune_gemm_residual_rmsnorm_gemm_swiglu(
    rows: int,
    hidden: int,
    inner0: int | None = None,
    mlp: int | None = None,
    profile_backend: str = "event",
    warmup: int = 3,
    rep: int = 20,
    skip_check: bool = True,
):
    inner0 = hidden if inner0 is None else inner0
    mlp = hidden * 4 if mlp is None else mlp

    def kernel(
        block_M=None,
        block_N=None,
        block_K=None,
        num_stages=None,
        threads=None,
        enable_swizzle=None,
    ):
        return gemm_rmsnorm_swiglu_stage2_with_config(
            rows=rows,
            mlp=mlp,
            hidden=hidden,
            block_M=block_M,
            block_N=block_N,
            block_K=block_K,
            threads=threads,
            num_stages=num_stages,
            enable_swizzle=enable_swizzle,
        )

    # Autotune the dominant second GEMM epilogue in isolation. The end-to-end
    # profile script times all three kernels together after choosing a config.
    autotuner = (
        AutoTuner.from_kernel(kernel=kernel, configs=get_autotune_configs())
        .set_compile_args(out_idx=[-1], target="auto")
        .set_profile_args(
            supply_type=tl.TensorSupplyType.Normal,
            ref_prog=None if skip_check else lambda weighted, w1, inv: _stage2_reference(weighted, w1, inv),
            skip_check=skip_check,
            rtol=3e-2,
            atol=3e-2,
            backend=profile_backend,
        )
    )
    return autotuner.run(warmup=warmup, rep=rep)


def _stage2_reference(
    weighted_residual: torch.Tensor,
    weight1: torch.Tensor,
    inv_rms: torch.Tensor,
) -> torch.Tensor:
    paired = (weighted_residual.float() @ weight1.float()) * inv_rms.float().unsqueeze(-1)
    gate = paired[:, 0::2]
    up = paired[:, 1::2]
    return (torch.nn.functional.silu(gate) * up).to(dtype=weighted_residual.dtype)


def run_demo(
    rows: int = 1024,
    hidden: int = 1024,
    inner0: int | None = None,
    mlp: int | None = None,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    inputs = make_inputs(rows=rows, hidden=hidden, inner0=inner0, mlp=mlp, dtype=dtype)
    return gemm_residual_rmsnorm_gemm_swiglu(*inputs)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the CODA GEMM-Residual-RMSNorm-GEMM-SwiGLU TileLang kernels.")
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--inner0", type=int, default=None)
    parser.add_argument("--mlp", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    output = run_demo(rows=args.rows, hidden=args.hidden, inner0=args.inner0, mlp=args.mlp)
    print(output[:2, :8])
