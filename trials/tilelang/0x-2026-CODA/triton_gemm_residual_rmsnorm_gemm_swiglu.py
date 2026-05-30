import argparse

import torch
import triton
import triton.language as tl

from gemm_residual_rmsnorm_gemm_swiglu_kernel import (
    DEFAULT_EPS,
    make_inputs,
    naive_torch_compile_reference,
)


@triton.jit
def _stage1_kernel(
    x,
    weight0,
    residual,
    gamma,
    weighted,
    partial_squares,
    ROWS: tl.constexpr,
    HIDDEN: tl.constexpr,
    INNER0: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k0 in range(0, INNER0, BLOCK_K):
        k = k0 + offs_k
        a = tl.load(
            x + offs_m[:, None] * INNER0 + k[None, :],
            mask=(offs_m[:, None] < ROWS) & (k[None, :] < INNER0),
            other=0.0,
        )
        b = tl.load(
            weight0 + k[:, None] * HIDDEN + offs_n[None, :],
            mask=(k[:, None] < INNER0) & (offs_n[None, :] < HIDDEN),
            other=0.0,
        )
        acc += tl.dot(a, b)

    res = tl.load(
        residual + offs_m[:, None] * HIDDEN + offs_n[None, :],
        mask=(offs_m[:, None] < ROWS) & (offs_n[None, :] < HIDDEN),
        other=0.0,
    ).to(tl.float32)
    val = acc + res
    g = tl.load(gamma + offs_n, mask=offs_n < HIDDEN, other=0.0).to(tl.float32)
    out = val * g[None, :]
    mask = (offs_m[:, None] < ROWS) & (offs_n[None, :] < HIDDEN)
    tl.store(weighted + offs_m[:, None] * HIDDEN + offs_n[None, :], out, mask=mask)

    squares = tl.where(mask, val * val, 0.0)
    row_sum = tl.sum(squares, axis=1) / HIDDEN
    tl.store(partial_squares + pid_n * ROWS + offs_m, row_sum, mask=offs_m < ROWS)


@triton.jit
def _reduce_kernel(
    partial_squares,
    inv_rms,
    ROWS: tl.constexpr,
    PARTIAL_COLS: tl.constexpr,
    BLOCK_PARTIALS: tl.constexpr,
    EPS: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_PARTIALS)
    vals = tl.load(
        partial_squares + offs * ROWS + row,
        mask=offs < PARTIAL_COLS,
        other=0.0,
    )
    total = tl.sum(vals, axis=0)
    tl.store(inv_rms + row, tl.rsqrt(total + EPS))


@triton.jit
def _stage2_kernel(
    weighted,
    weight1,
    inv_rms,
    out,
    ROWS: tl.constexpr,
    HIDDEN: tl.constexpr,
    MLP: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_O: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_o = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_o = pid_o * BLOCK_O + tl.arange(0, BLOCK_O)
    offs_k = tl.arange(0, BLOCK_K)

    gate_acc = tl.zeros((BLOCK_M, BLOCK_O), tl.float32)
    up_acc = tl.zeros((BLOCK_M, BLOCK_O), tl.float32)
    paired_cols = MLP * 2
    gate_cols = offs_o * 2
    up_cols = offs_o * 2 + 1

    for k0 in range(0, HIDDEN, BLOCK_K):
        k = k0 + offs_k
        a = tl.load(
            weighted + offs_m[:, None] * HIDDEN + k[None, :],
            mask=(offs_m[:, None] < ROWS) & (k[None, :] < HIDDEN),
            other=0.0,
        )
        gate_w = tl.load(
            weight1 + k[:, None] * paired_cols + gate_cols[None, :],
            mask=(k[:, None] < HIDDEN) & (offs_o[None, :] < MLP),
            other=0.0,
        )
        up_w = tl.load(
            weight1 + k[:, None] * paired_cols + up_cols[None, :],
            mask=(k[:, None] < HIDDEN) & (offs_o[None, :] < MLP),
            other=0.0,
        )
        gate_acc += tl.dot(a, gate_w)
        up_acc += tl.dot(a, up_w)

    scale = tl.load(inv_rms + offs_m, mask=offs_m < ROWS, other=0.0).to(tl.float32)
    gate = gate_acc * scale[:, None]
    up = up_acc * scale[:, None]
    silu = gate / (1.0 + tl.exp2(-gate * 1.4426950408889634))
    result = silu * up
    mask = (offs_m[:, None] < ROWS) & (offs_o[None, :] < MLP)
    tl.store(out + offs_m[:, None] * MLP + offs_o[None, :], result, mask=mask)


@triton.jit
def _stage2_interleaved_acc_kernel(
    weighted,
    weight1,
    inv_rms,
    out,
    ROWS: tl.constexpr,
    HIDDEN: tl.constexpr,
    MLP: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_O: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_o = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_o = pid_o * BLOCK_O + tl.arange(0, BLOCK_O)
    offs_p = pid_o * (BLOCK_O * 2) + tl.arange(0, BLOCK_O * 2)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_O * 2), tl.float32)
    paired_cols = MLP * 2

    for k0 in range(0, HIDDEN, BLOCK_K):
        k = k0 + offs_k
        a = tl.load(
            weighted + offs_m[:, None] * HIDDEN + k[None, :],
            mask=(offs_m[:, None] < ROWS) & (k[None, :] < HIDDEN),
            other=0.0,
        )
        b = tl.load(
            weight1 + k[:, None] * paired_cols + offs_p[None, :],
            mask=(k[:, None] < HIDDEN) & (offs_p[None, :] < paired_cols),
            other=0.0,
        )
        acc += tl.dot(a, b)

    acc_pairs = tl.reshape(acc, (BLOCK_M, BLOCK_O, 2))
    pair_lane = tl.arange(0, 2)
    gate_acc = tl.sum(tl.where(pair_lane[None, None, :] == 0, acc_pairs, 0.0), axis=2)
    up_acc = tl.sum(tl.where(pair_lane[None, None, :] == 1, acc_pairs, 0.0), axis=2)

    scale = tl.load(inv_rms + offs_m, mask=offs_m < ROWS, other=0.0).to(tl.float32)
    gate = gate_acc * scale[:, None]
    up = up_acc * scale[:, None]
    silu = gate / (1.0 + tl.exp2(-gate * 1.4426950408889634))
    result = silu * up
    mask = (offs_m[:, None] < ROWS) & (offs_o[None, :] < MLP)
    tl.store(out + offs_m[:, None] * MLP + offs_o[None, :], result, mask=mask)


def triton_gemm_residual_rmsnorm_gemm_swiglu(
    x: torch.Tensor,
    weight0: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    weight1: torch.Tensor,
    *,
    block_m: int = 128,
    block_n: int = 128,
    block_o: int = 64,
    block_k: int = 32,
    num_warps: int = 4,
    num_stages: int = 3,
    interleaved_acc: bool = False,
) -> torch.Tensor:
    rows, inner0 = x.shape
    hidden = residual.shape[1]
    mlp = weight1.shape[1] // 2
    partial_cols = triton.cdiv(hidden, block_n)
    weighted = torch.empty((rows, hidden), device=x.device, dtype=x.dtype)
    partial_squares = torch.empty((partial_cols, rows), device=x.device, dtype=torch.float32)
    inv_rms = torch.empty((rows,), device=x.device, dtype=torch.float32)
    out = torch.empty((rows, mlp), device=x.device, dtype=x.dtype)

    _stage1_kernel[(triton.cdiv(rows, block_m), partial_cols)](
        x,
        weight0,
        residual,
        gamma,
        weighted,
        partial_squares,
        rows,
        hidden,
        inner0,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    block_partials = triton.next_power_of_2(partial_cols)
    _reduce_kernel[(rows,)](
        partial_squares,
        inv_rms,
        rows,
        partial_cols,
        BLOCK_PARTIALS=block_partials,
        EPS=DEFAULT_EPS,
        num_warps=1,
    )
    stage2_kernel = _stage2_interleaved_acc_kernel if interleaved_acc else _stage2_kernel
    stage2_kernel[(triton.cdiv(rows, block_m), triton.cdiv(mlp, block_o))](
        weighted,
        weight1,
        inv_rms,
        out,
        rows,
        hidden,
        mlp,
        BLOCK_M=block_m,
        BLOCK_O=block_o,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile Triton CODA GEMM-Residual-RMSNorm-GEMM-SwiGLU.")
    parser.add_argument("--rows", type=int, default=16 * 1024)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--inner0", type=int, default=None)
    parser.add_argument("--mlp", type=int, default=None)
    parser.add_argument("--block-m", type=int, default=128)
    parser.add_argument("--block-n", type=int, default=128)
    parser.add_argument("--block-o", type=int, default=64)
    parser.add_argument("--block-k", type=int, default=32)
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--num-stages", type=int, default=3)
    parser.add_argument("--interleaved-acc", action="store_true")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    inner0 = args.hidden if args.inner0 is None else args.inner0
    mlp = args.hidden * 4 if args.mlp is None else args.mlp
    inputs = make_inputs(
        rows=args.rows,
        hidden=args.hidden,
        inner0=inner0,
        mlp=mlp,
        dtype=torch.float16,
    )

    def triton_fn(x, weight0, residual, gamma, weight1):
        return triton_gemm_residual_rmsnorm_gemm_swiglu(
            x,
            weight0,
            residual,
            gamma,
            weight1,
            block_m=args.block_m,
            block_n=args.block_n,
            block_o=args.block_o,
            block_k=args.block_k,
            num_warps=args.num_warps,
            num_stages=args.num_stages,
            interleaved_acc=args.interleaved_acc,
        )

    triton_out = triton_fn(*inputs)
    compiled_reference = torch.compile(naive_torch_compile_reference, fullgraph=True)
    torch_out = compiled_reference(*inputs)
    torch.testing.assert_close(triton_out, torch_out, rtol=5e-2, atol=5e-2)

    triton_ms = benchmark_callable(triton_fn, *inputs, warmup=args.warmup, iters=args.iters)
    torch_ms = benchmark_callable(compiled_reference, *inputs, warmup=args.warmup, iters=args.iters)
    flops = 2 * args.rows * inner0 * args.hidden + 2 * args.rows * args.hidden * (mlp * 2)

    print(f"rows={args.rows}")
    print(f"hidden={args.hidden}")
    print(f"inner0={inner0}")
    print(f"mlp={mlp}")
    print(
        "config="
        f"block_m={args.block_m}, block_n={args.block_n}, block_o={args.block_o}, "
        f"block_k={args.block_k}, num_warps={args.num_warps}, num_stages={args.num_stages}, "
        f"interleaved_acc={args.interleaved_acc}"
    )
    print(f"triton_latency_ms={triton_ms:.4f}")
    print(f"triton_effective_tflops={flops / (triton_ms * 1e9):.2f}")
    print(f"torch_compile_latency_ms={torch_ms:.4f}")
    print(f"torch_compile_effective_tflops={flops / (torch_ms * 1e9):.2f}")
    print(f"triton_vs_torch_compile_speedup={torch_ms / triton_ms:.2f}x")


if __name__ == "__main__":
    main()
