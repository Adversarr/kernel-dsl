import argparse
import statistics
import time

import torch
import torch.nn.functional as F

import tilelang
from tilelang.autotuner import autotune
import tilelang.language as T


def torch_naive_sliding_window_attention(q, k, v, window_size):
    batch, seq_len, heads, dim = q.shape
    out = torch.empty_like(q)
    scale = dim**-0.5

    for b in range(batch):
        for h in range(heads):
            for qi in range(seq_len):
                start = max(0, qi - window_size + 1)
                acc_s = torch.matmul(q[b, qi, h].float(), k[b, start : qi + 1, h].float().T) * scale
                probs = F.softmax(acc_s, dim=-1).to(v.dtype)
                out[b, qi, h] = torch.matmul(probs, v[b, start : qi + 1, h])
    return out


def get_interior_autotune_configs():
    return [
        {"block_M": 64, "block_N": 16, "num_stages": 1, "threads": 128},
        {"block_M": 64, "block_N": 16, "num_stages": 2, "threads": 128},
        {"block_M": 64, "block_N": 32, "num_stages": 1, "threads": 128},
        {"block_M": 64, "block_N": 64, "num_stages": 1, "threads": 128},
        {"block_M": 128, "block_N": 64, "num_stages": 1, "threads": 128},
    ]


def get_fast_autotune_configs():
    return [
        {"block_M": 64, "block_N": 32, "num_stages": 1, "threads": 128, "min_blocks_per_sm": 1, "swizzle_panel": 8},
        {"block_M": 64, "block_N": 32, "num_stages": 1, "threads": 128, "min_blocks_per_sm": 3, "swizzle_panel": 4},
        {"block_M": 64, "block_N": 32, "num_stages": 2, "threads": 128, "min_blocks_per_sm": 1, "swizzle_panel": 8},
        {"block_M": 64, "block_N": 32, "num_stages": 2, "threads": 128, "min_blocks_per_sm": 1, "swizzle_panel": 16},
        {"block_M": 64, "block_N": 64, "num_stages": 1, "threads": 128, "min_blocks_per_sm": 1, "swizzle_panel": 8},
        {"block_M": 64, "block_N": 16, "num_stages": 2, "threads": 128, "min_blocks_per_sm": 4, "swizzle_panel": 8},
    ]


@tilelang.jit(
    out_idx=[3],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def sliding_window_attention_kernel(
    batch,
    heads,
    seq_len,
    dim,
    window_size,
    block_M=64,
    block_N=64,
    num_stages=2,
    threads=128,
):
    dtype = T.float16
    accum_dtype = T.float32
    scale = (1.0 / dim) ** 0.5 * 1.4426950408889634

    @T.prim_func
    def main(
        Q: T.Tensor((batch, seq_len, heads, dim), dtype),
        K: T.Tensor((batch, seq_len, heads, dim), dtype),
        V: T.Tensor((batch, seq_len, heads, dim), dtype),
        Output: T.Tensor((batch, seq_len, heads, dim), dtype),
    ):
        with T.Kernel(T.ceildiv(seq_len, block_M), heads, batch, threads=threads) as (bx, by, bz):
            Q_shared = T.alloc_shared((block_M, dim), dtype)
            K_shared = T.alloc_shared((block_N, dim), dtype)
            V_shared = T.alloc_shared((block_N, dim), dtype)
            O_shared = T.alloc_shared((block_M, dim), dtype)

            acc_s = T.alloc_fragment((block_M, block_N), accum_dtype)
            acc_s_cast = T.alloc_fragment((block_M, block_N), dtype)
            acc_o = T.alloc_fragment((block_M, dim), accum_dtype)
            scores_max = T.alloc_fragment((block_M,), accum_dtype)
            scores_max_prev = T.alloc_fragment((block_M,), accum_dtype)
            scores_scale = T.alloc_fragment((block_M,), accum_dtype)
            scores_sum = T.alloc_fragment((block_M,), accum_dtype)
            logsum = T.alloc_fragment((block_M,), accum_dtype)

            T.copy(Q[bz, bx * block_M : (bx + 1) * block_M, by, :], Q_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            start_block = T.max(0, (bx * block_M - window_size + 1) // block_N)
            end_block = T.min(T.ceildiv(seq_len, block_N), T.ceildiv((bx + 1) * block_M, block_N))

            for k_block in T.serial(end_block - start_block):
                kv_block = start_block + k_block
                T.copy(K[bz, kv_block * block_N : (kv_block + 1) * block_N, by, :], K_shared)

                for i, j in T.Parallel(block_M, block_N):
                    q_idx = bx * block_M + i
                    k_idx = kv_block * block_N + j
                    acc_s[i, j] = T.if_then_else(
                        q_idx < seq_len and k_idx <= q_idx and k_idx + window_size > q_idx,
                        0,
                        -T.infinity(accum_dtype),
                    )

                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)

                for i in T.Parallel(block_M):
                    q_idx = bx * block_M + i
                    row_has_values = q_idx < seq_len and kv_block * block_N <= q_idx and (
                        kv_block * block_N + block_N - 1 + window_size > q_idx
                    )
                    scores_max[i] = T.if_then_else(
                        row_has_values,
                        T.max(scores_max[i], scores_max_prev[i]),
                        T.if_then_else(logsum[i] > 0, scores_max_prev[i], 0),
                    )
                    scores_scale[i] = T.if_then_else(
                        row_has_values and logsum[i] > 0,
                        T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale),
                        1,
                    )

                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                    q_idx = bx * block_M + i
                    k_idx = kv_block * block_N + j
                    acc_s[i, j] = T.if_then_else(
                        q_idx < seq_len and k_idx <= q_idx and k_idx + window_size > q_idx,
                        acc_s[i, j],
                        0,
                    )

                T.reduce_sum(acc_s, scores_sum, dim=1)

                for i in T.Parallel(block_M):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]

                T.copy(acc_s, acc_s_cast)

                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] *= scores_scale[i]

                T.copy(V[bz, kv_block * block_N : (kv_block + 1) * block_N, by, :], V_shared)
                T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] /= logsum[i]

            T.copy(acc_o, O_shared)
            T.copy(O_shared, Output[bz, bx * block_M : (bx + 1) * block_M, by, :])

    return main


@autotune(configs=get_interior_autotune_configs(), warmup=25, rep=100)
@tilelang.jit(
    out_idx=[3],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def sliding_window_attention_interior_kernel(
    batch,
    heads,
    seq_len,
    dim,
    window_size,
    block_M=64,
    block_N=64,
    num_stages=2,
    threads=128,
):
    dtype = T.float16
    accum_dtype = T.float32
    scale = (1.0 / dim) ** 0.5 * 1.4426950408889634
    window_blocks = T.ceildiv(window_size + block_M - 1, block_N)

    @T.prim_func
    def main(
        Q: T.Tensor((batch, seq_len, heads, dim), dtype),
        K: T.Tensor((batch, seq_len, heads, dim), dtype),
        V: T.Tensor((batch, seq_len, heads, dim), dtype),
        Output: T.Tensor((batch, seq_len, heads, dim), dtype),
    ):
        with T.Kernel(T.ceildiv(seq_len, block_M), heads, batch, threads=threads) as (bx, by, bz):
            Q_shared = T.alloc_shared((block_M, dim), dtype)
            K_shared = T.alloc_shared((block_N, dim), dtype)
            V_shared = T.alloc_shared((block_N, dim), dtype)
            O_shared = T.alloc_shared((block_M, dim), dtype)

            acc_s = T.alloc_fragment((block_M, block_N), accum_dtype)
            acc_s_cast = T.alloc_fragment((block_M, block_N), dtype)
            acc_o = T.alloc_fragment((block_M, dim), accum_dtype)
            scores_max = T.alloc_fragment((block_M,), accum_dtype)
            scores_max_prev = T.alloc_fragment((block_M,), accum_dtype)
            scores_scale = T.alloc_fragment((block_M,), accum_dtype)
            scores_sum = T.alloc_fragment((block_M,), accum_dtype)
            logsum = T.alloc_fragment((block_M,), accum_dtype)

            T.copy(Q[bz, bx * block_M : (bx + 1) * block_M, by, :], Q_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            end_block = T.ceildiv((bx + 1) * block_M, block_N)

            if end_block >= window_blocks:
                for rel_block in T.Pipelined(window_blocks, num_stages=num_stages):
                    kv_block = end_block - window_blocks + rel_block
                    T.copy(K[bz, kv_block * block_N : (kv_block + 1) * block_N, by, :], K_shared)

                    for i, j in T.Parallel(block_M, block_N):
                        q_idx = bx * block_M + i
                        k_idx = kv_block * block_N + j
                        acc_s[i, j] = T.if_then_else(
                            k_idx <= q_idx and k_idx + window_size > q_idx,
                            0,
                            -T.infinity(accum_dtype),
                        )

                    T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                    T.copy(scores_max, scores_max_prev)
                    T.fill(scores_max, -T.infinity(accum_dtype))
                    T.reduce_max(acc_s, scores_max, dim=1, clear=False)

                    for i in T.Parallel(block_M):
                        scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                        scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)

                    for i, j in T.Parallel(block_M, block_N):
                        q_idx = bx * block_M + i
                        k_idx = kv_block * block_N + j
                        acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                        acc_s[i, j] = T.if_then_else(
                            k_idx <= q_idx and k_idx + window_size > q_idx,
                            acc_s[i, j],
                            0,
                        )

                    T.reduce_sum(acc_s, scores_sum, dim=1)

                    for i in T.Parallel(block_M):
                        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]

                    T.copy(acc_s, acc_s_cast)

                    for i, j in T.Parallel(block_M, dim):
                        acc_o[i, j] *= scores_scale[i]

                    T.copy(V[bz, kv_block * block_N : (kv_block + 1) * block_N, by, :], V_shared)
                    T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
            else:
                for kv_block in T.serial(end_block):
                    T.copy(K[bz, kv_block * block_N : (kv_block + 1) * block_N, by, :], K_shared)

                    for i, j in T.Parallel(block_M, block_N):
                        q_idx = bx * block_M + i
                        k_idx = kv_block * block_N + j
                        acc_s[i, j] = T.if_then_else(
                            k_idx <= q_idx,
                            0,
                            -T.infinity(accum_dtype),
                        )

                    T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                    T.copy(scores_max, scores_max_prev)
                    T.fill(scores_max, -T.infinity(accum_dtype))
                    T.reduce_max(acc_s, scores_max, dim=1, clear=False)

                    for i in T.Parallel(block_M):
                        scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                        scores_scale[i] = T.if_then_else(
                            logsum[i] > 0,
                            T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale),
                            1,
                        )

                    for i, j in T.Parallel(block_M, block_N):
                        q_idx = bx * block_M + i
                        k_idx = kv_block * block_N + j
                        acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                        acc_s[i, j] = T.if_then_else(k_idx <= q_idx, acc_s[i, j], 0)

                    T.reduce_sum(acc_s, scores_sum, dim=1)

                    for i in T.Parallel(block_M):
                        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]

                    T.copy(acc_s, acc_s_cast)

                    for i, j in T.Parallel(block_M, dim):
                        acc_o[i, j] *= scores_scale[i]

                    T.copy(V[bz, kv_block * block_N : (kv_block + 1) * block_N, by, :], V_shared)
                    T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] /= logsum[i]

            T.copy(acc_o, O_shared)
            T.copy(O_shared, Output[bz, bx * block_M : (bx + 1) * block_M, by, :])

    return main


@autotune(configs=get_fast_autotune_configs(), warmup=25, rep=100)
@tilelang.jit(
    out_idx=[3],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def sliding_window_attention_fast_kernel(
    batch,
    heads,
    seq_len,
    dim,
    window_size,
    block_M=64,
    block_N=64,
    num_stages=1,
    threads=128,
    min_blocks_per_sm=1,
    swizzle_panel=0,
):
    assert window_size % block_N == 0
    dtype = T.float16
    accum_dtype = T.float32
    scale = (1.0 / dim) ** 0.5 * 1.4426950408889634

    @T.prim_func
    def main(
        Q: T.Tensor((batch, seq_len, heads, dim), dtype),
        K: T.Tensor((batch, seq_len, heads, dim), dtype),
        V: T.Tensor((batch, seq_len, heads, dim), dtype),
        Output: T.Tensor((batch, seq_len, heads, dim), dtype),
    ):
        with T.Kernel(T.ceildiv(seq_len, block_M), heads, batch, threads=threads) as (bx, by, bz):
            if swizzle_panel > 0:
                T.use_swizzle(panel_size=swizzle_panel)
            T.annotate_min_blocks_per_sm(min_blocks_per_sm)
            Q_shared = T.alloc_shared((block_M, dim), dtype)
            K_shared = T.alloc_shared((block_N, dim), dtype)
            V_shared = T.alloc_shared((block_N, dim), dtype)
            O_shared = T.alloc_shared((block_M, dim), dtype)

            acc_s = T.alloc_fragment((block_M, block_N), accum_dtype)
            acc_s_cast = T.alloc_fragment((block_M, block_N), dtype)
            acc_o = T.alloc_fragment((block_M, dim), accum_dtype)
            scores_max = T.alloc_fragment((block_M,), accum_dtype)
            scores_max_prev = T.alloc_fragment((block_M,), accum_dtype)
            scores_scale = T.alloc_fragment((block_M,), accum_dtype)
            scores_sum = T.alloc_fragment((block_M,), accum_dtype)
            logsum = T.alloc_fragment((block_M,), accum_dtype)

            T.copy(Q[bz, bx * block_M : (bx + 1) * block_M, by, :], Q_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            start = T.max(0, (bx * block_M - window_size) // block_N)
            end = T.min(T.ceildiv(seq_len, block_N), T.ceildiv((bx + 1) * block_M, block_N))
            loop_range = end - start

            for k in T.Pipelined(loop_range, num_stages=num_stages):
                actual_k = k + start
                T.copy(K[bz, actual_k * block_N : (actual_k + 1) * block_N, by, :], K_shared)

                for i, j in T.Parallel(block_M, block_N):
                    q_idx = bx * block_M + i
                    k_idx = actual_k * block_N + j
                    acc_s[i, j] = T.if_then_else(
                        q_idx >= k_idx and q_idx < k_idx + window_size,
                        0,
                        -T.infinity(accum_dtype),
                    )

                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)

                for i in T.Parallel(block_M):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                    scores_max[i] = T.if_then_else(scores_max[i] == -T.infinity(accum_dtype), 0, scores_max[i])

                for i in T.Parallel(block_M):
                    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)

                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)

                T.reduce_sum(acc_s, scores_sum, dim=1)

                for i in T.Parallel(block_M):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]

                T.copy(acc_s, acc_s_cast)

                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] *= scores_scale[i]

                T.copy(V[bz, actual_k * block_N : (actual_k + 1) * block_N, by, :], V_shared)
                T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] /= logsum[i]

            T.copy(acc_o, O_shared)
            T.copy(O_shared, Output[bz, bx * block_M : (bx + 1) * block_M, by, :])

    return main


def build_kernel(batch, heads, seq_len, dim, window_size, block_m, block_n, num_stages, threads):
    return sliding_window_attention_kernel.compile(
        batch,
        heads,
        seq_len,
        dim,
        window_size,
        block_M=block_m,
        block_N=block_n,
        num_stages=num_stages,
        threads=threads,
    )


def build_interior_kernel(batch, heads, seq_len, dim, window_size, block_m, block_n, num_stages, threads):
    return sliding_window_attention_interior_kernel.compile(
        batch,
        heads,
        seq_len,
        dim,
        window_size,
        block_M=block_m,
        block_N=block_n,
        num_stages=num_stages,
        threads=threads,
    )


def build_autotuned_interior_kernel(batch, heads, seq_len, dim, window_size):
    return sliding_window_attention_interior_kernel(
        batch,
        heads,
        seq_len,
        dim,
        window_size,
    )


def build_fast_kernel(
    batch,
    heads,
    seq_len,
    dim,
    window_size,
    block_m,
    block_n,
    num_stages,
    threads,
    min_blocks_per_sm=1,
    swizzle_panel=0,
):
    return sliding_window_attention_fast_kernel.compile(
        batch,
        heads,
        seq_len,
        dim,
        window_size,
        block_M=block_m,
        block_N=block_n,
        num_stages=num_stages,
        threads=threads,
        min_blocks_per_sm=min_blocks_per_sm,
        swizzle_panel=swizzle_panel,
    )


def build_autotuned_fast_kernel(batch, heads, seq_len, dim, window_size):
    return sliding_window_attention_fast_kernel(
        batch,
        heads,
        seq_len,
        dim,
        window_size,
    )


def bench_cuda(fn, warmup=10, rep=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(rep):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def bench_torch_naive(fn, warmup=1, rep=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(rep):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(times)


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.block_m != 64 or args.block_n != 64:
        raise ValueError("This trial currently supports --block-m 64 --block-n 64")

    torch.manual_seed(args.seed)
    q = torch.randn(args.batch, args.seq_len, args.heads, args.dim, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    out = torch.empty_like(q)

    kernel = build_kernel(
        args.batch,
        args.heads,
        args.seq_len,
        args.dim,
        args.window_size,
        args.block_m,
        args.block_n,
        args.num_stages,
        args.threads,
    )

    actual = kernel(q, k, v)
    expected = torch_naive_sliding_window_attention(q, k, v, args.window_size)
    torch.testing.assert_close(actual, expected, rtol=args.rtol, atol=args.atol)

    tilelang_ms = bench_cuda(lambda: kernel(q, k, v), warmup=args.warmup, rep=args.rep)
    torch_ms = bench_torch_naive(
        lambda: torch_naive_sliding_window_attention(q, k, v, args.window_size),
        warmup=max(1, args.warmup // 10),
        rep=max(3, args.rep // 10),
    )
    speedup = torch_ms / tilelang_ms

    work = 4.0 * args.batch * args.heads * args.seq_len * min(args.window_size, args.seq_len) * args.dim
    print(f"shape: B={args.batch} S={args.seq_len} H={args.heads} D={args.dim} W={args.window_size}")
    print(f"tilelang: {tilelang_ms:.4f} ms, {work / tilelang_ms / 1e9:.2f} TFLOP/s")
    print(f"torch_naive: {torch_ms:.4f} ms")
    print(f"speedup: {speedup:.2f}x")
    if speedup < args.min_speedup:
        raise RuntimeError(f"speedup {speedup:.2f}x is below target {args.min_speedup:.2f}x")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--block-m", type=int, default=64)
    parser.add_argument("--block-n", type=int, default=64)
    parser.add_argument("--num-stages", type=int, default=2)
    parser.add_argument("--threads", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--min-speedup", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
