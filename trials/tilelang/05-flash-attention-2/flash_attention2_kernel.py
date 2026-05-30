import argparse
import itertools
import tilelang as tl
import tilelang.language as T
import torch
from tilelang.autotuner import AutoTuner


DEFAULT_BLOCK_M = 64
DEFAULT_BLOCK_N = 64
DEFAULT_THREADS = 128
DEFAULT_NUM_STAGES = 2
DEFAULT_ENABLE_SWIZZLE = False
DEFAULT_CAUSAL = False
LOG2E = 1.4426950408889634


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


def resolve_softmax_scale(head_dim: int, softmax_scale: float | None) -> float:
    if softmax_scale is not None:
        return softmax_scale
    return head_dim ** -0.5


def validate_problem_size(seq_len: int, head_dim: int, block_M: int, block_N: int) -> None:
    if seq_len % block_M != 0:
        raise ValueError(f"seq_len={seq_len} must be divisible by block_M={block_M}")
    if seq_len % block_N != 0:
        raise ValueError(f"seq_len={seq_len} must be divisible by block_N={block_N}")
    if head_dim <= 0:
        raise ValueError("head_dim must be positive")


def flash_attention2_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = DEFAULT_CAUSAL,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    scale = resolve_softmax_scale(q.shape[-1], softmax_scale)
    q_bhsd = q.permute(0, 2, 1, 3).float()
    k_bhsd = k.permute(0, 2, 1, 3).float()
    v_bhsd = v.permute(0, 2, 1, 3).float()

    scores = torch.matmul(q_bhsd, k_bhsd.transpose(-1, -2)) * scale
    if causal:
        seq_len = q.shape[1]
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, v_bhsd)
    return out.permute(0, 2, 1, 3).to(dtype=q.dtype)


def make_flash_attention2_prim_func(
    batch: int,
    heads: int,
    seq_len: int,
    head_dim: int,
    causal: bool,
    softmax_scale: float,
    block_M: int,
    block_N: int,
    threads: int,
    num_stages: int,
    enable_swizzle: bool,
    dtype=T.float16,
    accum_dtype=T.float32,
):
    softmax_scale_log2e = softmax_scale * LOG2E

    @T.prim_func
    def main(
        q: T.Tensor((batch, seq_len, heads, head_dim), dtype),
        k: T.Tensor((batch, seq_len, heads, head_dim), dtype),
        v: T.Tensor((batch, seq_len, heads, head_dim), dtype),
        out: T.Tensor((batch, seq_len, heads, head_dim), dtype),
    ):
        with T.Kernel(T.ceildiv(seq_len, block_M), heads, batch, threads=threads) as (
            bx,
            by,
            bz,
        ):
            q_shared = T.alloc_shared((block_M, head_dim), dtype)
            k_shared = T.alloc_shared((block_N, head_dim), dtype)
            v_shared = T.alloc_shared((block_N, head_dim), dtype)
            acc_s = T.alloc_fragment((block_M, block_N), accum_dtype)
            acc_s_cast = T.alloc_fragment((block_M, block_N), dtype)
            acc_o = T.alloc_fragment((block_M, head_dim), accum_dtype)
            scores_max = T.alloc_fragment((block_M,), accum_dtype)
            scores_max_prev = T.alloc_fragment((block_M,), accum_dtype)
            scores_scale = T.alloc_fragment((block_M,), accum_dtype)
            scores_sum = T.alloc_fragment((block_M,), accum_dtype)
            logsum = T.alloc_fragment((block_M,), accum_dtype)

            T.use_swizzle(panel_size=10, enable=enable_swizzle)
            T.copy(q[bz, bx * block_M : (bx + 1) * block_M, by, :], q_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            loop_range = T.ceildiv(seq_len, block_N)

            for kv_tile in T.Pipelined(loop_range, num_stages=num_stages):
                T.copy(k[bz, kv_tile * block_N : (kv_tile + 1) * block_N, by, :], k_shared)

                if causal:
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.if_then_else(
                            bx * block_M + i >= kv_tile * block_N + j,
                            T.cast(0.0, accum_dtype),
                            -T.infinity(accum_dtype),
                        )
                else:
                    T.clear(acc_s)

                T.gemm(
                    q_shared,
                    k_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                T.copy(v[bz, kv_tile * block_N : (kv_tile + 1) * block_N, by, :], v_shared)

                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = acc_s[i, j] * T.cast(softmax_scale_log2e, accum_dtype)

                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)

                for i in T.Parallel(block_M):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                    scores_scale[i] = T.exp2(scores_max_prev[i] - scores_max[i])

                for i, j in T.Parallel(block_M, head_dim):
                    acc_o[i, j] = acc_o[i, j] * scores_scale[i]

                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.exp2(acc_s[i, j] - scores_max[i])

                T.copy(acc_s, acc_s_cast)
                T.gemm(acc_s_cast, v_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
                T.reduce_sum(acc_s, scores_sum, dim=1, clear=True)

                for i in T.Parallel(block_M):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]

            for i, j in T.Parallel(block_M, head_dim):
                acc_o[i, j] = acc_o[i, j] / logsum[i]

            T.copy(acc_o, out[bz, bx * block_M : (bx + 1) * block_M, by, :])

    return main


def get_autotune_configs(seq_len: int) -> list[dict[str, int | bool]]:
    block_M_values = [64, 128]
    block_N_values = [32, 64, 128]
    iter_params = dict(
        block_M=[value for value in block_M_values if seq_len % value == 0],
        block_N=[value for value in block_N_values if seq_len % value == 0],
        num_stages=[1, 2, 3],
        threads=[128, 256],
        enable_swizzle=[False, True],
    )
    return [
        dict(zip(iter_params.keys(), values))
        for values in itertools.product(*iter_params.values())
    ]


def autotune_flash_attention2(
    batch: int,
    heads: int,
    seq_len: int,
    head_dim: int,
    causal: bool = DEFAULT_CAUSAL,
    softmax_scale: float | None = None,
    profile_backend: str = "event",
    warmup: int = 3,
    rep: int = 20,
):
    scale = resolve_softmax_scale(head_dim, softmax_scale)

    def kernel(
        block_M=None,
        block_N=None,
        num_stages=None,
        threads=None,
        enable_swizzle=None,
    ):
        validate_problem_size(
            seq_len=seq_len,
            head_dim=head_dim,
            block_M=block_M,
            block_N=block_N,
        )
        return make_flash_attention2_prim_func(
            batch=batch,
            heads=heads,
            seq_len=seq_len,
            head_dim=head_dim,
            causal=causal,
            softmax_scale=scale,
            block_M=block_M,
            block_N=block_N,
            threads=threads,
            num_stages=num_stages,
            enable_swizzle=enable_swizzle,
        )

    autotuner = (
        AutoTuner.from_kernel(kernel=kernel, configs=get_autotune_configs(seq_len=seq_len))
        .set_compile_args(
            out_idx=[-1],
            target="auto",
        )
        .set_profile_args(
            supply_type=tl.TensorSupplyType.Normal,
            ref_prog=lambda q, k, v: flash_attention2_reference(
                q,
                k,
                v,
                causal=causal,
                softmax_scale=scale,
            ),
            skip_check=False,
            rtol=3e-2,
            atol=3e-2,
            backend=profile_backend,
        )
    )
    return autotuner.run(warmup=warmup, rep=rep)


@tl.jit(out_idx=[-1])
def flash_attention2_with_config(
    batch: int,
    heads: int,
    seq_len: int,
    head_dim: int,
    causal: bool = DEFAULT_CAUSAL,
    softmax_scale: float = 1.0,
    block_M: int = DEFAULT_BLOCK_M,
    block_N: int = DEFAULT_BLOCK_N,
    threads: int = DEFAULT_THREADS,
    num_stages: int = DEFAULT_NUM_STAGES,
    enable_swizzle: bool = DEFAULT_ENABLE_SWIZZLE,
    dtype=T.float16,
    accum_dtype=T.float32,
):
    validate_problem_size(
        seq_len=seq_len,
        head_dim=head_dim,
        block_M=block_M,
        block_N=block_N,
    )
    return make_flash_attention2_prim_func(
        batch=batch,
        heads=heads,
        seq_len=seq_len,
        head_dim=head_dim,
        causal=causal,
        softmax_scale=softmax_scale,
        block_M=block_M,
        block_N=block_N,
        threads=threads,
        num_stages=num_stages,
        enable_swizzle=enable_swizzle,
        dtype=dtype,
        accum_dtype=accum_dtype,
    )


def flash_attention2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = DEFAULT_CAUSAL,
    softmax_scale: float | None = None,
    block_M: int = DEFAULT_BLOCK_M,
    block_N: int = DEFAULT_BLOCK_N,
    threads: int = DEFAULT_THREADS,
    num_stages: int = DEFAULT_NUM_STAGES,
    enable_swizzle: bool = DEFAULT_ENABLE_SWIZZLE,
) -> torch.Tensor:
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, and v must all have the same shape")

    batch, seq_len, heads, head_dim = q.shape
    scale = resolve_softmax_scale(head_dim, softmax_scale)
    kernel = flash_attention2_with_config(
        batch,
        heads,
        seq_len,
        head_dim,
        causal=causal,
        softmax_scale=scale,
        block_M=block_M,
        block_N=block_N,
        threads=threads,
        num_stages=num_stages,
        enable_swizzle=enable_swizzle,
        dtype=torch_dtype_to_tilelang_dtype(q.dtype),
    )
    return kernel(q, k, v)


def make_inputs(
    batch: int,
    heads: int,
    seq_len: int,
    head_dim: int,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    q = torch.randn(
        batch,
        seq_len,
        heads,
        head_dim,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    k = torch.randn(
        batch,
        seq_len,
        heads,
        head_dim,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    v = torch.randn(
        batch,
        seq_len,
        heads,
        head_dim,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    return q, k, v


def run_demo(
    batch: int = 4,
    heads: int = 8,
    seq_len: int = 512,
    head_dim: int = 64,
    causal: bool = DEFAULT_CAUSAL,
    block_M: int = DEFAULT_BLOCK_M,
    block_N: int = DEFAULT_BLOCK_N,
    threads: int = DEFAULT_THREADS,
    num_stages: int = DEFAULT_NUM_STAGES,
    enable_swizzle: bool = DEFAULT_ENABLE_SWIZZLE,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    q, k, v = make_inputs(
        batch=batch,
        heads=heads,
        seq_len=seq_len,
        head_dim=head_dim,
        dtype=dtype,
    )
    return flash_attention2(
        q,
        k,
        v,
        causal=causal,
        block_M=block_M,
        block_N=block_N,
        threads=threads,
        num_stages=num_stages,
        enable_swizzle=enable_swizzle,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the TileLang FlashAttention-2 forward kernel.")
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
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    output = run_demo(
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
    )
    print(output[0, :2, :2, :8])
