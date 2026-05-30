import argparse
import itertools

import tilelang as tl
import tilelang.language as T
import torch
from tilelang.autotuner import AutoTuner, set_autotune_inputs


DEFAULT_THREADS = 32
DEFAULT_ELEMENTS_PER_THREAD = 1
DEFAULT_ENABLE_SWIZZLE = False

AUTOTUNE_THREADS = (8, 16, 32, 64, 128, 256)
AUTOTUNE_ELEMENTS_PER_THREAD = (1, 2, 4)


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


def resolve_compute_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.float32 if dtype == torch.float32 else dtype


def validate_inputs(q: torch.Tensor, k: torch.Tensor, freqs_cis: torch.Tensor) -> None:
    if q.ndim != 4 or k.ndim != 4:
        raise ValueError(f"q and k must be rank-4 tensors, got {q.ndim=} and {k.ndim=}")
    if q.shape[0] != k.shape[0] or q.shape[1] != k.shape[1] or q.shape[3] != k.shape[3]:
        raise ValueError(f"q and k must match in batch, seq, and head_dim, got {tuple(q.shape)} and {tuple(k.shape)}")
    if q.shape[3] % 2 != 0:
        raise ValueError(f"head_dim must be even for RoPE, got {q.shape[3]}")
    if q.device.type != "cuda" or k.device.type != "cuda":
        raise ValueError(f"TileLang RoPE requires CUDA tensors, got {q.device=} and {k.device=}")
    if freqs_cis.device != q.device:
        raise ValueError(f"freqs_cis must live on the same device as q/k, got {freqs_cis.device} and {q.device}")


def normalize_freqs_cis(
    freqs_cis: torch.Tensor,
    *,
    seq_len: int,
    head_dim_half: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if freqs_cis.is_complex():
        freqs_cis = freqs_cis.reshape(-1, freqs_cis.shape[-1])
        freqs_cis = torch.view_as_real(freqs_cis)
    elif freqs_cis.ndim == 2 and freqs_cis.shape[-1] == head_dim_half * 2:
        freqs_cis = freqs_cis.reshape(freqs_cis.shape[0], head_dim_half, 2)
    elif freqs_cis.ndim != 3 or freqs_cis.shape[-1] != 2:
        raise ValueError(
            "freqs_cis must be complex or real-valued with trailing dimension 2, "
            f"got shape {tuple(freqs_cis.shape)}"
        )

    freqs_cis = freqs_cis.reshape(-1, head_dim_half, 2)
    if freqs_cis.shape[0] < seq_len:
        raise ValueError(f"freqs_cis sequence length {freqs_cis.shape[0]} is smaller than seq_len={seq_len}")
    if freqs_cis.shape[1] != head_dim_half:
        raise ValueError(f"freqs_cis head_dim_half mismatch: expected {head_dim_half}, got {freqs_cis.shape[1]}")

    freqs_cis = freqs_cis[:seq_len].contiguous()
    cos = freqs_cis[..., 0].contiguous()
    sin = freqs_cis[..., 1].contiguous()
    return cos, sin


def prepare_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.dtype]:
    validate_inputs(q, k, freqs_cis)

    output_dtype = q.dtype
    compute_dtype = resolve_compute_dtype(q.dtype)
    if k.dtype != q.dtype:
        k = k.to(q.dtype)

    q_work = q.to(compute_dtype).contiguous()
    k_work = k.to(compute_dtype).contiguous()
    cos, sin = normalize_freqs_cis(
        freqs_cis=freqs_cis,
        seq_len=q.shape[1],
        head_dim_half=q.shape[3] // 2,
    )
    return q_work, k_work, cos, sin, output_dtype


def rotate_half_reference(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x_first = x[..., :half]
    x_second = x[..., half:]
    return torch.cat((-x_second, x_first), dim=-1)


def qwen_rope_reference_prepared(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    rotation_sign: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.to(dtype=q.dtype).view(1, q.shape[1], 1, q.shape[3] // 2)
    sin = (sin * rotation_sign).to(dtype=q.dtype).view(1, q.shape[1], 1, q.shape[3] // 2)

    def rotate(x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        x_first = x[..., :half]
        x_second = x[..., half:]
        first_out = x_first * cos - x_second * sin
        second_out = x_second * cos + x_first * sin
        return torch.cat((first_out, second_out), dim=-1)

    return rotate(q), rotate(k)


def qwen_rope_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cis: torch.Tensor,
    *,
    rotation_sign: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_work, k_work, cos, sin, output_dtype = prepare_inputs(q, k, freqs_cis)
    q_out, k_out = qwen_rope_reference_prepared(q_work, k_work, cos, sin, rotation_sign=rotation_sign)
    if q_out.dtype != output_dtype:
        q_out = q_out.to(output_dtype)
    if k_out.dtype != output_dtype:
        k_out = k_out.to(output_dtype)
    return q_out, k_out


def validate_kernel_config(head_dim_half: int, threads: int, elements_per_thread: int) -> None:
    tile_half = threads * elements_per_thread
    if tile_half <= 0:
        raise ValueError(f"tile_half must be positive, got {tile_half}")
    if head_dim_half % tile_half != 0:
        raise ValueError(
            f"head_dim_half={head_dim_half} must be divisible by tile_half={tile_half} "
            f"(threads={threads}, elements_per_thread={elements_per_thread})"
        )


def get_autotune_configs(head_dim_half: int) -> list[dict[str, int | bool]]:
    configs = []
    for threads, elements_per_thread, enable_swizzle in itertools.product(
        AUTOTUNE_THREADS,
        AUTOTUNE_ELEMENTS_PER_THREAD,
        (False, True),
    ):
        tile_half = threads * elements_per_thread
        if head_dim_half % tile_half != 0:
            continue
        configs.append(
            {
                "threads": threads,
                "elements_per_thread": elements_per_thread,
                "enable_swizzle": enable_swizzle,
            }
        )
    if not configs:
        raise ValueError(f"No legal autotune configs for head_dim_half={head_dim_half}")
    return configs


def select_default_config(head_dim_half: int) -> dict[str, int | bool]:
    configs = get_autotune_configs(head_dim_half)
    return max(
        configs,
        key=lambda config: (
            config["threads"] * config["elements_per_thread"],
            int(config["enable_swizzle"]),
            config["threads"],
        ),
    )


def make_qwen_rope_prim_func(
    batch_size: int,
    seq_len: int,
    n_q_heads: int,
    n_k_heads: int,
    head_dim: int,
    threads: int,
    elements_per_thread: int,
    enable_swizzle: bool,
    rotation_sign: float,
    dtype,
    freq_dtype,
    compute_dtype,
):
    head_dim_half = head_dim // 2
    tile_half = threads * elements_per_thread
    validate_kernel_config(head_dim_half=head_dim_half, threads=threads, elements_per_thread=elements_per_thread)
    max_heads = max(n_q_heads, n_k_heads)
    rotation_sign_value = float(rotation_sign)

    @T.prim_func
    def main(
        q: T.Tensor((batch_size, seq_len, n_q_heads, head_dim), dtype),
        k: T.Tensor((batch_size, seq_len, n_k_heads, head_dim), dtype),
        cos: T.Tensor((seq_len, head_dim_half), freq_dtype),
        sin: T.Tensor((seq_len, head_dim_half), freq_dtype),
        q_out: T.Tensor((batch_size, seq_len, n_q_heads, head_dim), dtype),
        k_out: T.Tensor((batch_size, seq_len, n_k_heads, head_dim), dtype),
    ):
        with T.Kernel(head_dim_half // tile_half, max_heads, batch_size * seq_len, threads=threads) as (bx, by, bz):
            batch_idx = bz // seq_len
            seq_idx = bz % seq_len
            half_start = bx * tile_half
            second_start = head_dim_half + half_start

            cos_tile = T.alloc_fragment((tile_half,), freq_dtype)
            sin_tile = T.alloc_fragment((tile_half,), freq_dtype)
            q_first_tile = T.alloc_fragment((tile_half,), dtype)
            q_second_tile = T.alloc_fragment((tile_half,), dtype)
            k_first_tile = T.alloc_fragment((tile_half,), dtype)
            k_second_tile = T.alloc_fragment((tile_half,), dtype)
            q_first_rot = T.alloc_fragment((tile_half,), dtype)
            q_second_rot = T.alloc_fragment((tile_half,), dtype)
            k_first_rot = T.alloc_fragment((tile_half,), dtype)
            k_second_rot = T.alloc_fragment((tile_half,), dtype)

            T.use_swizzle(panel_size=10, enable=enable_swizzle)
            T.copy(cos[seq_idx, half_start : half_start + tile_half], cos_tile)
            T.copy(sin[seq_idx, half_start : half_start + tile_half], sin_tile)

            if by < n_q_heads:
                T.copy(q[batch_idx, seq_idx, by, half_start : half_start + tile_half], q_first_tile)
                T.copy(q[batch_idx, seq_idx, by, second_start : second_start + tile_half], q_second_tile)
                for tid in T.Parallel(threads):
                    for lane in T.vectorized(elements_per_thread):
                        half_idx = tid * elements_per_thread + lane
                        q_first = T.cast(q_first_tile[half_idx], compute_dtype)
                        q_second = T.cast(q_second_tile[half_idx], compute_dtype)
                        cos_value = T.cast(cos_tile[half_idx], compute_dtype)
                        sin_value = T.cast(sin_tile[half_idx], compute_dtype) * T.cast(rotation_sign_value, compute_dtype)
                        q_first_rot[half_idx] = T.cast(q_first * cos_value - q_second * sin_value, dtype)
                        q_second_rot[half_idx] = T.cast(q_second * cos_value + q_first * sin_value, dtype)
                T.copy(q_first_rot, q_out[batch_idx, seq_idx, by, half_start : half_start + tile_half])
                T.copy(q_second_rot, q_out[batch_idx, seq_idx, by, second_start : second_start + tile_half])

            if by < n_k_heads:
                T.copy(k[batch_idx, seq_idx, by, half_start : half_start + tile_half], k_first_tile)
                T.copy(k[batch_idx, seq_idx, by, second_start : second_start + tile_half], k_second_tile)
                for tid in T.Parallel(threads):
                    for lane in T.vectorized(elements_per_thread):
                        half_idx = tid * elements_per_thread + lane
                        k_first = T.cast(k_first_tile[half_idx], compute_dtype)
                        k_second = T.cast(k_second_tile[half_idx], compute_dtype)
                        cos_value = T.cast(cos_tile[half_idx], compute_dtype)
                        sin_value = T.cast(sin_tile[half_idx], compute_dtype) * T.cast(rotation_sign_value, compute_dtype)
                        k_first_rot[half_idx] = T.cast(k_first * cos_value - k_second * sin_value, dtype)
                        k_second_rot[half_idx] = T.cast(k_second * cos_value + k_first * sin_value, dtype)
                T.copy(k_first_rot, k_out[batch_idx, seq_idx, by, half_start : half_start + tile_half])
                T.copy(k_second_rot, k_out[batch_idx, seq_idx, by, second_start : second_start + tile_half])

    return main


@tl.jit(out_idx=[-2, -1])
def qwen_rope_with_config(
    batch_size: int,
    seq_len: int,
    n_q_heads: int,
    n_k_heads: int,
    head_dim: int,
    threads: int = DEFAULT_THREADS,
    elements_per_thread: int = DEFAULT_ELEMENTS_PER_THREAD,
    enable_swizzle: bool = DEFAULT_ENABLE_SWIZZLE,
    rotation_sign: float = 1.0,
    dtype=T.float16,
    freq_dtype=T.float32,
    compute_dtype=T.float16,
):
    return make_qwen_rope_prim_func(
        batch_size=batch_size,
        seq_len=seq_len,
        n_q_heads=n_q_heads,
        n_k_heads=n_k_heads,
        head_dim=head_dim,
        threads=threads,
        elements_per_thread=elements_per_thread,
        enable_swizzle=enable_swizzle,
        rotation_sign=rotation_sign,
        dtype=dtype,
        freq_dtype=freq_dtype,
        compute_dtype=compute_dtype,
    )


def make_qwen_rope_jit_kernel(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    *,
    threads: int = DEFAULT_THREADS,
    elements_per_thread: int = DEFAULT_ELEMENTS_PER_THREAD,
    enable_swizzle: bool = DEFAULT_ENABLE_SWIZZLE,
    rotation_sign: float = 1.0,
):
    head_dim_half = q.shape[3] // 2
    if head_dim_half % (threads * elements_per_thread) != 0:
        config = select_default_config(head_dim_half)
        threads = int(config["threads"])
        elements_per_thread = int(config["elements_per_thread"])
        enable_swizzle = bool(config["enable_swizzle"])

    batch_size, seq_len, n_q_heads, head_dim = q.shape
    _, _, n_k_heads, _ = k.shape
    return qwen_rope_with_config(
        batch_size=batch_size,
        seq_len=seq_len,
        n_q_heads=n_q_heads,
        n_k_heads=n_k_heads,
        head_dim=head_dim,
        threads=threads,
        elements_per_thread=elements_per_thread,
        enable_swizzle=enable_swizzle,
        rotation_sign=rotation_sign,
        dtype=torch_dtype_to_tilelang_dtype(q.dtype),
        freq_dtype=torch_dtype_to_tilelang_dtype(cos.dtype),
        compute_dtype=torch_dtype_to_tilelang_dtype(q.dtype),
    )


def autotune_qwen_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cis: torch.Tensor,
    *,
    rotation_sign: float = 1.0,
    warmup: int = 3,
    rep: int = 20,
    profile_backend: str = "event",
):
    q_work, k_work, cos, sin, _ = prepare_inputs(q, k, freqs_cis)
    batch_size, seq_len, n_q_heads, head_dim = q_work.shape
    _, _, n_k_heads, _ = k_work.shape
    head_dim_half = head_dim // 2
    configs = get_autotune_configs(head_dim_half)
    rope_dtype = torch_dtype_to_tilelang_dtype(q_work.dtype)
    rope_freq_dtype = torch_dtype_to_tilelang_dtype(cos.dtype)

    def kernel(threads=None, elements_per_thread=None, enable_swizzle=None):
        return make_qwen_rope_prim_func(
            batch_size=batch_size,
            seq_len=seq_len,
            n_q_heads=n_q_heads,
            n_k_heads=n_k_heads,
            head_dim=head_dim,
            threads=threads,
            elements_per_thread=elements_per_thread,
            enable_swizzle=enable_swizzle,
            rotation_sign=rotation_sign,
            dtype=rope_dtype,
            freq_dtype=rope_freq_dtype,
            compute_dtype=rope_dtype,
        )

    with set_autotune_inputs(q_work, k_work, cos, sin):
        autotuner = (
            AutoTuner.from_kernel(kernel=kernel, configs=configs)
            .set_compile_args(
                out_idx=[-2, -1],
                target="auto",
            )
            .set_profile_args(
                supply_type=tl.TensorSupplyType.Normal,
                ref_prog=lambda q_in, k_in, cos_in, sin_in: qwen_rope_reference_prepared(
                    q_in,
                    k_in,
                    cos_in,
                    sin_in,
                    rotation_sign=rotation_sign,
                ),
                skip_check=False,
                rtol=2e-2,
                atol=2e-2,
                backend=profile_backend,
            )
        )
    return autotuner.run(warmup=warmup, rep=rep)


def qwen_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cis: torch.Tensor,
    *,
    threads: int = DEFAULT_THREADS,
    elements_per_thread: int = DEFAULT_ELEMENTS_PER_THREAD,
    enable_swizzle: bool = DEFAULT_ENABLE_SWIZZLE,
    rotation_sign: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_work, k_work, cos, sin, output_dtype = prepare_inputs(q, k, freqs_cis)
    kernel = make_qwen_rope_jit_kernel(
        q_work,
        k_work,
        cos,
        threads=threads,
        elements_per_thread=elements_per_thread,
        enable_swizzle=enable_swizzle,
        rotation_sign=rotation_sign,
    )
    q_out, k_out = kernel(q_work, k_work, cos, sin)
    if q_out.dtype != output_dtype:
        q_out = q_out.to(output_dtype)
    if k_out.dtype != output_dtype:
        k_out = k_out.to(output_dtype)
    return q_out, k_out


class QwenRopeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, freqs_cis, threads: int = DEFAULT_THREADS, elements_per_thread: int = DEFAULT_ELEMENTS_PER_THREAD):
        q_out, k_out = qwen_rope(
            q,
            k,
            freqs_cis,
            threads=threads,
            elements_per_thread=elements_per_thread,
            rotation_sign=1.0,
        )
        ctx.save_for_backward(freqs_cis.detach() if isinstance(freqs_cis, torch.Tensor) else freqs_cis)
        ctx.threads = threads
        ctx.elements_per_thread = elements_per_thread
        return q_out, k_out

    @staticmethod
    def backward(ctx, dq, dk):
        (freqs_cis,) = ctx.saved_tensors
        dq_out, dk_out = qwen_rope(
            dq,
            dk,
            freqs_cis,
            threads=ctx.threads,
            elements_per_thread=ctx.elements_per_thread,
            rotation_sign=-1.0,
        )
        return dq_out, dk_out, None, None, None


LigerQwenRopeFunction = QwenRopeFunction


def make_freqs_cis(
    seq_len: int,
    head_dim_half: int,
    *,
    dtype: torch.dtype = torch.float32,
    device: str = "cuda",
    seed: int = 0,
) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    angles = torch.randn(seq_len, head_dim_half, device=device, dtype=dtype, generator=generator)
    return torch.polar(torch.ones_like(angles), angles)


def make_inputs(
    batch_size: int,
    seq_len: int,
    n_q_heads: int,
    n_k_heads: int,
    head_dim: int,
    *,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    q = torch.randn(
        batch_size,
        seq_len,
        n_q_heads,
        head_dim,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    k = torch.randn(
        batch_size,
        seq_len,
        n_k_heads,
        head_dim,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    freqs_cis = make_freqs_cis(
        seq_len=seq_len,
        head_dim_half=head_dim // 2,
        device=device,
        seed=seed + 1,
    )
    return q, k, freqs_cis


def run_demo(
    batch_size: int = 4,
    seq_len: int = 1024,
    n_q_heads: int = 32,
    n_k_heads: int = 8,
    head_dim: int = 128,
    dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor, torch.Tensor]:
    q, k, freqs_cis = make_inputs(
        batch_size=batch_size,
        seq_len=seq_len,
        n_q_heads=n_q_heads,
        n_k_heads=n_k_heads,
        head_dim=head_dim,
        dtype=dtype,
    )
    return qwen_rope(q, k, freqs_cis)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the TileLang Qwen rotate-half RoPE kernel.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--q-heads", type=int, default=32)
    parser.add_argument("--k-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--elements-per-thread", type=int, default=DEFAULT_ELEMENTS_PER_THREAD)
    parser.add_argument("--enable-swizzle", action="store_true", default=DEFAULT_ENABLE_SWIZZLE)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    q, k, freqs_cis = make_inputs(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        n_q_heads=args.q_heads,
        n_k_heads=args.k_heads,
        head_dim=args.head_dim,
    )
    q_out, k_out = qwen_rope(
        q,
        k,
        freqs_cis,
        threads=args.threads,
        elements_per_thread=args.elements_per_thread,
        enable_swizzle=args.enable_swizzle,
    )
    print(q_out[0, 0, 0, :8])
    print(k_out[0, 0, 0, :8])
