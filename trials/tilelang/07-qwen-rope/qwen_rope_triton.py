import torch
import triton
import triton.language as tl

from qwen_rope_kernel import normalize_freqs_cis, resolve_compute_dtype


def _cast_and_contiguous(q, k, freqs_cis):
    compute_dtype = resolve_compute_dtype(q.dtype)

    if k.dtype != q.dtype:
        k = k.to(q.dtype)

    q = q.to(compute_dtype).contiguous()
    k = k.to(compute_dtype).contiguous()
    cos, sin = normalize_freqs_cis(
        freqs_cis,
        seq_len=q.shape[1],
        head_dim_half=q.shape[3] // 2,
    )
    return q, k, cos.contiguous(), sin.contiguous()


@triton.jit
def _qwen_rope_kernel(
    q_ptr,
    k_ptr,
    cos_ptr,
    sin_ptr,
    q_row_stride,
    k_row_stride,
    q_head_stride,
    k_head_stride,
    cos_row_stride,
    sin_row_stride,
    seq_len,
    batch_size,
    rotation_sign,
    head_dim_half: tl.constexpr,
    n_q_heads: tl.constexpr,
    n_k_heads: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_bs = tl.program_id(0)
    pid_h = tl.program_id(1)

    batch_idx = pid_bs // seq_len
    seq_idx = pid_bs % seq_len

    if batch_idx >= batch_size or seq_idx >= seq_len:
        return

    base_offset = batch_idx * seq_len + seq_idx
    q_base = q_ptr + base_offset * q_row_stride
    k_base = k_ptr + base_offset * k_row_stride
    cos_base = seq_idx * cos_row_stride
    sin_base = seq_idx * sin_row_stride

    for d_start in tl.static_range(0, head_dim_half, BLOCK_SIZE):
        d_indices = d_start + tl.arange(0, BLOCK_SIZE)
        mask_d = d_indices < head_dim_half

        cos = tl.load(cos_ptr + cos_base + d_indices, mask=mask_d, other=0.0)
        sin = tl.load(sin_ptr + sin_base + d_indices, mask=mask_d, other=0.0) * rotation_sign

        if pid_h < n_q_heads:
            q_head_ptr = q_base + pid_h * q_head_stride
            q_first = tl.load(q_head_ptr + d_indices, mask=mask_d, other=0.0)
            q_second = tl.load(q_head_ptr + head_dim_half + d_indices, mask=mask_d, other=0.0)
            new_q_first = tl.math.fma(q_first, cos, -(q_second * sin))
            new_q_second = tl.math.fma(q_second, cos, q_first * sin)
            tl.store(q_head_ptr + d_indices, new_q_first, mask=mask_d)
            tl.store(q_head_ptr + head_dim_half + d_indices, new_q_second, mask=mask_d)

        if pid_h < n_k_heads:
            k_head_ptr = k_base + pid_h * k_head_stride
            k_first = tl.load(k_head_ptr + d_indices, mask=mask_d, other=0.0)
            k_second = tl.load(k_head_ptr + head_dim_half + d_indices, mask=mask_d, other=0.0)
            new_k_first = tl.math.fma(k_first, cos, -(k_second * sin))
            new_k_second = tl.math.fma(k_second, cos, k_first * sin)
            tl.store(k_head_ptr + d_indices, new_k_first, mask=mask_d)
            tl.store(k_head_ptr + head_dim_half + d_indices, new_k_second, mask=mask_d)


def _select_kernel_meta(head_dim_half: int):
    if head_dim_half >= 256:
        return 128, 8
    if head_dim_half >= 96:
        return 128, 4
    if head_dim_half >= 48:
        return 64, 4
    if head_dim_half >= 24:
        return 32, 2
    return 16, 2


def qwen_rope_forward(q, k, freqs_cis, BLOCK_SIZE: int = None, rotation_sign: float = 1.0):
    original_dtype = q.dtype

    batch_size, seq_len, n_q_heads, head_dim = q.shape
    _, _, n_k_heads, _ = k.shape
    head_dim_half = head_dim // 2

    q, k, cos, sin = _cast_and_contiguous(q, k, freqs_cis)

    if BLOCK_SIZE is None:
        BLOCK_SIZE, num_warps = _select_kernel_meta(head_dim_half)
    else:
        _, num_warps = _select_kernel_meta(head_dim_half)

    n_heads_max = max(n_q_heads, n_k_heads)
    grid = (batch_size * seq_len, n_heads_max)

    _qwen_rope_kernel[grid](
        q,
        k,
        cos,
        sin,
        q.stride(1),
        k.stride(1),
        q.stride(2),
        k.stride(2),
        cos.stride(0),
        sin.stride(0),
        seq_len,
        batch_size,
        rotation_sign,
        head_dim_half,
        n_q_heads,
        n_k_heads,
        BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=2,
    )

    if q.dtype != original_dtype:
        q = q.to(original_dtype)
    if k.dtype != original_dtype:
        k = k.to(original_dtype)

    return q, k


class LigerQwenRopeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, freqs_cis, BLOCK_SIZE: int = None):
        q_out, k_out = qwen_rope_forward(q, k, freqs_cis, BLOCK_SIZE, rotation_sign=1.0)
        ctx.save_for_backward(freqs_cis.detach() if isinstance(freqs_cis, torch.Tensor) else freqs_cis)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        return q_out, k_out

    @staticmethod
    def backward(ctx, dq, dk):
        (freqs_cis,) = ctx.saved_tensors
        block_size = getattr(ctx, "BLOCK_SIZE", None)
        dq_out, dk_out = qwen_rope_forward(dq, dk, freqs_cis, block_size, rotation_sign=-1.0)
        return dq_out, dk_out, None
