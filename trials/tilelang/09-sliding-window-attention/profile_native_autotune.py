import importlib.util
import pathlib

import torch


HERE = pathlib.Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("sliding_window_attention", HERE / "sliding_window_attention.py")
swa = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(swa)


def useful_tflops(batch, heads, seq_len, dim, window_size, ms):
    return 4.0 * batch * heads * seq_len * min(window_size, seq_len) * dim / ms / 1e9


def actual_tile_tflops(batch, heads, seq_len, dim, window_size, block_m, block_n, ms):
    window_blocks = (window_size + block_m - 1 + block_n - 1) // block_n
    computed_k = window_blocks * block_n
    return 4.0 * batch * heads * seq_len * computed_k * dim / ms / 1e9


def main():
    small_q = torch.randn(1, 128, 1, 128, device="cuda", dtype=torch.float16)
    small_k = torch.randn_like(small_q)
    small_v = torch.randn_like(small_q)
    small_kernel = swa.build_fast_kernel(
        1,
        1,
        128,
        128,
        64,
        64,
        32,
        2,
        128,
        min_blocks_per_sm=1,
        swizzle_panel=8,
    )
    small_actual = small_kernel(small_q, small_k, small_v)
    small_expected = swa.torch_naive_sliding_window_attention(small_q, small_k, small_v, 64)
    torch.testing.assert_close(small_actual, small_expected, rtol=1e-2, atol=1e-2)

    batch, seq_len, heads, dim = 4, 2048, 8, 128
    torch.manual_seed(0)
    q = torch.randn(batch, seq_len, heads, dim, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    for window_size in (256, 512):
        kernel = swa.build_autotuned_fast_kernel(batch, heads, seq_len, dim, window_size)
        out = kernel(q, k, v)
        torch.cuda.synchronize()
        ms = swa.bench_cuda(lambda: kernel(q, k, v), warmup=50, rep=200)
        useful = useful_tflops(batch, heads, seq_len, dim, window_size, ms)
        print(f"W={window_size}: {ms:.4f} ms")
        print(f"  useful attention throughput: {useful:.2f} TFLOP/s")
        print(f"  has_nan: {torch.isnan(out).any().item()}")
        print(f"  output checksum: {out.float().mean().item():.6f}")


if __name__ == "__main__":
    main()
