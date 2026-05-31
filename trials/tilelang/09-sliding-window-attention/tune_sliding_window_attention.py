import importlib.util
import pathlib
import traceback

import torch


HERE = pathlib.Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("sliding_window_attention", HERE / "sliding_window_attention.py")
swa = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(swa)


def tflops(batch, heads, seq_len, dim, window_size, ms):
    work = 4.0 * batch * heads * seq_len * min(window_size, seq_len) * dim
    return work / ms / 1e9


def run_one(builder_name, window_size, block_m, block_n, num_stages, threads, q, k, v, check=False):
    batch, seq_len, heads, dim = q.shape
    builder = getattr(swa, builder_name)
    kernel = builder(batch, heads, seq_len, dim, window_size, block_m, block_n, num_stages, threads)
    out = kernel(q, k, v)
    torch.cuda.synchronize()
    if check:
        expected = swa.torch_naive_sliding_window_attention(
            q[:1, :128, :1, :].contiguous(),
            k[:1, :128, :1, :].contiguous(),
            v[:1, :128, :1, :].contiguous(),
            min(window_size, 128),
        )
        small_kernel = builder(1, 1, 128, dim, min(window_size, 128), block_m, block_n, num_stages, threads)
        actual = small_kernel(
            q[:1, :128, :1, :].contiguous(),
            k[:1, :128, :1, :].contiguous(),
            v[:1, :128, :1, :].contiguous(),
        )
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
    ms = swa.bench_cuda(lambda: kernel(q, k, v), warmup=25, rep=100)
    return ms, tflops(batch, heads, seq_len, dim, window_size, ms)


def main():
    batch, seq_len, heads, dim = 4, 2048, 8, 128
    torch.manual_seed(0)
    q = torch.randn(batch, seq_len, heads, dim, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    configs = [
        ("build_kernel", 64, 64, 2, 128),
        ("build_fast_kernel", 64, 64, 1, 128),
        ("build_fast_kernel", 64, 64, 2, 128),
        ("build_fast_kernel", 64, 16, 1, 128),
        ("build_fast_kernel", 64, 16, 2, 128),
        ("build_fast_kernel", 64, 32, 1, 128),
        ("build_fast_kernel", 64, 32, 2, 128),
        ("build_fast_kernel", 128, 64, 1, 128),
        ("build_fast_kernel", 128, 64, 2, 128),
        ("build_interior_kernel", 64, 64, 1, 128),
        ("build_interior_kernel", 64, 64, 2, 128),
        ("build_interior_kernel", 64, 64, 3, 128),
        ("build_interior_kernel", 64, 16, 1, 128),
        ("build_interior_kernel", 64, 16, 2, 128),
        ("build_interior_kernel", 64, 32, 1, 128),
        ("build_interior_kernel", 64, 32, 2, 128),
        ("build_interior_kernel", 64, 32, 1, 256),
        ("build_interior_kernel", 64, 64, 1, 256),
        ("build_interior_kernel", 128, 64, 1, 128),
        ("build_interior_kernel", 128, 64, 2, 128),
    ]

    for window_size in (256, 512):
        print(f"=== B={batch} S={seq_len} H={heads} D={dim} W={window_size}")
        best = None
        for builder_name, block_m, block_n, num_stages, threads in configs:
            label = (
                f"{builder_name} bm={block_m} bn={block_n} "
                f"stages={num_stages} threads={threads}"
            )
            try:
                ms, tf = run_one(
                    builder_name,
                    window_size,
                    block_m,
                    block_n,
                    num_stages,
                    threads,
                    q,
                    k,
                    v,
                    check=(builder_name == "build_interior_kernel" and block_m == 64 and block_n == 64),
                )
                print(f"{label}: {ms:.4f} ms {tf:.2f} TFLOP/s")
                if best is None or tf > best[0]:
                    best = (tf, ms, label)
            except Exception as exc:
                tail = traceback.format_exc(limit=1).strip().splitlines()[-1]
                print(f"{label}: FAIL {type(exc).__name__}: {tail}")
        print(f"BEST: {best[2]} {best[1]:.4f} ms {best[0]:.2f} TFLOP/s")


if __name__ == "__main__":
    main()
