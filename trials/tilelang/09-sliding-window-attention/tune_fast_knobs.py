import importlib.util
import pathlib
import traceback

import torch


HERE = pathlib.Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("sliding_window_attention", HERE / "sliding_window_attention.py")
swa = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(swa)


def tflops(batch, heads, seq_len, dim, window_size, ms):
    return 4.0 * batch * heads * seq_len * min(window_size, seq_len) * dim / ms / 1e9


def main():
    batch, seq_len, heads, dim = 4, 2048, 8, 128
    torch.manual_seed(0)
    q = torch.randn(batch, seq_len, heads, dim, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    configs = []
    for block_n in (16, 32, 64):
        for num_stages in (1, 2):
            for min_blocks_per_sm in (1, 2, 3, 4):
                for swizzle_panel in (0, 4, 8, 16):
                    configs.append((64, block_n, num_stages, 128, min_blocks_per_sm, swizzle_panel))

    for window_size in (256, 512):
        print(f"=== fast knobs W={window_size}")
        best = None
        for block_m, block_n, num_stages, threads, min_blocks_per_sm, swizzle_panel in configs:
            label = (
                f"bm={block_m} bn={block_n} stages={num_stages} threads={threads} "
                f"minblk={min_blocks_per_sm} swz={swizzle_panel}"
            )
            try:
                kernel = swa.build_fast_kernel(
                    batch,
                    heads,
                    seq_len,
                    dim,
                    window_size,
                    block_m,
                    block_n,
                    num_stages,
                    threads,
                    min_blocks_per_sm=min_blocks_per_sm,
                    swizzle_panel=swizzle_panel,
                )
                kernel(q, k, v)
                torch.cuda.synchronize()
                ms = swa.bench_cuda(lambda: kernel(q, k, v), warmup=20, rep=80)
                tf = tflops(batch, heads, seq_len, dim, window_size, ms)
                print(f"{label}: {ms:.4f} ms {tf:.2f} TFLOP/s")
                if best is None or tf > best[0]:
                    best = (tf, ms, label)
            except Exception as exc:
                tail = traceback.format_exc(limit=1).strip().splitlines()[-1]
                print(f"{label}: FAIL {type(exc).__name__}: {tail}")
        print(f"BEST: {best[2]} {best[1]:.4f} ms {best[0]:.2f} TFLOP/s")


if __name__ == "__main__":
    main()
