# FlashAttention-2

This trial implements a self-contained TileLang forward FlashAttention-2 kernel for:

`out = softmax((q @ k^T) * scale) @ v`

with `q`, `k`, and `v` laid out as `(batch, seq_len, heads, head_dim)`.

## Files

- `flash_attention2_kernel.py`: importable TileLang kernel, PyTorch references, fixed-config compilation helper, and programmatic autotune helper.
- `test_flash_attention2_kernel.py`: correctness checks against a PyTorch attention reference for aligned non-causal and causal cases.
- `profile_flash_attention2_kernel.py`: timing harness that autotunes by default, compares against `torch.compile`, and also reports explicit PyTorch SDPA flash-kernel performance.

## Operator Contract

- `q`, `k`, and `v` must have identical shape `(batch, seq_len, heads, head_dim)`.
- Inputs and outputs are dense CUDA tensors.
- Inputs default to `torch.float16`.
- The kernel accumulates scores and output tiles in `float32`, then writes the final result back in the input dtype.
- The current implementation assumes `seq_len` is divisible by both `block_M` and `block_N`.
- The exposed tuning knobs are `block_M`, `block_N`, `threads`, `num_stages`, and `enable_swizzle`.

## Kernel Structure

- Uses `T.Kernel(T.ceildiv(seq_len, block_M), heads, batch, threads=threads)`.
- Stages `Q`, `K`, and `V` tiles through shared memory with `T.copy(...)`.
- Computes `Q @ K^T` with `T.gemm(..., transpose_B=True, policy=T.GemmWarpPolicy.FullRow)`.
- Applies online softmax normalization by tracking per-row `scores_max`, `scores_scale`, and `logsum`.
- Rescales the running output tile before fusing the `P @ V` update into `acc_o`.

## Verification

Run:

```bash
uv run python test_flash_attention2_kernel.py
```

Verified cases:

- non-causal: `(batch=1, heads=4, seq_len=64, head_dim=32)`
- non-causal: `(batch=2, heads=4, seq_len=128, head_dim=64)`
- causal: `(batch=1, heads=8, seq_len=128, head_dim=64)`

## Profiling

Run:

```bash
uv run python profile_flash_attention2_kernel.py
```

Use `--no-autotune` to benchmark a single fixed configuration instead of searching the config space.

## Autotuning

- The autotuner searches `block_M`, `block_N`, `num_stages`, `threads`, and `enable_swizzle`.
- A verified autotuned run on an RTX 3090 with `batch=4`, `heads=8`, `seq_len=512`, `head_dim=64`, and `causal=False` reached `6.13x` speedup versus `torch.compile`.
- The best verified autotuned config for that run was:
  `{'block_M': 64, 'block_N': 64, 'num_stages': 2, 'threads': 128, 'enable_swizzle': True}`
- The same run also measured `1.62x` versus explicit PyTorch SDPA with the flash-attention backend forced.

## Official Reference

The closest official implementation in this workspace is:

- `../../.agents/skills/tilelang-wiki/examples/flash_attention/example_mha_fwd_bshd.py`

Related official validation entry points are:

- `../../.agents/skills/tilelang-wiki/examples/flash_attention/test_example_flash_attention.py`
- `../../.agents/skills/tilelang-wiki/examples/flash_attention/README.md`

## What This Trial Adds

- Keeps the forward kernel importable and separate from correctness and profiling harnesses.
- Exposes a clean host wrapper for tensors already in `(batch, seq_len, heads, head_dim)` format.
- Uses programmatic autotuning so the benchmark script can directly report the best config and latency.
- Adds an explicit comparison against both `torch.compile` and PyTorch's flash SDPA path.

## Documentation Improvements

- Document the layout contract more explicitly: the official example is `BSHD`, while the directory also contains `BHSD` variants and the README does not clearly spell out when to choose each.
- Explain the online softmax math in one short block: why the kernel multiplies by `log2(e)`, why `scores_max_prev` is retained, and why `acc_o` must be rescaled before the next `V` update.
- Call out the practical tuning surface: `block_M`, `block_N`, `threads`, `num_stages`, and swizzle are the real knobs, but the official example only ships a tiny default config list.
- Add a note about shape constraints and boundary handling. The docs show the aligned happy path but do not clearly distinguish between aligned kernels and kernels that additionally mask partial tail tiles.
- Add a benchmarking note that separates three baselines: eager PyTorch reference, `torch.compile`, and explicit SDPA flash attention. They answer different questions and can differ by a lot.
- Mention a fixed-loop causal fallback for environments where a causal `T.Pipelined(...)` loop bound that depends on `bx` is harder to lower reliably than full-range iteration plus in-tile masking.
