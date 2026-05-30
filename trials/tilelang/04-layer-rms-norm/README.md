# Layer RMSNorm

This trial implements a TileLang kernel for the operator:

`out = x * rsqrt(mean(x * x) + eps) * weight`

## Files

- `layer_rms_norm_kernel.py`: importable TileLang kernel, PyTorch reference, fixed-config compilation helper, and autotune helper.
- `test_layer_rms_norm_kernel.py`: correctness checks against the PyTorch RMSNorm reference.
- `profile_layer_rms_norm_kernel.py`: timing harness that autotunes by default and compares the best TileLang kernel with `torch.compile`.

## Operator Contract

- `x` has shape `(rows, cols)`.
- `weight` has shape `(cols,)`.
- Inputs and outputs are dense CUDA tensors.
- Inputs default to `torch.float16`.
- The kernel accumulates the row square sum in `float32`, computes `rsqrt(mean(x^2) + eps)`, and writes back a weighted normalized result in the original dtype.
- The exposed tuning knobs are `threads` and `elements_per_thread`.

## Kernel Structure

- Uses `T.Kernel(rows, threads=threads)` so one thread block handles one row.
- Sweeps each row in tiles of `threads * elements_per_thread`.
- Accumulates the RMS denominator with a tile-local `T.reduce_sum(...)`.
- Replays the same row tiles to apply the normalization scale and `weight`.

## Verification

Run:

```bash
uv run python test_layer_rms_norm_kernel.py
```

## Profiling

Run:

```bash
uv run python profile_layer_rms_norm_kernel.py
```

Use `--no-autotune` to benchmark a single fixed configuration.

## Autotuning

- The autotuner searches `threads` and `elements_per_thread`.
- A verified run on an RTX 3090 with `rows=4096` and `cols=4096` reached `1.05x` speedup versus `torch.compile`.
- The best verified config for that run was `{'threads': 128, 'elements_per_thread': 16}`.

## Official Reference

The closest official TileLang implementation in this workspace is:

- `../../.agents/skills/tilelang-wiki/examples/norm/rms_norm.py`

That official example confirms the same core operator structure:

- accumulate `x * x` across the row,
- reduce the row square sum with `T.reduce_sum(...)`,
- compute `T.rsqrt(mean + eps)`,
- replay the row values to apply the normalization factor.

## Documentation Improvements

- Clarify that the official `rms_norm.py` example is an unweighted RMSNorm baseline, while production-style RMSNorm usually includes a learned `weight` vector applied in the output pass.
- Explain when to use the full-row fragment variant versus the split-K streaming variant, since the example ships both but does not spell out the trade-off in memory footprint versus tiling flexibility.
- Add a norm-specific autotuning example or a short link from the norm example to the autotuning guide, because the official RMSNorm example verifies correctness and benchmarks latency but does not show how to search launch parameters.
- Call out mixed-precision adaptation explicitly: the official example uses `T.float`, while practical kernels often load `float16` or `bfloat16`, accumulate in `float32`, and cast back on writeback.
- Recommend separating importable kernel code from correctness and profiling harnesses for reusable operator trials, because the official example is monolithic and less convenient to benchmark or extend in isolation.
