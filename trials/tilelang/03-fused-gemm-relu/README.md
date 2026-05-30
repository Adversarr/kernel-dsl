# Fused GEMM + ReLU

This trial implements a TileLang kernel for the operator:

`out = relu(lhs @ rhs)`

## Files

- `fused_gemm_relu_kernel.py`: importable TileLang kernel, PyTorch reference, fixed-config compilation helpers, and autotune helpers.
- `test_fused_gemm_relu_kernel.py`: correctness checks against `torch.relu(lhs @ rhs)`.
- `profile_fused_gemm_relu_kernel.py`: timing harness that autotunes the kernel by default and compares the best result with `torch.compile`.

## Operator Contract

- `lhs` has shape `(M, K)` and `rhs` has shape `(K, N)`.
- The current implementation targets dense GPU tensors on CUDA.
- Inputs default to `torch.float16`.
- Accumulation happens in `float32`, then the fused epilogue applies ReLU before the result is written back to the output tensor.
- Tile sizes and pipeline depth are exposed as parameters: `block_M`, `block_N`, `block_K`, `threads`, and `num_stages`.

## Kernel Structure

- Uses `T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads)`.
- Stages operand tiles through shared memory with `T.copy(...)`.
- Accumulates the matrix product with `T.gemm(...)`.
- Applies the ReLU epilogue with a tile-wide `T.Parallel(...)` loop over the accumulator fragment.

## Verification

Run:

```bash
uv run python test_fused_gemm_relu_kernel.py
```

## Profiling

Run:

```bash
uv run python profile_fused_gemm_relu_kernel.py
```

Use `--no-autotune` to benchmark a single fixed configuration instead of searching the config space.

## Autotuning

- The autotuner searches `block_M`, `block_N`, `block_K`, `num_stages`, `threads`, and `enable_swizzle`.
- The benchmark script uses programmatic TileLang autotuning and validates candidates against `torch.relu(lhs @ rhs)`.
- A verified run on `M=N=K=1024` reached `1.06x` speedup versus `torch.compile` with:
  `{'block_M': 64, 'block_N': 128, 'block_K': 32, 'num_stages': 2, 'threads': 256, 'enable_swizzle': False}`

## Official Reference

The closest official examples in this workspace are:

- `../../skills/tilelang-wiki/examples/quickstart.py`
- `../../skills/tilelang-wiki/examples/gemm/example_gemm_autotune.py`

Those examples use the same core structure:

- `T.Kernel(...)` over `(N, M)` tiles
- `T.copy(...)` into shared memory
- `T.gemm(...)` for the matmul tile
- `T.Parallel(...)` with `T.max(...)` for the fused ReLU epilogue
- Programmatic autotuning over tile shape, pipeline depth, and launch configuration

## What This Trial Adds

- Separates the importable kernel from correctness and profiling harnesses.
- Exposes tuning knobs through function arguments instead of only through one top-level example.
- Verifies more than one shape, rather than only one aligned `1024 x 1024 x 1024` configuration.
- Compares runtime with a PyTorch baseline in a dedicated benchmark script and supports autotuned search by default.

## Documentation Improvements

- Record that the official quickstart is the direct fused GEMM+ReLU reference for this trial, while `example_gemm_autotune.py` is the autotuning reference pattern.
- Mention that `@tilelang.jit` can infer the target from the input tensors, as shown in the official example comments.
- Call out `T.use_swizzle(...)` as an autotune dimension for cache locality, because the official examples show it as an important optimization knob.
- Mention `kernel.get_profiler()` as the native TileLang profiling path alongside the simpler event-based benchmark script used here.
- Note that the official examples demonstrate the minimal happy path and the autotune API separately, while this trial combines reusable module boundaries, correctness checks, and autotuned benchmarking for one operator.

## Follow-Up

Useful next documentation upgrades:

- Add a short note on how to choose `block_M`, `block_N`, and `block_K` for different GPU generations.
- Add a brief explanation of why the epilogue is applied on the accumulator fragment before the final `T.copy(...)`.
- Add explicit notes on which shape combinations have been verified so boundary coverage is visible without opening the test file.
