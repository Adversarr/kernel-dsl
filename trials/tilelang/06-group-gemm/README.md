# Group GEMM

This trial implements a self-contained TileLang grouped GEMM kernel for a list of
independent `A_i @ B_i` problems with per-group `(M, N, K)` sizes.

## Files

- `group_gemm_kernel.py`: importable TileLang kernel, packing helpers, PyTorch
  references, fixed-config wrapper, and programmatic autotune entrypoint.
- `test_group_gemm_kernel.py`: correctness coverage for the fixed wrapper, a
  compiled fixed configuration, and the autotune path.
- `profile_group_gemm_kernel.py`: autotuned benchmark harness that compares
  against `torch.compile` and the local Triton baseline in
  `group_gemm_triton.py`.
- `group_gemm_triton.py`: existing Triton baseline kept for comparison.

## Implementation Status

- Status: implemented and verified.
- Kernel form: batch-packed grouped GEMM using a single TileLang kernel over
  `(group, tile_m, tile_n)`.
- Data model: each group is padded into a common packed tensor shape
  `(group_size, padded_M, padded_K)` and `(group_size, padded_K, padded_N)`.
- Metadata: a device-side `group_sizes` tensor stores per-group `(M, N, K)`.
- Tuning knobs: `block_M`, `block_N`, `block_K`, `threads`, `num_stages`, and
  `enable_swizzle`.
- Scope boundary: this implementation is original to this trial and does not
  depend on the official grouped-GEMM example code or anything under `3rdparty`.

## Operator Contract

- Inputs are two equal-length Python sequences of CUDA tensors.
- Every `lhs` and `rhs` pair must be rank-2, have matching inner dimension, and
  share device and dtype across the group.
- The current implementation expects row-major inner-contiguous operands, i.e.
  `lhs.stride(1) == 1` and `rhs.stride(1) == 1`.
- Inputs default to `torch.float16`; accumulation is performed in `float32`,
  then written back in the input dtype.

## Verification

Run:

```bash
uv run --with pytest python -m pytest trials/tilelang/06-group-gemm/test_group_gemm_kernel.py -q
```

Verified coverage:

- fixed wrapper on irregular group shapes
- compiled fixed configuration on packed inputs
- autotune smoke test using real packed inputs and metadata tensors

Latest local result:

- `3 passed in 6.61s`

## Profiling

Run:

```bash
uv run python trials/tilelang/06-group-gemm/profile_group_gemm_kernel.py --autotune-warmup 2 --autotune-rep 5 --benchmark-warmup 10 --benchmark-iters 30
```

Verified local benchmark on the current RTX 3090 setup:

- shapes:
  `[(2048, 2048, 2048), (1536, 1536, 2048), (1024, 1024, 2048), (512, 512, 2048)]`
- best config:
  `{'block_M': 64, 'block_N': 64, 'block_K': 32, 'num_stages': 2, 'threads': 128, 'enable_swizzle': False}`
- TileLang latency: `0.4867 ms`
- `torch.compile` latency: `1.0322 ms`
- Triton latency: `0.5385 ms`
- TileLang vs `torch.compile`: `2.12x`
- TileLang vs Triton: `1.11x`

These measurements satisfy the target goals for this trial:

- at least `1.0x` versus `torch.compile`: achieved
- near `1.0x` versus Triton: achieved

## Official Reference

After verification, the closest official TileLang implementations in this
workspace are:

- `../../skills/tilelang-wiki/examples/grouped_gemm/example_grouped_gemm_fwd.py`
- `../../skills/tilelang-wiki/examples/grouped_gemm/example_grouped_gemm_fwd_ptr.py`
- `../../skills/tilelang-wiki/examples/grouped_gemm/test_example_grouped_gemm.py`

The official examples cover a different operator shape than this trial:

- `example_grouped_gemm_fwd.py` packs one large `A` and uses group-dependent row
  segments with shared `K` and `N`
- `example_grouped_gemm_fwd_ptr.py` keeps per-group tensors separate and uses
  pointer tables
- this trial instead benchmarks a packed list of fully independent
  `(M_i, N_i, K_i)` GEMMs against PyTorch compile and Triton

## Documentation Improvements

- Clarify the two grouped-GEMM problem formulations. The example catalog says
  "grouped GEMM", but the official files actually cover two distinct contracts:
  segmented rows over a shared `A`, and pointer-table dispatch over per-group
  tensors.
- Add an operator-contract block near the top of each example. The reader should
  not need to infer whether `K` and `N` are shared, which dimensions may vary,
  or whether group padding is required.
- Document padding semantics explicitly. `example_grouped_gemm_fwd.py` uses
  `batch_padded_offsets`, but the docs do not explain why padded tile offsets
  are needed or how correctness is preserved for partial row tiles.
- Call out ptr-path limitations more prominently. The ptr example comments that
  runtime-varying ptr-backed shapes and multi-stage pipelining are not stable
  yet; that should be elevated from inline comments into the user-facing docs.
- Add autotune and baseline guidance. The official grouped-GEMM examples verify
  correctness, but they do not show how to compare against `torch.compile`,
  Triton, or how to report the best TileLang config.
- Split "correctness benchmark" from "performance benchmark". Readers benefit
  from seeing a small reproducible correctness case first, then a separate
  larger-shape performance harness with explicit FLOP accounting.
