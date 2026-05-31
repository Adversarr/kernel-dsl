# TileLang Language API Guide V2

This directory is organized by `tilelang.language` API area. Each topic has two
files:

- `basic.md`: common APIs and patterns used across the examples.
- `advanced.md`: explicit scheduling, target-specific, lower-level, or rarely
  used APIs.

Start with the basic page for the topic you need. Open the advanced page only
when the basic page points there or when the kernel uses an explicit hardware
path such as TMA, WGMMA, TCGEN05, cluster launch control, raw pointers, or
manual synchronization.

## Topics

- `loop/`: `T.Parallel`, `T.Pipelined`, serial loops, unroll/vectorized loops,
  and persistent scheduling.
- `allocate/`: shared/local/fragment buffers, scalar variables, eager outputs,
  barriers, tensor memory, descriptors, and reducers.
- `copy_op/`: `T.copy`, `T.im2col`, async copy, TMA copy, cluster copy,
  gather/scatter TMA, and transpose.
- `gemm_op/`: dense GEMM, sparse GEMM, WGMMA, TCGEN05, and block-scaled GEMM.
- `basic_operations/`: tensor annotations, fill/clear, elementwise work,
  proxy buffers, pointer views, and math intrinsics.
- `kernel_warpgroup_cluster_builtins/`: launch frames, thread/block bindings,
  cluster helpers, mbarriers, warpgroup helpers, and low-level MMA builtins.
- `annotations/`: swizzle, layout, restrict-buffer, launch-bound, safe-value,
  L2, compile-flag, and pass-config annotations.
- `reduce_op/`: tile reductions, cumulative sum, generic reduction controls,
  batched reductions, NaN propagation, and warp reductions.
- `misc/`: atomics, debug helpers, dynamic symbols, boolean buffer reductions,
  access pointers, explicit loads/stores, random numbers, PDL, and raw TIR
  exports.