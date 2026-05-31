# Programming Guides Overview

This directory is split into fast-reference pages and API detail pages.
Use the top-level pages first when you need the shortest path to a working
kernel. Open the nested API directories when you need exact function behavior.

## Fast Reference Pages

- `overview.md`: this map of the guide structure.
- `language_basics.md`: basic TileLang language cheatsheet: kernel shape,
  launch region, allocation, copy, compute, reductions, and the flash-attention
  reading pattern.
- `jit_autotune.md`: basic `tilelang.jit` and `tilelang.autotune` workflow:
  compile a kernel, run it, scan configs, and avoid common cache surprises.
- `python_compat.md`: Python syntax, symbolic dimensions, tensor annotations,
  dtype forms, casts, and type-related gotchas inside TileLang kernels.

These pages intentionally avoid advanced scheduling and backend-specific detail.
Their job is to help you recognize the right construct quickly.

## API Detail Pages

- `tilelang/`: top-level `import tilelang` APIs.
  - `jit/`: `tilelang.jit`, explicit compile paths, `JITKernel`, source
    inspection, and cache behavior.
  - `autotune/`: decorator and programmatic tuning, validation, captured inputs,
    config spaces, grouped compile, and tuning cache controls.
  - `env/`: environment variables that affect cache, target/backend selection,
    verbosity, temporary files, native libraries, and import behavior.
  - `autodd_tools/`: AutoDD CLI, freeze annotations, layout plotting, analyzer
    helpers, and debug hooks.
- `tilelang_language/`: `import tilelang.language as T` APIs.
  - `loop/`: serial, unrolled, parallel, pipelined, and persistent loop forms.
  - `allocate/`: global, shared, local, fragment, scalar, barrier, tensor-memory,
    descriptor, and reducer allocations.
  - `copy_op/`: `T.copy`, `T.async_copy`, TMA, cluster copy, gather/scatter, and
    transpose helpers.
  - `gemm_op/`: dense GEMM, sparse GEMM, WGMMA, TCGEN05, and block-scaled GEMM.
  - `basic_operations/`: fill/clear, tensor annotations, proxy buffers, pointer
    views, and math intrinsics.
  - `kernel_warpgroup_cluster_builtins/`: kernel launch frames, thread/block
    builtins, warpgroup helpers, cluster helpers, mbarriers, and low-level MMA.
  - `annotations/`: swizzle, layout, pass config, compile flags, launch bounds,
    safe value, restrict buffer, and L2 hints.
  - `reduce_op/`: reductions, cumulative sum, warp reductions, and generic
    reduction controls.
  - `misc/`: atomics, debug helpers, random numbers, dynamic symbols, explicit
    loads/stores, PDL, raw TIR exports, and less common helpers.

Each API topic has `basic.md` for common usage and `advanced.md` for
target-specific, lower-level, or rarely used behavior.

## Legacy Pages

The previous narrative guides are preserved under `old/`. They are useful when
you want a longer walkthrough or historical context, but the fast-reference and
API-detail pages above should be the default sources for new answers.

## Reading Order

1. Start with `language_basics.md` for kernel structure.
2. Use `jit_autotune.md` when the question is about compiling, calling, tuning,
   or cache behavior.
3. Use `python_compat.md` when the question is about supported Python syntax,
   shapes, tensors, dtypes, or casts.
4. Jump into `tilelang_language/<topic>/basic.md` for a specific `T.*` API.
5. Jump into `tilelang/<topic>/basic.md` for a top-level `tilelang.*` API.
6. Open `advanced.md` only when a basic page points there or the kernel uses an
   explicit advanced path such as TMA, WGMMA, TCGEN05, cluster launch control,
   manual synchronization, AutoDD, or grouped autotune compile.
