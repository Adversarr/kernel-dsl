# TileLang Reference Guide

This README is the main reference entry point for the local TileLang skill.
Use it first when you need to understand how TileLang kernels are written,
which APIs matter most, how to choose a starting template, and where to go next
inside the bundled reference pages.

TileLang is a Python-first DSL for writing high-performance kernels for targets
such as CUDA and HIP. The local material in this directory is the source of
truth for semantics and guidance in this skill. Use `../examples/README.md` for
the example catalogue and operator inventory. Use `../FAQs.md` for known
failure modes and issue-driven debugging shortcuts.

## Start Here

Use this file first for:

- the mental model of a TileLang kernel
- the most important DSL primitives
- a quick API reference
- common kernel templates and adaptation patterns
- routing to deeper guides under `references/`

Use the rest of the local docs by intent:

- setup and targets:
  - `get_started/Installation.md`
  - `get_started/overview.md`
  - `get_started/targets.md`
- core DSL semantics:
  - `programming_guides/language_basics.md`
  - `programming_guides/control_flow.md`
  - `programming_guides/python_compatibility.md`
  - `programming_guides/type_system.md`
- instruction behavior and pipelining:
  - `programming_guides/instructions.md`
  - `programming_guides/software_pipeline.md`
  - `programming_guides/cluster_tma.md`
- tuning and debugging:
  - `programming_guides/autotuning.md`
  - `tutorials/auto_tuning.md`
  - `tutorials/debug_tools_for_tilelang.md`
  - `tutorials/logging.md`
  - `../FAQs.md`
  Prefer the Decorator Workflow for autotuning by default; use the Programmatic Workflow only when explicitly requested or when finer control is required.
- operator walkthroughs:
  - `deeplearning_operators/elementwise.md`
  - `deeplearning_operators/gemv.md`
  - `deeplearning_operators/matmul.md`
  - `deeplearning_operators/matmul_sparse.md`
  - `deeplearning_operators/deepseek_mla.md`
- compiler and runtime internals:
  - `compiler_internals/letstmt_inline.md`
  - `compiler_internals/inject_fence_proxy.md`
  - `compiler_internals/tensor_checks.md`
  - `runtime_internals/stubs.md`

## TileLang In One Minute

Most TileLang work follows the same shape:

1. Define a kernel with `@tilelang.jit` or return a nested `@T.prim_func`.
2. Declare symbolic sizes with `T.const(...)` or `T.dynamic(...)`.
3. Describe buffers with `T.Tensor(...)` and outputs with `T.empty(...)`.
4. Open a launch region with `T.Kernel(...)`.
5. Allocate shared, fragment, local, or scalar storage.
6. Move tiles with `T.copy(...)`.
7. Compute with `T.gemm(...)`, reductions, or elementwise loops.
8. Use `T.Pipelined(...)`, swizzle, autotuning, or specialized instructions to optimize.
9. Validate against a reference and inspect generated code or profiles when needed.

The canonical local example is `../examples/quickstart.py`.

## Language Basics

The most important TileLang surface area is small.

### Kernel Entry Points

Most examples start with:

```python
import tilelang
import tilelang.language as T
from tilelang import jit

@tilelang.jit
def kernel(...):
    ...
```

Two common styles exist:

- eager-style JIT function that returns outputs allocated with `T.empty(...)`
- JIT factory that returns a nested `@T.prim_func` with explicit output buffers

Use the first style for straightforward kernels and the second when you want
more explicit signatures or `out_idx`-style control.

### Shapes And Buffer Types

Use `T.const(...)` for dimensions inferred from input tensors:

```python
M, N, K = T.const("M, N, K")
```

Use `T.dynamic(...)` when the generated kernel must stay shape-polymorphic:

```python
M = T.dynamic("M")
N = T.dynamic("N")
K = T.dynamic("K")
```

`T.dyn["K"]` is the shorthand to use when you want a named symbolic dimension
directly in a tensor annotation and can read the concrete extent back from a
buffer shape inside the kernel body.

Then annotate buffers:

```python
A: T.Tensor((M, K), T.float16)
B: T.Tensor((K, N), T.float16)
C = T.empty((M, N), T.float16)
```

### Launch Regions

`T.Kernel(...)` defines the launch geometry:

```python
with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
    ...
```

- positional arguments map to grid dimensions
- `threads=` sets thread geometry
- `cluster_dims=` is available on supported GPU targets
- most high-level kernels use `T.Parallel(...)` or tile ops rather than raw thread bindings

### Memory Scopes

The common allocation helpers are:

```python
A_shared = T.alloc_shared((block_M, block_K), T.float16)
B_shared = T.alloc_shared((block_K, block_N), T.float16)
C_frag = T.alloc_fragment((block_M, block_N), T.float32)
tmp = T.alloc_local((4,), T.float16)
scale = T.alloc_var("float32", init=1.0)
```

Use them like this:

- `T.Tensor(...)` and `T.empty(...)`: global memory
- `T.alloc_shared(...)`: block-visible shared memory
- `T.alloc_fragment(...)`: fragment/register-style storage for tile compute
- `T.alloc_local(...)`: thread-local storage
- `T.alloc_var(...)`: scalar-like local storage

### Loops And Conditions

TileLang provides loop helpers that describe intent:

```python
for i in T.serial(N):
    ...

for i, j in T.Parallel(block_M, block_N):
    ...

for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
    ...
```

- `T.serial(...)`: ordinary sequential loops
- `T.unroll(...)`: unrolled loops
- `T.Parallel(...)`: elementwise or tile-parallel loops
- `T.Pipelined(...)`: staged copy/compute loops used heavily in GEMM and attention

Use Python `if` / `elif` / `else` for control flow.
Use `T.if_then_else(...)` when you need a value-producing conditional expression.

### Data Movement And Compute

The core data-movement primitive is:

```python
T.copy(src, dst)
```

Treat `T.copy(...)` as the default, synchronously consumable movement primitive.
Reach for `T.async_copy(...)` only when you are intentionally building a manual
copy/compute overlap and are prepared to insert the required wait before
consuming the destination tile.

Typical usage:

```python
T.copy(A[by * block_M, ko * block_K], A_shared)
T.copy(B[ko * block_K, bx * block_N], B_shared)
T.copy(C_frag, C[by * block_M, bx * block_N])
```

The core tile compute primitive is:

```python
T.gemm(A_shared, B_shared, C_frag)
```

Use `T.async_copy(...)`, TMA helpers, or lower-level synchronization only when
you are intentionally managing an advanced pipeline.

### Debugging, Validation, And Profiling

The most common host-side and kernel-side tools are:

- `T.print(...)`
- `T.device_assert(...)`
- `kernel.get_kernel_source()`
- `kernel.get_profiler()`
- `profiler.do_bench(...)`
- `torch.testing.assert_close(...)`

## Quick API Reference

### Structure

| API | Purpose |
| --- | --- |
| `@tilelang.jit(...)` | JIT compile a Python kernel wrapper or kernel factory |
| `@jit` | Short alias for `@tilelang.jit` often used by examples |
| `@T.prim_func` | Declare the TileLang kernel body |
| `T.Tensor((shape), dtype)` | Annotate a kernel buffer |
| `T.empty(shape, dtype)` | Declare an eager-style output |
| `T.Kernel(...)` | Define launch geometry |
| `tilelang.compile(...)` | Explicit compilation path |

### Shapes And Dtypes

| API | Purpose |
| --- | --- |
| `T.const("M, N")` | Infer symbolic dimensions from input tensors |
| `T.dynamic("name")` | Keep a dimension symbolic in the compiled kernel |
| `T.dyn["name"]` | Shorthand symbolic dimension for tensor annotations |
| `T.ceildiv(a, b)` | Compute ceil division for grid sizing |
| `T.float16`, `T.bfloat16`, `T.float32` | Common floating-point dtypes |
| `T.float64` | Additional floating-point dtype used by some helpers and tests |
| `T.int8`, `T.int16`, `T.int32`, `T.int64` | Common integer dtypes |
| `T.uint8`, `T.uint16`, `T.uint32` | Common unsigned integer dtypes |
| `T.float8_e4m3fn`, `T.float8_e5m2` | Common FP8 dtypes used by examples |

### Allocation And Movement

| API | Purpose |
| --- | --- |
| `T.alloc_shared(shape, dtype)` | Allocate shared memory |
| `T.alloc_fragment(shape, dtype)` | Allocate fragment/register-style storage |
| `T.alloc_local(shape, dtype)` | Allocate local thread-private storage |
| `T.alloc_var(dtype, init=...)` | Allocate a scalar-like local value |
| `T.alloc_global(shape, dtype)` | Dynamically allocate a global buffer when needed |
| `T.copy(src, dst)` | Move data between memory scopes |
| `T.async_copy(src, dst)` | Explicit asynchronous copy path that requires an explicit wait before use |

### Compute And Helpers

| API | Purpose |
| --- | --- |
| `T.gemm(A, B, C)` | Tile GEMM into a fragment accumulator |
| `T.gemm(..., transpose_B=True)` | GEMM when B is staged as `(N, K)` |
| `T.reduce_sum`, `T.reduce_max`, `T.reduce_min` | Tile reductions |
| `T.cumsum(...)` | Prefix-sum style scan |
| `T.clear(buf)` | Zero a buffer |
| `T.fill(buf, value)` | Fill a buffer with a scalar |
| `T.atomic_add(dst, value)` | Atomic update |
| `T.cast(value, dtype)` | Explicit cast |
| `T.max`, `T.min`, `T.exp`, `T.log`, `T.rsqrt`, `T.sigmoid` | Common math helpers |
| `T.infinity(dtype)` | Typed positive infinity constant |
| `T.if_then_else(cond, x, y)` | Value-producing conditional |

### Loops And Scheduling

| API | Purpose |
| --- | --- |
| `T.serial(...)` | Sequential loop |
| `T.unroll(...)` | Unrolled loop |
| `T.Parallel(...)` | Parallel loop nest |
| `T.Pipelined(..., num_stages=N)` | Software-pipelined loop |
| `T.Persistent(...)` | Persistent thread-block loop form used by some streaming kernels |
| `T.use_swizzle(panel_size=..., enable=True)` | L2/rasterization hint |
| `T.annotate_layout({...})` | Explicit layout hint |
| `T.annotate_l2_hit_ratio(buf, ratio)` | Hint expected L2 reuse for a buffer |

### Debugging And Profiling

| API | Purpose |
| --- | --- |
| `T.print(obj, msg="")` | Print a scalar or buffer |
| `T.device_assert(cond, msg="")` | Device-side assertion |
| `kernel.get_kernel_source()` | Inspect generated kernel source |
| `kernel.get_profiler()` | Build a profiler |
| `profiler.do_bench(...)` | Benchmark latency |
| `profiler.assert_allclose(...)` | Compare against a reference |
| `profiler.assert_consistent(...)` | Re-run to catch output instability or races |

## Mental Model

TileLang is primarily tile-level programming, not elementwise programming with a
GPU-flavored syntax.

When writing a kernel, first decide:

- what tile-shaped data is staged at each step
- what tile-shaped state must persist across loop iterations
- what operator family already has a canonical example

Treat the shapes of shared and fragment buffers as part of the algorithm design,
not as incidental storage details.

A kernel can be shape-legal and still be the wrong starting structure if it
scalarizes state that is naturally tile-shaped, or if it is derived from a
generic template instead of the canonical operator example.

## Kernel Templates

The local examples are the actual runnable templates. Use this section to choose
the right pattern before opening `../examples/README.md`.

### 1D Or 2D Elementwise

Use this shape when every output element depends on only a small local region of
input data:

```python
with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
    X_shared = T.alloc_shared((block_M, block_N), dtype)
    Y_local = T.alloc_fragment((block_M, block_N), accum_dtype)
    T.copy(X[by * block_M, bx * block_N], X_shared)
    for i, j in T.Parallel(block_M, block_N):
        Y_local[i, j] = op(X_shared[i, j])
    T.copy(Y_local, Y[by * block_M, bx * block_N])
```

Start from:

- `../examples/elementwise/example_elementwise_add.py`
- `../examples/cast/`
- `../examples/norm/`

### Row Reduction

Use this pattern when each output row accumulates across a tiled reduction
dimension:

This template teaches tiled reduction mechanics. It is not a default derivation
path for every row-wise operator.

```python
acc = T.alloc_fragment((block_M,), accum_dtype)
T.clear(acc)
for ko in T.serial(T.ceildiv(N, block_N)):
    T.copy(A[bx * block_M, ko * block_N], A_shared)
    T.reduce_sum(A_shared, local_sum, dim=1)
    for i in T.Parallel(block_M):
        acc[i] = acc[i] + local_sum[i]
T.copy(acc, Out[bx * block_M])
```

If an operator family has an example under `examples/`, prefer that operator
example over rewriting the operator from generic reduction primitives. In
TileLang, the buffer shapes and loop-carried state are part of the kernel
design.

Start from:

- `../examples/topk/`
- `../examples/online_softmax/`
- `../examples/norm/`

### Minimal GEMM

Use this as the default matrix-multiplication template:

```python
with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
    A_shared = T.alloc_shared((block_M, block_K), dtype)
    B_shared = T.alloc_shared((block_K, block_N), dtype)
    C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
    T.clear(C_local)
    for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
        T.copy(A[by * block_M, ko * block_K], A_shared)
        T.copy(B[ko * block_K, bx * block_N], B_shared)
        T.gemm(A_shared, B_shared, C_local)
    T.copy(C_local, C[by * block_M, bx * block_N])
```

Start from:

- `../examples/quickstart.py`
- `../examples/gemm/`

### Fused GEMM Epilogue

Use this when the output of GEMM is immediately transformed before writeback:

```python
for i, j in T.Parallel(block_M, block_N):
    C_local[i, j] = epilogue(C_local[i, j])
T.copy(C_local, C[by * block_M, bx * block_N])
```

Typical epilogues include ReLU, sigmoid, scaling, bias, or partial softmax work.

Start from:

- `../examples/quickstart.py`
- `../examples/fusedmoe/`

### Dynamic-Shape GEMM

Use this when one compiled kernel must support multiple runtime sizes:

```python
M = T.dynamic("M")
N = T.dynamic("N")
K = T.dynamic("K")
```

Then use the same tiled GEMM structure with symbolic shapes.

Start from:

- `../examples/dynamic_shape/example_dynamic.py`

### GEMM With Transposed B

Use `transpose_B=True` when B is staged as `(block_N, block_K)` instead of
`(block_K, block_N)`:

```python
T.copy(B[bx * block_N, ko * block_K], B_shared)
T.gemm(A_shared, B_shared, C_local, transpose_B=True)
```

This pattern is common in attention and weight-transposed linear layers.

Start from:

- `../examples/gemm/`
- `../examples/flash_attention/`

## Practical Advice

- Start with the highest-level legal primitive first, especially `T.copy`, `T.gemm`, and `T.Pipelined`.
- Separate semantic questions from performance questions.
- Prefer `T.copy` over `T.async_copy` unless you are intentionally managing overlap.
- Validate against a PyTorch reference before tuning.
- Use `../examples/README.md` to choose a matching operator family after you know the template shape.
- Use the programming guides for semantics and the examples for concrete implementation details.

## Deep Dives

Open these pages next depending on the question:

- language and kernel structure:
  - `programming_guides/language_basics.md`
  - `programming_guides/control_flow.md`
  - `programming_guides/python_compatibility.md`
- instruction semantics:
  - `programming_guides/instructions.md`
  - `programming_guides/software_pipeline.md`
  - `programming_guides/cluster_tma.md`
- performance:
  - `programming_guides/autotuning.md`
  - `tutorials/auto_tuning.md`
- debugging:
  - `tutorials/debug_tools_for_tilelang.md`
  - `tutorials/logging.md`
- internals:
  - `compiler_internals/letstmt_inline.md`
  - `compiler_internals/inject_fence_proxy.md`
  - `compiler_internals/tensor_checks.md`
  - `runtime_internals/stubs.md`
