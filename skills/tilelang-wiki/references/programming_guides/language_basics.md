# Language Basics

This page introduces the TileLang surface used by current working examples:
`@tilelang.jit`, `T.const`, `T.Tensor`, `T.empty`, `T.Kernel`, scoped
allocations, `T.copy`, and the minimal tiled kernel structure that later guides
assume.

The examples use the conventional imports:

```python
import tilelang
import tilelang.language as T
from tilelang import jit
```

## Defining A JIT Kernel

Most examples define a Python function decorated with `@tilelang.jit`. Tensor
arguments are ordinary Python parameters first, then annotated inside the body
with TileLang tensor types. Return tensors are allocated with `T.empty`. The
shorter `@jit` alias is also common in example code.

```python
@tilelang.jit
def add(A, B, block_M: int, block_N: int, dtype=T.float32):
    M, N = T.const("M, N")

    A: T.Tensor((M, N), dtype)
    B: T.Tensor((M, N), dtype)
    C = T.empty((M, N), dtype)

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
        for i, j in T.Parallel(block_M, block_N):
            C[by * block_M + i, bx * block_N + j] = (
                A[by * block_M + i, bx * block_N + j]
                + B[by * block_M + i, bx * block_N + j]
            )

    return C
```

`T.const("M, N")` declares dimensions that TileLang infers from the actual
tensor arguments when the JIT function is compiled or called. Use it for shape
symbols that should be known from inputs.

For dtypes, examples commonly use either TileLang dtypes such as `T.float16`
and `T.float32` or string forms such as `"float16"` and `"float32"`. The
deeper normalization details live in `type_system.md`; for most kernels, pick
one clear style and use it consistently.

Use `T.dynamic(name)` when a symbolic dimension must stay dynamic in the
generated kernel:

```python
@tilelang.jit
def dynamic_matmul(A, B, block_M, block_N, block_K):
    M = T.dynamic("m")
    N = T.dynamic("n")
    K = T.dynamic("k")

    A: T.Tensor((M, K), T.float16)
    B: T.Tensor((K, N), T.float16)
    C = T.empty((M, N), T.float16)
    ...
    return C
```

`T.symbolic(...)` exists as a deprecated alias of `T.dynamic(...)`; prefer
`T.dynamic`.

For annotation-heavy code, `T.dyn["K"]` is a convenient shorthand when you want
to bind a named symbolic dimension directly in a tensor type:

```python
K = T.dyn["K"]

@T.prim_func
def uses_dyn(A: T.Tensor((K,), "float32")):
    N = A.shape[0]
    for i in T.serial(N):
        ...
```

Use `T.dyn[...]` when the symbol mainly lives in the function signature. Use
`T.dynamic(...)` when you need a first-class symbolic variable inside loop
bounds, indexing expressions, or helper calculations.

## Nested `@T.prim_func`

Some examples use `@tilelang.jit` as a factory that returns a nested
`@T.prim_func`. This is useful when the output buffers are explicit arguments or
when compile options such as `out_idx` are attached to the JIT wrapper.

```python
@tilelang.jit(out_idx=[2])
def add_factory(M: int, N: int, dtype=T.float32):
    @T.prim_func
    def add_kernel(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, 32), T.ceildiv(M, 32), threads=128) as (bx, by):
            for i, j in T.Parallel(32, 32):
                row = by * 32 + i
                col = bx * 32 + j
                C[row, col] = A[row, col] + B[row, col]

    return add_kernel
```

Both styles share the same body language.

## Launch Regions With `T.Kernel`

`with T.Kernel(...)` creates a launch region. Positional arguments are grid
extents for `blockIdx.x`, `blockIdx.y`, and `blockIdx.z`; `threads=` describes
the block dimensions.

```python
with T.Kernel(grid_x, grid_y, grid_z, threads=128) as (bx, by, bz):
    ...

with T.Kernel(grid_x, threads=(block_x, block_y)) as bx:
    tx = T.get_thread_binding(0)
    ty = T.get_thread_binding(1)
    ...
```

Most tile-level code uses `T.Parallel` and tile operations instead of direct
thread indices. Direct thread bindings are still useful for lower-level kernels
such as GEMV, reductions, atomics, and vectorized loads.

`T.Kernel(..., is_cpu=True)` is available for CPU-style launch frames. Target
names are handled by the compile/JIT target selection layer; common target
strings include `auto`, `cuda`, `hip`, `metal`, `llvm`, `c`, `webgpu`, and
`cutedsl`. `cluster_dims=` is available for cluster launch on supported GPU
targets.

## Memory Scopes

TileLang exposes several allocation helpers:

```python
A_shared = T.alloc_shared((block_M, block_K), dtype)
B_shared = T.alloc_shared((block_K, block_N), dtype)
C_frag = T.alloc_fragment((block_M, block_N), T.float32)
tmp = T.alloc_local((4,), dtype)
scale = T.alloc_var("float32", init=1.0)
```

Common scopes:

- `T.Tensor(...)` arguments and `T.empty(...)` outputs live in global memory.
- `T.alloc_shared(...)` allocates block-visible shared memory. Its default
  scope is dynamic shared memory (`shared.dyn`).
- `T.alloc_fragment(...)` allocates fragment/register storage that tile
  operators such as `T.gemm` and reductions can use.
- `T.alloc_local(...)` allocates thread-local local storage.
- `T.alloc_var(...)` allocates a single scalar-like local buffer.

The shape of a fragment or shared buffer often reflects the intended
computation structure and should usually be chosen from the operator's
canonical tiled formulation, not invented ad hoc from scalar reasoning.

Use `T.clear(buffer)` or `T.fill(buffer, value)` to initialize accumulators and
temporary buffers.

## Moving Data

`T.copy(src, dst)` is the standard tile copy primitive. It accepts buffer slices,
buffer regions, and buffers; the compiler lowers it according to source and
destination scopes. Treat it as the default movement primitive with synchronous
semantics: after the statement, it is safe to consume `dst`.

```python
T.copy(A[by * block_M, k * block_K], A_shared)
T.copy(B[k * block_K, bx * block_N], B_shared)
T.copy(C_frag, C[by * block_M, bx * block_N])
```

`T.copy` is used for global-to-shared, shared-to-fragment, fragment-to-shared,
and fragment/shared-to-global movement in the examples. For explicit async-copy
work, use the lower-level async copy and wait primitives covered in the
instructions and software-pipeline guides.

Use `T.async_copy(src, dst)` only when you intentionally want manual overlap
between copy and compute. Unlike `T.copy`, it does not imply the wait needed to
consume `dst`: insert the appropriate wait operation before the first read of
the asynchronously produced tile, then rely on the relevant synchronization
rules for the scope you are using.

## Loops

TileLang loop helpers describe both logical iteration and lowering intent:

```python
for i in T.serial(N):
    ...

for i in T.serial(0, N, 2):
    ...

for i, j, k in T.grid(M, N, K):
    ...

for k in T.unroll(4):
    ...

for k in T.vectorized(tile):
    ...

for i, j in T.Parallel(block_M, block_N):
    ...

for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
    ...
```

`T.Parallel` creates a parallel loop nest and is the default for elementwise
tile work. `T.Pipelined` is the common loop form around repeated copy/compute
stages, especially GEMM and attention.

These loop forms describe work over tiles, not a recommendation to design
kernels one scalar at a time.

In practice, first choose the tile-shaped buffers and loop-carried state that
match the operator. Then use `T.Parallel`, `T.reduce_*`, `T.copy`, and
`T.Pipelined` to express the work over those tiles.

In eager JIT code, ordinary Python `range(...)` is also supported and maps to a
serial TileLang loop. Use `T.grid(...)` when you want a compact Cartesian
product loop nest, most often in CPU-style scalar kernels or simple nested
iteration.

For the full loop and branching surface, including `while`, `break`,
`continue`, guard patterns, and thread bindings, see `control_flow.md`. For the
exact semantics of `T.Pipelined`, see `software_pipeline.md`.

## Conditions And Selection

Use Python `if`/`elif`/`else` inside kernels for control flow:

```python
if trans_A:
    T.copy(A[k * block_K, by * block_M], A_shared)
else:
    T.copy(A[by * block_M, k * block_K], A_shared)
```

Use `T.if_then_else(cond, true_value, false_value)` when you need an expression
value inside an assignment:

```python
acc_s[i, j] = T.if_then_else(q_idx >= k_idx, 0, -T.infinity(acc_s.dtype))
```

`T.if_then_else` is also useful for guarding value-producing loads because the
untaken branch is not evaluated.

Python boolean operators are commonly used in examples for compound predicates:

```python
if i_s <= i_t and i_s >= 0:
    ...
```

`T.any_of(buffer_or_region)` and `T.all_of(buffer_or_region)` are reductions
over boolean buffers or regions; they are not variadic predicate builders.

This page keeps conditions brief because `control_flow.md` owns the full
branching and boundary-handling guidance.

## Tile Operators And Math

Useful operations seen throughout the examples include:

```python
T.gemm(A_shared, B_shared, C_frag)
T.reduce_max(scores, scores_max, dim=1)
T.reduce_sum(scores, scores_sum, dim=1)
T.atomic_add(dst, value)
T.exp2(x)
T.log2(x)
T.max(a, b)
T.min(a, b)
T.infinity(dtype)
T.cast(value, dtype)
```

TileLang also exposes target-specific instructions and layout annotations for
advanced kernels. Keep simple kernels at the tile-operator level until
profiling shows a need for lower-level control.

For the full instruction surface, including async copy, TMA, reductions,
atomics, synchronization, and warp intrinsics, see `instructions.md`.

## Calling, Compiling, And Profiling

You can call a JIT function directly:

```python
C = add(A, B, block_M=32, block_N=32, dtype=T.float32)
```

Or compile it explicitly, then call the compiled object:

```python
kernel = matmul.compile(M=1024, N=1024, K=1024, block_M=128, block_N=128, block_K=32)
C = kernel(A, B)

print(kernel.get_kernel_source())
latency_ms = kernel.get_profiler().do_bench()
```

Direct calls are concise for testing. Explicit `.compile(...)` is useful when
you want generated source, profiler handles, or repeated calls with a fixed
configuration.

## Canonical Minimal GEMM Pattern

This is the canonical TileLang teaching skeleton used by the quickstart and
GEMM examples. Other programming guides refer back to this pattern rather than
repeating the full kernel:

```python
@tilelang.jit
def matmul(A, B, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
    M, N, K = T.const("M, N, K")
    A: T.Tensor((M, K), dtype)
    B: T.Tensor((K, N), dtype)
    C = T.empty((M, N), dtype)

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_K), dtype)
        B_shared = T.alloc_shared((block_K, block_N), dtype)
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

        T.clear(C_local)
        for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
            T.copy(A[by * block_M, k * block_K], A_shared)
            T.copy(B[k * block_K, bx * block_N], B_shared)
            T.gemm(A_shared, B_shared, C_local)

        T.copy(C_local, C[by * block_M, bx * block_N])

    return C
```

Use this skeleton as the starting point for tiled matrix operations, then add
post-ops, layout annotations, swizzling, or autotuning only when needed.

Follow-on guides assume this structure:

- `instructions.md` explains what `T.copy`, `T.gemm`, reductions, and sync
  primitives do.
- `control_flow.md` explains loop forms, branch semantics, and boundary guards.
- `software_pipeline.md` explains how `T.Pipelined(..., num_stages=...)` is
  lowered and how manual stage/order annotations work.
- `autotuning.md` explains how to turn this kind of kernel into a tuned kernel
  factory.
