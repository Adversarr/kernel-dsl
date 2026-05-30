# Frequently Asked Questions

## [Question] Layout infer conflict

If you hit an error like:

```text
tvm.error.InternalError: Layout infer conflict between acc_s and acc_s_cast in T.Parallel loop
```

the usual cause is that two `T.gemm(...)` calls expect different layouts for the
same intermediate buffer. In the report from issue
`tile-ai/tilelang#1165`, the first GEMM writes `acc_s` with
`policy=T.GemmWarpPolicy.FullCol`, while the later GEMM path expects a layout
compatible with `FullRow`.

Two known fixes are:

1. Change the casted buffer from fragment memory to shared memory:

```python
acc_s_cast = T.alloc_shared([block_N, block_M * heads], dtype)
```

This works because shared memory layout is more flexible.

2. Keep the casted buffer as a fragment, but align the GEMM layout policy:

```python
T.gemm(
    K_shared,
    Q_shared,
    acc_s,
    transpose_B=True,
    policy=T.GemmWarpPolicy.FullRow,
)
```

This is the preferred fix when valid for your kernel, because it avoids an
extra register-to-shared-memory copy and is typically faster.

Rule of thumb: if a fragment buffer is produced by one `T.gemm(...)` and later
consumed by another GEMM-related path, make sure both operations agree on the
fragment layout policy. If they do not, either align the policies or move the
intermediate through shared memory.

## [Question] `no available layout found` with two reductions

If you hit an error like:

```text
tvm.error.InternalError: Check failed: (min_reg_num < INT64_MAX) is false: no available layout found
```

and your kernel applies multiple reductions to the same fragment buffer, the
cause may be conflicting layout constraints from the reductions themselves. In
issue `tile-ai/tilelang#1714`, the kernel does:

```python
b = T.alloc_fragment([tilesize, nstr], dtype=dtype)
T.reduce_sum(R, b, dim=-1)
T.reduce_sum(R, b, dim=-2)
```

The problem is that `T.reduce_sum(...)` constrains both source and destination
layouts. Reducing into the same fragment buffer `b` with different reduction
dimensions can attach incompatible layout requirements, and layout inference
fails with `no available layout found`.

A simple workaround is to allocate the destination as shared memory instead of
a fragment:

```python
b = T.alloc_shared([tilesize, nstr], dtype=dtype)
```

Rule of thumb: if multiple reductions write into the same intermediate and the
reduction dimensions differ, avoid reusing a fragment buffer for all of them.
Use shared memory for the intermediate, or split the computation so each
reduction gets a layout-compatible destination.

## [Question] Observed `blockDim` does not match `threads=...`

If you write a kernel like:

```python
with T.Kernel(T.ceildiv(seq_len, block_m), heads, batch, threads=128) as (bx, by, bz):
```

but Nsight Compute shows `blockDim=(256, 1, 1)`, the usual reason is that TMA
was enabled and the compiler inserted an extra producer warp group.

In issue `tile-ai/tilelang#1523`, the TileLang maintainers explained that when
the compiler detects TMA copy usage, it may launch an extra warp group to issue
those TMA operations. That means the runtime CUDA block size can be larger than
the `threads=` value you passed in `T.Kernel(...)`.

So in this situation, the extra threads do not necessarily mean your
`threads=128` argument was ignored. Instead, TileLang augmented the launch
configuration to support TMA.

Rule of thumb: if profilers show more threads than expected, check whether the
kernel or pass configuration allowed TMA. When TMA is active, TileLang may add
producer warps on top of your requested worker threads.
