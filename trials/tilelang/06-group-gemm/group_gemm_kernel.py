import itertools
from collections.abc import Sequence

import tilelang as tl
import tilelang.language as T
import torch
from tilelang.autotuner import AutoTuner, set_autotune_inputs


DEFAULT_BLOCK_M = 128
DEFAULT_BLOCK_N = 128
DEFAULT_BLOCK_K = 32
DEFAULT_THREADS = 128
DEFAULT_NUM_STAGES = 3
DEFAULT_ENABLE_SWIZZLE = False

AUTOTUNE_BLOCK_M_VALUES = [64, 128]
AUTOTUNE_BLOCK_N_VALUES = [64, 128, 256]
AUTOTUNE_BLOCK_K_VALUES = [32, 64]
AUTOTUNE_THREADS_VALUES = [128, 256]
AUTOTUNE_NUM_STAGES_VALUES = [2, 3]

PACK_ALIGNMENT_M = max(AUTOTUNE_BLOCK_M_VALUES)
PACK_ALIGNMENT_N = max(AUTOTUNE_BLOCK_N_VALUES)
PACK_ALIGNMENT_K = max(AUTOTUNE_BLOCK_K_VALUES)


def torch_dtype_to_tilelang_dtype(dtype: torch.dtype):
    mapping = {
        torch.float16: T.float16,
        torch.float32: T.float32,
        torch.bfloat16: T.bfloat16,
    }
    try:
        return mapping[dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported torch dtype: {dtype}") from exc


def round_up(value: int, multiple: int) -> int:
    if value <= 0:
        raise ValueError(f"value must be positive, got {value}")
    return ((value + multiple - 1) // multiple) * multiple


def validate_group_gemm_inputs(
    group_lhs: Sequence[torch.Tensor],
    group_rhs: Sequence[torch.Tensor],
) -> None:
    if len(group_lhs) == 0:
        raise ValueError("group_lhs must not be empty")
    if len(group_lhs) != len(group_rhs):
        raise ValueError("group_lhs and group_rhs must have the same length")

    first_dtype = group_lhs[0].dtype
    first_device = group_lhs[0].device
    if first_device.type != "cuda":
        raise ValueError(f"group GEMM requires CUDA tensors, got {first_device}")

    for index, (lhs, rhs) in enumerate(zip(group_lhs, group_rhs)):
        if lhs.ndim != 2 or rhs.ndim != 2:
            raise ValueError(f"group entry {index} must be rank-2, got {lhs.ndim=} and {rhs.ndim=}")
        if lhs.device != first_device or rhs.device != first_device:
            raise ValueError("all tensors must live on the same CUDA device")
        if lhs.dtype != first_dtype or rhs.dtype != first_dtype:
            raise ValueError("all tensors must use the same dtype")
        if lhs.shape[1] != rhs.shape[0]:
            raise ValueError(
                f"group entry {index} has incompatible shapes: {tuple(lhs.shape)} x {tuple(rhs.shape)}"
            )
        if lhs.stride(1) != 1 or rhs.stride(1) != 1:
            raise ValueError(
                "this trial expects row-major contiguous inner dimensions for both operands"
            )


def pack_group_gemm_inputs(
    group_lhs: Sequence[torch.Tensor],
    group_rhs: Sequence[torch.Tensor],
    *,
    align_m: int = PACK_ALIGNMENT_M,
    align_n: int = PACK_ALIGNMENT_N,
    align_k: int = PACK_ALIGNMENT_K,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    validate_group_gemm_inputs(group_lhs, group_rhs)

    max_m = max(lhs.shape[0] for lhs in group_lhs)
    max_n = max(rhs.shape[1] for rhs in group_rhs)
    max_k = max(lhs.shape[1] for lhs in group_lhs)
    padded_m = round_up(max_m, align_m)
    padded_n = round_up(max_n, align_n)
    padded_k = round_up(max_k, align_k)

    group_size = len(group_lhs)
    device = group_lhs[0].device
    dtype = group_lhs[0].dtype

    packed_lhs = torch.zeros((group_size, padded_m, padded_k), device=device, dtype=dtype)
    packed_rhs = torch.zeros((group_size, padded_k, padded_n), device=device, dtype=dtype)
    group_sizes = torch.empty((group_size, 3), device=device, dtype=torch.int32)

    for index, (lhs, rhs) in enumerate(zip(group_lhs, group_rhs)):
        rows, inner = lhs.shape
        _, cols = rhs.shape
        packed_lhs[index, :rows, :inner].copy_(lhs)
        packed_rhs[index, :inner, :cols].copy_(rhs)
        group_sizes[index, 0] = rows
        group_sizes[index, 1] = cols
        group_sizes[index, 2] = inner

    return packed_lhs, packed_rhs, group_sizes


def unpack_group_gemm_output(
    packed_out: torch.Tensor,
    group_sizes: torch.Tensor,
) -> list[torch.Tensor]:
    group_out = []
    for index in range(packed_out.shape[0]):
        rows = int(group_sizes[index, 0].item())
        cols = int(group_sizes[index, 1].item())
        group_out.append(packed_out[index, :rows, :cols].clone())
    return group_out


def group_gemm_reference(
    group_lhs: Sequence[torch.Tensor],
    group_rhs: Sequence[torch.Tensor],
) -> list[torch.Tensor]:
    validate_group_gemm_inputs(group_lhs, group_rhs)
    return [lhs @ rhs for lhs, rhs in zip(group_lhs, group_rhs)]


def packed_group_gemm_reference(
    packed_lhs: torch.Tensor,
    packed_rhs: torch.Tensor,
    group_sizes: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.bmm(packed_lhs, packed_rhs)


def effective_group_gemm_flops(group_sizes: torch.Tensor) -> int:
    total_flops = 0
    for index in range(group_sizes.shape[0]):
        rows = int(group_sizes[index, 0].item())
        cols = int(group_sizes[index, 1].item())
        inner = int(group_sizes[index, 2].item())
        total_flops += 2 * rows * cols * inner
    return total_flops


def make_group_gemm_prim_func(
    group_size: int,
    padded_m: int,
    padded_n: int,
    padded_k: int,
    block_M: int,
    block_N: int,
    block_K: int,
    threads: int,
    num_stages: int,
    enable_swizzle: bool,
    dtype=T.float16,
    accum_dtype=T.float32,
):
    if padded_m % block_M != 0:
        raise ValueError(f"padded_m={padded_m} must be divisible by block_M={block_M}")
    if padded_n % block_N != 0:
        raise ValueError(f"padded_n={padded_n} must be divisible by block_N={block_N}")
    if padded_k % block_K != 0:
        raise ValueError(f"padded_k={padded_k} must be divisible by block_K={block_K}")

    @T.prim_func
    def main(
        lhs: T.Tensor((group_size, padded_m, padded_k), dtype),
        rhs: T.Tensor((group_size, padded_k, padded_n), dtype),
        group_sizes: T.Tensor((group_size, 3), "int32"),
        out: T.Tensor((group_size, padded_m, padded_n), dtype),
    ):
        with T.Kernel(
            T.ceildiv(padded_n, block_N),
            T.ceildiv(padded_m, block_M),
            group_size,
            threads=threads,
        ) as (bx, by, bz):
            rows = group_sizes[bz, 0]
            cols = group_sizes[bz, 1]
            lhs_shared = T.alloc_shared((block_M, block_K), dtype)
            rhs_shared = T.alloc_shared((block_K, block_N), dtype)
            acc = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(acc)
            if by * block_M < rows:
                if bx * block_N < cols:
                    T.use_swizzle(panel_size=10, enable=enable_swizzle)
                    for ko in T.Pipelined(padded_k // block_K, num_stages=num_stages):
                        T.copy(
                            lhs[
                                bz,
                                by * block_M : (by + 1) * block_M,
                                ko * block_K : (ko + 1) * block_K,
                            ],
                            lhs_shared,
                        )
                        T.copy(
                            rhs[
                                bz,
                                ko * block_K : (ko + 1) * block_K,
                                bx * block_N : (bx + 1) * block_N,
                            ],
                            rhs_shared,
                        )
                        T.gemm(lhs_shared, rhs_shared, acc)

            T.copy(
                acc,
                out[
                    bz,
                    by * block_M : (by + 1) * block_M,
                    bx * block_N : (bx + 1) * block_N,
                ],
            )

    return main


def get_autotune_configs() -> list[dict[str, int | bool]]:
    iter_params = dict(
        block_M=AUTOTUNE_BLOCK_M_VALUES,
        block_N=AUTOTUNE_BLOCK_N_VALUES,
        block_K=AUTOTUNE_BLOCK_K_VALUES,
        num_stages=AUTOTUNE_NUM_STAGES_VALUES,
        threads=AUTOTUNE_THREADS_VALUES,
        enable_swizzle=[False, True],
    )
    return [
        dict(zip(iter_params.keys(), values))
        for values in itertools.product(*iter_params.values())
    ]


def make_group_gemm_jit_kernel(
    packed_lhs: torch.Tensor,
    packed_rhs: torch.Tensor,
    *,
    block_M: int = DEFAULT_BLOCK_M,
    block_N: int = DEFAULT_BLOCK_N,
    block_K: int = DEFAULT_BLOCK_K,
    threads: int = DEFAULT_THREADS,
    num_stages: int = DEFAULT_NUM_STAGES,
    enable_swizzle: bool = DEFAULT_ENABLE_SWIZZLE,
):
    return group_gemm_with_config(
        packed_lhs.shape[0],
        packed_lhs.shape[1],
        packed_rhs.shape[2],
        packed_lhs.shape[2],
        block_M=block_M,
        block_N=block_N,
        block_K=block_K,
        threads=threads,
        num_stages=num_stages,
        enable_swizzle=enable_swizzle,
        dtype=torch_dtype_to_tilelang_dtype(packed_lhs.dtype),
    )


def autotune_group_gemm(
    packed_lhs: torch.Tensor,
    packed_rhs: torch.Tensor,
    group_sizes: torch.Tensor,
    *,
    warmup: int = 3,
    rep: int = 20,
    profile_backend: str = "event",
):
    group_size, padded_m, padded_k = packed_lhs.shape
    padded_n = packed_rhs.shape[2]
    dtype = torch_dtype_to_tilelang_dtype(packed_lhs.dtype)

    def kernel(
        block_M=None,
        block_N=None,
        block_K=None,
        num_stages=None,
        threads=None,
        enable_swizzle=None,
    ):
        return make_group_gemm_prim_func(
            group_size=group_size,
            padded_m=padded_m,
            padded_n=padded_n,
            padded_k=padded_k,
            block_M=block_M,
            block_N=block_N,
            block_K=block_K,
            threads=threads,
            num_stages=num_stages,
            enable_swizzle=enable_swizzle,
            dtype=dtype,
        )

    with set_autotune_inputs(packed_lhs, packed_rhs, group_sizes):
        autotuner = (
            AutoTuner.from_kernel(kernel=kernel, configs=get_autotune_configs())
            .set_compile_args(
                out_idx=[-1],
                target="auto",
            )
            .set_profile_args(
                supply_type=tl.TensorSupplyType.Normal,
                ref_prog=packed_group_gemm_reference,
                skip_check=False,
                rtol=2e-2,
                atol=2e-2,
                backend=profile_backend,
            )
        )
    return autotuner.run(warmup=warmup, rep=rep)


@tl.jit(out_idx=[-1])
def group_gemm_with_config(
    group_size: int,
    padded_m: int,
    padded_n: int,
    padded_k: int,
    block_M: int = DEFAULT_BLOCK_M,
    block_N: int = DEFAULT_BLOCK_N,
    block_K: int = DEFAULT_BLOCK_K,
    threads: int = DEFAULT_THREADS,
    num_stages: int = DEFAULT_NUM_STAGES,
    enable_swizzle: bool = DEFAULT_ENABLE_SWIZZLE,
    dtype=T.float16,
    accum_dtype=T.float32,
):
    return make_group_gemm_prim_func(
        group_size=group_size,
        padded_m=padded_m,
        padded_n=padded_n,
        padded_k=padded_k,
        block_M=block_M,
        block_N=block_N,
        block_K=block_K,
        threads=threads,
        num_stages=num_stages,
        enable_swizzle=enable_swizzle,
        dtype=dtype,
        accum_dtype=accum_dtype,
    )


def run_group_gemm(
    group_lhs: Sequence[torch.Tensor],
    group_rhs: Sequence[torch.Tensor],
    *,
    block_M: int = DEFAULT_BLOCK_M,
    block_N: int = DEFAULT_BLOCK_N,
    block_K: int = DEFAULT_BLOCK_K,
    threads: int = DEFAULT_THREADS,
    num_stages: int = DEFAULT_NUM_STAGES,
    enable_swizzle: bool = DEFAULT_ENABLE_SWIZZLE,
) -> list[torch.Tensor]:
    packed_lhs, packed_rhs, group_sizes = pack_group_gemm_inputs(group_lhs, group_rhs)
    kernel = make_group_gemm_jit_kernel(
        packed_lhs,
        packed_rhs,
        block_M=block_M,
        block_N=block_N,
        block_K=block_K,
        threads=threads,
        num_stages=num_stages,
        enable_swizzle=enable_swizzle,
    )
    packed_out = kernel(packed_lhs, packed_rhs, group_sizes)
    return unpack_group_gemm_output(packed_out, group_sizes)


def make_demo_inputs(
    shapes: Sequence[tuple[int, int, int]] = (
        (1024, 1024, 1024),
        (768, 768, 1024),
        (512, 512, 1024),
        (256, 256, 1024),
    ),
    *,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
    seed: int = 0,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    torch.manual_seed(seed)

    group_lhs = []
    group_rhs = []
    for rows, cols, inner in shapes:
        lhs = torch.randn(rows, inner, device=device, dtype=dtype)
        rhs = torch.randn(inner, cols, device=device, dtype=dtype)
        group_lhs.append(lhs)
        group_rhs.append(rhs)
    return group_lhs, group_rhs
