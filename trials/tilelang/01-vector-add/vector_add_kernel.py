import argparse

import tilelang
import tilelang.language as T
import torch


DEFAULT_THREADS = 128
DEFAULT_ELEMENTS_PER_THREAD = 8


@tilelang.jit
def vector_add(
    lhs,
    rhs,
    threads: int = DEFAULT_THREADS,
    elements_per_thread: int = DEFAULT_ELEMENTS_PER_THREAD,
    dtype=T.float32,
):
    N = T.const("N")

    lhs: T.Tensor((N,), dtype)
    rhs: T.Tensor((N,), dtype)
    out = T.empty((N,), dtype)

    tile = threads * elements_per_thread
    with T.Kernel(T.ceildiv(N, tile), threads=threads) as bx:
        lhs_tile = T.alloc_fragment((tile,), dtype)
        rhs_tile = T.alloc_fragment((tile,), dtype)
        out_tile = T.alloc_fragment((tile,), dtype)

        start = bx * tile
        stop = start + tile

        T.copy(lhs[start:stop], lhs_tile)
        T.copy(rhs[start:stop], rhs_tile)

        for tid, lane in T.Parallel(threads, elements_per_thread):
            idx = tid * elements_per_thread + lane
            out_tile[idx] = lhs_tile[idx] + rhs_tile[idx]

        T.copy(out_tile, out[start:stop])

    return out


def vector_add_reference(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    return lhs + rhs


def make_inputs(
    length: int,
    dtype: torch.dtype = torch.float32,
    device: str = "cuda",
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    lhs = torch.randn(length, device=device, dtype=dtype, generator=generator)
    rhs = torch.randn(length, device=device, dtype=dtype, generator=generator)
    return lhs, rhs


def run_demo(
    length: int = 1 << 20,
    dtype: torch.dtype = torch.float32,
    threads: int = DEFAULT_THREADS,
    elements_per_thread: int = DEFAULT_ELEMENTS_PER_THREAD,
) -> torch.Tensor:
    lhs, rhs = make_inputs(length=length, dtype=dtype)
    return vector_add(
        lhs,
        rhs,
        threads=threads,
        elements_per_thread=elements_per_thread,
        dtype=getattr(T, str(dtype).split(".")[-1]),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the TileLang vector add kernel.")
    parser.add_argument("--length", type=int, default=1 << 20)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument(
        "--elements-per-thread",
        type=int,
        default=DEFAULT_ELEMENTS_PER_THREAD,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    output = run_demo(
        length=args.length,
        threads=args.threads,
        elements_per_thread=args.elements_per_thread,
    )
    print(output[:8])
