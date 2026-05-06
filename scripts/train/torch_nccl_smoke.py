#!/usr/bin/env python3
"""Small torch.distributed smoke test for multi-node NCCL launches."""

from __future__ import annotations

import argparse
import os
import socket
import time

import torch
import torch.distributed as dist


def parse_sizes(value: str) -> list[int]:
    sizes = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        sizes.append(int(item))
    if not sizes:
        raise argparse.ArgumentTypeError("at least one size in MiB is required")
    return sizes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="nccl", choices=["nccl", "gloo"])
    parser.add_argument("--sizes-mib", type=parse_sizes, default=parse_sizes("16,64,256"))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    args = parser.parse_args()

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    host = socket.gethostname()

    use_cuda = args.backend == "nccl"
    if use_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL backend requested but CUDA is not available")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    dist.init_process_group(backend=args.backend)

    info = {
        "rank": rank,
        "local_rank": local_rank,
        "host": host,
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(device) if use_cuda else "cpu",
    }
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, info)
    if rank == 0:
        print("rank mapping:")
        for row in gathered:
            print(
                f"  rank={row['rank']:03d} local_rank={row['local_rank']} "
                f"host={row['host']} device={row['device']} name={row['cuda_name']}",
                flush=True,
            )
        for name in (
            "MASTER_ADDR",
            "MASTER_PORT",
            "NCCL_SOCKET_IFNAME",
            "NCCL_IB_HCA",
            "NCCL_IB_GID_INDEX",
            "NCCL_IB_DISABLE",
            "NCCL_DEBUG",
        ):
            print(f"{name}={os.environ.get(name, '')}", flush=True)

    dtype = getattr(torch, args.dtype)
    dist.barrier()

    if rank == 0:
        print("all_reduce bandwidth:", flush=True)
        print("  size_mib,max_time_ms,alg_bw_gbps,bus_bw_gbps", flush=True)

    for size_mib in args.sizes_mib:
        element_size = torch.empty((), dtype=dtype).element_size()
        numel = size_mib * 1024 * 1024 // element_size
        tensor = torch.ones(numel, dtype=dtype, device=device)

        for _ in range(args.warmup):
            dist.all_reduce(tensor)
        if use_cuda:
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(args.iters):
            dist.all_reduce(tensor)
        if use_cuda:
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / args.iters

        elapsed_tensor = torch.tensor([elapsed], dtype=torch.float64, device=device)
        dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
        max_elapsed = float(elapsed_tensor.item())
        bytes_per_rank = size_mib * 1024 * 1024
        alg_bw_gbps = bytes_per_rank / max_elapsed * 8 / 1e9
        bus_bw_gbps = alg_bw_gbps * (2 * (world_size - 1) / world_size)

        if rank == 0:
            print(
                f"  {size_mib},{max_elapsed * 1000:.3f},"
                f"{alg_bw_gbps:.3f},{bus_bw_gbps:.3f}",
                flush=True,
            )

    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
