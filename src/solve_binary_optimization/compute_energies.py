
import torch
import torch.multiprocessing as mp
import time
import os
import math
import argparse
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor

import sys
sys.path.insert(0, os.getcwd())

from util import binom


def combination_unrank(n, k, rank):
    """
    Return the `rank`-th combination (0-indexed) from
    itertools.combinations(range(n), k) in lexicographic order.

    Uses the greedy combinatorial unranking algorithm:
    for each position i, find the smallest element j such that
    C(n - 1 - j, k - 1 - i) > remaining_rank.

    Time complexity: O(n * k), effectively O(1) for typical n, k.

    Args:
        n: total number of elements (range is 0..n-1)
        k: number of elements to choose
        rank: 0-based index in the lexicographic ordering

    Returns:
        tuple of k integers, identical to list(combinations(range(n), k))[rank]

    Raises:
        ValueError: if rank < 0 or rank >= C(n, k)
    """
    total = math.comb(n, k)
    if rank < 0 or rank >= total:
        raise ValueError(
            f"rank must be in [0, {total}), got {rank}"
        )
    if k == 0:
        return ()
    result = []
    start = 0
    for i in range(k):
        for j in range(start, n):
            # Number of combinations with j at position i:
            # choose remaining (k - 1 - i) elements from {j+1, ..., n-1}
            c = math.comb(n - 1 - j, k - 1 - i)
            if c > rank:
                result.append(j)
                start = j + 1
                break
            rank -= c
    return tuple(result)


def combinations_from_range(n, k, start_rank, end_rank):
    """
    Generate combinations from rank `start_rank` (inclusive) to `end_rank`
    (exclusive), equivalent to:
        islice(itertools.combinations(range(n), k), start_rank, end_rank)

    Uses `combination_unrank` to jump directly to the start position in
    O(n * k) time, then iterates via the standard "next combination"
    algorithm (find rightmost incrementable position).

    Args:
        n: total number of elements (range is 0..n-1)
        k: number of elements to choose
        start_rank: 0-based starting rank (inclusive)
        end_rank: 0-based ending rank (exclusive)

    Yields:
        tuples of k integers, identical to itertools.combinations output
    """
    if start_rank >= end_rank:
        return

    comb = list(combination_unrank(n, k, start_rank))
    for _ in range(start_rank, end_rank):
        yield tuple(comb)
        # Standard "next combination in lex order":
        # Find rightmost position i where comb[i] can be incremented
        # (i.e., comb[i] < n - k + i), increment it, and reset all
        # subsequent positions to consecutive values.
        i = k - 1
        while i >= 0 and comb[i] == n - k + i:
            i -= 1
        if i < 0:
            # This was the last combination; stop.
            return
        comb[i] += 1
        for j in range(i + 1, k):
            comb[j] = comb[j - 1] + 1


@torch.no_grad()
def bitset_batches_from_iter(
    comb_iter,
    n_bits: int,
    exact_zeros: int,
    batch_size: int,
    device=None,
):
    """
    Generate bitstrings from a custom combination iterator, in batches.
    Each yielded batch contains only combinations from the provided iterator,
    allowing each GPU worker to supply its own sliced range via itertools.islice.

    Yields per batch:
        indices: (B,)        int64 tensor with integer indices (0..2**n_bits-1)
        vecs:    (B, n_bits) float64 tensor of 0/1 bits (column 0 = MSB)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    n = n_bits
    k = exact_zeros

    # Trivial cases
    if k == 0:
        vecs = torch.ones((1, n), dtype=torch.float64, device=device)
        idx_val = (1 << n) - 1
        indices = torch.tensor([idx_val], dtype=torch.int64, device=device)
        yield indices, vecs
        return

    if k == n:
        vecs = torch.zeros((1, n), dtype=torch.float64, device=device)
        indices = torch.zeros((1,), dtype=torch.int64, device=device)
        yield indices, vecs
        return

    # Precompute powers of two for index computation (MSB at position 0)
    pow2 = 1 << torch.arange(n - 1, -1, -1, dtype=torch.int64, device=device)
    full_index = (1 << n) - 1

    batch_positions = []

    def flush():
        nonlocal batch_positions
        if not batch_positions:
            return None

        # (B, k) tensor of zero positions
        pos_tensor = torch.tensor(batch_positions, dtype=torch.long, device=device)
        B = pos_tensor.shape[0]

        # Start from all ones; set chosen positions to zero
        vecs = torch.ones((B, n), dtype=torch.float64, device=device)
        vecs.scatter_(1, pos_tensor, 0.0)

        # integer indices: all-ones index minus contributions of zeroed bits
        zero_contrib = pow2[pos_tensor]            # (B, k)
        indices = full_index - zero_contrib.sum(dim=1)

        batch_positions = []
        return indices, vecs

    for comb in comb_iter:
        batch_positions.append(comb)
        if len(batch_positions) >= batch_size:
            out = flush()
            if out is not None:
                yield out

    # Flush last partial batch
    out = flush()
    if out is not None:
        yield out


@torch.no_grad()
def bitset_batches_exact_zeros(
    n_bits: int,
    exact_zeros: int,
    batch_size: int,
    device=None,
):
    """
    Original interface: generates all bitstrings with exactly `exact_zeros` zeros.
    Delegates to bitset_batches_from_iter with the full combination iterator.
    """
    comb_iter = combinations(range(n_bits), exact_zeros)
    yield from bitset_batches_from_iter(comb_iter, n_bits, exact_zeros, batch_size, device)


def energy(vecs, H):
    """
    vec: (B, n_bits), float64
    H:   (n_bits, n_bits), float64
    Returns: (B,) energies, one per vector in the batch.
    Optimized: matmul instead of einsum for better cuBLAS utilization.
    v^T H v = sum_i (Hv)_i * v_i  where Hv = v @ H
    """
    Hv = vecs @ H               # (B, n_bits) @ (n_bits, n_bits) -> (B, n_bits)
    return (Hv * vecs).sum(dim=1)  # element-wise multiply + sum -> (B,)


def worker(rank, world_size, args):
    """
    GPU worker: processes its share of the combination space.

    The total C(n_bits, exact_ones) combinations are split across
    world_size GPUs by rank. Each worker uses combination_unrank
    to jump directly to its assigned range in O(n*k) time, then
    iterates via the standard next-combination algorithm.
    """
    gpu_id = rank
    device = f"cuda:{gpu_id}"

    # Load A and compute H on this GPU
    A = torch.load(f"{args.A_directory}/A.pt")
    n = A.shape[0]
    A = A.to(torch.double)
    H = (A.T @ A / n).to(device)

    if rank == 0:
        print(H)
        print(f"A.shape={A.shape}, H.shape={H.shape}")

    n_bits = H.shape[0]
    exact_ones = args.ndel
    exact_zeros = n_bits - exact_ones
    batch_size = args.batch_size
    top_k = args.top_k

    # Compute this worker's share of combinations
    total_combs = int(binom(n_bits, exact_ones).item())
    start_rank = rank * total_combs // world_size
    end_rank = (rank + 1) * total_combs // world_size
    my_combs = end_rank - start_rank

    num_my_batches = math.ceil(my_combs / batch_size) if my_combs > 0 else 0
    num_total_batches = math.ceil(total_combs / batch_size)

    out_dir = f"{args.output_directory}/energies_del{exact_ones}"
    os.makedirs(out_dir, exist_ok=True)

    print(f"[GPU {gpu_id}] rank={rank}, combinations [{start_rank}, {end_rank}), "
          f"my_combs={my_combs}, my_batches={num_my_batches}")

    # Create the combination iterator for this worker's range
    # combinations_from_range uses O(n*k) unranking to jump directly
    # to start_rank, eliminating the O(start_rank) islice linear scan.
    comb_iter = combinations_from_range(n_bits, exact_zeros, start_rank, end_rank)

    global_min_E = None
    global_min_vec = None
    global_min_index = None

    # Async I/O: overlap disk writes with GPU computation
    save_executor = ThreadPoolExecutor(max_workers=1)
    save_future = None

    tini = time.time()
    i = 0

    for indices, vecs in bitset_batches_from_iter(
        comb_iter, n_bits, exact_zeros, batch_size, device=device
    ):
        E = energy(vecs, H)

        # topk instead of full sort: O(B log K) vs O(B log B), K << B
        # Also dramatically reduces I/O: save only top-K instead of all B
        actual_k = min(top_k, E.shape[0])
        topk_E, topk_idx = torch.topk(E, k=actual_k, largest=False)
        topk_vecs = vecs[topk_idx].to(torch.int64)
        topk_indices = indices[topk_idx].to(torch.int64)

        batch_min_E = topk_E[0]
        batch_min_vec = topk_vecs[0]
        batch_min_index = topk_indices[0]

        if (global_min_E is None) or (batch_min_E < global_min_E):
            global_min_E = batch_min_E.detach()
            global_min_vec = batch_min_vec.detach()
            global_min_index = batch_min_index.detach()

        one_pos = (batch_min_vec == 1).nonzero(as_tuple=True)[0]

        print(f"[GPU {gpu_id}] Batch {i+1}/{num_my_batches} "
              f"(global ~{start_rank // batch_size + i + 1}/{num_total_batches})")
        print(f"  min E={batch_min_E.item():.6f}")
        print(f"  vec={batch_min_vec.detach().cpu().numpy()}")
        print(f"  del={one_pos.detach().cpu().numpy()}")

        # Wait for previous async save to complete
        if save_future is not None:
            save_future.result()

        # Transfer top-K results to CPU and save asynchronously
        result = {
            "energies": topk_E.detach().cpu(),
            "vecs": topk_vecs.detach().cpu(),
            "indices": topk_indices.detach().cpu(),
        }
        save_future = save_executor.submit(
            torch.save,
            result,
            os.path.join(out_dir,
                         f"results_del{exact_ones}_bs={batch_size}"
                         f"_b={i}_gpu{rank}.pt"),
        )

        i += 1
        elapsed = time.time() - tini
        print(f"[GPU {gpu_id}] Avg time per batch: {elapsed / i:.6f} seconds, "
              f"elapsed: {elapsed:.1f}s")

    # Wait for last async save to complete
    if save_future is not None:
        save_future.result()
    save_executor.shutdown(wait=True)

    # Save this GPU's local minimum for the main process to merge
    if global_min_E is not None:
        one_pos = (global_min_vec == 1).nonzero(as_tuple=True)[0]
        tfin = time.time()
        print(f"[GPU {gpu_id}] Total time: {tfin - tini:.6f} seconds")
        print(f"[GPU {gpu_id}] Local min energy E={global_min_E.item():.6f}")
        print(f"[GPU {gpu_id}] vec={global_min_vec.detach().cpu().numpy()}, "
              f"del={one_pos.detach().cpu().numpy()}")

        torch.save({
            "min_E": global_min_E.detach().cpu(),
            "min_vec": global_min_vec.detach().cpu(),
            "min_index": global_min_index.detach().cpu(),
        }, os.path.join(out_dir, f"_gpu{rank}_min.pt"))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-A_directory", type=str,
                        default="Amatrices/Llama-3-8B-Instruct_n_samples_2048/")
    parser.add_argument("-output_directory", type=str,
                        default="/data2/work/Block_removal_through_constrained_binary_optimization/"
                                "Amatrices/Qwen3-8B_n_samples_2048_think/")
    parser.add_argument("-ndel", type=int, default=8)
    parser.add_argument("-batch_size", type=int, default=1024*1024*8)
    parser.add_argument("-top_k", type=int, default=100,
                        help="Number of lowest-energy states to save per batch (default: 100). "
                             "Global top-K is guaranteed when per-batch top-K >= desired global K.")
    parser.add_argument("-num_gpus", type=int, default=None,
                        help="Number of GPUs to use (default: all available)")
    parser.add_argument("-single_gpu", type=int, default=None,
                        help="If set, run on a single GPU with this ID "
                             "(for debugging / backward compatibility)")
    args = parser.parse_args()

    if args.single_gpu is not None:
        # Single GPU mode (backward compatible)
        worker(args.single_gpu, 1, args)
    else:
        # Multi-GPU mode
        num_gpus = args.num_gpus or torch.cuda.device_count()
        print(f"Launching {num_gpus} GPU workers...")
        t_total_start = time.time()

        mp.spawn(
            worker,
            args=(num_gpus, args),
            nprocs=num_gpus,
            join=True,
        )

        t_total_end = time.time()
        print(f"\nAll workers finished. Wall time: {t_total_end - t_total_start:.2f} seconds")

        # Merge local minima from all GPUs to find the global minimum
        out_dir = f"{args.output_directory}/energies_del{args.ndel}"
        global_min = None
        for r in range(num_gpus):
            fpath = os.path.join(out_dir, f"_gpu{r}_min.pt")
            if os.path.exists(fpath):
                data = torch.load(fpath, map_location="cpu")
                if global_min is None or data["min_E"] < global_min["min_E"]:
                    global_min = data
                os.remove(fpath)  # clean up temp file

        if global_min is not None:
            one_pos = (global_min["min_vec"] == 1).nonzero(as_tuple=True)[0]
            print(f"\n{'='*60}")
            print(f"GLOBAL MINIMUM")
            print(f"  Energy E = {global_min['min_E'].item():.6f}")
            print(f"  vec      = {global_min['min_vec'].numpy()}")
            print(f"  del      = {one_pos.numpy()}")
            print(f"{'='*60}")
