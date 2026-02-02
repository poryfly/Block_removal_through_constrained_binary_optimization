
import torch
import time
import os
import math
import argparse
from itertools import combinations

import sys
sys.path.insert(0, os.getcwd())

from util import binom

@torch.no_grad()
def bitset_batches_exact_zeros(
    n_bits: int,
    exact_zeros: int,
    batch_size: int,
    device=None,
):
    """
    Generate all bitstrings of length n_bits that contain exactly `exact_zeros`
    zeros, in batches, without enumerating all 2**n_bits states.

    Yields per batch:
        indices: (B,)        int64 tensor with integer indices (0..2**n_bits-1)
        vecs:    (B, n_bits) float64 tensor of 0/1 bits (column 0 = MSB)
    """
    if n_bits <= 0:
        raise ValueError("n_bits must be positive")
    if not (0 <= exact_zeros <= n_bits):
        raise ValueError("exact_zeros must be in [0, n_bits]")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

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

    # Use itertools.combinations to generate all k-subsets of {0, …, n-1}
    for comb in combinations(range(n), k):
        batch_positions.append(comb)
        if len(batch_positions) >= batch_size:
            out = flush()
            if out is not None:
                yield out

    # Flush last partial batch
    out = flush()
    if out is not None:
        yield out


def energy(vec, H):
    """
    vec: (B, n_bits), float64
    H:   (n_bits, n_bits), float64
    Returns: (B,) energies, one per vector in the batch.
    """
    # Batched quadratic form: v^T H v for each v in vec
    return torch.einsum('bi,ij,bj->b', vec, H, vec)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-A_directory", type=str, default="Amatrices/Llama-3-8B-Instruct_n_samples_2048/")
    parser.add_argument("-ndel", type=int, default=8)
    parser.add_argument("-batch_size", type=int, default=1024*1024*8)
    args = parser.parse_args()

    device = "cuda:0"
    A = torch.load(f"{args.A_directory}/A.pt")
    n=A.shape[0]
    A = A.to(torch.double)
    
    H = torch.transpose(A, 0, 1) @ A / n
    print(H)
    print("A.shape=", A.shape, "H.shape=", H.shape)
    H = H.to(device)

    n_bits = H.shape[0]
    exact_ones = args.ndel
    exact_zeros = n_bits - exact_ones
    batch_size = args.batch_size
    num_combinations = binom(n_bits, exact_ones).item()
    num_batches = math.ceil(num_combinations / batch_size)
    out_dir = f"{args.A_directory}/energies_del{exact_ones}"
    os.makedirs(out_dir, exist_ok=True)
    print(f"Batch size: {batch_size}, Number of batches: {num_batches}")

    global_min_E = None
    global_min_vec = None
    global_min_index = None

    tini = time.time()
    i = 0
    for indices, vecs in bitset_batches_exact_zeros(n_bits, exact_zeros, batch_size, device=device):
        E = energy(vecs, H)

        sorted_E, sorted_idx = torch.sort(E)
        sorted_vecs = vecs[sorted_idx].to(torch.int64)
        sorted_indices = indices[sorted_idx].to(torch.int64)

        batch_min_E = sorted_E[0]
        batch_min_vec = sorted_vecs[0]
        batch_min_index = sorted_indices[0]

        if (global_min_E is None) or (batch_min_E < global_min_E):
            global_min_E = batch_min_E.detach()
            global_min_vec = batch_min_vec.detach()
            global_min_index = batch_min_index.detach()

        zero_pos = (batch_min_vec == 0).nonzero(as_tuple=True)[0]
        one_pos  = (batch_min_vec == 1).nonzero(as_tuple=True)[0]

        print(f"Batch {i+1}/{num_batches}")
        print(f"min E={batch_min_E.item():.6f}")
        print(f"vec={batch_min_vec.detach().cpu().numpy()}")
        print(f"del={one_pos.detach().cpu().numpy()}")

        torch.save(
            {
                "energies": sorted_E.detach().cpu(),
                "vecs": sorted_vecs.detach().cpu(),
                "indices": sorted_indices.detach().cpu(),
            },
         os.path.join(out_dir, f"results_del{exact_ones}_bs={batch_size}_b={i}_{num_batches}.pt"),
        )

        i += 1
        tit = time.time()
        print(f"Time per batch: {(tit - tini)/i:.6f} seconds")
    
    min_E     = global_min_E
    min_vec   = global_min_vec
    min_index = global_min_index
    one_pos   = (min_vec == 1).nonzero(as_tuple=True)[0]

    tfin = time.time()
    print(f"Total time: {tfin - tini:.6f} seconds")
    print(f"Global min energy E={min_E.item():.6f}")
    print(f"vec={min_vec.detach().cpu().numpy()}, del={one_pos.detach().cpu().numpy()}")
