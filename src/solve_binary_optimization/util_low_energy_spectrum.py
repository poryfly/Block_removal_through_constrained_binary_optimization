import torch
from collections import defaultdict
import heapq
import os, glob, heapq
import torch

from tqdm import tqdm
def _torch_load_maybe_mmap(path, map_location="cpu"):
    # mmap=True avoids materializing huge tensors in RAM on newer torch versions
    try:
        return torch.load(path, map_location=map_location, mmap=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _ones_positions_from_vec(v):
    return torch.nonzero(v, as_tuple=True)[0].tolist()


def is_phase_separated_ones_contiguous(v: torch.Tensor) -> bool:
    """
    Phase-separated here means: all 1s form a single contiguous block.
    v: 1D tensor of 0/1 (or bool) on CPU.
    """
    one_pos = (v == 1).nonzero(as_tuple=True)[0]
    if one_pos.numel() == 0:
        return False
    return bool(torch.all(one_pos[1:] - one_pos[:-1] == 1).item())


def iter_global_lowest_n_from_sorted_batches(batch_files, n):

    # """
    # Global lowest-n across many batch files with sorted 'energies'.

    # Yields tuples (energy_float, file_path, local_index, idx_tensor_or_None)

    # Memory: O(#files) heap state; does not allocate proportional to total levels.
    # """
    # print(batch_files)
    heap = []
    handles = []

    # # Load only energies (+ indices handle) per file; vecs loaded later only for selected states
    for fpath in batch_files:
        print(fpath)
        d = _torch_load_maybe_mmap(fpath, map_location="cpu")
        E = d["energies"]
        I = d.get("indices", None)
        if E.numel() == 0:
            continue
        h_id = len(handles)
        handles.append((fpath, E, I))
        heapq.heappush(heap, (float(E[0].item()), h_id, 0))

    produced = 0
    while heap and produced < n:
        e, h_id, j = heapq.heappop(heap)
        fpath, E, I = handles[h_id]
        idx = None if I is None else I[j]
        yield e, fpath, j, idx

        produced += 1
        j2 = j + 1
        if j2 < E.numel():
            heapq.heappush(heap, (float(E[j2].item()), h_id, j2))
