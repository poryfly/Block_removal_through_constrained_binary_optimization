import torch

@torch.no_grad()
def binom(n, k, device=None, dtype=torch.float64):
    n = torch.as_tensor(n, device=device, dtype=dtype)
    k = torch.as_tensor(k, device=device, dtype=dtype)
    return torch.exp(
        torch.lgamma(n + 1)
        - torch.lgamma(k + 1)
        - torch.lgamma(n - k + 1)
    )
