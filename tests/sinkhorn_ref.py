"""PyTorch reference Sinkhorn for testing.

Pure-PyTorch implementations (no Triton) used as ground truth in kernel tests.
"""
from __future__ import annotations

import math

import torch
from einops import einsum


def sinkhorn_log_ref(logits, num_iters=20, tau=1.0):
    """Pure-PyTorch log-domain Sinkhorn (no Triton needed).

    Uses separate row/col marginals so rectangular matrices converge properly.
    Output is scaled by rows so row sums = 1 (suitable for routing matmuls).
    """
    rows, cols = logits.shape[-2], logits.shape[-1]
    Z = logits / tau

    u = torch.zeros(*logits.shape[:-1], device=Z.device, dtype=Z.dtype)
    v = torch.zeros(*logits.shape[:-2], cols, device=Z.device, dtype=Z.dtype)

    log_row_marginal = -math.log(rows)
    log_col_marginal = -math.log(cols)

    for _ in range(num_iters):
        u = log_row_marginal - torch.logsumexp(Z + v.unsqueeze(-2), dim=-1)
        v = log_col_marginal - torch.logsumexp(Z + u.unsqueeze(-1), dim=-2)

    return torch.exp(Z + u.unsqueeze(-1) + v.unsqueeze(-2)) * rows


def sinkhorn_route_ref(logits, state, num_iters=20, tau=1.0):
    """Sinkhorn projection + routing: ``Sinkhorn(logits) @ state``."""
    P = sinkhorn_log_ref(logits, num_iters, tau)
    return einsum(P, state, "... i j, ... j c -> ... i c")


# ========================================================================== #
#  Edge case logit generators for stress testing
# ========================================================================== #

EDGE_CASES = ["near_permutation", "large_magnitude", "near_uniform", "one_hot_rows"]


def make_test_logits(kind, B, rows, cols, device="cuda", seed=300):
    """Generate edge-case logits for stress testing.

    Kinds:
    - near_permutation: diagonal dominant → P ≈ permutation (ill-conditioned KKT)
    - large_magnitude:  10× random → extreme Z values at low tau
    - near_uniform:     0.01× random → P ≈ doubly stochastic uniform
    - one_hot_rows:     one dominant entry per row (sparse routing pattern)
    """
    torch.manual_seed(seed)
    if kind == "random":
        return torch.randn(B, rows, cols, device=device)
    elif kind == "near_permutation":
        base = torch.randn(B, rows, cols, device=device) * 0.1
        base += torch.eye(rows, cols, device=device) * 10.0
        return base
    elif kind == "large_magnitude":
        return torch.randn(B, rows, cols, device=device) * 10.0
    elif kind == "near_uniform":
        return torch.randn(B, rows, cols, device=device) * 0.01
    elif kind == "one_hot_rows":
        base = torch.randn(B, rows, cols, device=device) * 0.1
        idx = torch.randint(0, cols, (B, rows), device=device)
        base.scatter_(2, idx.unsqueeze(-1), 10.0)
        return base
    raise ValueError(f"Unknown kind: {kind}")
