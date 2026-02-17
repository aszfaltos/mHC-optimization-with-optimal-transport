"""Fused Triton Sinkhorn kernel for doubly stochastic projection.

Drop-in replacement for the PyTorch sinkhorn_log: runs all iterations in a
single kernel launch with the matrix living in SRAM/registers.  Supports both
square and rectangular matrices, log-space computation for numerical stability,
and implicit-differentiation backward (no stored per-iteration intermediates).
"""
from __future__ import annotations

import math

import torch
from torch.autograd import Function

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False


def is_available() -> bool:
    return _TRITON_AVAILABLE


# ---------------------------------------------------------------------------
# Forward kernel
# ---------------------------------------------------------------------------

if _TRITON_AVAILABLE:

    @triton.jit
    def _sinkhorn_fwd_kernel(
        logits_ptr,
        output_ptr,
        stride_b,
        stride_r,
        stride_c,
        o_stride_b,
        o_stride_r,
        o_stride_c,
        ROWS: tl.constexpr,
        COLS: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_COLS: tl.constexpr,
        NUM_ITERS: tl.constexpr,
        tau,
        log_marginal,  # precomputed -log(COLS) as float
        n_scale,  # precomputed float(COLS) for output scaling
    ):
        pid = tl.program_id(0)

        rows = tl.arange(0, BLOCK_ROWS)
        cols = tl.arange(0, BLOCK_COLS)
        row_mask = rows < ROWS
        col_mask = cols < COLS
        mask_2d = row_mask[:, None] & col_mask[None, :]

        # Load logits into registers (as fp32) and scale by 1/tau
        base = pid * stride_b
        offsets = base + rows[:, None] * stride_r + cols[None, :] * stride_c
        Z = tl.load(logits_ptr + offsets, mask=mask_2d, other=0.0).to(tl.float32) / tau

        # Mask out-of-bounds with -inf so they don't affect logsumexp
        Z = tl.where(mask_2d, Z, float("-inf"))

        u = tl.zeros([BLOCK_ROWS], dtype=tl.float32)
        v = tl.zeros([BLOCK_COLS], dtype=tl.float32)

        for _it in range(NUM_ITERS):
            # Row update: u_i = log_marg - logsumexp_j(Z_ij + v_j)
            Zv = Z + v[None, :]
            max_Zv = tl.max(Zv, axis=1)
            lse_row = max_Zv + tl.log(tl.sum(tl.exp(Zv - max_Zv[:, None]), axis=1))
            u = tl.where(row_mask, log_marginal - lse_row, 0.0)

            # Col update: v_j = log_marg - logsumexp_i(Z_ij + u_i)
            Zu = Z + u[:, None]
            max_Zu = tl.max(Zu, axis=0)
            lse_col = max_Zu + tl.log(tl.sum(tl.exp(Zu - max_Zu[None, :]), axis=0))
            v = tl.where(col_mask, log_marginal - lse_col, 0.0)

        # Output: exp(Z + u + v) * n_cols
        P = tl.exp(Z + u[:, None] + v[None, :]) * n_scale
        P = tl.where(mask_2d, P, 0.0)

        o_base = pid * o_stride_b
        o_offsets = o_base + rows[:, None] * o_stride_r + cols[None, :] * o_stride_c
        tl.store(output_ptr + o_offsets, P, mask=mask_2d)

    # -----------------------------------------------------------------------
    # Backward kernel (implicit differentiation of the Sinkhorn fixed point)
    # -----------------------------------------------------------------------

    @triton.jit
    def _sinkhorn_bwd_kernel(
        grad_out_ptr,
        output_ptr,  # saved forward output (includes * n_cols)
        grad_in_ptr,
        stride_b,
        stride_r,
        stride_c,
        ROWS: tl.constexpr,
        COLS: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_COLS: tl.constexpr,
        NUM_ITERS: tl.constexpr,
        tau,
        n_scale,  # precomputed float(COLS) for de-scaling
    ):
        pid = tl.program_id(0)

        rows = tl.arange(0, BLOCK_ROWS)
        cols = tl.arange(0, BLOCK_COLS)
        row_mask = rows < ROWS
        col_mask = cols < COLS
        mask_2d = row_mask[:, None] & col_mask[None, :]

        base = pid * stride_b
        offsets = base + rows[:, None] * stride_r + cols[None, :] * stride_c

        # Load grad_output and the saved forward output P_scaled = P * n_cols
        G = tl.load(grad_out_ptr + offsets, mask=mask_2d, other=0.0).to(tl.float32)
        P_scaled = tl.load(output_ptr + offsets, mask=mask_2d, other=0.0).to(tl.float32)

        # P_norm: doubly stochastic (row/col sums = 1)
        P = P_scaled / n_scale

        # H = P * grad_output (element-wise)
        H = P * G

        # Marginals of H
        r = tl.sum(H, axis=1)  # (BLOCK_ROWS,)
        c = tl.sum(H, axis=0)  # (BLOCK_COLS,)

        # Solve the linear system for alpha, beta via Sinkhorn-like iteration:
        #   alpha_i + sum_j P_ij * beta_j = r_i
        #   sum_i P_ij * alpha_i + beta_j = c_j
        alpha = tl.zeros([BLOCK_ROWS], dtype=tl.float32)
        beta = tl.zeros([BLOCK_COLS], dtype=tl.float32)

        for _it in range(NUM_ITERS):
            # alpha = r - P @ beta
            P_beta = tl.sum(P * beta[None, :], axis=1)
            alpha = tl.where(row_mask, r - P_beta, 0.0)

            # beta = c - P^T @ alpha
            Pt_alpha = tl.sum(P * alpha[:, None], axis=0)
            beta = tl.where(col_mask, c - Pt_alpha, 0.0)

        # grad_Z = P * (G - alpha_broadcast - beta_broadcast)
        grad_Z = P * (G - alpha[:, None] - beta[None, :])

        # Chain rule: Z = logits / tau  =>  grad_logits = grad_Z / tau
        grad_logits = grad_Z / tau
        grad_logits = tl.where(mask_2d, grad_logits, 0.0)

        tl.store(grad_in_ptr + offsets, grad_logits, mask=mask_2d)


# ---------------------------------------------------------------------------
# Autograd Function
# ---------------------------------------------------------------------------


class _TritonSinkhornFn(Function):
    @staticmethod
    def forward(ctx, logits: torch.Tensor, num_iters: int, tau: float) -> torch.Tensor:
        orig_shape = logits.shape
        R, C = orig_shape[-2], orig_shape[-1]
        B = logits.numel() // (R * C)
        flat = logits.reshape(B, R, C).contiguous()
        output = torch.empty_like(flat)

        BLOCK_ROWS = triton.next_power_of_2(R)
        BLOCK_COLS = triton.next_power_of_2(C)

        log_marginal = -math.log(C)
        n_scale = float(C)

        _sinkhorn_fwd_kernel[(B,)](
            flat,
            output,
            flat.stride(0),
            flat.stride(1),
            flat.stride(2),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            ROWS=R,
            COLS=C,
            BLOCK_ROWS=BLOCK_ROWS,
            BLOCK_COLS=BLOCK_COLS,
            NUM_ITERS=num_iters,
            tau=tau,
            log_marginal=log_marginal,
            n_scale=n_scale,
        )

        ctx.save_for_backward(output)
        ctx.num_iters = num_iters
        ctx.tau = tau
        ctx.R = R
        ctx.C = C
        ctx.orig_shape = orig_shape
        return output.reshape(orig_shape)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (output,) = ctx.saved_tensors
        R, C = ctx.R, ctx.C
        B = output.shape[0]

        grad_flat = grad_output.reshape(B, R, C).contiguous()
        grad_logits = torch.empty_like(grad_flat)

        BLOCK_ROWS = triton.next_power_of_2(R)
        BLOCK_COLS = triton.next_power_of_2(C)

        n_scale = float(C)

        _sinkhorn_bwd_kernel[(B,)](
            grad_flat,
            output,
            grad_logits,
            grad_flat.stride(0),
            grad_flat.stride(1),
            grad_flat.stride(2),
            ROWS=R,
            COLS=C,
            BLOCK_ROWS=BLOCK_ROWS,
            BLOCK_COLS=BLOCK_COLS,
            NUM_ITERS=ctx.num_iters,
            tau=ctx.tau,
            n_scale=n_scale,
        )

        return grad_logits.reshape(ctx.orig_shape), None, None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def triton_sinkhorn(logits: torch.Tensor, num_iters: int = 10, tau: float = 0.05) -> torch.Tensor:
    """Fused Triton Sinkhorn projection.

    Drop-in replacement for ``sinkhorn_log``.  Accepts ``(..., rows, cols)``
    logits and returns the doubly stochastic projection scaled by ``cols``
    (so row/col sums ≈ 1).  Requires CUDA tensors and Triton.
    """
    return _TritonSinkhornFn.apply(logits.contiguous(), num_iters, tau)
