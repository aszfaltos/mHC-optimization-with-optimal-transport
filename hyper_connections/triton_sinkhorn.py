"""Fused Triton Sinkhorn kernel for doubly stochastic projection.

Drop-in replacement for the PyTorch sinkhorn_log: runs all iterations in a
single kernel launch with the matrix living in SRAM/registers.  Supports both
square and rectangular matrices, log-space computation for numerical stability.

Uses separate row/col marginals (-log(ROWS), -log(COLS)) so that rectangular
matrices converge to the correct OT coupling between uniform distributions.
Output is scaled by ROWS so row sums = 1 (suitable for routing matmuls).
For square matrices this is identical to the single-marginal formulation.

Forward: fused Triton kernel (single launch, all iters in SRAM).
Backward: fused Triton kernel using implicit differentiation of the Sinkhorn
fixed point.  The KKT adjoint system has a rank-1 null space (constant shift
between α and β).  We center α and β at each Gauss-Seidel iteration to project
out this gauge freedom, ensuring convergence even when P is near-permutation.
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
        log_row_marginal,  # precomputed -log(ROWS)
        log_col_marginal,  # precomputed -log(COLS)
        n_scale,  # precomputed float(ROWS) for output scaling (row sums = 1)
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
            # Row update: u_i = log_row_marg - logsumexp_j(Z_ij + v_j)
            Zv = Z + v[None, :]
            max_Zv = tl.max(Zv, axis=1)
            lse_row = max_Zv + tl.log(tl.sum(tl.exp(Zv - max_Zv[:, None]), axis=1))
            u = tl.where(row_mask, log_row_marginal - lse_row, 0.0)

            # Col update: v_j = log_col_marg - logsumexp_i(Z_ij + u_i)
            Zu = Z + u[:, None]
            max_Zu = tl.max(Zu, axis=0)
            lse_col = max_Zu + tl.log(tl.sum(tl.exp(Zu - max_Zu[None, :]), axis=0))
            v = tl.where(col_mask, log_col_marginal - lse_col, 0.0)

        # Output: exp(Z + u + v) * ROWS → row sums = 1, col sums = ROWS/COLS
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
        output_ptr,  # saved forward output (row sums = 1, col sums = ROWS/COLS)
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
        col_row_ratio,  # COLS/ROWS — 1.0 for square matrices
    ):
        pid = tl.program_id(0)

        rows = tl.arange(0, BLOCK_ROWS)
        cols = tl.arange(0, BLOCK_COLS)
        row_mask = rows < ROWS
        col_mask = cols < COLS
        mask_2d = row_mask[:, None] & col_mask[None, :]

        base = pid * stride_b
        offsets = base + rows[:, None] * stride_r + cols[None, :] * stride_c

        # Load grad_output and saved forward output P (row sums = 1)
        G = tl.load(grad_out_ptr + offsets, mask=mask_2d, other=0.0).to(tl.float32)
        P = tl.load(output_ptr + offsets, mask=mask_2d, other=0.0).to(tl.float32)

        # H = P * grad_output (element-wise)
        H = P * G

        # Marginals of H
        r = tl.sum(H, axis=1)  # (BLOCK_ROWS,)
        c = tl.sum(H, axis=0)  # (BLOCK_COLS,)

        # Solve the adjoint system via Gauss-Seidel iteration.
        # With row sums = 1, col sums = ROWS/COLS, the KKT system is:
        #   1 · α_i + Σ_j P_ij · β_j = r_i          →  α = r - P·β
        #   (ROWS/COLS) · β_j + Σ_i P_ij · α_i = c_j →  β = (c - P^T·α) · COLS/ROWS
        alpha = tl.zeros([BLOCK_ROWS], dtype=tl.float32)
        beta = tl.zeros([BLOCK_COLS], dtype=tl.float32)

        for _it in range(NUM_ITERS):
            P_beta = tl.sum(P * beta[None, :], axis=1)
            alpha = tl.where(row_mask, r - P_beta, 0.0)

            Pt_alpha = tl.sum(P * alpha[:, None], axis=0)
            beta = tl.where(col_mask, (c - Pt_alpha) * col_row_ratio, 0.0)

            # Project out rank-1 null space (α+t, β-t) of KKT system.
            # Sets sum(α) = sum(β), eliminating the gauge degree of freedom.
            t = (tl.sum(alpha) - tl.sum(beta)) / (ROWS + COLS)
            alpha = alpha - t
            beta = beta + t

        # grad_Z = P * (G - α - β)
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
    def forward(ctx, logits: torch.Tensor, num_iters: int, tau: float, num_iters_bwd: int = 0) -> torch.Tensor:
        orig_shape = logits.shape
        R, C = orig_shape[-2], orig_shape[-1]
        B = logits.numel() // (R * C)
        input_dtype = logits.dtype
        flat = logits.reshape(B, R, C).contiguous()
        output = torch.empty(B, R, C, device=flat.device, dtype=torch.float32)

        BLOCK_ROWS = triton.next_power_of_2(R)
        BLOCK_COLS = triton.next_power_of_2(C)

        log_row_marginal = -math.log(R)
        log_col_marginal = -math.log(C)
        n_scale = float(R)

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
            log_row_marginal=log_row_marginal,
            log_col_marginal=log_col_marginal,
            n_scale=n_scale,
        )

        ctx.save_for_backward(output)
        ctx.num_iters_bwd = num_iters_bwd if num_iters_bwd > 0 else num_iters
        ctx.tau = tau
        ctx.R = R
        ctx.C = C
        ctx.orig_shape = orig_shape
        ctx.input_dtype = input_dtype
        return output.to(input_dtype).reshape(orig_shape)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        (output_flat,) = ctx.saved_tensors
        R, C = ctx.R, ctx.C
        B = output_flat.shape[0]
        tau = ctx.tau

        grad_flat = grad_output.reshape(B, R, C).contiguous()
        output_flat = output_flat.contiguous()
        grad_input = torch.empty_like(output_flat)

        BLOCK_ROWS = triton.next_power_of_2(R)
        BLOCK_COLS = triton.next_power_of_2(C)

        _sinkhorn_bwd_kernel[(B,)](
            grad_flat,
            output_flat,
            grad_input,
            output_flat.stride(0),
            output_flat.stride(1),
            output_flat.stride(2),
            ROWS=R,
            COLS=C,
            BLOCK_ROWS=BLOCK_ROWS,
            BLOCK_COLS=BLOCK_COLS,
            NUM_ITERS=ctx.num_iters_bwd,
            tau=tau,
            col_row_ratio=float(C) / float(R),
        )

        return grad_input.to(grad_output.dtype).reshape(ctx.orig_shape), None, None, None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def triton_sinkhorn(
    logits: torch.Tensor,
    num_iters: int = 20,
    tau: float = 1.0,
    num_iters_bwd: int | None = None,
) -> torch.Tensor:
    """Fused Triton Sinkhorn projection.

    Drop-in replacement for ``sinkhorn_log``.  Accepts ``(..., rows, cols)``
    logits and returns the doubly stochastic projection scaled by ``rows``
    (so row sums ≈ 1, col sums ≈ rows/cols).  Requires CUDA tensors and Triton.

    Args:
        num_iters_bwd: Gauss-Seidel iterations for the backward adjoint solve.
            Defaults to ``num_iters``.  Increase for large m or near-permutation P.
    """
    return _TritonSinkhornFn.apply(
        logits.contiguous(), num_iters, tau, num_iters_bwd or num_iters,
    )
