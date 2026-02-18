"""Fused Triton Sinkhorn + routing kernel.

Computes ``P @ state`` where ``P = Sinkhorn(logits)`` in a single kernel
launch — P never touches global memory.  The backward kernel computes both
``grad_logits`` (via implicit differentiation) and ``grad_state`` (via P^T)
in one launch.

Uses separate row/col marginals (-log(ROWS), -log(COLS)) so that rectangular
matrices converge to the correct OT coupling.  Output P has row sums = 1 (scaled
by ROWS), col sums = ROWS/COLS.  For square matrices this reduces to the
single-marginal formulation.

Public API:
    triton_sinkhorn_route(logits, state, num_iters=20, tau=1.0) -> result
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
# Forward kernel: Sinkhorn → P @ state, P stays in registers
# ---------------------------------------------------------------------------

if _TRITON_AVAILABLE:

    # ------------------------------------------------------------------
    # Alpha+bias fused variant: absorbs res_logits = alpha * raw + bias
    # into the kernel, eliminating 2 elementwise launches + intermediate
    # ------------------------------------------------------------------

    @triton.jit
    def _sinkhorn_route_ab_fwd_kernel(
        raw_ptr,       # (B, R, M) raw logits from phi projection
        alpha_ptr,     # scalar
        bias_ptr,      # (R, M) static bias (shared across batch)
        state_ptr,
        output_ptr,
        P_out_ptr,
        stride_b,
        stride_r,
        stride_c,
        s_stride_b,
        s_stride_r,
        s_stride_c,
        o_stride_b,
        o_stride_r,
        o_stride_c,
        p_stride_b,
        p_stride_r,
        p_stride_c,
        b_stride_r,   # bias strides
        b_stride_c,
        ROWS: tl.constexpr,
        COLS: tl.constexpr,
        C_DIM: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_COLS: tl.constexpr,
        BLOCK_C: tl.constexpr,
        NUM_ITERS: tl.constexpr,
        tau,
        log_row_marginal,
        log_col_marginal,
        n_scale,
    ):
        pid = tl.program_id(0)

        rows = tl.arange(0, BLOCK_ROWS)
        cols = tl.arange(0, BLOCK_COLS)
        row_mask = rows < ROWS
        col_mask = cols < COLS
        mask_2d = row_mask[:, None] & col_mask[None, :]

        # Load raw logits and fuse alpha * raw + bias
        base = pid * stride_b
        offsets = base + rows[:, None] * stride_r + cols[None, :] * stride_c
        raw = tl.load(raw_ptr + offsets, mask=mask_2d, other=0.0).to(tl.float32)

        alpha = tl.load(alpha_ptr).to(tl.float32)

        bias_offsets = rows[:, None] * b_stride_r + cols[None, :] * b_stride_c
        bias = tl.load(bias_ptr + bias_offsets, mask=mask_2d, other=0.0).to(tl.float32)

        # Fused: (alpha * raw + bias) / tau
        Z = (alpha * raw + bias) / tau
        Z = tl.where(mask_2d, Z, float("-inf"))

        # Sinkhorn iterations
        u = tl.zeros([BLOCK_ROWS], dtype=tl.float32)
        v = tl.zeros([BLOCK_COLS], dtype=tl.float32)

        for _it in range(NUM_ITERS):
            Zv = Z + v[None, :]
            max_Zv = tl.max(Zv, axis=1)
            lse_row = max_Zv + tl.log(tl.sum(tl.exp(Zv - max_Zv[:, None]), axis=1))
            u = tl.where(row_mask, log_row_marginal - lse_row, 0.0)

            Zu = Z + u[:, None]
            max_Zu = tl.max(Zu, axis=0)
            lse_col = max_Zu + tl.log(tl.sum(tl.exp(Zu - max_Zu[None, :]), axis=0))
            v = tl.where(col_mask, log_col_marginal - lse_col, 0.0)

        P = tl.exp(Z + u[:, None] + v[None, :]) * n_scale
        P = tl.where(mask_2d, P, 0.0)

        # Save P for backward (always fp32)
        p_base = pid * p_stride_b
        p_offsets = p_base + rows[:, None] * p_stride_r + cols[None, :] * p_stride_c
        tl.store(P_out_ptr + p_offsets, P, mask=mask_2d)

        # Route: P @ state, tiled over c dimension
        s_base = pid * s_stride_b
        o_base = pid * o_stride_b
        c_range = tl.arange(0, BLOCK_C)

        for c_start in range(0, C_DIM, BLOCK_C):
            c_idx = c_start + c_range
            c_mask = c_idx < C_DIM
            s_offsets = s_base + cols[:, None] * s_stride_r + c_idx[None, :] * s_stride_c
            s_mask = col_mask[:, None] & c_mask[None, :]
            state_tile = tl.load(state_ptr + s_offsets, mask=s_mask, other=0.0).to(tl.float32)
            result_tile = tl.dot(P, state_tile)
            o_offsets = o_base + rows[:, None] * o_stride_r + c_idx[None, :] * o_stride_c
            o_mask = row_mask[:, None] & c_mask[None, :]
            tl.store(output_ptr + o_offsets, result_tile, mask=o_mask)

    @triton.jit
    def _sinkhorn_route_fwd_kernel(
        logits_ptr,
        state_ptr,
        output_ptr,
        P_out_ptr,  # save P for backward
        stride_b,
        stride_r,
        stride_c,
        s_stride_b,
        s_stride_r,
        s_stride_c,
        o_stride_b,
        o_stride_r,
        o_stride_c,
        p_stride_b,
        p_stride_r,
        p_stride_c,
        ROWS: tl.constexpr,
        COLS: tl.constexpr,
        C_DIM: tl.constexpr,  # chunk size (state cols)
        BLOCK_ROWS: tl.constexpr,
        BLOCK_COLS: tl.constexpr,
        BLOCK_C: tl.constexpr,
        NUM_ITERS: tl.constexpr,
        tau,
        log_row_marginal,  # -log(ROWS)
        log_col_marginal,  # -log(COLS)
        n_scale,  # float(ROWS) — row sums = 1
    ):
        pid = tl.program_id(0)

        rows = tl.arange(0, BLOCK_ROWS)
        cols = tl.arange(0, BLOCK_COLS)
        row_mask = rows < ROWS
        col_mask = cols < COLS
        mask_2d = row_mask[:, None] & col_mask[None, :]

        # ---- Sinkhorn in registers ----
        base = pid * stride_b
        offsets = base + rows[:, None] * stride_r + cols[None, :] * stride_c
        Z = tl.load(logits_ptr + offsets, mask=mask_2d, other=0.0).to(tl.float32) / tau
        Z = tl.where(mask_2d, Z, float("-inf"))

        u = tl.zeros([BLOCK_ROWS], dtype=tl.float32)
        v = tl.zeros([BLOCK_COLS], dtype=tl.float32)

        for _it in range(NUM_ITERS):
            Zv = Z + v[None, :]
            max_Zv = tl.max(Zv, axis=1)
            lse_row = max_Zv + tl.log(tl.sum(tl.exp(Zv - max_Zv[:, None]), axis=1))
            u = tl.where(row_mask, log_row_marginal - lse_row, 0.0)

            Zu = Z + u[:, None]
            max_Zu = tl.max(Zu, axis=0)
            lse_col = max_Zu + tl.log(tl.sum(tl.exp(Zu - max_Zu[None, :]), axis=0))
            v = tl.where(col_mask, log_col_marginal - lse_col, 0.0)

        # P = exp(Z + u + v) * ROWS → row sums = 1, col sums = ROWS/COLS
        P = tl.exp(Z + u[:, None] + v[None, :]) * n_scale
        P = tl.where(mask_2d, P, 0.0)

        # Save P for backward
        p_base = pid * p_stride_b
        p_offsets = p_base + rows[:, None] * p_stride_r + cols[None, :] * p_stride_c
        tl.store(P_out_ptr + p_offsets, P, mask=mask_2d)

        # ---- Route: P @ state, tiled over c dimension ----
        # P is (BLOCK_ROWS, BLOCK_COLS) in registers, stays there
        # state is (COLS, C_DIM) — we tile over C_DIM in chunks of BLOCK_C
        s_base = pid * s_stride_b
        o_base = pid * o_stride_b

        c_range = tl.arange(0, BLOCK_C)

        for c_start in range(0, C_DIM, BLOCK_C):
            c_idx = c_start + c_range
            c_mask = c_idx < C_DIM

            # Load state tile: (COLS, BLOCK_C)
            s_offsets = s_base + cols[:, None] * s_stride_r + c_idx[None, :] * s_stride_c
            s_mask = col_mask[:, None] & c_mask[None, :]
            state_tile = tl.load(state_ptr + s_offsets, mask=s_mask, other=0.0).to(tl.float32)

            # result_tile = P @ state_tile: (ROWS, BLOCK_C)
            result_tile = tl.dot(P, state_tile)

            # Store result tile
            o_offsets = o_base + rows[:, None] * o_stride_r + c_idx[None, :] * o_stride_c
            o_mask = row_mask[:, None] & c_mask[None, :]
            tl.store(output_ptr + o_offsets, result_tile, mask=o_mask)

    # -----------------------------------------------------------------------
    # Backward kernel: grad_logits (implicit diff) + grad_state (P^T @ grad)
    # -----------------------------------------------------------------------

    @triton.jit
    def _sinkhorn_route_bwd_kernel(
        grad_out_ptr,  # grad w.r.t. output: (B, ROWS, C_DIM)
        P_ptr,  # saved forward P (row sums = 1): (B, ROWS, COLS)
        state_ptr,  # saved state: (B, COLS, C_DIM)
        grad_logits_ptr,  # output: (B, ROWS, COLS)
        grad_state_ptr,  # output: (B, COLS, C_DIM)
        stride_b,
        stride_r,
        stride_c,
        p_stride_b,
        p_stride_r,
        p_stride_c,
        s_stride_b,
        s_stride_r,
        s_stride_c,
        gs_stride_b,
        gs_stride_r,
        gs_stride_c,
        ROWS: tl.constexpr,
        COLS: tl.constexpr,
        C_DIM: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_COLS: tl.constexpr,
        BLOCK_C: tl.constexpr,
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

        # Load P (row sums = 1, col sums = ROWS/COLS) into registers
        p_base = pid * p_stride_b
        p_offsets = p_base + rows[:, None] * p_stride_r + cols[None, :] * p_stride_c
        P = tl.load(P_ptr + p_offsets, mask=mask_2d, other=0.0).to(tl.float32)

        # ---- Tile over c: accumulate grad_P and compute grad_state ----
        grad_P = tl.zeros([BLOCK_ROWS, BLOCK_COLS], dtype=tl.float32)

        c_range = tl.arange(0, BLOCK_C)
        go_base = pid * stride_b
        s_base = pid * s_stride_b
        gs_base = pid * gs_stride_b

        for c_start in range(0, C_DIM, BLOCK_C):
            c_idx = c_start + c_range
            c_mask = c_idx < C_DIM

            # Load grad_output tile: (ROWS, BLOCK_C)
            go_offsets = go_base + rows[:, None] * stride_r + c_idx[None, :] * stride_c
            go_mask = row_mask[:, None] & c_mask[None, :]
            grad_out_tile = tl.load(grad_out_ptr + go_offsets, mask=go_mask, other=0.0).to(tl.float32)

            # Load state tile: (COLS, BLOCK_C)
            s_offsets = s_base + cols[:, None] * s_stride_r + c_idx[None, :] * s_stride_c
            s_mask = col_mask[:, None] & c_mask[None, :]
            state_tile = tl.load(state_ptr + s_offsets, mask=s_mask, other=0.0).to(tl.float32)

            # grad_P += grad_out_tile @ state_tile^T: (ROWS, BLOCK_C) @ (BLOCK_C, COLS)
            grad_P += tl.dot(grad_out_tile, tl.trans(state_tile))

            # grad_state_tile = P^T @ grad_out_tile: (COLS, ROWS) @ (ROWS, BLOCK_C)
            grad_state_tile = tl.dot(tl.trans(P), grad_out_tile)

            # Store grad_state tile
            gs_offsets = gs_base + cols[:, None] * gs_stride_r + c_idx[None, :] * gs_stride_c
            tl.store(grad_state_ptr + gs_offsets, grad_state_tile, mask=s_mask)

        # ---- Implicit differentiation for grad_logits ----
        G = grad_P

        H = P * G
        r = tl.sum(H, axis=1)
        c_marg = tl.sum(H, axis=0)

        # Solve adjoint system via Gauss-Seidel.
        # With row sums = 1, col sums = ROWS/COLS:
        #   α = r - P·β            (row marginal = 1, no rescale)
        #   β = (c - P^T·α) · COLS/ROWS  (col marginal = ROWS/COLS)
        alpha = tl.zeros([BLOCK_ROWS], dtype=tl.float32)
        beta = tl.zeros([BLOCK_COLS], dtype=tl.float32)

        for _it in range(NUM_ITERS):
            P_beta = tl.sum(P * beta[None, :], axis=1)
            alpha = tl.where(row_mask, r - P_beta, 0.0)

            Pt_alpha = tl.sum(P * alpha[:, None], axis=0)
            beta = tl.where(col_mask, (c_marg - Pt_alpha) * col_row_ratio, 0.0)

            # Project out rank-1 null space (α+t, β-t) of KKT system.
            # Sets sum(α) = sum(β), eliminating the gauge degree of freedom.
            t = (tl.sum(alpha) - tl.sum(beta)) / (ROWS + COLS)
            alpha = alpha - t
            beta = beta + t

        grad_Z = P * (G - alpha[:, None] - beta[None, :])
        grad_logits = grad_Z / tau
        grad_logits = tl.where(mask_2d, grad_logits, 0.0)

        # Store grad_logits
        gl_base = pid * p_stride_b  # reuse P strides (same shape)
        gl_offsets = gl_base + rows[:, None] * p_stride_r + cols[None, :] * p_stride_c
        tl.store(grad_logits_ptr + gl_offsets, grad_logits, mask=mask_2d)


# ---------------------------------------------------------------------------
# Autograd Function
# ---------------------------------------------------------------------------


def _next_power_of_2(n: int) -> int:
    """Triton-compatible next power of 2."""
    if _TRITON_AVAILABLE:
        return triton.next_power_of_2(n)
    v = 1
    while v < n:
        v <<= 1
    return v


def _block_size(n: int) -> int:
    """Block size for tl.dot: must be power of 2 and >= 16."""
    return max(16, _next_power_of_2(n))


class _SinkhornRouteFn(Function):
    @staticmethod
    def forward(
        ctx,
        logits: torch.Tensor,
        state: torch.Tensor,
        num_iters: int,
        tau: float,
        num_iters_bwd: int = 0,
    ) -> torch.Tensor:
        # logits: (..., rows, cols), state: (..., cols, c_dim)
        orig_logits_shape = logits.shape
        orig_state_shape = state.shape
        R, M = orig_logits_shape[-2], orig_logits_shape[-1]
        C_DIM = orig_state_shape[-1]
        B = logits.numel() // (R * M)

        logits_flat = logits.reshape(B, R, M).contiguous()
        state_flat = state.reshape(B, M, C_DIM).contiguous()
        output = torch.empty(B, R, C_DIM, device=logits.device, dtype=logits.dtype)
        P_saved = torch.empty(B, R, M, device=logits.device, dtype=torch.float32)

        # tl.dot requires block dims >= 16 (power of 2)
        BLOCK_ROWS = _block_size(R)
        BLOCK_COLS = _block_size(M)
        BLOCK_C = _block_size(min(C_DIM, 64))

        log_row_marginal = -math.log(R)
        log_col_marginal = -math.log(M)
        n_scale = float(R)

        _sinkhorn_route_fwd_kernel[(B,)](
            logits_flat,
            state_flat,
            output,
            P_saved,
            logits_flat.stride(0),
            logits_flat.stride(1),
            logits_flat.stride(2),
            state_flat.stride(0),
            state_flat.stride(1),
            state_flat.stride(2),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            P_saved.stride(0),
            P_saved.stride(1),
            P_saved.stride(2),
            ROWS=R,
            COLS=M,
            C_DIM=C_DIM,
            BLOCK_ROWS=BLOCK_ROWS,
            BLOCK_COLS=BLOCK_COLS,
            BLOCK_C=BLOCK_C,
            NUM_ITERS=num_iters,
            tau=tau,
            log_row_marginal=log_row_marginal,
            log_col_marginal=log_col_marginal,
            n_scale=n_scale,
        )

        ctx.save_for_backward(P_saved, state_flat)
        ctx.num_iters_bwd = num_iters_bwd if num_iters_bwd > 0 else num_iters
        ctx.tau = tau
        ctx.R = R
        ctx.M = M
        ctx.C_DIM = C_DIM
        ctx.orig_logits_shape = orig_logits_shape
        ctx.orig_state_shape = orig_state_shape

        out_shape = list(orig_state_shape)
        out_shape[-2] = R
        return output.reshape(out_shape)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        P_saved, state_flat = ctx.saved_tensors
        R, M, C_DIM = ctx.R, ctx.M, ctx.C_DIM
        B = P_saved.shape[0]

        grad_out_flat = grad_output.reshape(B, R, C_DIM).contiguous()
        P_saved = P_saved.contiguous()
        state_flat = state_flat.contiguous()

        grad_logits = torch.empty(B, R, M, device=P_saved.device, dtype=P_saved.dtype)
        grad_state = torch.empty(B, M, C_DIM, device=P_saved.device, dtype=P_saved.dtype)

        BLOCK_ROWS = _block_size(R)
        BLOCK_COLS = _block_size(M)
        BLOCK_C = _block_size(min(C_DIM, 64))

        _sinkhorn_route_bwd_kernel[(B,)](
            grad_out_flat,
            P_saved,
            state_flat,
            grad_logits,
            grad_state,
            grad_out_flat.stride(0),
            grad_out_flat.stride(1),
            grad_out_flat.stride(2),
            P_saved.stride(0),
            P_saved.stride(1),
            P_saved.stride(2),
            state_flat.stride(0),
            state_flat.stride(1),
            state_flat.stride(2),
            grad_state.stride(0),
            grad_state.stride(1),
            grad_state.stride(2),
            ROWS=R,
            COLS=M,
            C_DIM=C_DIM,
            BLOCK_ROWS=BLOCK_ROWS,
            BLOCK_COLS=BLOCK_COLS,
            BLOCK_C=BLOCK_C,
            NUM_ITERS=ctx.num_iters_bwd,
            tau=ctx.tau,
            col_row_ratio=float(M) / float(R),
        )

        grad_logits_out = grad_logits.to(grad_output.dtype).reshape(ctx.orig_logits_shape)
        grad_state_out = grad_state.to(grad_output.dtype).reshape(ctx.orig_state_shape)
        return grad_logits_out, grad_state_out, None, None, None


class _SinkhornRouteABFn(Function):
    """Fused alpha*raw+bias → Sinkhorn → P@state in one kernel launch."""

    @staticmethod
    def forward(
        ctx,
        raw_logits: torch.Tensor,
        alpha: torch.Tensor,
        bias: torch.Tensor,
        state: torch.Tensor,
        num_iters: int,
        tau: float,
        num_iters_bwd: int = 0,
    ) -> torch.Tensor:
        orig_raw_shape = raw_logits.shape
        orig_state_shape = state.shape
        R, M = orig_raw_shape[-2], orig_raw_shape[-1]
        C_DIM = orig_state_shape[-1]
        B = raw_logits.numel() // (R * M)

        raw_flat = raw_logits.reshape(B, R, M).contiguous()
        state_flat = state.reshape(B, M, C_DIM).contiguous()
        alpha_cont = alpha.contiguous()
        bias_cont = bias.contiguous()

        output = torch.empty(B, R, C_DIM, device=raw_logits.device, dtype=state.dtype)
        P_saved = torch.empty(B, R, M, device=raw_logits.device, dtype=torch.float32)

        BLOCK_ROWS = _block_size(R)
        BLOCK_COLS = _block_size(M)
        BLOCK_C = _block_size(min(C_DIM, 64))

        log_row_marginal = -math.log(R)
        log_col_marginal = -math.log(M)
        n_scale = float(R)

        _sinkhorn_route_ab_fwd_kernel[(B,)](
            raw_flat,
            alpha_cont,
            bias_cont,
            state_flat,
            output,
            P_saved,
            raw_flat.stride(0),
            raw_flat.stride(1),
            raw_flat.stride(2),
            state_flat.stride(0),
            state_flat.stride(1),
            state_flat.stride(2),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            P_saved.stride(0),
            P_saved.stride(1),
            P_saved.stride(2),
            bias_cont.stride(0),
            bias_cont.stride(1),
            ROWS=R,
            COLS=M,
            C_DIM=C_DIM,
            BLOCK_ROWS=BLOCK_ROWS,
            BLOCK_COLS=BLOCK_COLS,
            BLOCK_C=BLOCK_C,
            NUM_ITERS=num_iters,
            tau=tau,
            log_row_marginal=log_row_marginal,
            log_col_marginal=log_col_marginal,
            n_scale=n_scale,
        )

        ctx.save_for_backward(P_saved, state_flat, raw_flat, alpha_cont)
        ctx.num_iters_bwd = num_iters_bwd if num_iters_bwd > 0 else num_iters
        ctx.tau = tau
        ctx.R = R
        ctx.M = M
        ctx.C_DIM = C_DIM
        ctx.orig_raw_shape = orig_raw_shape
        ctx.orig_state_shape = orig_state_shape

        out_shape = list(orig_state_shape)
        out_shape[-2] = R
        return output.reshape(out_shape)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        P_saved, state_flat, raw_flat, alpha = ctx.saved_tensors
        R, M, C_DIM = ctx.R, ctx.M, ctx.C_DIM
        B = P_saved.shape[0]

        grad_out_flat = grad_output.reshape(B, R, C_DIM).contiguous()
        P_saved = P_saved.contiguous()
        state_flat = state_flat.contiguous()

        # Reuse existing backward kernel to get grad_logits (w.r.t. alpha*raw+bias)
        grad_logits = torch.empty(B, R, M, device=P_saved.device, dtype=torch.float32)
        grad_state = torch.empty(B, M, C_DIM, device=P_saved.device, dtype=torch.float32)

        BLOCK_ROWS = _block_size(R)
        BLOCK_COLS = _block_size(M)
        BLOCK_C = _block_size(min(C_DIM, 64))

        _sinkhorn_route_bwd_kernel[(B,)](
            grad_out_flat,
            P_saved,
            state_flat,
            grad_logits,
            grad_state,
            grad_out_flat.stride(0),
            grad_out_flat.stride(1),
            grad_out_flat.stride(2),
            P_saved.stride(0),
            P_saved.stride(1),
            P_saved.stride(2),
            state_flat.stride(0),
            state_flat.stride(1),
            state_flat.stride(2),
            grad_state.stride(0),
            grad_state.stride(1),
            grad_state.stride(2),
            ROWS=R,
            COLS=M,
            C_DIM=C_DIM,
            BLOCK_ROWS=BLOCK_ROWS,
            BLOCK_COLS=BLOCK_COLS,
            BLOCK_C=BLOCK_C,
            NUM_ITERS=ctx.num_iters_bwd,
            tau=ctx.tau,
            col_row_ratio=float(M) / float(R),
        )

        # Chain rule through alpha * raw + bias:
        #   grad_raw = grad_logits * alpha
        #   grad_alpha = sum(grad_logits * raw)  — scalar, sums over B×R×M
        #   grad_bias = grad_logits summed over batch
        alpha_val = alpha.float()
        raw_fp32 = raw_flat.float()
        grad_raw = (grad_logits * alpha_val).to(grad_output.dtype).reshape(ctx.orig_raw_shape)
        grad_alpha = (grad_logits * raw_fp32).sum().to(alpha.dtype)
        grad_bias = grad_logits.to(grad_output.dtype).sum(dim=0)  # sum over batch
        grad_state_out = grad_state.to(grad_output.dtype).reshape(ctx.orig_state_shape)

        return grad_raw, grad_alpha, grad_bias, grad_state_out, None, None, None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def triton_sinkhorn_route_fused(
    raw_logits: torch.Tensor,
    alpha: torch.Tensor,
    bias: torch.Tensor,
    state: torch.Tensor,
    num_iters: int = 20,
    tau: float = 1.0,
    num_iters_bwd: int | None = None,
) -> torch.Tensor:
    """Fused alpha*raw+bias → Sinkhorn → P@state.

    Eliminates 2 elementwise kernel launches and the intermediate logits tensor.

    Args:
        num_iters_bwd: Gauss-Seidel iterations for backward adjoint solve.
            Defaults to ``num_iters``.
    """
    return _SinkhornRouteABFn.apply(
        raw_logits.contiguous(), alpha.contiguous(), bias.contiguous(),
        state.contiguous(), num_iters, tau, num_iters_bwd or num_iters,
    )


def triton_sinkhorn_route(
    logits: torch.Tensor,
    state: torch.Tensor,
    num_iters: int = 20,
    tau: float = 1.0,
    num_iters_bwd: int | None = None,
) -> torch.Tensor:
    """Fused Sinkhorn projection + routing matmul.

    Computes ``Sinkhorn(logits) @ state`` in a single Triton kernel — the
    doubly stochastic matrix P never leaves SRAM.

    Args:
        logits: ``(..., rows, cols)`` raw Sinkhorn input.
        state:  ``(..., cols, c_dim)`` state to route.
        num_iters: Number of Sinkhorn iterations.
        tau: Temperature parameter.
        num_iters_bwd: Gauss-Seidel iterations for backward adjoint solve.
            Defaults to ``num_iters``.

    Returns:
        ``(..., rows, c_dim)`` routed state.
    """
    return _SinkhornRouteFn.apply(
        logits.contiguous(), state.contiguous(), num_iters, tau,
        num_iters_bwd or num_iters,
    )
