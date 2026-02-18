"""Manifold-constrained Hyper-Connections with fine-grained doubly stochastic routing.

Cleaned-up version: mHC-only (no non-mHC alpha/beta path, no frac-connections,
no channel-first, no orthostochastic projection).
"""
from __future__ import annotations

from functools import partial
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from einops import einsum

"""
ein notation:
b - batch
d - feature dimension
s - residual streams
"""

# helper functions


def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


# core: log-domain Sinkhorn (doubly stochastic projection)

from hyper_connections.triton_sinkhorn import triton_sinkhorn as _triton_sinkhorn
from hyper_connections.triton_rmsnorm import triton_rmsnorm as _triton_rmsnorm
from hyper_connections.triton_sinkhorn_route import (
    triton_sinkhorn_route as _triton_sinkhorn_route,
    triton_sinkhorn_route_fused as _triton_sinkhorn_route_fused,
)


def _sinkhorn_log_pytorch(logits, num_iters=20, tau=1.0):
    """Pure-PyTorch log-domain Sinkhorn (CPU-only fallback for testing).

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


def sinkhorn_log(logits, num_iters=20, tau=1.0, num_iters_bwd=None):
    """Project logits onto doubly stochastic matrices via Sinkhorn.

    Triton kernel on CUDA, pure-PyTorch on CPU.
    """
    if logits.is_cuda:
        return _triton_sinkhorn(logits, num_iters, tau, num_iters_bwd=num_iters_bwd)
    return _sinkhorn_log_pytorch(logits, num_iters, tau)


def sinkhorn_route(logits, state, num_iters=20, tau=1.0, num_iters_bwd=None):
    """Fused Sinkhorn projection + routing: ``Sinkhorn(logits) @ state``."""
    if logits.is_cuda:
        return _triton_sinkhorn_route(logits, state, num_iters, tau, num_iters_bwd=num_iters_bwd)
    P = _sinkhorn_log_pytorch(logits, num_iters, tau)
    return einsum(P, state, "... i j, ... j c -> ... i c")


def sinkhorn_route_fused(raw_logits, alpha, bias, state, num_iters=20, tau=1.0, num_iters_bwd=None):
    """Fused alpha*raw+bias → Sinkhorn → P@state."""
    if raw_logits.is_cuda:
        return _triton_sinkhorn_route_fused(raw_logits, alpha, bias, state, num_iters, tau, num_iters_bwd=num_iters_bwd)
    logits = alpha * raw_logits + bias
    P = _sinkhorn_log_pytorch(logits, num_iters, tau)
    return einsum(P, state, "... i j, ... j c -> ... i c")


# norms


class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.zeros(dim))
        self._dim = dim

    def forward(self, x):
        if x.is_cuda:
            return _triton_rmsnorm(x, self.gamma, self._dim)
        return F.normalize(x, dim=-1) * self.scale * (self.gamma + 1)


# stream expand/reduce


class ExpandStreams(Module):
    """Expand ``(B, ..., dim)`` → ``(B, ..., n, dim)`` via repeat (free view)."""

    def __init__(self, num_streams):
        super().__init__()
        self.n = num_streams

    def forward(self, x):
        return x.unsqueeze(-2).expand(*x.shape[:-1], self.n, x.shape[-1])


class ReduceStreams(Module):
    """Reduce ``(B, ..., n, dim)`` → ``(B, ..., dim)`` via sum."""

    def __init__(self, num_streams):
        super().__init__()
        self.n = num_streams

    def forward(self, x):
        return x.sum(dim=-2)


class ExpandChunks(Module):
    """Expand ``(B, ..., dim)`` → ``(B, ..., m, c)`` via reshape+repeat (for FTR)."""

    def __init__(self, num_streams, m, c):
        super().__init__()
        self.n = num_streams
        self.m = m
        self.c = c

    def forward(self, x):
        # (B, ..., dim) → (B, ..., m, c) via repeat n times then reshape
        if self.n == 1:
            return x.reshape(*x.shape[:-1], self.m, self.c)
        # n>1: (B, ..., dim) → (B, ..., n*dim) → (B, ..., m, c)
        expanded = x.unsqueeze(-2).expand(*x.shape[:-1], self.n, x.shape[-1])
        flat = expanded.reshape(*x.shape[:-1], self.n * x.shape[-1])
        return flat.reshape(*x.shape[:-1], self.m, self.c)


class ReduceChunks(Module):
    """Reduce ``(B, ..., m, c)`` → ``(B, ..., dim)`` via reshape+sum (for FTR)."""

    def __init__(self, num_streams, dim):
        super().__init__()
        self.n = num_streams
        self.dim = dim

    def forward(self, x):
        if self.n == 1:
            return x.reshape(*x.shape[:-2], self.dim)
        # n>1: (B, ..., m, c) → (B, ..., n, dim) → sum → (B, ..., dim)
        return x.reshape(*x.shape[:-2], self.n, self.dim).sum(dim=-2)


def get_expand_reduce_stream_functions(num_streams, disable=False, layout="streams", m=None, c=None):
    if num_streams == 1 or disable:
        return (nn.Identity(), nn.Identity())

    if layout == "chunks":
        assert m is not None and c is not None
        dim = num_streams * c * m // (num_streams)  # c * m / n = dim when m*c = n*dim
        # Actually: n*dim = m*c, so dim = m*c/n
        dim = m * c // num_streams
        return ExpandChunks(num_streams, m, c), ReduceChunks(num_streams, dim)

    # Default: "streams" layout → (B, ..., n, dim)
    return ExpandStreams(num_streams), ReduceStreams(num_streams)


def get_init_and_expand_reduce_stream_functions(
    num_streams, disable=None, layout="streams", m=None, c=None, **kwargs
):
    """Return ``(init_hc, expand_stream, reduce_stream)`` factory triple.

    Extra ``**kwargs`` (e.g. legacy ``num_fracs``, ``add_stream_embed``) are
    accepted and silently ignored for backwards compatibility.
    """
    disable = default(disable, num_streams == 1)

    hyper_conn_klass = HyperConnections if not disable else Residual

    init_hyper_conn_fn = partial(hyper_conn_klass, num_streams)
    expand_reduce_fns = get_expand_reduce_stream_functions(
        num_streams, disable=disable, layout=layout, m=m, c=c
    )

    return (init_hyper_conn_fn, *expand_reduce_fns)


# residual (baseline skip connection)


class Residual(Module):
    """Simple skip connection: ``output = branch(x) + x``.

    Accepts and ignores extra constructor kwargs so it can be used as a
    drop-in replacement for :class:`HyperConnections` via ``partial()``.
    """

    def __init__(self, *args, branch=None, **kwargs):
        super().__init__()
        self.branch = branch

    def forward(self, residuals, *branch_args, **branch_kwargs):
        if not exists(self.branch):
            return residuals, lambda branch_out: branch_out + residuals

        branch_output = self.branch(residuals, *branch_args, **branch_kwargs)
        return branch_output + residuals


# hyper-connections (mHC routing)


class HyperConnections(Module):
    """Manifold-constrained Hyper-Connections with fine-grained routing.

    Implements doubly stochastic routing on multi-stream residual states.
    See https://arxiv.org/abs/2409.19606 (Algorithm 2) and
    https://arxiv.org/abs/2512.24880 for mHC.

    Supports fine-grained routing (``routing_granularity``) and bottleneck
    conditioning (``routing_bottleneck_dim``).
    """

    def __init__(
        self,
        num_residual_streams,
        *,
        dim,
        branch=None,
        layer_index=None,
        dropout=0.0,
        sinkhorn_iters=20,
        sinkhorn_iters_bwd=None,
        sinkhorn_tau=1.0,
        mhc_residual_identity_mix=False,
        mhc_residual_alpha=0.01,
        routing_granularity=None,
        routing_bottleneck_dim=None,
        sinkhorn_checkpoint=False,
        **kwargs,  # absorb unused params (mhc, mhc_h_res_proj, ns_*, etc.)
    ):
        super().__init__()
        self.branch = branch

        assert num_residual_streams > 0
        self.num_residual_streams = num_residual_streams
        self._single_stream = num_residual_streams == 1
        init_residual_index = (
            default(layer_index, 0) % num_residual_streams
        )

        # Fine-grained routing: m controls H^res resolution.
        # When m == num_residual_streams (default), behavior is identical to original mHC.
        m = routing_granularity if routing_granularity is not None else num_residual_streams
        total_bus_dim = num_residual_streams * dim
        assert total_bus_dim % m == 0, (
            f"routing_granularity ({m}) must divide n_streams * dim ({total_bus_dim})"
        )
        self.routing_m = m
        self.routing_c = total_bus_dim // m

        # RMSNorm for dynamic projection input (paper Eq. 7)
        self.mhc_norm = RMSNorm(total_bus_dim)

        self.sinkhorn_iters = sinkhorn_iters
        self.sinkhorn_iters_bwd = sinkhorn_iters_bwd
        self.sinkhorn_tau = sinkhorn_tau
        self.sinkhorn_checkpoint = sinkhorn_checkpoint

        # H^res: static bias b^res + dynamic α·mat(x̃'·φ^res) (paper Eq. 7)
        H_res_init = torch.full((m, m), -8.0)
        H_res_init.fill_diagonal_(0.0)
        self.H_res_logits = nn.Parameter(H_res_init)
        self.alpha_res = nn.Parameter(torch.ones(()) * 1e-2)

        # Conditioning pathway: direct (nC → m²) or bottleneck (nC → d → m²)
        self.routing_bottleneck = routing_bottleneck_dim is not None
        if self.routing_bottleneck:
            d = routing_bottleneck_dim
            self.compress_res = nn.Parameter(torch.zeros(total_bus_dim, d))
            self.to_logits_res = nn.Parameter(torch.zeros(d, m * m))
        else:
            self.phi_res = nn.Parameter(torch.zeros(total_bus_dim, m * m))

        # H^pre: selects stream for branch input (paper Eq. 7-8)
        H_pre_init = torch.full((num_residual_streams,), -8.0)
        H_pre_init[init_residual_index] = 0.0
        self.H_pre_logits = nn.Parameter(H_pre_init)
        self.phi_pre = nn.Parameter(torch.zeros(total_bus_dim, num_residual_streams))
        self.alpha_pre = nn.Parameter(torch.ones(()) * 1e-2)

        # H^post: distributes branch output back to streams (paper Eq. 7-8)
        self.H_post_logits = nn.Parameter(torch.zeros(num_residual_streams))
        self.phi_post = nn.Parameter(torch.zeros(total_bus_dim, num_residual_streams))
        self.alpha_post = nn.Parameter(torch.ones(()) * 1e-2)

        # Identity mix mode: H_res = (1-α)·I + α·S
        self.mhc_residual_identity_mix = mhc_residual_identity_mix
        if mhc_residual_identity_mix:
            alpha_clamped = max(1e-4, min(1 - 1e-4, mhc_residual_alpha))
            alpha_logit_init = math.log(alpha_clamped / (1 - alpha_clamped))
            self.H_res_alpha_logit = nn.Parameter(torch.tensor(alpha_logit_init))

        self.dropout = nn.Dropout(dropout)

    def width_connection(self, residuals):
        # residuals: (B, T, n, dim) for n>1, or (B, T, dim) for n=1 (Identity expand)
        streams = self.num_residual_streams

        # For n=1, add stream dim so all paths see (B, T, n, dim)
        if self._single_stream and residuals.shape[-1] != streams:
            residuals = residuals.unsqueeze(-2)  # (B,T,dim) → (B,T,1,dim) — free view

        # Flatten bus and normalize for dynamic projections (paper Eq. 7)
        bus_flat = residuals.reshape(*residuals.shape[:-2], streams * residuals.shape[-1])
        bus_normed = self.mhc_norm(bus_flat)

        # --- Pre-cast all routing params to activation dtype once ---
        # Prevents autocast from triggering aten::copy_ per-matmul per-layer.
        dtype = residuals.dtype
        if self.routing_bottleneck:
            _compress_res = self.compress_res.to(dtype)
            _to_logits_res = self.to_logits_res.to(dtype)
        else:
            _phi_res = self.phi_res.to(dtype)
        if not self._single_stream:
            _phi_pre = self.phi_pre.to(dtype)
            _phi_post = self.phi_post.to(dtype)
            _alpha_pre = self.alpha_pre.to(dtype)
            _H_pre_logits = self.H_pre_logits.to(dtype)
            _alpha_post = self.alpha_post.to(dtype)
            _H_post_logits = self.H_post_logits.to(dtype)

        # --- 1A + 1D: Batched φ projections ---
        m = self.routing_m
        if self.routing_bottleneck:
            if self._single_stream:
                # 1D: n=1, skip phi_pre entirely
                code = bus_normed @ _compress_res
                res_raw = code @ _to_logits_res
            else:
                # 1A: Batch compress_res + phi_pre into one matmul
                both = bus_normed @ torch.cat([_compress_res, _phi_pre], dim=-1)
                code, pre_raw = both.split([_compress_res.shape[-1], streams], dim=-1)
                res_raw = code @ _to_logits_res
        else:
            if self._single_stream:
                # 1D: n=1, skip phi_pre entirely
                res_raw = bus_normed @ _phi_res
            else:
                # 1A: Batch phi_res + phi_pre into one matmul
                both = bus_normed @ torch.cat([_phi_res, _phi_pre], dim=-1)
                res_raw, pre_raw = both.split([m * m, streams], dim=-1)

        res_raw = res_raw.view(*bus_normed.shape[:-1], m, m)

        # Fine-grained routing: reshape bus to (m, c) before H^res, back to (n, dim) after.
        orig_shape = residuals.shape  # (..., n, dim)
        reshaped = residuals.reshape(
            *orig_shape[:-2], self.routing_m, self.routing_c
        )

        # Use fused sinkhorn_route when we don't need intermediate S
        _need_S = self.mhc_residual_identity_mix or getattr(self, "collect_stats", False)
        if _need_S:
            # Unfused path: need access to the doubly stochastic matrix
            _alpha_res = self.alpha_res.to(dtype)
            _H_res_logits = self.H_res_logits.to(dtype)
            res_logits = _alpha_res * res_raw + _H_res_logits
            if self.sinkhorn_checkpoint and res_logits.requires_grad:
                S = torch_checkpoint(sinkhorn_log, res_logits, self.sinkhorn_iters, self.sinkhorn_tau, self.sinkhorn_iters_bwd, use_reentrant=False)
            else:
                S = sinkhorn_log(res_logits, self.sinkhorn_iters, self.sinkhorn_tau, self.sinkhorn_iters_bwd)

            if self.mhc_residual_identity_mix:
                alpha = torch.sigmoid(self.H_res_alpha_logit)
                I = torch.eye(self.routing_m, device=S.device, dtype=S.dtype)
                H_res = (1 - alpha) * I + alpha * S
            else:
                H_res = S

            mixed = einsum(H_res, reshaped, "... s t, ... s c -> ... t c")
        else:
            # 2A: Fused alpha*raw+bias → Sinkhorn → P@state in one kernel
            mixed = sinkhorn_route_fused(
                res_raw, self.alpha_res, self.H_res_logits, reshaped,
                self.sinkhorn_iters, self.sinkhorn_tau, self.sinkhorn_iters_bwd,
            )

        residuals_mixed = mixed.reshape(orig_shape)

        # --- 1D: Short-circuit H^pre/H^post for n=1 ---
        if self._single_stream:
            # softmax([x]) = [1.0] always; einsum is identity squeeze
            branch_input = residuals[..., 0, :]
            H_post = None
        else:
            # Dynamic H^pre: softmax(α·(x̃'·φ^pre) + b^pre) (paper Eq. 7-8)
            H_pre = F.softmax(_alpha_pre * pre_raw + _H_pre_logits, dim=-1)
            branch_input = einsum(H_pre, residuals, "... s, ... s d -> ... d")

            # --- 1B: H^post from routed bus state (not stale original) ---
            routed_flat = residuals_mixed.reshape(*residuals_mixed.shape[:-2], streams * residuals_mixed.shape[-1])
            routed_normed = self.mhc_norm(routed_flat)
            H_post = F.softmax(
                _alpha_post * (routed_normed @ _phi_post) + _H_post_logits,
                dim=-1,
            )

        # Stats collection (only on unfused path where H_res is available)
        if getattr(self, "collect_stats", False) and not self._single_stream:
            with torch.no_grad():
                H_res_clamped = H_res.clamp(min=1e-8)
                stats = dict(
                    h_res_min=H_res.min(),
                    h_res_max=H_res.max(),
                    h_res_row_sum=H_res.sum(dim=-1).mean(),
                    h_res_col_sum=H_res.sum(dim=-2).mean(),
                    h_res_entropy=-(H_res_clamped * H_res_clamped.log()).sum(dim=(-2, -1)).mean(),
                    h_res_sparsity=(H_res < 0.01).float().mean(),
                    h_pre_min=H_pre.min(),
                    h_post_min=H_post.min(),
                )
                if self.mhc_residual_identity_mix:
                    stats["h_res_alpha"] = torch.sigmoid(self.H_res_alpha_logit)
                self.last_stats = {k: v.detach() for k, v in stats.items()}

        return (
            branch_input,
            residuals_mixed,  # (B, T, n, dim) — no copy needed
            dict(beta=H_post, residuals_mixed=residuals_mixed),
        )

    def depth_connection(self, branch_output, residuals, *, beta, residuals_mixed):
        # Returns (B, T, n, dim) for n>1, or (B, T, dim) for n=1
        if self._single_stream:
            # 1D: n=1, H_post=[1.0], just add directly (no stream dim in output)
            output = residuals_mixed.squeeze(-2) + branch_output
        else:
            branch_to_streams = einsum(branch_output, beta, "... d, ... s -> ... s d")
            output = residuals_mixed + branch_to_streams
        return self.dropout(output)

    def forward(self, residuals, *branch_args, **branch_kwargs):
        branch_input, residuals, residual_kwargs = self.width_connection(residuals)

        def add_residual_fn(branch_out):
            return self.depth_connection(branch_out, residuals, **residual_kwargs)

        if not exists(self.branch):
            return branch_input, add_residual_fn

        branch_output = self.branch(branch_input, *branch_args, **branch_kwargs)

        return add_residual_fn(branch_output)


HyperConnections.get_expand_reduce_stream_functions = staticmethod(  # type: ignore[attr-defined]
    get_expand_reduce_stream_functions
)
HyperConnections.get_init_and_expand_reduce_stream_functions = staticmethod(  # type: ignore[attr-defined]
    get_init_and_expand_reduce_stream_functions
)


# full transport routing (Level 4)


class FullTransportRouting(Module):
    """Full read/align/prepare transport cycle with distinct coupling matrices.

    Unlike HyperConnections (which wraps width+depth connections around a branch),
    this class wraps the entire layer function in a read→compute→write cycle with
    three (or four) separate doubly stochastic coupling matrices:
      - H^pre:   assemble layer input from state (k×m, where k=m/n)
      - H^align: position state for output absorption (m×m)
      - H^next:  prepare combined state for next layer (m×m)
      - H^post:  distribute output back into state (m×k, only when n>1)

    When n=1, k=m so H^pre is square and H^post is not needed.

    All couplings share a single conditioning bottleneck (norm → compress → code).
    """

    def __init__(self, dim, m, d, n_streams=1, sinkhorn_iters=20, sinkhorn_iters_bwd=None, sinkhorn_tau=1.0, sinkhorn_checkpoint=False):
        super().__init__()
        self.m = m
        self.n = n_streams
        self.state_dim = n_streams * dim
        self.c_s = self.state_dim // m
        self.sinkhorn_iters = sinkhorn_iters
        self.sinkhorn_iters_bwd = sinkhorn_iters_bwd
        self.sinkhorn_tau = sinkhorn_tau
        self.sinkhorn_checkpoint = sinkhorn_checkpoint

        assert self.state_dim % m == 0, (
            f"m ({m}) must divide n_streams * dim ({self.state_dim})"
        )

        # k = layer-input chunks (k*c_s = dim). For n=1, k=m.
        self.k = dim // self.c_s
        assert dim % self.c_s == 0, (
            f"dim ({dim}) must be divisible by c_s ({self.c_s})"
        )

        # Shared conditioning bottleneck
        self.norm = RMSNorm(self.state_dim)
        self.compress = nn.Parameter(torch.zeros(self.state_dim, d))

        # H^pre: k×m -- read k layer-input chunks from m state chunks
        H_pre_init = torch.full((self.k, m), -8.0)
        for i in range(min(self.k, m)):
            H_pre_init[i, i] = 0.0
        self.H_pre_logits = nn.Parameter(H_pre_init)
        self.alpha_pre = nn.Parameter(torch.ones(()) * 1e-2)
        self.to_logits_pre = nn.Parameter(torch.zeros(d, self.k * m))

        # H^align: m×m -- position state for output absorption
        H_align_init = torch.full((m, m), -8.0)
        H_align_init.fill_diagonal_(0.0)
        self.H_align_logits = nn.Parameter(H_align_init)
        self.alpha_align = nn.Parameter(torch.ones(()) * 1e-2)
        self.to_logits_align = nn.Parameter(torch.zeros(d, m * m))

        # H^next: m×m -- prepare combined state for next layer
        H_next_init = torch.full((m, m), -8.0)
        H_next_init.fill_diagonal_(0.0)
        self.H_next_logits = nn.Parameter(H_next_init)
        self.alpha_next = nn.Parameter(torch.ones(()) * 1e-2)
        self.to_logits_next = nn.Parameter(torch.zeros(d, m * m))

        # H^post: m×k -- write k output chunks back into m state chunks (only when n>1)
        self.needs_hpost = n_streams > 1
        if self.needs_hpost:
            H_post_init = torch.full((self.m, self.k), -8.0)
            for i in range(min(self.m, self.k)):
                H_post_init[i, i % self.k] = 0.0
            self.H_post_logits = nn.Parameter(H_post_init)
            self.alpha_post = nn.Parameter(torch.ones(()) * 1e-2)
            self.to_logits_post = nn.Parameter(torch.zeros(d, m * self.k))

    def _coupling(self, code, alpha, to_logits, bias, rows, cols):
        logits = alpha * (code @ to_logits).view(*code.shape[:-1], rows, cols) + bias
        if self.sinkhorn_checkpoint and logits.requires_grad:
            return torch_checkpoint(sinkhorn_log, logits, self.sinkhorn_iters, self.sinkhorn_tau, self.sinkhorn_iters_bwd, use_reentrant=False)
        return sinkhorn_log(logits, self.sinkhorn_iters, self.sinkhorn_tau, self.sinkhorn_iters_bwd)

    def _coupling_route(self, code, alpha, to_logits, bias, state, rows, cols):
        """Fused coupling + routing: computes Sinkhorn(alpha*raw+bias) @ state."""
        raw = (code @ to_logits).view(*code.shape[:-1], rows, cols)
        return sinkhorn_route_fused(raw, alpha, bias, state, self.sinkhorn_iters, self.sinkhorn_tau, self.sinkhorn_iters_bwd)

    def read(self, s_flat, code, alpha_pre, to_logits_pre, H_pre_logits, need_coupling=False):
        """Assemble layer input from state via k×m coupling."""
        if need_coupling:
            H_pre = self._coupling(
                code, alpha_pre, to_logits_pre, H_pre_logits, self.k, self.m
            )
            layer_in = einsum(H_pre, s_flat, "... i j, ... j c -> ... i c")
            return layer_in.reshape(*s_flat.shape[:-2], -1), H_pre

        # Fused path: Sinkhorn + routing in one kernel
        layer_in = self._coupling_route(
            code, alpha_pre, to_logits_pre, H_pre_logits,
            s_flat, self.k, self.m,
        )
        return layer_in.reshape(*s_flat.shape[:-2], -1), None

    def write(self, s_flat, layer_output, code,
              alpha_align, to_logits_align, H_align_logits,
              alpha_next, to_logits_next, H_next_logits,
              alpha_post, to_logits_post, H_post_logits,
              need_coupling=False):
        """Align state, add output, rearrange for next layer."""
        if need_coupling:
            # Unfused path: need H matrices for stats
            H_align = self._coupling(
                code, alpha_align, to_logits_align, H_align_logits,
                self.m, self.m,
            )
            H_next = self._coupling(
                code, alpha_next, to_logits_next, H_next_logits,
                self.m, self.m,
            )
            s_aligned = einsum(H_align, s_flat, "... i j, ... j c -> ... i c")

            H_post = None
            if self.needs_hpost:
                H_post = self._coupling(
                    code, alpha_post, to_logits_post, H_post_logits,
                    self.m, self.k,
                )
                out_flat = layer_output.reshape(*layer_output.shape[:-1], self.k, -1)
                s_updated = s_aligned + einsum(H_post, out_flat, "... i j, ... j c -> ... i c")
            else:
                out_flat = layer_output.reshape(*s_flat.shape[:-2], self.m, self.c_s)
                s_updated = s_aligned + out_flat

            s_next = einsum(H_next, s_updated, "... i j, ... j c -> ... i c")
            return s_next, H_align, H_next, H_post

        # Fused path: Sinkhorn + routing in one kernel per coupling
        s_aligned = self._coupling_route(
            code, alpha_align, to_logits_align, H_align_logits,
            s_flat, self.m, self.m,
        )

        if self.needs_hpost:
            out_flat = layer_output.reshape(*layer_output.shape[:-1], self.k, -1)
            s_updated = s_aligned + self._coupling_route(
                code, alpha_post, to_logits_post, H_post_logits,
                out_flat, self.m, self.k,
            )
        else:
            out_flat = layer_output.reshape(*s_flat.shape[:-2], self.m, self.c_s)
            s_updated = s_aligned + out_flat

        s_next = self._coupling_route(
            code, alpha_next, to_logits_next, H_next_logits,
            s_updated, self.m, self.m,
        )
        return s_next, None, None, None

    def forward(self, s, layer_fn):
        """Full transport cycle: read → compute → write.

        Accepts either ``(B, T, m, c)`` (chunks layout) or ``(B, T, state_dim)``
        (flat layout).  Both reshape to ``(m, c)`` internally as free views.
        """
        if s.ndim >= 3 and s.shape[-2] == self.m and s.shape[-1] == self.c_s:
            s_flat = s  # already (B, T, m, c) — no reshape needed
        else:
            s_flat = s.reshape(*s.shape[:-1], self.m, self.c_s)
        s_norm = self.norm(s_flat.reshape(*s_flat.shape[:-2], self.state_dim))

        # Pre-cast matmul weights to activation dtype once (bf16 under autocast).
        # Alpha scalars and H_*_logits biases stay fp32 — the fused kernel casts internally.
        dtype = s.dtype
        _compress = self.compress.to(dtype)
        _to_logits_pre = self.to_logits_pre.to(dtype)
        _to_logits_align = self.to_logits_align.to(dtype)
        _to_logits_next = self.to_logits_next.to(dtype)
        if self.needs_hpost:
            _to_logits_post = self.to_logits_post.to(dtype)

        code = s_norm @ _compress

        _need_stats = getattr(self, "collect_stats", False)
        layer_input, H_pre = self.read(
            s_flat, code, self.alpha_pre, _to_logits_pre, self.H_pre_logits,
            need_coupling=_need_stats,
        )
        layer_output = layer_fn(layer_input)
        s_next, H_align, H_next, H_post = self.write(
            s_flat, layer_output, code,
            self.alpha_align, _to_logits_align, self.H_align_logits,
            self.alpha_next, _to_logits_next, self.H_next_logits,
            self.alpha_post if self.needs_hpost else None,
            _to_logits_post if self.needs_hpost else None,
            self.H_post_logits if self.needs_hpost else None,
            need_coupling=_need_stats,
        )

        if _need_stats:
            with torch.no_grad():
                def _coupling_stats(H, prefix):
                    H_c = H.clamp(min=1e-8)
                    return {
                        f"{prefix}_min": H.min(),
                        f"{prefix}_max": H.max(),
                        f"{prefix}_row_sum": H.sum(dim=-1).mean(),
                        f"{prefix}_col_sum": H.sum(dim=-2).mean(),
                        f"{prefix}_entropy": -(H_c * H_c.log()).sum(dim=(-2, -1)).mean(),
                        f"{prefix}_sparsity": (H < 0.01).float().mean(),
                    }
                stats = {}
                stats.update(_coupling_stats(H_pre, "h_pre"))
                stats.update(_coupling_stats(H_align, "h_align"))
                stats.update(_coupling_stats(H_next, "h_next"))
                stats["alpha_pre"] = self.alpha_pre.detach()
                stats["alpha_align"] = self.alpha_align.detach()
                stats["alpha_next"] = self.alpha_next.detach()
                if H_post is not None:
                    stats.update(_coupling_stats(H_post, "h_post"))
                    stats["alpha_post"] = self.alpha_post.detach()
                self.last_stats = {k: v.detach() for k, v in stats.items()}

        if s_next.shape == s.shape:
            return s_next
        return s_next.reshape_as(s)
