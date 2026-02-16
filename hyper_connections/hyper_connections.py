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

from einops import rearrange, einsum
from einops.layers.torch import Reduce

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


def sinkhorn_log(logits, num_iters=10, tau=0.05):
    """Project logits onto the set of doubly stochastic matrices via Sinkhorn.

    Operates in log-space for numerical stability.  Accepts batched inputs
    ``(..., m, m)`` and returns ``exp(...) * n`` so that row/col sums ≈ 1.
    """
    n = logits.shape[-1]
    Z = logits / tau

    u = torch.zeros(*logits.shape[:-1], device=Z.device, dtype=Z.dtype)
    v = torch.zeros(*logits.shape[:-2], n, device=Z.device, dtype=Z.dtype)

    log_marginal = -math.log(n)

    for _ in range(num_iters):
        u = log_marginal - torch.logsumexp(Z + v.unsqueeze(-2), dim=-1)
        v = log_marginal - torch.logsumexp(Z + u.unsqueeze(-1), dim=-2)

    return torch.exp(Z + u.unsqueeze(-1) + v.unsqueeze(-2)) * n


# norms


class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * (self.gamma + 1)


# stream expand/reduce


def get_expand_reduce_stream_functions(num_streams, disable=False):
    if num_streams == 1 or disable:
        return (nn.Identity(), nn.Identity())

    expand_fn = Reduce(
        pattern="b ... -> (b s) ...", reduction="repeat", s=num_streams
    )
    reduce_fn = Reduce(
        pattern="(b s) ... -> b ...", reduction="sum", s=num_streams
    )

    return expand_fn, reduce_fn


def get_init_and_expand_reduce_stream_functions(
    num_streams, disable=None, **kwargs
):
    """Return ``(init_hc, expand_stream, reduce_stream)`` factory triple.

    Extra ``**kwargs`` (e.g. legacy ``num_fracs``, ``add_stream_embed``) are
    accepted and silently ignored for backwards compatibility.
    """
    disable = default(disable, num_streams == 1)

    hyper_conn_klass = HyperConnections if not disable else Residual

    init_hyper_conn_fn = partial(hyper_conn_klass, num_streams)
    expand_reduce_fns = get_expand_reduce_stream_functions(
        num_streams, disable=disable
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
        sinkhorn_iters=10,
        sinkhorn_tau=0.05,
        mhc_residual_identity_mix=False,
        mhc_residual_alpha=0.01,
        routing_granularity=None,
        routing_bottleneck_dim=None,
        **kwargs,  # absorb unused params (mhc, mhc_h_res_proj, ns_*, etc.)
    ):
        super().__init__()
        self.branch = branch

        assert num_residual_streams > 0
        self.num_residual_streams = num_residual_streams
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
        self.sinkhorn_tau = sinkhorn_tau

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
        streams = self.num_residual_streams

        # Split out streams: (B*n, T, dim) → (B, T, n, dim)
        residuals = rearrange(residuals, "(b s) ... d -> b ... s d", s=streams)

        # Flatten bus and normalize for dynamic projections (paper Eq. 7)
        bus_flat = rearrange(residuals, "b ... s d -> b ... (s d)")
        bus_normed = self.mhc_norm(bus_flat)

        # Dynamic H^res: α·mat(x̃'·φ^res) + b^res → Sinkhorn (paper Eq. 7-8)
        if self.routing_bottleneck:
            code = bus_normed @ self.compress_res
            raw = (code @ self.to_logits_res).view(
                *bus_normed.shape[:-1], self.routing_m, self.routing_m
            )
        else:
            raw = (bus_normed @ self.phi_res).view(
                *bus_normed.shape[:-1], self.routing_m, self.routing_m
            )
        res_logits = self.alpha_res * raw + self.H_res_logits
        S = sinkhorn_log(res_logits, self.sinkhorn_iters, self.sinkhorn_tau)

        if self.mhc_residual_identity_mix:
            alpha = torch.sigmoid(self.H_res_alpha_logit)
            I = torch.eye(self.routing_m, device=S.device, dtype=S.dtype)
            H_res = (1 - alpha) * I + alpha * S
        else:
            H_res = S

        # Dynamic H^pre: softmax(α·(x̃'·φ^pre) + b^pre) (paper Eq. 7-8)
        H_pre = F.softmax(
            self.alpha_pre * (bus_normed @ self.phi_pre) + self.H_pre_logits, dim=-1
        )

        # Dynamic H^post: softmax(α·(x̃'·φ^post) + b^post) (paper Eq. 7-8)
        H_post = F.softmax(
            self.alpha_post * (bus_normed @ self.phi_post) + self.H_post_logits,
            dim=-1,
        )

        # Fine-grained routing: reshape bus to (m, c) before H^res, back to (n, dim) after.
        orig_shape = residuals.shape  # (..., n, dim)
        reshaped = residuals.reshape(
            *orig_shape[:-2], self.routing_m, self.routing_c
        )
        mixed = einsum(H_res, reshaped, "... s t, ... s c -> ... t c")
        residuals_mixed = mixed.reshape(orig_shape)

        # Apply H^pre: weighted sum of streams → branch input
        branch_input = einsum(H_pre, residuals, "... s, ... s d -> ... d")

        # Stats collection
        if getattr(self, "collect_stats", False):
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

        residuals_out = rearrange(residuals_mixed, "b ... s d -> (b s) ... d")

        return (
            branch_input,
            residuals_out,
            dict(beta=H_post, residuals_mixed=residuals_mixed),
        )

    def depth_connection(self, branch_output, residuals, *, beta, residuals_mixed):
        branch_to_streams = einsum(branch_output, beta, "... d, ... s -> ... s d")
        output = residuals_mixed + branch_to_streams
        output = rearrange(output, "b ... s d -> (b s) ... d")
        return self.dropout(output)

    def forward(self, residuals, *branch_args, **branch_kwargs):
        branch_input, residuals, residual_kwargs = self.width_connection(residuals)

        def add_residual_fn(branch_out):
            return self.depth_connection(branch_out, residuals, **residual_kwargs)

        if not exists(self.branch):
            return branch_input, add_residual_fn

        branch_output = self.branch(branch_input, *branch_args, **branch_kwargs)

        return add_residual_fn(branch_output)


HyperConnections.get_expand_reduce_stream_functions = staticmethod(
    get_expand_reduce_stream_functions
)
HyperConnections.get_init_and_expand_reduce_stream_functions = staticmethod(
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

    def __init__(self, dim, m, d, n_streams=1, sinkhorn_iters=20, sinkhorn_tau=0.05):
        super().__init__()
        self.m = m
        self.n = n_streams
        self.state_dim = n_streams * dim
        self.c_s = self.state_dim // m
        self.sinkhorn_iters = sinkhorn_iters
        self.sinkhorn_tau = sinkhorn_tau

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
        return sinkhorn_log(logits, self.sinkhorn_iters, self.sinkhorn_tau)

    def read(self, s_flat, code):
        """Assemble layer input from state via k×m coupling."""
        H_pre = self._coupling(
            code, self.alpha_pre, self.to_logits_pre, self.H_pre_logits, self.k, self.m
        )
        # (k, m) @ (m, c_s) → (k, c_s), reshape to (k*c_s) = dim
        layer_in = einsum(H_pre, s_flat, "... i j, ... j c -> ... i c")
        return layer_in.reshape(*s_flat.shape[:-2], -1), H_pre

    def write(self, s_flat, layer_output, code):
        """Align state, add output, rearrange for next layer."""
        H_align = self._coupling(
            code, self.alpha_align, self.to_logits_align, self.H_align_logits,
            self.m, self.m,
        )
        H_next = self._coupling(
            code, self.alpha_next, self.to_logits_next, self.H_next_logits,
            self.m, self.m,
        )

        s_aligned = einsum(H_align, s_flat, "... i j, ... j c -> ... i c")

        H_post = None
        if self.needs_hpost:
            H_post = self._coupling(
                code, self.alpha_post, self.to_logits_post, self.H_post_logits,
                self.m, self.k,
            )
            out_flat = layer_output.reshape(*layer_output.shape[:-1], self.k, -1)
            s_updated = s_aligned + einsum(H_post, out_flat, "... i j, ... j c -> ... i c")
        else:
            out_flat = layer_output.reshape(*s_flat.shape[:-2], self.m, self.c_s)
            s_updated = s_aligned + out_flat

        s_next = einsum(H_next, s_updated, "... i j, ... j c -> ... i c")
        return s_next, H_align, H_next, H_post

    def forward(self, s, layer_fn):
        """Full transport cycle: read → compute → write."""
        s_flat = s.reshape(*s.shape[:-1], self.m, self.c_s)
        s_norm = self.norm(s.reshape(*s.shape[:-1], self.state_dim))
        code = s_norm @ self.compress

        layer_input, H_pre = self.read(s_flat, code)
        layer_output = layer_fn(layer_input)
        s_next, H_align, H_next, H_post = self.write(s_flat, layer_output, code)

        if getattr(self, "collect_stats", False):
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

        return s_next.reshape_as(s)
