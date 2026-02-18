"""Fused Triton RMSNorm kernel.

Replaces F.normalize + scale + gamma with a single kernel launch.
Computes: out = x / sqrt(mean(x²) + eps) * (gamma + 1)

Public API:
    triton_rmsnorm(x, gamma, dim) -> result
"""
from __future__ import annotations

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


if _TRITON_AVAILABLE:

    @triton.jit
    def _rmsnorm_fwd_kernel(
        X_ptr,
        Gamma_ptr,
        Out_ptr,
        RMS_ptr,  # save rms per row for backward
        stride_x,
        N: tl.constexpr,
        BLOCK: tl.constexpr,
        eps: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        mask = offs < N

        x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)
        gamma = tl.load(Gamma_ptr + offs, mask=mask, other=0.0).to(tl.float32)

        # RMS = sqrt(mean(x²) + eps)
        var = tl.sum(x * x) / N
        rms = tl.sqrt(var + eps)

        # Normalize and scale
        x_norm = x / rms
        out = x_norm * (gamma + 1.0)

        tl.store(Out_ptr + row * stride_x + offs, out, mask=mask)
        tl.store(RMS_ptr + row, rms)

    @triton.jit
    def _rmsnorm_bwd_kernel(
        GradOut_ptr,
        X_ptr,
        RMS_ptr,
        Gamma_ptr,
        GradX_ptr,
        stride_x,
        N: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        mask = offs < N

        g_out = tl.load(GradOut_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)
        rms = tl.load(RMS_ptr + row).to(tl.float32)
        gamma = tl.load(Gamma_ptr + offs, mask=mask, other=0.0).to(tl.float32)

        g = gamma + 1.0
        x_hat = x / rms
        grad_x_hat = g_out * g

        # grad_x = (grad_x_hat - x_hat * mean(grad_x_hat * x_hat)) / rms
        dot = tl.sum(grad_x_hat * x_hat) / N
        grad_x = (grad_x_hat - x_hat * dot) / rms

        tl.store(GradX_ptr + row * stride_x + offs, grad_x, mask=mask)


class _TritonRMSNormFn(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, gamma: torch.Tensor, dim: int) -> torch.Tensor:
        orig_shape = x.shape
        N = orig_shape[-1]
        B = x.numel() // N
        x_flat = x.reshape(B, N).contiguous()

        out = torch.empty_like(x_flat)
        rms_saved = torch.empty(B, device=x.device, dtype=torch.float32)

        BLOCK = triton.next_power_of_2(N)

        _rmsnorm_fwd_kernel[(B,)](
            x_flat, gamma, out, rms_saved,
            x_flat.stride(0),
            N=N, BLOCK=BLOCK, eps=1e-8,
        )

        ctx.save_for_backward(x_flat, rms_saved, gamma)
        ctx.N = N
        ctx.orig_shape = orig_shape
        return out.reshape(orig_shape)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        x_flat, rms_saved, gamma = ctx.saved_tensors
        N = ctx.N
        B = x_flat.shape[0]

        grad_flat = grad_output.reshape(B, N).contiguous()
        grad_x = torch.empty_like(x_flat)

        BLOCK = triton.next_power_of_2(N)

        _rmsnorm_bwd_kernel[(B,)](
            grad_flat, x_flat, rms_saved, gamma, grad_x,
            x_flat.stride(0),
            N=N, BLOCK=BLOCK,
        )

        # grad_gamma = sum over batch of (grad_out * x_hat)
        x_hat = x_flat / rms_saved.unsqueeze(1)
        grad_gamma = (grad_flat.float() * x_hat).sum(dim=0).to(gamma.dtype)

        return grad_x.to(grad_output.dtype).reshape(ctx.orig_shape), grad_gamma, None


def triton_rmsnorm(x: torch.Tensor, gamma: torch.Tensor, dim: int) -> torch.Tensor:
    """Fused RMSNorm: x / sqrt(mean(x²) + eps) * (gamma + 1)."""
    return _TritonRMSNormFn.apply(x, gamma, dim)
