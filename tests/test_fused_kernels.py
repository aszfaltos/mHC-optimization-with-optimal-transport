"""Tests for fused Triton kernels: RMSNorm and sinkhorn_route_fused (alpha+bias).

Run: uv run pytest tests/test_fused_kernels.py -v
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from hyper_connections.hyper_connections import RMSNorm, sinkhorn_route_fused
from tests.sinkhorn_ref import sinkhorn_log_ref, make_test_logits, EDGE_CASES

# --- Triton availability checks ---

try:
    from hyper_connections.triton_rmsnorm import triton_rmsnorm, is_available as rmsnorm_available
except ImportError:
    rmsnorm_available = lambda: False
    triton_rmsnorm = None

try:
    from hyper_connections.triton_sinkhorn_route import (
        triton_sinkhorn_route_fused,
        is_available as route_available,
    )
except ImportError:
    route_available = lambda: False
    triton_sinkhorn_route_fused = None

requires_triton = pytest.mark.skipif(
    not torch.cuda.is_available() or triton_rmsnorm is None or not rmsnorm_available(),
    reason="Triton + CUDA required",
)

requires_triton_route = pytest.mark.skipif(
    not torch.cuda.is_available()
    or triton_sinkhorn_route_fused is None
    or not route_available(),
    reason="Triton + CUDA required",
)


# ========================================================================== #
#  RMSNorm tests
# ========================================================================== #


def _rmsnorm_pytorch_ref(x, gamma):
    """Reference RMSNorm: x / sqrt(mean(x²) + eps) * (gamma + 1)."""
    rms = torch.sqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + 1e-8)
    return (x.float() / rms * (gamma.float() + 1.0)).to(x.dtype)


@requires_triton
@pytest.mark.parametrize("N", [64, 128, 256, 512])
def test_rmsnorm_forward(N):
    """Triton RMSNorm matches reference."""
    torch.manual_seed(42)
    x = torch.randn(32, N, device="cuda")
    gamma = torch.randn(N, device="cuda") * 0.01

    ref = _rmsnorm_pytorch_ref(x, gamma)
    out = triton_rmsnorm(x, gamma, N)

    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)


@requires_triton
def test_rmsnorm_matches_module():
    """Triton RMSNorm matches the RMSNorm Module (which uses F.normalize)."""
    torch.manual_seed(43)
    N = 256
    x = torch.randn(16, 64, N, device="cuda")
    norm = RMSNorm(N).to("cuda")

    # Module forward (dispatches to Triton on CUDA)
    out_module = norm(x)

    # Manual F.normalize reference
    ref = F.normalize(x, dim=-1) * norm.scale * (norm.gamma + 1)

    torch.testing.assert_close(out_module, ref, atol=1e-4, rtol=1e-4)


@requires_triton
def test_rmsnorm_backward():
    """Triton RMSNorm backward matches PyTorch autograd."""
    torch.manual_seed(44)
    N = 256

    # Reference
    x_ref = torch.randn(16, N, device="cuda", requires_grad=True)
    gamma_ref = torch.randn(N, device="cuda", requires_grad=True)
    out_ref = _rmsnorm_pytorch_ref(x_ref, gamma_ref)
    out_ref.sum().backward()

    # Triton
    x_tri = x_ref.detach().clone().requires_grad_(True)
    gamma_tri = gamma_ref.detach().clone().requires_grad_(True)
    out_tri = triton_rmsnorm(x_tri, gamma_tri, N)
    out_tri.sum().backward()

    torch.testing.assert_close(x_tri.grad, x_ref.grad, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(gamma_tri.grad, gamma_ref.grad, atol=1e-3, rtol=1e-3)


@requires_triton
def test_rmsnorm_bf16():
    """Works with bf16 inputs."""
    torch.manual_seed(45)
    N = 256
    x = torch.randn(32, N, device="cuda", dtype=torch.bfloat16)
    gamma = torch.zeros(N, device="cuda")  # fp32 param (as in actual usage)

    out = triton_rmsnorm(x, gamma, N)
    assert torch.isfinite(out).all()


@requires_triton
def test_rmsnorm_3d_input():
    """Works with (B, T, N) shaped input."""
    torch.manual_seed(46)
    N = 128
    x = torch.randn(4, 32, N, device="cuda")
    gamma = torch.randn(N, device="cuda") * 0.01

    ref = _rmsnorm_pytorch_ref(x, gamma)
    out = triton_rmsnorm(x, gamma, N)

    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)


# ========================================================================== #
#  sinkhorn_route_fused (alpha+bias) tests
# ========================================================================== #


@requires_triton_route
@pytest.mark.parametrize("m", [4, 8, 16, 32])
@pytest.mark.parametrize("c", [16, 64])
def test_route_fused_forward(m, c):
    """Fused alpha*raw+bias → Sinkhorn → route matches separate ops."""
    torch.manual_seed(50)
    B = 32
    raw = torch.randn(B, m, m, device="cuda")
    alpha = torch.tensor(0.01, device="cuda")
    bias = torch.randn(m, m, device="cuda") * 0.5
    state = torch.randn(B, m, c, device="cuda")

    # Reference: manual alpha*raw+bias then sinkhorn+route
    logits = alpha * raw + bias
    P = sinkhorn_log_ref(logits, num_iters=20, tau=1.0)
    ref = torch.bmm(P, state)

    # Fused
    out = triton_sinkhorn_route_fused(raw, alpha, bias, state, num_iters=20, tau=1.0)

    torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)


@requires_triton_route
@pytest.mark.parametrize("m", [4, 8, 16])
def test_route_fused_backward_grad_raw(m):
    """grad_raw from fused kernel matches PyTorch autograd."""
    torch.manual_seed(51)
    c = 32
    raw = torch.randn(16, m, m, device="cuda")
    alpha = torch.tensor(0.01, device="cuda")
    bias = torch.randn(m, m, device="cuda") * 0.5
    state = torch.randn(16, m, c, device="cuda")

    # Reference
    raw_ref = raw.clone().requires_grad_(True)
    alpha_ref = alpha.clone().requires_grad_(True)
    bias_ref = bias.clone().requires_grad_(True)
    logits = alpha_ref * raw_ref + bias_ref
    P = sinkhorn_log_ref(logits, num_iters=20, tau=1.0)
    (P @ state).sum().backward()

    # Fused
    raw_tri = raw.clone().requires_grad_(True)
    alpha_tri = alpha.clone().requires_grad_(True)
    bias_tri = bias.clone().requires_grad_(True)
    out = triton_sinkhorn_route_fused(raw_tri, alpha_tri, bias_tri, state, num_iters=20, tau=1.0)
    out.sum().backward()

    torch.testing.assert_close(raw_tri.grad, raw_ref.grad, atol=5e-3, rtol=5e-3)


@requires_triton_route
@pytest.mark.parametrize("m", [4, 8, 16])
def test_route_fused_backward_grad_alpha(m):
    """grad_alpha from fused kernel matches PyTorch autograd."""
    torch.manual_seed(52)
    c = 32
    raw = torch.randn(16, m, m, device="cuda")
    alpha = torch.tensor(0.01, device="cuda")
    bias = torch.randn(m, m, device="cuda") * 0.5
    state = torch.randn(16, m, c, device="cuda")

    # Reference
    raw_ref = raw.clone().detach()
    alpha_ref = alpha.clone().requires_grad_(True)
    logits = alpha_ref * raw_ref + bias
    P = sinkhorn_log_ref(logits, num_iters=20, tau=1.0)
    (P @ state).sum().backward()

    # Fused
    alpha_tri = alpha.clone().requires_grad_(True)
    out = triton_sinkhorn_route_fused(raw, alpha_tri, bias, state, num_iters=20, tau=1.0)
    out.sum().backward()

    torch.testing.assert_close(alpha_tri.grad, alpha_ref.grad, atol=5e-3, rtol=5e-3)


@requires_triton_route
@pytest.mark.parametrize("m", [4, 8, 16])
def test_route_fused_backward_grad_bias(m):
    """grad_bias from fused kernel matches PyTorch autograd."""
    torch.manual_seed(53)
    c = 32
    raw = torch.randn(16, m, m, device="cuda")
    alpha = torch.tensor(0.01, device="cuda")
    bias = torch.randn(m, m, device="cuda")
    state = torch.randn(16, m, c, device="cuda")

    # Reference
    bias_ref = bias.clone().requires_grad_(True)
    logits = alpha * raw + bias_ref
    P = sinkhorn_log_ref(logits, num_iters=20, tau=1.0)
    (P @ state).sum().backward()

    # Fused
    bias_tri = bias.clone().requires_grad_(True)
    out = triton_sinkhorn_route_fused(raw, alpha, bias_tri, state, num_iters=20, tau=1.0)
    out.sum().backward()

    torch.testing.assert_close(bias_tri.grad, bias_ref.grad, atol=5e-3, rtol=5e-3)


@requires_triton_route
def test_route_fused_backward_grad_state(m=16):
    """grad_state from fused kernel matches PyTorch autograd."""
    torch.manual_seed(54)
    c = 32
    raw = torch.randn(16, m, m, device="cuda")
    alpha = torch.tensor(0.01, device="cuda")
    bias = torch.randn(m, m, device="cuda")
    state = torch.randn(16, m, c, device="cuda")

    # Reference
    state_ref = state.clone().requires_grad_(True)
    logits = alpha * raw + bias
    P = sinkhorn_log_ref(logits, num_iters=20, tau=1.0)
    (P @ state_ref).sum().backward()

    # Fused
    state_tri = state.clone().requires_grad_(True)
    out = triton_sinkhorn_route_fused(raw, alpha, bias, state_tri, num_iters=20, tau=1.0)
    out.sum().backward()

    torch.testing.assert_close(state_tri.grad, state_ref.grad, atol=1e-3, rtol=1e-3)


@requires_triton_route
def test_route_fused_bf16():
    """Works with bf16 inputs."""
    torch.manual_seed(55)
    m, c = 16, 32
    raw = torch.randn(16, m, m, device="cuda", dtype=torch.bfloat16)
    alpha = torch.tensor(0.01, device="cuda")
    bias = torch.randn(m, m, device="cuda", dtype=torch.bfloat16)
    state = torch.randn(16, m, c, device="cuda", dtype=torch.bfloat16)

    out = triton_sinkhorn_route_fused(raw, alpha, bias, state, num_iters=20, tau=1.0)
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out).all()


@requires_triton_route
def test_route_fused_all_grads_flow():
    """All gradients are non-zero and finite."""
    torch.manual_seed(56)
    m, c = 16, 32
    raw = torch.randn(8, m, m, device="cuda", requires_grad=True)
    alpha = torch.tensor(0.5, device="cuda", requires_grad=True)
    bias = torch.randn(m, m, device="cuda", requires_grad=True)
    state = torch.randn(8, m, c, device="cuda", requires_grad=True)

    out = triton_sinkhorn_route_fused(raw, alpha, bias, state, num_iters=20, tau=1.0)
    out.sum().backward()

    for name, t in [("raw", raw), ("alpha", alpha), ("bias", bias), ("state", state)]:
        assert t.grad is not None, f"{name}.grad is None"
        assert torch.isfinite(t.grad).all(), f"{name}.grad has non-finite values"
        assert t.grad.abs().max() > 0, f"{name}.grad is all zeros"


@requires_triton_route
def test_route_fused_dispatcher():
    """The sinkhorn_route_fused dispatcher routes CUDA tensors to Triton."""
    torch.manual_seed(57)
    m, c = 16, 32
    raw = torch.randn(8, m, m, device="cuda")
    alpha = torch.tensor(0.01, device="cuda")
    bias = torch.randn(m, m, device="cuda")
    state = torch.randn(8, m, c, device="cuda")

    out_dispatch = sinkhorn_route_fused(raw, alpha, bias, state, num_iters=20, tau=1.0)
    out_triton = triton_sinkhorn_route_fused(raw, alpha, bias, state, num_iters=20, tau=1.0)
    torch.testing.assert_close(out_dispatch, out_triton)


# ========================================================================== #
#  Integration: HyperConnections with fused ops
# ========================================================================== #


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_hyperconnections_n1_forward():
    """HyperConnections n=1 forward still works with all fusions."""
    from hyper_connections import HyperConnections

    torch.manual_seed(60)
    hc = HyperConnections(
        1, dim=64, routing_granularity=4, mhc=True,
    ).to("cuda")

    x = torch.randn(2, 16, 64, device="cuda")
    branch_input, residuals_out, kwargs = hc.width_connection(x)

    assert branch_input.shape == (2, 16, 64)
    # width_connection internally unsqueezes n=1 to (B,T,1,dim); depth_connection squeezes back
    assert residuals_out.shape == (2, 16, 1, 64)
    assert torch.isfinite(branch_input).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_hyperconnections_n1_backward():
    """HyperConnections n=1 backward works with all fusions."""
    from hyper_connections import HyperConnections

    torch.manual_seed(61)
    hc = HyperConnections(
        1, dim=64, routing_granularity=4, mhc=True,
        branch=torch.nn.Linear(64, 64),
    ).to("cuda")

    x = torch.randn(2, 16, 64, device="cuda", requires_grad=True)
    out = hc(x)
    out.sum().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_hyperconnections_n2_forward():
    """HyperConnections n=2 forward works (H_pre/H_post not short-circuited)."""
    from hyper_connections import HyperConnections

    torch.manual_seed(62)
    hc = HyperConnections(
        2, dim=64, routing_granularity=8, mhc=True,
    ).to("cuda")

    # n=2: width_connection expects (B, T, n, dim) layout
    x = torch.randn(2, 16, 2, 64, device="cuda")
    branch_input, residuals_out, kwargs = hc.width_connection(x)

    assert branch_input.shape == (2, 16, 64)
    assert residuals_out.shape == (2, 16, 2, 64)
    assert kwargs["beta"] is not None  # H_post should be computed


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_hyperconnections_bottleneck_forward():
    """HyperConnections with bottleneck works with batched phi."""
    from hyper_connections import HyperConnections

    torch.manual_seed(63)
    hc = HyperConnections(
        1, dim=64, routing_granularity=4,
        routing_bottleneck_dim=8, mhc=True,
    ).to("cuda")

    x = torch.randn(2, 16, 64, device="cuda")
    branch_input, residuals_out, kwargs = hc.width_connection(x)

    assert branch_input.shape == (2, 16, 64)
    assert torch.isfinite(branch_input).all()


@requires_triton_route
@pytest.mark.parametrize("m", [4, 8, 16])
def test_route_fused_num_iters_bwd(m):
    """Fused AB variant accepts num_iters_bwd and produces finite gradients."""
    torch.manual_seed(80)
    c = 32
    raw = torch.randn(16, m, m, device="cuda", requires_grad=True)
    alpha = torch.tensor(0.01, device="cuda", requires_grad=True)
    bias = torch.randn(m, m, device="cuda", requires_grad=True)
    state = torch.randn(16, m, c, device="cuda", requires_grad=True)

    out = triton_sinkhorn_route_fused(
        raw, alpha, bias, state, num_iters=20, tau=1.0, num_iters_bwd=40,
    )
    out.sum().backward()

    for name, t in [("raw", raw), ("alpha", alpha), ("bias", bias), ("state", state)]:
        assert t.grad is not None, f"{name}.grad is None"
        assert torch.isfinite(t.grad).all(), f"{name}.grad has non-finite values"


# ========================================================================== #
#  Low iteration + edge case stress tests
# ========================================================================== #


@requires_triton_route
@pytest.mark.parametrize("num_iters", [5, 10, 15])
@pytest.mark.parametrize("m", [4, 16, 32])
def test_route_fused_low_iters_forward(num_iters, m):
    """Fused AB variant matches reference at low iteration counts."""
    torch.manual_seed(200)
    B, c = 16, 32
    raw = torch.randn(B, m, m, device="cuda")
    alpha = torch.tensor(0.01, device="cuda")
    bias = torch.randn(m, m, device="cuda") * 0.5
    state = torch.randn(B, m, c, device="cuda")

    logits = alpha * raw + bias
    P = sinkhorn_log_ref(logits, num_iters=num_iters, tau=1.0)
    ref = torch.bmm(P, state)

    out = triton_sinkhorn_route_fused(raw, alpha, bias, state, num_iters=num_iters, tau=1.0)
    torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)


@requires_triton_route
@pytest.mark.parametrize("num_iters", [5, 10, 15])
@pytest.mark.parametrize("kind", EDGE_CASES)
def test_route_fused_edge_case_forward(num_iters, kind):
    """Fused AB variant matches reference on badly-conditioned inputs."""
    m, c = 16, 32
    raw = make_test_logits(kind, 16, m, m, seed=201)
    alpha = torch.tensor(0.5, device="cuda")
    bias = torch.randn(m, m, device="cuda") * 0.1
    state = torch.randn(16, m, c, device="cuda")

    logits = alpha * raw + bias
    P = sinkhorn_log_ref(logits, num_iters=num_iters, tau=0.05)
    ref = torch.bmm(P, state)

    out = triton_sinkhorn_route_fused(raw, alpha, bias, state, num_iters=num_iters, tau=0.05)
    # Wider tolerance: P differences amplified by state matmul on extreme inputs.
    torch.testing.assert_close(out, ref, atol=5e-3, rtol=5e-3)


@requires_triton_route
@pytest.mark.parametrize("num_iters", [5, 10, 15])
@pytest.mark.parametrize("kind", ["random"] + EDGE_CASES)
def test_route_fused_low_iters_backward_finite(num_iters, kind):
    """All gradients finite at low iters on all input types."""
    m, c = 16, 32
    raw = make_test_logits(kind, 16, m, m, seed=202).requires_grad_(True)
    alpha = torch.tensor(0.5, device="cuda", requires_grad=True)
    bias = torch.randn(m, m, device="cuda", requires_grad=True)
    state = torch.randn(16, m, c, device="cuda", requires_grad=True)

    out = triton_sinkhorn_route_fused(raw, alpha, bias, state, num_iters=num_iters, tau=0.05)
    out.sum().backward()

    for name, t in [("raw", raw), ("alpha", alpha), ("bias", bias), ("state", state)]:
        assert t.grad is not None, f"{name}.grad is None (kind={kind}, iters={num_iters})"
        assert torch.isfinite(t.grad).all(), (
            f"{name}.grad non-finite (kind={kind}, iters={num_iters})"
        )
