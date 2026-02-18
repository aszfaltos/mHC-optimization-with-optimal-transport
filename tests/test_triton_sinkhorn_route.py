"""Correctness tests for the fused Triton Sinkhorn + routing kernel.

Run tests:   uv run pytest tests/test_triton_sinkhorn_route.py -v
"""
from __future__ import annotations

import pytest
import torch

from tests.sinkhorn_ref import sinkhorn_log_ref, make_test_logits, EDGE_CASES
from hyper_connections.hyper_connections import sinkhorn_route

try:
    from hyper_connections.triton_sinkhorn_route import (
        triton_sinkhorn_route,
        is_available as triton_route_available,
    )
except ImportError:
    triton_route_available = lambda: False
    triton_sinkhorn_route = None

requires_triton = pytest.mark.skipif(
    not torch.cuda.is_available()
    or triton_sinkhorn_route is None
    or not triton_route_available(),
    reason="Triton + CUDA required",
)


# ========================================================================== #
#  Forward correctness
# ========================================================================== #


@requires_triton
@pytest.mark.parametrize("m", [4, 8, 16, 32])
@pytest.mark.parametrize("c", [16, 64, 256])
def test_forward_matches_separate(m, c):
    """Fused sinkhorn_route matches sinkhorn_log + matmul."""
    torch.manual_seed(42)
    B = 32
    logits = torch.randn(B, m, m, device="cuda")
    state = torch.randn(B, m, c, device="cuda")

    # Reference: separate Sinkhorn + matmul
    P = sinkhorn_log_ref(logits, num_iters=20, tau=1.0)
    ref = torch.bmm(P, state)

    # Fused kernel
    out = triton_sinkhorn_route(logits, state, num_iters=20, tau=1.0)

    torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)


@requires_triton
@pytest.mark.parametrize(
    "rows,cols",
    [(4, 16), (16, 4), (8, 32)],
    ids=["4x16", "16x4", "8x32"],
)
def test_rectangular_forward(rows, cols):
    """Fused kernel works with rectangular logits (FTR H^pre / H^post)."""
    torch.manual_seed(43)
    B, c = 16, 32
    logits = torch.randn(B, rows, cols, device="cuda")
    state = torch.randn(B, cols, c, device="cuda")

    P = sinkhorn_log_ref(logits, num_iters=20, tau=1.0)
    ref = torch.bmm(P, state)

    out = triton_sinkhorn_route(logits, state, num_iters=20, tau=1.0)
    torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)


@requires_triton
def test_batched_leading_dims():
    """Works with arbitrary leading batch dims (..., m, m) + (..., m, c)."""
    torch.manual_seed(44)
    logits = torch.randn(4, 8, 16, 16, device="cuda")
    state = torch.randn(4, 8, 16, 32, device="cuda")

    out = triton_sinkhorn_route(logits, state, num_iters=20, tau=1.0)
    assert out.shape == (4, 8, 16, 32)

    # Verify against flat batch
    logits_flat = logits.reshape(32, 16, 16)
    state_flat = state.reshape(32, 16, 32)
    P = sinkhorn_log_ref(logits_flat, num_iters=20, tau=1.0)
    ref = torch.bmm(P, state_flat).reshape(4, 8, 16, 32)
    torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)


# ========================================================================== #
#  Backward correctness
# ========================================================================== #


@requires_triton
@pytest.mark.parametrize("m", [4, 8, 16, 32])
def test_backward_grad_state(m):
    """grad_state from fused kernel matches PyTorch autograd."""
    torch.manual_seed(45)
    c = 32
    logits = torch.randn(16, m, m, device="cuda")
    state = torch.randn(16, m, c, device="cuda")

    # Reference
    state_ref = state.clone().requires_grad_(True)
    logits_ref = logits.clone().detach()
    P = sinkhorn_log_ref(logits_ref, num_iters=20, tau=1.0)
    (P @ state_ref).sum().backward()

    # Fused
    state_tri = state.clone().requires_grad_(True)
    logits_tri = logits.clone().detach().requires_grad_(True)
    triton_sinkhorn_route(logits_tri, state_tri, num_iters=20, tau=1.0).sum().backward()

    torch.testing.assert_close(
        state_tri.grad, state_ref.grad, atol=1e-3, rtol=1e-3
    )


@requires_triton
@pytest.mark.parametrize("m", [4, 8, 16])
def test_backward_grad_logits(m):
    """grad_logits from fused kernel matches PyTorch autograd (via Sinkhorn)."""
    torch.manual_seed(46)
    c = 32
    logits = torch.randn(16, m, m, device="cuda")
    state = torch.randn(16, m, c, device="cuda")

    # Reference: PyTorch autograd through sinkhorn_log + matmul
    logits_ref = logits.clone().requires_grad_(True)
    state_ref = state.clone().detach()
    P = sinkhorn_log_ref(logits_ref, num_iters=20, tau=1.0)
    (P @ state_ref).sum().backward()

    # Fused kernel (implicit diff backward for grad_logits)
    logits_tri = logits.clone().requires_grad_(True)
    state_tri = state.clone().detach()
    triton_sinkhorn_route(logits_tri, state_tri, num_iters=20, tau=1.0).sum().backward()

    torch.testing.assert_close(
        logits_tri.grad, logits_ref.grad, atol=5e-3, rtol=5e-3
    )


@requires_triton
def test_backward_both_grads_flow():
    """Both grad_logits and grad_state are non-zero and finite."""
    torch.manual_seed(47)
    logits = torch.randn(8, 16, 16, device="cuda", requires_grad=True)
    state = torch.randn(8, 16, 32, device="cuda", requires_grad=True)

    out = triton_sinkhorn_route(logits, state, num_iters=20, tau=1.0)
    out.sum().backward()

    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert state.grad is not None and torch.isfinite(state.grad).all()
    assert logits.grad.abs().max() > 0
    assert state.grad.abs().max() > 0


@requires_triton
def test_backward_rectangular_gradients():
    """Gradients flow through rectangular sinkhorn_route (FTR-like shapes)."""
    torch.manual_seed(48)
    # k×m coupling with k=4, m=16
    logits = torch.randn(8, 4, 16, device="cuda", requires_grad=True)
    state = torch.randn(8, 16, 32, device="cuda", requires_grad=True)

    out = triton_sinkhorn_route(logits, state, num_iters=20, tau=1.0)
    assert out.shape == (8, 4, 32)
    out.sum().backward()

    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert state.grad is not None and torch.isfinite(state.grad).all()


@requires_triton
@pytest.mark.parametrize(
    "rows,cols",
    [(4, 16), (16, 4), (8, 32)],
    ids=["4x16", "16x4", "8x32"],
)
def test_rectangular_backward_grad_logits(rows, cols):
    """Rectangular grad_logits from fused kernel matches PyTorch autograd."""
    torch.manual_seed(51)
    B, c = 16, 32
    logits = torch.randn(B, rows, cols, device="cuda")
    state = torch.randn(B, cols, c, device="cuda")

    # Reference: PyTorch autograd through sinkhorn_log + matmul
    logits_ref = logits.clone().requires_grad_(True)
    state_ref = state.clone().detach()
    P = sinkhorn_log_ref(logits_ref, num_iters=20, tau=1.0)
    (P @ state_ref).sum().backward()

    # Fused kernel (implicit diff backward)
    logits_tri = logits.clone().requires_grad_(True)
    state_tri = state.clone().detach()
    triton_sinkhorn_route(logits_tri, state_tri, num_iters=20, tau=1.0).sum().backward()

    torch.testing.assert_close(
        logits_tri.grad, logits_ref.grad, atol=5e-3, rtol=5e-3
    )


@requires_triton
@pytest.mark.parametrize(
    "rows,cols",
    [(4, 16), (16, 4), (8, 32)],
    ids=["4x16", "16x4", "8x32"],
)
def test_rectangular_backward_grad_state(rows, cols):
    """Rectangular grad_state from fused kernel matches PyTorch autograd."""
    torch.manual_seed(52)
    B, c = 16, 32
    logits = torch.randn(B, rows, cols, device="cuda")
    state = torch.randn(B, cols, c, device="cuda")

    # Reference
    state_ref = state.clone().requires_grad_(True)
    logits_ref = logits.clone().detach()
    P = sinkhorn_log_ref(logits_ref, num_iters=20, tau=1.0)
    (P @ state_ref).sum().backward()

    # Fused
    state_tri = state.clone().requires_grad_(True)
    logits_tri = logits.clone().detach().requires_grad_(True)
    triton_sinkhorn_route(logits_tri, state_tri, num_iters=20, tau=1.0).sum().backward()

    torch.testing.assert_close(
        state_tri.grad, state_ref.grad, atol=1e-3, rtol=1e-3
    )


# ========================================================================== #
#  Dtype and autocast
# ========================================================================== #


@requires_triton
def test_bf16_forward():
    """Works with bf16 inputs."""
    torch.manual_seed(49)
    logits = torch.randn(16, 8, 8, device="cuda", dtype=torch.bfloat16)
    state = torch.randn(16, 8, 32, device="cuda", dtype=torch.bfloat16)

    out = triton_sinkhorn_route(logits, state, num_iters=20, tau=1.0)
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out).all()


@requires_triton
def test_dispatcher_routes_to_triton():
    """The sinkhorn_route dispatcher routes CUDA tensors to Triton."""
    torch.manual_seed(50)
    logits = torch.randn(8, 16, 16, device="cuda")
    state = torch.randn(8, 16, 32, device="cuda")

    out_dispatch = sinkhorn_route(logits, state, num_iters=20, tau=1.0)
    out_triton = triton_sinkhorn_route(logits, state, num_iters=20, tau=1.0)
    torch.testing.assert_close(out_dispatch, out_triton)


@requires_triton
@pytest.mark.parametrize("m", [4, 8, 16])
def test_num_iters_bwd(m):
    """num_iters_bwd=40 produces gradients closer to converged than num_iters_bwd=20."""
    torch.manual_seed(70)
    c = 32
    logits = torch.randn(16, m, m, device="cuda", dtype=torch.float32)
    state = torch.randn(16, m, c, device="cuda", dtype=torch.float32)

    # Converged reference
    logits_hi = logits.clone().requires_grad_(True)
    triton_sinkhorn_route(logits_hi, state, num_iters=20, tau=1.0, num_iters_bwd=80).sum().backward()

    # 20 backward iters
    logits_20 = logits.clone().requires_grad_(True)
    triton_sinkhorn_route(logits_20, state, num_iters=20, tau=1.0, num_iters_bwd=20).sum().backward()

    # 40 backward iters
    logits_40 = logits.clone().requires_grad_(True)
    triton_sinkhorn_route(logits_40, state, num_iters=20, tau=1.0, num_iters_bwd=40).sum().backward()

    err_20 = (logits_20.grad - logits_hi.grad).abs().max().item()
    err_40 = (logits_40.grad - logits_hi.grad).abs().max().item()
    # With centering, GS converges fast. Once converged, extra iters only add
    # fp32 rounding noise (~1e-7). Different NUM_ITERS constexprs also compile
    # to different kernels with different reduction order.
    assert err_40 <= err_20 + 1e-5, (
        f"More bwd iters should be closer to converged: err_20={err_20}, err_40={err_40}"
    )


@requires_triton
def test_bf16_backward_finite():
    """bf16 logits input produces finite gradients (P saved in fp32 internally)."""
    torch.manual_seed(71)
    m, c = 16, 32
    logits = torch.randn(16, m, m, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    state = torch.randn(16, m, c, device="cuda", dtype=torch.bfloat16)

    out = triton_sinkhorn_route(logits, state, num_iters=20, tau=1.0)
    out.sum().backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all(), (
        f"Non-finite grads with bf16 input: has_nan={logits.grad.isnan().any()}"
    )


# ========================================================================== #
#  Low iteration + edge case stress tests
# ========================================================================== #


@requires_triton
@pytest.mark.parametrize("num_iters", [5, 10, 15])
@pytest.mark.parametrize("m", [4, 16, 32])
def test_low_iters_forward_matches_separate(num_iters, m):
    """Fused sinkhorn_route matches sinkhorn_log + matmul at low iters."""
    torch.manual_seed(200)
    B, c = 16, 32
    logits = torch.randn(B, m, m, device="cuda")
    state = torch.randn(B, m, c, device="cuda")

    P = sinkhorn_log_ref(logits, num_iters=num_iters, tau=1.0)
    ref = torch.bmm(P, state)

    out = triton_sinkhorn_route(logits, state, num_iters=num_iters, tau=1.0)
    torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)


@requires_triton
@pytest.mark.parametrize("num_iters", [5, 10, 15])
@pytest.mark.parametrize("kind", EDGE_CASES)
def test_edge_case_forward_matches_separate(num_iters, kind):
    """Fused sinkhorn_route matches reference on badly-conditioned inputs."""
    m, c = 16, 32
    logits = make_test_logits(kind, 16, m, m, seed=201)
    state = torch.randn(16, m, c, device="cuda")

    P = sinkhorn_log_ref(logits, num_iters=num_iters, tau=0.05)
    ref = torch.bmm(P, state)

    out = triton_sinkhorn_route(logits, state, num_iters=num_iters, tau=0.05)
    # Wider tolerance: small P differences from fp32 reduction order get
    # amplified by the state matmul, especially for extreme inputs.
    torch.testing.assert_close(out, ref, atol=5e-3, rtol=5e-3)


@requires_triton
@pytest.mark.parametrize("num_iters", [5, 10, 15])
@pytest.mark.parametrize("kind", ["random"] + EDGE_CASES)
def test_low_iters_backward_finite(num_iters, kind):
    """Both grad_logits and grad_state are finite at low iters on all input types."""
    m, c = 16, 32
    logits = make_test_logits(kind, 16, m, m, seed=202).requires_grad_(True)
    state = torch.randn(16, m, c, device="cuda", requires_grad=True)

    triton_sinkhorn_route(logits, state, num_iters=num_iters, tau=0.05).sum().backward()

    assert logits.grad is not None and torch.isfinite(logits.grad).all(), (
        f"Non-finite logits grad: kind={kind}, iters={num_iters}"
    )
    assert state.grad is not None and torch.isfinite(state.grad).all(), (
        f"Non-finite state grad: kind={kind}, iters={num_iters}"
    )


@requires_triton
@pytest.mark.parametrize("num_iters", [5, 10, 15])
@pytest.mark.parametrize("kind", EDGE_CASES)
def test_edge_case_backward_grad_state(num_iters, kind):
    """grad_state on edge cases matches reference at low iters."""
    m, c = 16, 32
    logits = make_test_logits(kind, 16, m, m, seed=203)
    state = torch.randn(16, m, c, device="cuda")

    # Reference
    state_ref = state.clone().requires_grad_(True)
    P = sinkhorn_log_ref(logits, num_iters=num_iters, tau=1.0)
    (P @ state_ref).sum().backward()

    # Fused
    state_tri = state.clone().requires_grad_(True)
    triton_sinkhorn_route(logits, state_tri, num_iters=num_iters, tau=1.0).sum().backward()

    torch.testing.assert_close(
        state_tri.grad, state_ref.grad, atol=1e-3, rtol=1e-3
    )


@requires_triton
@pytest.mark.parametrize("num_iters", [5, 10, 15])
@pytest.mark.parametrize(
    "rows,cols",
    [(4, 16), (16, 4)],
    ids=["4x16", "16x4"],
)
def test_low_iters_rectangular_forward(num_iters, rows, cols):
    """Rectangular fused route matches reference at low iters."""
    torch.manual_seed(204)
    B, c = 16, 32
    logits = torch.randn(B, rows, cols, device="cuda")
    state = torch.randn(B, cols, c, device="cuda")

    P = sinkhorn_log_ref(logits, num_iters=num_iters, tau=1.0)
    ref = torch.bmm(P, state)

    out = triton_sinkhorn_route(logits, state, num_iters=num_iters, tau=1.0)
    torch.testing.assert_close(out, ref, atol=5e-3, rtol=5e-3)
