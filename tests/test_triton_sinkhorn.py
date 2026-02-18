"""Correctness tests and benchmarks for the fused Triton Sinkhorn kernel.

Run tests:   uv run pytest tests/test_triton_sinkhorn.py -v
Run bench:   uv run python tests/test_triton_sinkhorn.py
"""
from __future__ import annotations

import math
import sys
import time

import pytest
import torch

# PyTorch reference (always available, no Triton needed)
from tests.sinkhorn_ref import sinkhorn_log_ref, make_test_logits, EDGE_CASES

# Triton kernel (may not be available on CPU-only machines)
try:
    from hyper_connections.triton_sinkhorn import triton_sinkhorn, is_available as triton_available
except ImportError:
    triton_available = lambda: False
    triton_sinkhorn = None

requires_triton = pytest.mark.skipif(
    not torch.cuda.is_available() or triton_sinkhorn is None or not triton_available(),
    reason="Triton + CUDA required",
)


# ========================================================================== #
#  Correctness tests
# ========================================================================== #


@requires_triton
@pytest.mark.parametrize("m", [4, 8, 16, 32])
@pytest.mark.parametrize("tau", [0.05, 0.1])
def test_forward_matches_pytorch(m, tau):
    """Triton forward matches PyTorch sinkhorn_log to tight tolerance."""
    torch.manual_seed(42)
    logits = torch.randn(64, m, m, device="cuda")

    out_triton = triton_sinkhorn(logits, num_iters=20, tau=tau)
    out_ref = sinkhorn_log_ref(logits, num_iters=20, tau=tau)

    torch.testing.assert_close(out_triton, out_ref, atol=1e-4, rtol=1e-4)


@requires_triton
@pytest.mark.parametrize("m", [4, 8, 16, 32])
def test_doubly_stochastic(m):
    """Output has row and col sums ≈ 1.

    Uses tau=1.0 so Sinkhorn converges in 20 iters.  With tau=0.05 the
    near-permutation output has Hilbert-metric contraction rate ≈ 1 and
    row sums only converge after hundreds of iters — that's fine for training
    (col sums are exact, mean is correct) but not for an exact row-sum test.
    Kernel correctness at tau=0.05 is covered by test_forward_matches_pytorch.
    """
    torch.manual_seed(0)
    logits = torch.randn(32, m, m, device="cuda")
    P = triton_sinkhorn(logits, num_iters=20, tau=1.0)

    ones_row = torch.ones(32, m, device="cuda")
    ones_col = torch.ones(32, m, device="cuda")
    torch.testing.assert_close(P.sum(dim=-1), ones_row, atol=1e-3, rtol=0)
    torch.testing.assert_close(P.sum(dim=-2), ones_col, atol=1e-3, rtol=0)


@requires_triton
@pytest.mark.parametrize(
    "rows,cols",
    [(4, 16), (16, 4), (8, 32), (32, 8)],
    ids=["4x16", "16x4", "8x32", "32x8"],
)
def test_rectangular_marginals(rows, cols):
    """Rectangular output has row sums = 1 and col sums = rows/cols.

    With separate row/col marginals, the OT coupling between Uniform(rows)
    and Uniform(cols) has: unscaled row sums = 1/rows, col sums = 1/cols.
    After scaling by rows: row sums = 1, col sums = rows/cols.
    """
    torch.manual_seed(12)
    B = 32
    logits = torch.randn(B, rows, cols, device="cuda")
    P = triton_sinkhorn(logits, num_iters=20, tau=1.0)

    expected_row_sum = torch.ones(B, rows, device="cuda")
    expected_col_sum = torch.full((B, cols), rows / cols, device="cuda")
    torch.testing.assert_close(P.sum(dim=-1), expected_row_sum, atol=1e-3, rtol=0)
    torch.testing.assert_close(P.sum(dim=-2), expected_col_sum, atol=1e-3, rtol=0)


@requires_triton
def test_gradcheck():
    """Backward math passes finite-difference gradcheck (float64).

    Gradcheck runs on sinkhorn_log_ref because the Triton forward and
    PyTorch forward have different floating-point reduction order, so numerical
    derivatives of one vs analytical derivatives of the other diverge.
    Triton backward correctness is validated by test_backward_matches_pytorch.
    """
    torch.manual_seed(1)
    logits = torch.randn(4, 8, 8, device="cuda", dtype=torch.float64, requires_grad=True)

    def fn(x):
        return sinkhorn_log_ref(x, num_iters=20, tau=1.0)

    torch.autograd.gradcheck(fn, logits, eps=1e-6, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "rows,cols",
    [(4, 16), (16, 4), (8, 32)],
    ids=["4x16", "16x4", "8x32"],
)
def test_rectangular_gradcheck(rows, cols):
    """Rectangular PyTorch Sinkhorn backward passes finite-difference gradcheck (float64).

    Tests sinkhorn_log_ref (not Triton) — Triton backward correctness
    for rectangular is validated by test_rectangular_gradients which compares
    Triton implicit-diff against this PyTorch reference.
    """
    torch.manual_seed(13)
    logits = torch.randn(4, rows, cols, device="cuda", dtype=torch.float64, requires_grad=True)

    def fn(x):
        return sinkhorn_log_ref(x, num_iters=20, tau=1.0)

    torch.autograd.gradcheck(fn, logits, eps=1e-6, atol=1e-3, rtol=1e-3)


@requires_triton
@pytest.mark.parametrize("m", [4, 8, 16, 32])
def test_backward_matches_pytorch(m):
    """Triton implicit-diff gradient matches PyTorch autograd gradient.

    Uses tau=1.0 where Sinkhorn converges well in 20 iterations.
    Implicit diff assumes convergence, so it diverges from unrolled autograd
    at low tau (e.g. 0.1) where 20 iterations are insufficient — that's
    expected and not a bug.
    """
    torch.manual_seed(7)
    logits = torch.randn(32, m, m, device="cuda", dtype=torch.float32)

    # PyTorch reference gradient
    logits_ref = logits.clone().requires_grad_(True)
    out_ref = sinkhorn_log_ref(logits_ref, num_iters=20, tau=1.0)
    loss_ref = out_ref.sum()
    loss_ref.backward()

    # Triton gradient (implicit differentiation backward)
    logits_tri = logits.clone().requires_grad_(True)
    out_tri = triton_sinkhorn(logits_tri, num_iters=20, tau=1.0)
    loss_tri = out_tri.sum()
    loss_tri.backward()

    # Implicit diff may have slightly looser tolerance than autograd recompute
    torch.testing.assert_close(
        logits_tri.grad, logits_ref.grad, atol=5e-3, rtol=5e-3
    )


@requires_triton
def test_backward_gradient_flow():
    """Gradients propagate and have reasonable magnitude."""
    torch.manual_seed(2)
    logits = torch.randn(32, 16, 16, device="cuda", requires_grad=True)
    P = triton_sinkhorn(logits, num_iters=20, tau=1.0)
    P.sum().backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().max() > 0


@requires_triton
@pytest.mark.parametrize("m", [4, 8, 16, 32])
def test_backward_implicit_diff_convergence(m):
    """Implicit-diff backward converges: gradient changes negligibly with more iters.

    The Gauss-Seidel solver for the KKT system converges geometrically.
    With tau=1.0 the spectral gap is large.  We verify that doubling the
    iteration count doesn't change the gradient significantly.
    """
    torch.manual_seed(11)
    logits = torch.randn(16, m, m, device="cuda", dtype=torch.float32)

    # Run with 20 iterations (default)
    logits_20 = logits.clone().requires_grad_(True)
    triton_sinkhorn(logits_20, num_iters=20, tau=1.0).sum().backward()

    # Run with 40 iterations (should be converged further)
    logits_40 = logits.clone().requires_grad_(True)
    triton_sinkhorn(logits_40, num_iters=40, tau=1.0).sum().backward()

    # Gradient should be nearly identical — convergence means more iters don't help
    diff = (logits_20.grad - logits_40.grad).abs().max().item()
    assert diff < 1e-3, f"Backward not converged: max grad diff = {diff}"


@requires_triton
def test_bf16_autocast():
    """Works correctly under bf16 autocast."""
    torch.manual_seed(3)
    logits = torch.randn(32, 16, 16, device="cuda")

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        P_bf16 = triton_sinkhorn(logits.to(torch.bfloat16), num_iters=20, tau=0.05)
    assert P_bf16.dtype == torch.bfloat16

    P_fp32 = triton_sinkhorn(logits, num_iters=20, tau=0.05)
    torch.testing.assert_close(P_bf16.float(), P_fp32, atol=5e-2, rtol=1e-2)


@requires_triton
@pytest.mark.parametrize(
    "shape",
    [(16, 16), (4, 8, 16, 16), (2, 3, 4, 8, 8)],
    ids=["2d", "4d", "5d"],
)
def test_batched_leading_dims(shape):
    """Works with arbitrary leading batch dims (..., rows, cols)."""
    torch.manual_seed(4)
    logits = torch.randn(*shape, device="cuda", requires_grad=True)
    P = triton_sinkhorn(logits, num_iters=20, tau=0.05)
    assert P.shape == logits.shape
    P.sum().backward()
    assert logits.grad is not None
    assert logits.grad.shape == logits.shape


@requires_triton
def test_numerical_stability():
    """No NaN/Inf with large logits (typical regime: logits * 5 / tau=0.05 → Z ~ 500)."""
    torch.manual_seed(5)
    logits = torch.randn(32, 16, 16, device="cuda") * 5
    P = triton_sinkhorn(logits, num_iters=20, tau=0.05)
    assert torch.isfinite(P).all(), f"Non-finite values in output: {P}"


@requires_triton
@pytest.mark.parametrize(
    "rows,cols",
    [(4, 16), (16, 4), (8, 32), (32, 8)],
    ids=["4x16", "16x4", "8x32", "32x8"],
)
def test_rectangular(rows, cols):
    """Non-square matrices matching FTR H^pre (k×m) and H^post (m×k)."""
    torch.manual_seed(6)
    logits = torch.randn(32, rows, cols, device="cuda")

    out_triton = triton_sinkhorn(logits, num_iters=20, tau=0.05)
    out_ref = sinkhorn_log_ref(logits, num_iters=20, tau=0.05)

    assert out_triton.shape == (32, rows, cols)
    torch.testing.assert_close(out_triton, out_ref, atol=1e-4, rtol=1e-4)


@requires_triton
@pytest.mark.parametrize(
    "rows,cols",
    [(4, 16), (16, 4), (8, 32), (32, 8)],
    ids=["4x16", "16x4", "8x32", "32x8"],
)
def test_rectangular_gradients(rows, cols):
    """Gradients flow through rectangular Sinkhorn and match PyTorch."""
    torch.manual_seed(8)
    logits = torch.randn(16, rows, cols, device="cuda", dtype=torch.float32)

    # PyTorch reference
    logits_ref = logits.clone().requires_grad_(True)
    sinkhorn_log_ref(logits_ref, num_iters=20, tau=1.0).sum().backward()

    # Triton implicit-diff
    logits_tri = logits.clone().requires_grad_(True)
    triton_sinkhorn(logits_tri, num_iters=20, tau=1.0).sum().backward()

    assert logits_tri.grad is not None
    assert torch.isfinite(logits_tri.grad).all()
    torch.testing.assert_close(
        logits_tri.grad, logits_ref.grad, atol=5e-3, rtol=5e-3
    )


@requires_triton
def test_single_matrix():
    """Works with a single matrix (no batch dim beyond the 2d matrix)."""
    torch.manual_seed(9)
    logits = torch.randn(8, 8, device="cuda", requires_grad=True)
    P = triton_sinkhorn(logits, num_iters=20, tau=0.05)
    assert P.shape == (8, 8)
    P.sum().backward()
    assert logits.grad is not None


@requires_triton
def test_dispatcher_uses_triton():
    """The sinkhorn_log dispatcher should route CUDA tensors to Triton."""
    from hyper_connections.hyper_connections import sinkhorn_log

    torch.manual_seed(10)
    logits = torch.randn(8, 16, 16, device="cuda")

    # sinkhorn_log should produce same result as triton_sinkhorn
    out_dispatch = sinkhorn_log(logits, num_iters=20, tau=0.05)
    out_triton = triton_sinkhorn(logits, num_iters=20, tau=0.05)
    torch.testing.assert_close(out_dispatch, out_triton)


@requires_triton
@pytest.mark.parametrize("m", [4, 8, 16])
def test_backward_near_permutation(m):
    """Backward produces finite gradients on near-permutation P (diagonal-dominant logits).

    This is the regime where the KKT adjoint has a near-singular Jacobian.
    The centering fix (projecting out rank-1 null space) prevents divergence.
    """
    torch.manual_seed(99)
    logits = torch.randn(16, m, m, device="cuda", dtype=torch.float32) * 0.1
    # Make diagonal dominant → P ≈ identity after Sinkhorn
    logits += torch.eye(m, device="cuda") * 10.0

    logits_tri = logits.clone().requires_grad_(True)
    triton_sinkhorn(logits_tri, num_iters=20, tau=0.05).sum().backward()

    assert logits_tri.grad is not None
    assert torch.isfinite(logits_tri.grad).all(), (
        f"Non-finite gradients on near-permutation: "
        f"max={logits_tri.grad.abs().max()}, has_nan={logits_tri.grad.isnan().any()}"
    )

    # Also verify against PyTorch reference
    logits_ref = logits.clone().requires_grad_(True)
    sinkhorn_log_ref(logits_ref, num_iters=20, tau=0.05).sum().backward()
    torch.testing.assert_close(
        logits_tri.grad, logits_ref.grad, atol=5e-3, rtol=5e-3
    )


@requires_triton
@pytest.mark.parametrize("m", [4, 8, 16])
def test_num_iters_bwd(m):
    """num_iters_bwd=40 produces gradients closer to converged than num_iters_bwd=20."""
    torch.manual_seed(100)
    logits = torch.randn(16, m, m, device="cuda", dtype=torch.float32)

    # Converged reference: many backward iters
    logits_hi = logits.clone().requires_grad_(True)
    triton_sinkhorn(logits_hi, num_iters=20, tau=1.0, num_iters_bwd=80).sum().backward()

    # 20 backward iters
    logits_20 = logits.clone().requires_grad_(True)
    triton_sinkhorn(logits_20, num_iters=20, tau=1.0, num_iters_bwd=20).sum().backward()

    # 40 backward iters
    logits_40 = logits.clone().requires_grad_(True)
    triton_sinkhorn(logits_40, num_iters=20, tau=1.0, num_iters_bwd=40).sum().backward()

    err_20 = (logits_20.grad - logits_hi.grad).abs().max().item()
    err_40 = (logits_40.grad - logits_hi.grad).abs().max().item()
    # At small m with tau=1.0, convergence is fast and both may hit machine precision.
    # Allow a small absolute tolerance for noise-level differences.
    assert err_40 <= err_20 + 1e-5, (
        f"More bwd iters should be closer to converged: err_20={err_20}, err_40={err_40}"
    )


# ========================================================================== #
#  Low iteration + edge case stress tests
# ========================================================================== #


@requires_triton
@pytest.mark.parametrize("num_iters", [5, 10, 15])
@pytest.mark.parametrize("m", [4, 16, 32])
@pytest.mark.parametrize("tau", [0.05, 1.0])
def test_low_iters_forward_matches_ref(num_iters, m, tau):
    """Triton forward matches PyTorch reference at low iteration counts."""
    logits = make_test_logits("random", 32, m, m, seed=200)
    out_triton = triton_sinkhorn(logits, num_iters=num_iters, tau=tau)
    out_ref = sinkhorn_log_ref(logits, num_iters=num_iters, tau=tau)
    torch.testing.assert_close(out_triton, out_ref, atol=1e-4, rtol=1e-4)


@requires_triton
@pytest.mark.parametrize("num_iters", [5, 10, 15])
@pytest.mark.parametrize("kind", EDGE_CASES)
@pytest.mark.parametrize("m", [8, 16])
def test_edge_case_forward_matches_ref(num_iters, kind, m):
    """Triton forward matches reference on badly-conditioned inputs at low tau."""
    logits = make_test_logits(kind, 16, m, m, seed=201)
    out_triton = triton_sinkhorn(logits, num_iters=num_iters, tau=0.05)
    out_ref = sinkhorn_log_ref(logits, num_iters=num_iters, tau=0.05)
    # Looser tolerance for ill-conditioned cases with extreme Z values
    torch.testing.assert_close(out_triton, out_ref, atol=1e-3, rtol=1e-3)


@requires_triton
@pytest.mark.parametrize("num_iters", [5, 10, 15])
@pytest.mark.parametrize("kind", ["random"] + EDGE_CASES)
@pytest.mark.parametrize("m", [8, 16])
def test_low_iters_backward_finite(num_iters, kind, m):
    """Backward produces finite gradients at low iters on all input types.

    Note: near_permutation at tau=0.05 produces zero gradients — P is a hard
    permutation in fp32 so there's no smooth curvature to differentiate through.
    This is correct behavior, not a bug.
    """
    logits = make_test_logits(kind, 16, m, m, seed=202).requires_grad_(True)
    triton_sinkhorn(logits, num_iters=num_iters, tau=0.05).sum().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all(), (
        f"Non-finite grads: kind={kind}, m={m}, iters={num_iters}"
    )


@requires_triton
@pytest.mark.parametrize("num_iters", [5, 10, 15])
@pytest.mark.parametrize(
    "rows,cols",
    [(4, 16), (16, 4), (8, 32)],
    ids=["4x16", "16x4", "8x32"],
)
def test_low_iters_rectangular_forward(num_iters, rows, cols):
    """Rectangular forward matches reference at low iteration counts."""
    logits = make_test_logits("random", 16, rows, cols, seed=203)
    out_triton = triton_sinkhorn(logits, num_iters=num_iters, tau=1.0)
    out_ref = sinkhorn_log_ref(logits, num_iters=num_iters, tau=1.0)
    torch.testing.assert_close(out_triton, out_ref, atol=1e-4, rtol=1e-4)


@requires_triton
@pytest.mark.parametrize("num_iters", [5, 10, 15])
@pytest.mark.parametrize(
    "rows,cols",
    [(4, 16), (16, 4)],
    ids=["4x16", "16x4"],
)
def test_low_iters_rectangular_backward_finite(num_iters, rows, cols):
    """Rectangular backward produces finite gradients at low iters."""
    logits = make_test_logits("random", 16, rows, cols, seed=204).requires_grad_(True)
    triton_sinkhorn(logits, num_iters=num_iters, tau=1.0).sum().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


@requires_triton
@pytest.mark.parametrize("num_iters", [5, 10, 15])
@pytest.mark.parametrize(
    "kind",
    ["near_permutation", "near_uniform", "one_hot_rows"],
)
def test_edge_case_backward_matches_ref(num_iters, kind):
    """Backward on edge cases matches PyTorch autograd reference.

    Uses tau=1.0 where these input types converge reasonably even at 5 iters.
    large_magnitude is excluded: Z = 10*randn has poor contraction at 5-15 iters,
    so implicit diff (assumes convergence) diverges fundamentally from unrolled
    autograd. That's not a kernel bug — it's the expected behavior of implicit
    differentiation on an unconverged fixed point. large_magnitude backward
    finiteness is tested separately in test_low_iters_backward_finite.
    """
    m = 16
    logits = make_test_logits(kind, 16, m, m, seed=205)

    # PyTorch reference (unrolled autograd)
    logits_ref = logits.clone().requires_grad_(True)
    sinkhorn_log_ref(logits_ref, num_iters=num_iters, tau=1.0).sum().backward()

    # Triton implicit-diff
    logits_tri = logits.clone().requires_grad_(True)
    triton_sinkhorn(logits_tri, num_iters=num_iters, tau=1.0).sum().backward()

    # Wider tolerance than the 20-iter tests: at 5 iters, implicit diff and
    # unrolled autograd can differ by ~1% even for convergent inputs.
    torch.testing.assert_close(
        logits_tri.grad, logits_ref.grad, atol=2e-2, rtol=2e-2
    )


# ========================================================================== #
#  Benchmark (run as standalone script)
# ========================================================================== #


def _time_fn(fn, warmup=10, repeat=100):
    """Time a CUDA function with proper synchronization."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeat


def benchmark():
    assert torch.cuda.is_available(), "CUDA required for benchmarks"
    assert triton_sinkhorn is not None, "Triton kernel not available"

    print("=" * 80)
    print("Fused Triton Sinkhorn Benchmark")
    print("=" * 80)

    configs = [
        # (batch, rows, cols, iters, label)
        (256, 4, 4, 20, "B=256  m=4   square"),
        (256, 8, 8, 20, "B=256  m=8   square"),
        (256, 16, 16, 20, "B=256  m=16  square"),
        (256, 32, 32, 20, "B=256  m=32  square"),
        (2048, 4, 4, 20, "B=2048 m=4   square"),
        (2048, 8, 8, 20, "B=2048 m=8   square"),
        (2048, 16, 16, 20, "B=2048 m=16  square"),
        (2048, 32, 32, 20, "B=2048 m=32  square"),
        (4096, 4, 4, 20, "B=4096 m=4   square"),
        (4096, 8, 8, 20, "B=4096 m=8   square"),
        (4096, 16, 16, 20, "B=4096 m=16  square"),
        (4096, 32, 32, 20, "B=4096 m=32  square"),
        # Rectangular (FTR-like)
        (2048, 4, 16, 20, "B=2048 4×16  rect"),
        (2048, 16, 4, 20, "B=2048 16×4  rect"),
    ]

    print(f"\n{'Config':<28} {'Triton fwd':>12} {'PyTorch fwd':>12} {'Speedup':>8}")
    print("-" * 64)

    for B, R, C, iters, label in configs:
        logits = torch.randn(B, R, C, device="cuda")

        t_triton = _time_fn(lambda: triton_sinkhorn(logits, iters, 0.05))
        t_pytorch = _time_fn(lambda: sinkhorn_log_ref(logits, iters, 0.05))

        speedup = t_pytorch / t_triton if t_triton > 0 else float("inf")
        print(
            f"{label:<28} {t_triton*1e3:>10.3f}ms {t_pytorch*1e3:>10.3f}ms {speedup:>7.1f}x"
        )

    # Forward + backward
    print(f"\n{'Config':<28} {'Triton f+b':>12} {'PyTorch f+b':>12} {'Speedup':>8}")
    print("-" * 64)

    for B, R, C, iters, label in configs:
        logits_t = torch.randn(B, R, C, device="cuda", requires_grad=True)
        logits_p = logits_t.detach().clone().requires_grad_(True)

        def triton_fwd_bwd():
            out = triton_sinkhorn(logits_t, iters, 0.05)
            out.sum().backward()
            logits_t.grad = None

        def pytorch_fwd_bwd():
            out = sinkhorn_log_ref(logits_p, iters, 0.05)
            out.sum().backward()
            logits_p.grad = None

        t_triton = _time_fn(triton_fwd_bwd)
        t_pytorch = _time_fn(pytorch_fwd_bwd)

        speedup = t_pytorch / t_triton if t_triton > 0 else float("inf")
        print(
            f"{label:<28} {t_triton*1e3:>10.3f}ms {t_pytorch*1e3:>10.3f}ms {speedup:>7.1f}x"
        )

    print("\nDone.")


if __name__ == "__main__":
    benchmark()
