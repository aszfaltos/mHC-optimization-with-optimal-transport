import pytest

import torch
from torch import nn


def test_disable_matches_residual():
    from hyper_connections import get_init_and_expand_reduce_stream_functions

    torch.manual_seed(0)

    branch = nn.Linear(32, 32)
    residual = torch.randn(2, 16, 32)
    expected = branch(residual) + residual

    init_hyper_conn, expand_stream, reduce_stream = (
        get_init_and_expand_reduce_stream_functions(4, disable=True)
    )

    hyper_conn_branch = init_hyper_conn(dim=32, branch=branch)
    output = reduce_stream(hyper_conn_branch(expand_stream(residual)))

    torch.testing.assert_close(output, expected)


def test_mhc_H_res_constraints():
    from hyper_connections.hyper_connections import HyperConnections, sinkhorn_log

    hc = HyperConnections(num_residual_streams=4, dim=64, mhc=True)
    H_res = sinkhorn_log(hc.H_res_logits, hc.sinkhorn_iters, hc.sinkhorn_tau)

    assert H_res.min().item() >= 0
    assert torch.allclose(
        H_res.sum(dim=-1),
        torch.ones(4, device=H_res.device, dtype=H_res.dtype),
        atol=1e-3,
    )
    assert torch.allclose(
        H_res.sum(dim=-2),
        torch.ones(4, device=H_res.device, dtype=H_res.dtype),
        atol=1e-3,
    )


def test_mhc_H_pre_H_post_constraints():
    from hyper_connections.hyper_connections import HyperConnections

    hc = HyperConnections(num_residual_streams=4, dim=64, mhc=True)
    H_pre = torch.softmax(hc.H_pre_logits, dim=-1)
    H_post = torch.softmax(hc.H_post_logits, dim=-1)

    assert H_pre.min().item() >= 0
    assert H_post.min().item() >= 0
    assert torch.allclose(
        H_pre.sum(),
        torch.ones((), device=H_pre.device, dtype=H_pre.dtype),
        atol=1e-6,
    )
    assert torch.allclose(
        H_post.sum(),
        torch.ones((), device=H_post.device, dtype=H_post.dtype),
        atol=1e-6,
    )


def test_mhc_forward_shapes():
    from hyper_connections.hyper_connections import HyperConnections

    streams, dim, batch, seq = 4, 64, 2, 8
    hc = HyperConnections(num_residual_streams=streams, dim=dim, mhc=True)
    x = torch.randn(batch, seq, streams, dim)

    branch_input, add_residual = hc(x)
    assert branch_input.shape == (batch, seq, dim)

    branch_output = torch.randn(batch, seq, dim)
    out = add_residual(branch_output)
    assert out.shape == (batch, seq, streams, dim)


def test_mhc_gradients_flow():
    from hyper_connections.hyper_connections import HyperConnections

    hc = HyperConnections(num_residual_streams=4, dim=64, mhc=True)
    x = torch.randn(2, 8, 4, 64, requires_grad=True)

    branch_input, add_residual = hc(x)
    out = add_residual(branch_input)
    out.sum().backward()

    assert hc.H_res_logits.grad is not None
    assert hc.H_pre_logits.grad is not None
    assert hc.H_post_logits.grad is not None
    assert not torch.isnan(hc.H_res_logits.grad).any()
    assert not torch.isnan(hc.H_pre_logits.grad).any()
    assert not torch.isnan(hc.H_post_logits.grad).any()


def test_mhc_identity_mix_H_res_constraints():
    from hyper_connections.hyper_connections import HyperConnections, sinkhorn_log

    hc = HyperConnections(
        num_residual_streams=4, dim=64, mhc=True, mhc_residual_identity_mix=True
    )
    S = sinkhorn_log(hc.H_res_logits, hc.sinkhorn_iters, hc.sinkhorn_tau)
    alpha = torch.sigmoid(hc.H_res_alpha_logit)
    I = torch.eye(4, device=S.device, dtype=S.dtype)
    H_res = (1 - alpha) * I + alpha * S

    assert H_res.min().item() >= 0
    assert torch.allclose(
        H_res.sum(dim=-1),
        torch.ones(4, device=H_res.device, dtype=H_res.dtype),
        atol=1e-3,
    )
    assert torch.allclose(
        H_res.sum(dim=-2),
        torch.ones(4, device=H_res.device, dtype=H_res.dtype),
        atol=1e-3,
    )


def test_mhc_identity_mix_alpha_init():
    from hyper_connections.hyper_connections import HyperConnections

    hc = HyperConnections(
        num_residual_streams=4,
        dim=64,
        mhc=True,
        mhc_residual_identity_mix=True,
        mhc_residual_alpha=0.01,
    )
    alpha = torch.sigmoid(hc.H_res_alpha_logit)
    assert torch.allclose(alpha, torch.tensor(0.01), atol=1e-3)


def test_mhc_identity_mix_gradients_flow():
    from hyper_connections.hyper_connections import HyperConnections

    hc = HyperConnections(
        num_residual_streams=4, dim=64, mhc=True, mhc_residual_identity_mix=True
    )
    x = torch.randn(2, 8, 4, 64, requires_grad=True)

    branch_input, add_residual = hc(x)
    out = add_residual(branch_input)
    out.sum().backward()

    assert hc.H_res_logits.grad is not None
    assert hc.H_pre_logits.grad is not None
    assert hc.H_post_logits.grad is not None
    assert hc.H_res_alpha_logit.grad is not None
    assert not torch.isnan(hc.H_res_alpha_logit.grad).any()


def test_mhc_identity_mix_forward_shapes():
    from hyper_connections.hyper_connections import HyperConnections

    streams, dim, batch, seq = 4, 64, 2, 8
    hc = HyperConnections(
        num_residual_streams=streams,
        dim=dim,
        mhc=True,
        mhc_residual_identity_mix=True,
    )
    x = torch.randn(batch, seq, streams, dim)

    branch_input, add_residual = hc(x)
    assert branch_input.shape == (batch, seq, dim)

    branch_output = torch.randn(batch, seq, dim)
    out = add_residual(branch_output)
    assert out.shape == (batch, seq, streams, dim)


# --------------------------------------------------------------------------- #
# Fine-grained routing (routing_granularity) tests
# --------------------------------------------------------------------------- #


def test_fine_grained_H_res_shape():
    """H_res_logits should be (m, m) when routing_granularity=m."""
    from hyper_connections.hyper_connections import HyperConnections

    hc = HyperConnections(num_residual_streams=4, dim=64, mhc=True, routing_granularity=16)
    assert hc.H_res_logits.shape == (16, 16)
    assert hc.routing_m == 16
    assert hc.routing_c == (4 * 64) // 16  # = 16


def test_fine_grained_H_res_doubly_stochastic():
    """H_res should still be doubly stochastic after Sinkhorn with fine-grained routing."""
    from hyper_connections.hyper_connections import HyperConnections, sinkhorn_log

    for m in [8, 16, 32]:
        hc = HyperConnections(num_residual_streams=4, dim=64, mhc=True, routing_granularity=m)
        H_res = sinkhorn_log(hc.H_res_logits, hc.sinkhorn_iters, hc.sinkhorn_tau)

        assert H_res.min().item() >= 0
        torch.testing.assert_close(
            H_res.sum(dim=-1),
            torch.ones(m, dtype=H_res.dtype),
            atol=1e-3,
            rtol=0,
        )
        torch.testing.assert_close(
            H_res.sum(dim=-2),
            torch.ones(m, dtype=H_res.dtype),
            atol=1e-3,
            rtol=0,
        )


def test_fine_grained_forward_shapes():
    """Forward pass shapes should be preserved with fine-grained routing."""
    from hyper_connections.hyper_connections import HyperConnections

    streams, dim, batch, seq = 4, 64, 2, 8
    hc = HyperConnections(
        num_residual_streams=streams, dim=dim, mhc=True, routing_granularity=16
    )
    x = torch.randn(batch, seq, streams, dim)

    branch_input, add_residual = hc(x)
    assert branch_input.shape == (batch, seq, dim)

    branch_output = torch.randn(batch, seq, dim)
    out = add_residual(branch_output)
    assert out.shape == (batch, seq, streams, dim)


def test_fine_grained_gradients_flow():
    """All parameters should receive gradients with fine-grained routing."""
    from hyper_connections.hyper_connections import HyperConnections

    hc = HyperConnections(
        num_residual_streams=4, dim=64, mhc=True, routing_granularity=16
    )
    x = torch.randn(2, 8, 4, 64, requires_grad=True)

    branch_input, add_residual = hc(x)
    out = add_residual(branch_input)
    out.sum().backward()

    assert hc.H_res_logits.grad is not None
    assert hc.H_res_logits.shape == (16, 16)
    assert not torch.isnan(hc.H_res_logits.grad).any()
    assert hc.H_pre_logits.grad is not None
    assert hc.H_post_logits.grad is not None


def test_fine_grained_default_matches_original():
    """routing_granularity=None should produce identical results to original mHC."""
    from hyper_connections.hyper_connections import HyperConnections

    torch.manual_seed(42)
    streams, dim, batch, seq = 4, 64, 2, 8
    x = torch.randn(batch, seq, streams, dim)

    hc_orig = HyperConnections(num_residual_streams=streams, dim=dim, mhc=True)
    hc_default = HyperConnections(
        num_residual_streams=streams, dim=dim, mhc=True, routing_granularity=None
    )

    # Copy parameters
    with torch.no_grad():
        hc_default.H_res_logits.copy_(hc_orig.H_res_logits)
        hc_default.H_pre_logits.copy_(hc_orig.H_pre_logits)
        hc_default.H_post_logits.copy_(hc_orig.H_post_logits)

    bi_orig, ar_orig = hc_orig(x)
    bi_def, ar_def = hc_default(x)

    torch.testing.assert_close(bi_orig, bi_def)

    branch_out = torch.randn(batch, seq, dim)
    out_orig = ar_orig(branch_out)
    out_def = ar_def(branch_out)
    torch.testing.assert_close(out_orig, out_def)


def test_fine_grained_m_equals_n_matches_original():
    """routing_granularity=n should be identical to original mHC (m==n is identity reshape)."""
    from hyper_connections.hyper_connections import HyperConnections

    torch.manual_seed(42)
    streams, dim, batch, seq = 4, 64, 2, 8
    x = torch.randn(batch, seq, streams, dim)

    hc_orig = HyperConnections(num_residual_streams=streams, dim=dim, mhc=True)
    hc_m4 = HyperConnections(
        num_residual_streams=streams, dim=dim, mhc=True, routing_granularity=4
    )

    with torch.no_grad():
        hc_m4.H_res_logits.copy_(hc_orig.H_res_logits)
        hc_m4.H_pre_logits.copy_(hc_orig.H_pre_logits)
        hc_m4.H_post_logits.copy_(hc_orig.H_post_logits)

    bi_orig, ar_orig = hc_orig(x)
    bi_m4, ar_m4 = hc_m4(x)

    torch.testing.assert_close(bi_orig, bi_m4)

    branch_out = torch.randn(batch, seq, dim)
    torch.testing.assert_close(ar_orig(branch_out), ar_m4(branch_out))


def test_fine_grained_level1_single_stream():
    """Level 1: n=1 with routing_granularity should work (vanilla transformer + routing)."""
    from hyper_connections import get_init_and_expand_reduce_stream_functions

    dim, batch, seq, m = 64, 2, 8, 16
    init_hc, expand_stream, reduce_stream = (
        get_init_and_expand_reduce_stream_functions(1, disable=False)
    )

    branch = nn.Linear(dim, dim)
    hc = init_hc(dim=dim, branch=branch, mhc=True, routing_granularity=m)

    x = torch.randn(batch, seq, dim)
    x = expand_stream(x)  # Identity for n=1
    out = hc(x)
    out = reduce_stream(out)

    assert out.shape == (batch, seq, dim)


def test_fine_grained_level1_gradients():
    """Level 1: gradients flow through n=1 fine-grained routing."""
    from hyper_connections.hyper_connections import HyperConnections

    dim, m = 64, 16
    hc = HyperConnections(
        num_residual_streams=1, dim=dim, mhc=True, routing_granularity=m
    )
    x = torch.randn(2, 8, dim, requires_grad=True)

    branch_input, add_residual = hc(x)
    out = add_residual(branch_input)
    out.sum().backward()

    assert hc.H_res_logits.grad is not None
    assert hc.H_res_logits.shape == (m, m)
    assert not torch.isnan(hc.H_res_logits.grad).any()


def test_fine_grained_rejects_bad_granularity():
    """routing_granularity must divide n_streams * dim."""
    from hyper_connections.hyper_connections import HyperConnections

    with pytest.raises(AssertionError, match="routing_granularity"):
        HyperConnections(
            num_residual_streams=4, dim=64, mhc=True, routing_granularity=7
        )


def test_fine_grained_identity_mix():
    """Fine-grained routing with identity mix should work and preserve doubly stochastic property."""
    from hyper_connections.hyper_connections import HyperConnections, sinkhorn_log

    m = 16
    hc = HyperConnections(
        num_residual_streams=4,
        dim=64,
        mhc=True,
        routing_granularity=m,
        mhc_residual_identity_mix=True,
        mhc_residual_alpha=0.01,
    )

    S = sinkhorn_log(hc.H_res_logits, hc.sinkhorn_iters, hc.sinkhorn_tau)
    alpha = torch.sigmoid(hc.H_res_alpha_logit)
    I = torch.eye(m, device=S.device, dtype=S.dtype)
    H_res = (1 - alpha) * I + alpha * S

    assert H_res.min().item() >= 0
    torch.testing.assert_close(
        H_res.sum(dim=-1),
        torch.ones(m, dtype=H_res.dtype),
        atol=1e-3,
        rtol=0,
    )

    # Forward pass should work
    x = torch.randn(2, 8, 4, 64, requires_grad=True)
    branch_input, add_residual = hc(x)
    out = add_residual(branch_input)
    out.sum().backward()
    assert hc.H_res_alpha_logit.grad is not None


def test_fine_grained_stats_collection():
    """Stats collection should include entropy and sparsity with fine-grained routing."""
    from hyper_connections.hyper_connections import HyperConnections

    hc = HyperConnections(
        num_residual_streams=4, dim=64, mhc=True, routing_granularity=16
    )
    hc.collect_stats = True
    x = torch.randn(2, 8, 4, 64)

    branch_input, add_residual = hc(x)
    add_residual(branch_input)

    assert hasattr(hc, "last_stats")
    assert "h_res_entropy" in hc.last_stats
    assert "h_res_sparsity" in hc.last_stats
    assert "h_res_max" in hc.last_stats


# --------------------------------------------------------------------------- #
# Conditioning bottleneck (routing_bottleneck_dim) tests
# --------------------------------------------------------------------------- #


def test_bottleneck_creates_two_projections():
    """With routing_bottleneck_dim, compress_res and to_logits_res replace phi_res."""
    from hyper_connections.hyper_connections import HyperConnections

    hc = HyperConnections(
        num_residual_streams=1, dim=256, mhc=True,
        routing_granularity=16, routing_bottleneck_dim=32,
    )
    assert hasattr(hc, "compress_res")
    assert hasattr(hc, "to_logits_res")
    assert not hasattr(hc, "phi_res")
    assert hc.compress_res.shape == (256, 32)
    assert hc.to_logits_res.shape == (32, 256)  # d -> m*m = 16*16


def test_bottleneck_forward_shapes():
    """Forward pass shapes should be correct with bottleneck."""
    from hyper_connections.hyper_connections import HyperConnections

    dim, batch, seq, m, d = 64, 2, 8, 16, 8
    hc = HyperConnections(
        num_residual_streams=1, dim=dim, mhc=True,
        routing_granularity=m, routing_bottleneck_dim=d,
    )
    x = torch.randn(batch, seq, dim)

    branch_input, add_residual = hc(x)
    assert branch_input.shape == (batch, seq, dim)

    branch_output = torch.randn(batch, seq, dim)
    out = add_residual(branch_output)
    assert out.shape == (batch, seq, dim)


def test_bottleneck_gradients_flow():
    """All bottleneck parameters should receive gradients."""
    from hyper_connections.hyper_connections import HyperConnections

    hc = HyperConnections(
        num_residual_streams=1, dim=64, mhc=True,
        routing_granularity=16, routing_bottleneck_dim=8,
    )
    x = torch.randn(2, 8, 64, requires_grad=True)

    branch_input, add_residual = hc(x)
    out = add_residual(branch_input)
    out.sum().backward()

    assert hc.compress_res.grad is not None
    assert hc.to_logits_res.grad is not None
    assert hc.H_res_logits.grad is not None
    assert not torch.isnan(hc.compress_res.grad).any()
    assert not torch.isnan(hc.to_logits_res.grad).any()


def test_bottleneck_no_bottleneck_has_phi_res():
    """Without routing_bottleneck_dim, phi_res should be used (not compress/to_logits)."""
    from hyper_connections.hyper_connections import HyperConnections

    hc = HyperConnections(
        num_residual_streams=1, dim=64, mhc=True,
        routing_granularity=16,
    )
    assert hasattr(hc, "phi_res")
    assert not hasattr(hc, "compress_res")


def test_bottleneck_with_multistream():
    """Bottleneck should work with n>1 (mHC bus)."""
    from hyper_connections.hyper_connections import HyperConnections

    streams, dim, batch, seq = 4, 64, 2, 8
    hc = HyperConnections(
        num_residual_streams=streams, dim=dim, mhc=True,
        routing_granularity=16, routing_bottleneck_dim=16,
    )
    x = torch.randn(batch, seq, streams, dim)

    branch_input, add_residual = hc(x)
    assert branch_input.shape == (batch, seq, dim)

    branch_output = torch.randn(batch, seq, dim)
    out = add_residual(branch_output)
    assert out.shape == (batch, seq, streams, dim)


# --------------------------------------------------------------------------- #
# FullTransportRouting tests
# --------------------------------------------------------------------------- #


def test_full_transport_forward_shapes():
    """FullTransportRouting forward should preserve shapes."""
    from hyper_connections.hyper_connections import FullTransportRouting

    dim, m, d, batch, seq = 64, 16, 8, 2, 8
    ftr = FullTransportRouting(dim=dim, m=m, d=d, n_streams=1)
    s = torch.randn(batch, seq, dim)

    layer_fn = nn.Linear(dim, dim)
    out = ftr(s, layer_fn)
    assert out.shape == s.shape


def test_full_transport_gradients_flow():
    """All FullTransportRouting parameters should receive gradients."""
    from hyper_connections.hyper_connections import FullTransportRouting

    dim, m, d = 64, 16, 8
    ftr = FullTransportRouting(dim=dim, m=m, d=d, n_streams=1)
    s = torch.randn(2, 8, dim, requires_grad=True)

    layer_fn = nn.Linear(dim, dim)
    out = ftr(s, layer_fn)
    out.sum().backward()

    assert ftr.compress.grad is not None
    assert ftr.to_logits_pre.grad is not None
    assert ftr.to_logits_align.grad is not None
    assert ftr.to_logits_next.grad is not None
    assert ftr.H_pre_logits.grad is not None
    assert ftr.H_align_logits.grad is not None
    assert ftr.H_next_logits.grad is not None
    assert not torch.isnan(ftr.compress.grad).any()


def test_full_transport_multistream():
    """FullTransportRouting with n>1 should create H^post and handle shape correctly."""
    from hyper_connections.hyper_connections import FullTransportRouting

    dim, m, d, n = 64, 16, 8, 4
    ftr = FullTransportRouting(dim=dim, m=m, d=d, n_streams=n)
    assert ftr.needs_hpost
    assert hasattr(ftr, "to_logits_post")
    assert ftr.k == m // n  # k = 4

    # H^pre should be k×m, H^post should be m×k
    assert ftr.H_pre_logits.shape == (ftr.k, m)
    assert ftr.H_post_logits.shape == (m, ftr.k)

    state_dim = n * dim  # 256
    batch, seq = 2, 8
    s = torch.randn(batch, seq, state_dim)

    # layer_fn receives dim-sized input (H^pre compresses m→k), outputs dim-sized
    layer_fn = nn.Linear(dim, dim)
    out = ftr(s, layer_fn)
    assert out.shape == s.shape


def test_full_transport_no_hpost_single_stream():
    """n=1 should NOT create H^post."""
    from hyper_connections.hyper_connections import FullTransportRouting

    ftr = FullTransportRouting(dim=64, m=16, d=8, n_streams=1)
    assert not ftr.needs_hpost
    assert not hasattr(ftr, "to_logits_post")
