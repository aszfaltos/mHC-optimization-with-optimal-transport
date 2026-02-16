"""Smoke test: verify all experiment configs can instantiate and forward pass."""
import sys
from pathlib import Path

import torch

# model.py imports value_residual from local dir
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "nanogpt"))

from model import GPT, GPTConfig


def _count_params(model):
    return sum(p.numel() for p in model.parameters())


def _make_and_forward(config_kwargs, label):
    cfg = GPTConfig(**config_kwargs)
    model = GPT(cfg)
    n = _count_params(model)
    x = torch.randint(0, 50304, (2, 256))
    y = torch.randint(0, 50304, (2, 256))
    logits, loss = model(x, y)
    assert logits.shape == (2, 256, 50304), f"{label}: bad logits shape {logits.shape}"
    assert loss.item() > 0, f"{label}: loss should be positive"
    return n


# Base model: C=256, 24 layers, 8 heads (~32M params)
COMMON = dict(
    n_layer=24, n_head=8, n_embd=256, block_size=256, vocab_size=50304,
    bias=False, dropout=0.0,
)


# --- L0: Scaling baselines ---

def test_baseline_32M():
    n = _make_and_forward({**COMMON, "hc_num_streams": 1, "hc_disable": True}, "baseline_32M")
    print(f"Baseline 32M: {n:,} params")


def test_baseline_50M():
    cfg = {**COMMON, "n_embd": 320, "hc_num_streams": 1, "hc_disable": True}
    n = _make_and_forward(cfg, "baseline_50M")
    print(f"Baseline 50M: {n:,} params")


def test_baseline_100M():
    cfg = {**COMMON, "n_embd": 512, "hc_num_streams": 1, "hc_disable": True}
    n = _make_and_forward(cfg, "baseline_100M")
    print(f"Baseline 100M: {n:,} params")


def test_mhc_original():
    n = _make_and_forward(
        {**COMMON, "hc_num_streams": 4, "hc_disable": False, "mhc": True},
        "mhc_original",
    )
    print(f"mHC original (n=4, m=4): {n:,} params")


# --- L1: Single H^res routing sweep (n=1, no bottleneck) ---

def test_L1_m4():
    n = _make_and_forward(
        {**COMMON, "hc_num_streams": 1, "hc_disable": False, "mhc": True,
         "routing_granularity": 4},
        "L1_m4",
    )
    print(f"L1 m=4: {n:,} params")


def test_L1_m8():
    n = _make_and_forward(
        {**COMMON, "hc_num_streams": 1, "hc_disable": False, "mhc": True,
         "routing_granularity": 8},
        "L1_m8",
    )
    print(f"L1 m=8: {n:,} params")


def test_L1_m16():
    n = _make_and_forward(
        {**COMMON, "hc_num_streams": 1, "hc_disable": False, "mhc": True,
         "routing_granularity": 16},
        "L1_m16",
    )
    print(f"L1 m=16: {n:,} params")


def test_L1_m32():
    n = _make_and_forward(
        {**COMMON, "hc_num_streams": 1, "hc_disable": False, "mhc": True,
         "routing_granularity": 32},
        "L1_m32",
    )
    print(f"L1 m=32: {n:,} params")


# --- L2: Conditioning bottleneck (n=1, m=16, sweep d) ---

def test_L2_m16_d4():
    n = _make_and_forward(
        {**COMMON, "hc_num_streams": 1, "hc_disable": False, "mhc": True,
         "routing_granularity": 16, "routing_bottleneck_dim": 4},
        "L2_m16_d4",
    )
    print(f"L2 m=16 d=4: {n:,} params")


def test_L2_m16_d16():
    n = _make_and_forward(
        {**COMMON, "hc_num_streams": 1, "hc_disable": False, "mhc": True,
         "routing_granularity": 16, "routing_bottleneck_dim": 16},
        "L2_m16_d16",
    )
    print(f"L2 m=16 d=16 (bin-level): {n:,} params")


def test_L2_m16_d32():
    n = _make_and_forward(
        {**COMMON, "hc_num_streams": 1, "hc_disable": False, "mhc": True,
         "routing_granularity": 16, "routing_bottleneck_dim": 32},
        "L2_m16_d32",
    )
    print(f"L2 m=16 d=32: {n:,} params")


def test_L2_m16_d256():
    n = _make_and_forward(
        {**COMMON, "hc_num_streams": 1, "hc_disable": False, "mhc": True,
         "routing_granularity": 16, "routing_bottleneck_dim": 256},
        "L2_m16_d256",
    )
    print(f"L2 m=16 d=256 (full): {n:,} params")


# --- Parameter overhead table ---

def test_param_overhead_table():
    """Print full parameter overhead table for all experiment levels."""
    configs = [
        ("Baseline 32M", {"hc_num_streams": 1, "hc_disable": True}),
        ("mHC orig n=4", {"hc_num_streams": 4, "hc_disable": False, "mhc": True}),
        ("L1 m=4", {"hc_num_streams": 1, "hc_disable": False, "mhc": True, "routing_granularity": 4}),
        ("L1 m=8", {"hc_num_streams": 1, "hc_disable": False, "mhc": True, "routing_granularity": 8}),
        ("L1 m=16", {"hc_num_streams": 1, "hc_disable": False, "mhc": True, "routing_granularity": 16}),
        ("L1 m=32", {"hc_num_streams": 1, "hc_disable": False, "mhc": True, "routing_granularity": 32}),
        ("L2 m=16 d=4", {"hc_num_streams": 1, "hc_disable": False, "mhc": True, "routing_granularity": 16, "routing_bottleneck_dim": 4}),
        ("L2 m=16 d=8", {"hc_num_streams": 1, "hc_disable": False, "mhc": True, "routing_granularity": 16, "routing_bottleneck_dim": 8}),
        ("L2 m=16 d=16", {"hc_num_streams": 1, "hc_disable": False, "mhc": True, "routing_granularity": 16, "routing_bottleneck_dim": 16}),
        ("L2 m=16 d=32", {"hc_num_streams": 1, "hc_disable": False, "mhc": True, "routing_granularity": 16, "routing_bottleneck_dim": 32}),
        ("L2 m=16 d=64", {"hc_num_streams": 1, "hc_disable": False, "mhc": True, "routing_granularity": 16, "routing_bottleneck_dim": 64}),
        ("L2 m=16 d=256", {"hc_num_streams": 1, "hc_disable": False, "mhc": True, "routing_granularity": 16, "routing_bottleneck_dim": 256}),
    ]

    results = []
    for label, kw in configs:
        cfg = GPTConfig(**{**COMMON, **kw})
        model = GPT(cfg)
        results.append((label, _count_params(model)))

    base = results[0][1]
    print("\n--- Parameter Overhead Table ---")
    print(f"{'Config':<25} {'Params':>12} {'Overhead':>12} {'%':>8}")
    print("-" * 60)
    for label, n in results:
        overhead = n - base
        pct = 100 * overhead / base if base > 0 else 0
        print(f"{label:<25} {n:>12,} {overhead:>+12,} {pct:>+7.2f}%")


# --- L4: Full Transport Routing ---

def test_ftr_n1_m16_d32():
    """FTR with n=1, m=16, d=32: basic integration."""
    n = _make_and_forward(
        {**COMMON, "hc_num_streams": 1, "hc_disable": False,
         "full_transport_routing": True, "routing_granularity": 16,
         "routing_bottleneck_dim": 32},
        "FTR_n1_m16_d32",
    )
    print(f"FTR n=1 m=16 d=32: {n:,} params")


def test_ftr_n1_m32_d32():
    """FTR with n=1, m=32, d=32: higher granularity."""
    n = _make_and_forward(
        {**COMMON, "hc_num_streams": 1, "hc_disable": False,
         "full_transport_routing": True, "routing_granularity": 32,
         "routing_bottleneck_dim": 32},
        "FTR_n1_m32_d32",
    )
    print(f"FTR n=1 m=32 d=32: {n:,} params")


def test_ftr_gradients_flow():
    """Verify all FTR parameters receive gradients."""
    cfg = GPTConfig(
        **COMMON, hc_num_streams=1, hc_disable=False,
        full_transport_routing=True, routing_granularity=16,
        routing_bottleneck_dim=32,
    )
    model = GPT(cfg)
    x = torch.randint(0, 50304, (2, 64))
    y = torch.randint(0, 50304, (2, 64))
    _, loss = model(x, y)
    loss.backward()

    no_grad = []
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is None:
            no_grad.append(name)
    assert len(no_grad) == 0, f"Parameters with no gradient: {no_grad}"


def test_ftr_stats_collection():
    """Verify stats collection works on FTR blocks."""
    from hyper_connections import FullTransportRouting

    cfg = GPTConfig(
        **{**COMMON, "n_layer": 2}, hc_num_streams=1, hc_disable=False,
        full_transport_routing=True, routing_granularity=16,
        routing_bottleneck_dim=32,
    )
    model = GPT(cfg)
    for block in model.transformer.h:
        block.ftr_attn.collect_stats = True
        block.ftr_mlp.collect_stats = True

    x = torch.randint(0, 50304, (2, 64))
    y = torch.randint(0, 50304, (2, 64))
    model(x, y)

    for block in model.transformer.h:
        for ftr in (block.ftr_attn, block.ftr_mlp):
            assert hasattr(ftr, "last_stats"), "FTR should have last_stats after forward"
            stats = ftr.last_stats
            assert "h_pre_entropy" in stats
            assert "h_align_entropy" in stats
            assert "h_next_entropy" in stats
            assert "alpha_pre" in stats
