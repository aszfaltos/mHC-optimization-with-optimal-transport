"""Count total and routing params for all experiment configs.

Usage:
    uv run python count_params.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
from examples.nanogpt.model import GPT, GPTConfig
from hyper_connections import HyperConnections

CONFIGS = {
    # L0
    "baseline_32M": dict(n_layer=24, n_head=8, n_embd=256, bias=False, hc_num_streams=1, hc_disable=True),
    "baseline_50M": dict(n_layer=24, n_head=8, n_embd=320, bias=False, hc_num_streams=1, hc_disable=True),
    "baseline_100M": dict(n_layer=24, n_head=8, n_embd=512, bias=False, hc_num_streams=1, hc_disable=True),
    "mhc_original": dict(n_layer=24, n_head=8, n_embd=256, bias=False, hc_num_streams=4, hc_disable=False, mhc=True, sinkhorn_iters=20, sinkhorn_tau=1.0, mhc_residual_alpha=0.01),
    # L1
    "L1_m4":  dict(n_layer=24, n_head=8, n_embd=256, bias=False, hc_num_streams=1, hc_disable=False, mhc=True, sinkhorn_iters=20, sinkhorn_tau=1.0, mhc_residual_alpha=0.01, routing_granularity=4),
    "L1_m8":  dict(n_layer=24, n_head=8, n_embd=256, bias=False, hc_num_streams=1, hc_disable=False, mhc=True, sinkhorn_iters=20, sinkhorn_tau=1.0, mhc_residual_alpha=0.01, routing_granularity=8),
    "L1_m16": dict(n_layer=24, n_head=8, n_embd=256, bias=False, hc_num_streams=1, hc_disable=False, mhc=True, sinkhorn_iters=20, sinkhorn_tau=1.0, mhc_residual_alpha=0.01, routing_granularity=16),
    "L1_m32": dict(n_layer=24, n_head=8, n_embd=256, bias=False, hc_num_streams=1, hc_disable=False, mhc=True, sinkhorn_iters=20, sinkhorn_tau=1.0, mhc_residual_alpha=0.01, routing_granularity=32),
    # L2
    "L2_m16_d4":   dict(n_layer=24, n_head=8, n_embd=256, bias=False, hc_num_streams=1, hc_disable=False, mhc=True, sinkhorn_iters=20, sinkhorn_tau=1.0, mhc_residual_alpha=0.01, routing_granularity=16, routing_bottleneck_dim=4),
    "L2_m16_d8":   dict(n_layer=24, n_head=8, n_embd=256, bias=False, hc_num_streams=1, hc_disable=False, mhc=True, sinkhorn_iters=20, sinkhorn_tau=1.0, mhc_residual_alpha=0.01, routing_granularity=16, routing_bottleneck_dim=8),
    "L2_m16_d16":  dict(n_layer=24, n_head=8, n_embd=256, bias=False, hc_num_streams=1, hc_disable=False, mhc=True, sinkhorn_iters=20, sinkhorn_tau=1.0, mhc_residual_alpha=0.01, routing_granularity=16, routing_bottleneck_dim=16),
    "L2_m16_d32":  dict(n_layer=24, n_head=8, n_embd=256, bias=False, hc_num_streams=1, hc_disable=False, mhc=True, sinkhorn_iters=20, sinkhorn_tau=1.0, mhc_residual_alpha=0.01, routing_granularity=16, routing_bottleneck_dim=32),
    "L2_m16_d64":  dict(n_layer=24, n_head=8, n_embd=256, bias=False, hc_num_streams=1, hc_disable=False, mhc=True, sinkhorn_iters=20, sinkhorn_tau=1.0, mhc_residual_alpha=0.01, routing_granularity=16, routing_bottleneck_dim=64),
    "L2_m16_d256": dict(n_layer=24, n_head=8, n_embd=256, bias=False, hc_num_streams=1, hc_disable=False, mhc=True, sinkhorn_iters=20, sinkhorn_tau=1.0, mhc_residual_alpha=0.01, routing_granularity=16, routing_bottleneck_dim=256),
}


# HC-only parameter names (excludes the wrapped branch)
HC_OWN_PARAMS = {
    "H_res_logits", "alpha_res", "phi_res", "compress_res", "to_logits_res",
    "H_pre_logits", "phi_pre", "alpha_pre",
    "H_post_logits", "phi_post", "alpha_post",
    "H_res_alpha_logit",
    "mhc_norm.gamma",  # RMSNorm
}


def count(name, kwargs):
    config = GPTConfig(block_size=256, vocab_size=50304, **kwargs)
    with torch.no_grad():
        model = GPT(config)

    total_params = sum(p.numel() for p in model.parameters())

    routing_params = 0
    for block in model.transformer.h:
        if hasattr(block, "use_ftr") and block.use_ftr:
            for ftr in (block.ftr_attn, block.ftr_mlp):
                routing_params += sum(p.numel() for p in ftr.parameters())
        elif hasattr(block, "hc_attn") and isinstance(block.hc_attn, HyperConnections):
            for hc in (block.hc_attn, block.hc_mlp):
                for pname, p in hc.named_parameters():
                    if pname in HC_OWN_PARAMS:
                        routing_params += p.numel()

    pct = routing_params / total_params * 100 if routing_params > 0 else 0
    return total_params, routing_params, pct


print(f"{'Config':<20} {'Total':>12} {'Routing':>12} {'Overhead':>8}")
print("-" * 56)
for name, kwargs in CONFIGS.items():
    total, routing, pct = count(name, kwargs)
    print(f"{name:<20} {total:>12,} {routing:>12,} {pct:>7.1f}%")
