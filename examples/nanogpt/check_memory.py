"""Estimate peak GPU memory for a training config (single forward+backward pass).

Usage:
    uv run python check_memory.py config/train_baseline_32M.py
    uv run python check_memory.py config/train_L1_m16.py --batch_size=8
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch

from examples.nanogpt.model import GPT, GPTConfig
from hyper_connections import HyperConnections


def check_memory(config_path: str, batch_size_override: int | None = None):
    # ---- load config via exec (same as train.py) ----
    cfg = {}
    exec(open(config_path).read(), cfg)

    batch_size = batch_size_override or cfg.get("batch_size", 16)
    block_size = cfg.get("block_size", 256)
    dtype_str = cfg.get("dtype", "bfloat16")
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype_str]

    model_config = GPTConfig(
        block_size=block_size,
        vocab_size=50304,
        n_layer=cfg.get("n_layer", 24),
        n_head=cfg.get("n_head", 8),
        n_embd=cfg.get("n_embd", 256),
        dropout=cfg.get("dropout", 0.0),
        bias=cfg.get("bias", False),
        hc_num_streams=cfg.get("hc_num_streams", 1),
        hc_disable=cfg.get("hc_disable", True),
        mhc=cfg.get("mhc", False),
        sinkhorn_iters=cfg.get("sinkhorn_iters", 10),
        sinkhorn_tau=cfg.get("sinkhorn_tau", 0.05),
        mhc_residual_identity_mix=cfg.get("mhc_residual_identity_mix", False),
        mhc_residual_alpha=cfg.get("mhc_residual_alpha", 0.01),
        routing_granularity=cfg.get("routing_granularity", None),
        routing_bottleneck_dim=cfg.get("routing_bottleneck_dim", None),
        full_transport_routing=cfg.get("full_transport_routing", False),
    )

    device = "cuda"
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()

    # ---- model ----
    model = GPT(model_config)
    model.to(device)
    after_model = torch.cuda.max_memory_allocated(device)

    # ---- optimizer (AdamW stores 2 extra fp32 copies per param) ----
    optimizer = model.configure_optimizers(
        weight_decay=cfg.get("weight_decay", 0.1),
        learning_rate=cfg.get("learning_rate", 6e-4),
        betas=(cfg.get("beta1", 0.9), cfg.get("beta2", 0.95)),
        device_type="cuda",
    )
    after_optimizer = torch.cuda.max_memory_allocated(device)

    # ---- single forward + backward ----
    ctx = torch.amp.autocast(device_type="cuda", dtype=ptdtype)
    x = torch.randint(0, 50304, (batch_size, block_size), device=device)
    y = torch.randint(0, 50304, (batch_size, block_size), device=device)

    optimizer.zero_grad(set_to_none=True)
    with ctx:
        _, loss = model(x, y)
    loss.backward()
    after_backward = torch.cuda.max_memory_allocated(device)

    # ---- optimizer step ----
    optimizer.step()
    after_step = torch.cuda.max_memory_allocated(device)

    peak = torch.cuda.max_memory_allocated(device)

    # ---- report ----
    total_params = sum(p.numel() for p in model.parameters())
    name = os.path.basename(config_path).replace("train_", "").replace(".py", "")

    def mb(x):
        return x / 1024**2

    print(f"Config:     {name}")
    print(f"Params:     {total_params:,}")
    print(f"Batch:      {batch_size} x {block_size}")
    print(f"Dtype:      {dtype_str}")
    print()
    print(f"Model load:       {mb(after_model):>8.1f} MB")
    print(f"+ Optimizer init: {mb(after_optimizer):>8.1f} MB")
    print(f"+ Fwd + Bwd:      {mb(after_backward):>8.1f} MB")
    print(f"+ Optimizer step: {mb(after_step):>8.1f} MB")
    print(f"Peak GPU memory:  {mb(peak):>8.1f} MB")
    print()
    print(f"GPU total:        {torch.cuda.get_device_properties(0).total_mem / 1024**2:.0f} MB")
    print(f"Headroom:         {(torch.cuda.get_device_properties(0).total_mem - peak) / 1024**2:.0f} MB")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to config file")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    args = parser.parse_args()

    check_memory(args.config, args.batch_size)
