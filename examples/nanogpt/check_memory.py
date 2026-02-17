"""Estimate peak GPU memory and step time for a training config.

Runs a few warmup iterations then times a forward+backward+step cycle.

Usage:
    uv run python check_memory.py config/train_baseline_32M.py
    uv run python check_memory.py config/train_L1_m16.py --batch_size=8
    uv run python check_memory.py config/train_L1_m16.py --sinkhorn_checkpoint
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch

from examples.nanogpt.model import GPT, GPTConfig
from hyper_connections import HyperConnections

WARMUP_ITERS = 3
BENCH_ITERS = 10


def check_memory(config_path: str, batch_size_override: int | None = None, sinkhorn_checkpoint: bool | None = None):
    # ---- load config via exec (same as train.py) ----
    cfg = {}
    exec(open(config_path).read(), cfg)

    batch_size = batch_size_override or cfg.get("batch_size", 16)
    block_size = cfg.get("block_size", 256)
    dtype_str = cfg.get("dtype", "bfloat16")
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype_str]

    sk_ckpt = sinkhorn_checkpoint if sinkhorn_checkpoint is not None else cfg.get("sinkhorn_checkpoint", False)

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
        sinkhorn_checkpoint=sk_ckpt,
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

    ctx = torch.amp.autocast(device_type="cuda", dtype=ptdtype)

    def train_step():
        x = torch.randint(0, 50304, (batch_size, block_size), device=device)
        y = torch.randint(0, 50304, (batch_size, block_size), device=device)
        optimizer.zero_grad(set_to_none=True)
        with ctx:
            _, loss = model(x, y)
        loss.backward()
        optimizer.step()
        return loss.item()

    # ---- warmup (also triggers lazy optimizer state alloc) ----
    for _ in range(WARMUP_ITERS):
        train_step()

    after_warmup = torch.cuda.max_memory_allocated(device)

    # ---- timed benchmark ----
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(BENCH_ITERS):
        train_step()
    end.record()
    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end)
    ms_per_step = elapsed_ms / BENCH_ITERS
    tok_per_s = batch_size * block_size / (ms_per_step / 1000)

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
    print(f"Sinkhorn checkpoint: {sk_ckpt}")
    print()
    print(f"Model load:       {mb(after_model):>8.1f} MB")
    print(f"+ Optimizer init: {mb(after_optimizer):>8.1f} MB")
    print(f"+ Warmup peak:    {mb(after_warmup):>8.1f} MB")
    print(f"Peak GPU memory:  {mb(peak):>8.1f} MB")
    print()
    print(f"Step time:        {ms_per_step:>8.1f} ms  ({BENCH_ITERS} iters)")
    print(f"Throughput:       {tok_per_s:>8.0f} tok/s")
    print()
    print(f"GPU total:        {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f} MB")
    print(f"Headroom:         {(torch.cuda.get_device_properties(0).total_memory - peak) / 1024**2:.0f} MB")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to config file")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--sinkhorn_checkpoint", action="store_true", default=None,
                        help="Force sinkhorn gradient checkpointing on")
    args = parser.parse_args()

    check_memory(args.config, args.batch_size, args.sinkhorn_checkpoint)
