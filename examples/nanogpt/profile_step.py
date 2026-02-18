"""Profile a full training step (fwd + bwd + optimizer) with torch.profiler.

Produces a Chrome-trace JSON and prints a table of the top CUDA/CPU ops
sorted by total GPU time.

Usage:
    uv run python profile_step.py config/train_baseline_32M.py
    uv run python profile_step.py config/train_L1_m16.py --batch_size=8
    uv run python profile_step.py config/train_L1_m16.py --sort=cpu_time_total
    uv run python profile_step.py config/train_L1_m16.py --trace=trace_L1.json
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
from torch.profiler import profile, ProfilerActivity, schedule

from examples.nanogpt.model import GPT, GPTConfig

WARMUP_ITERS = 3
ACTIVE_ITERS = 3


def profile_step(
    config_path: str,
    batch_size_override: int | None = None,
    sinkhorn_checkpoint: bool | None = None,
    sort_by: str = "cuda_time_total",
    trace_path: str | None = None,
    top_n: int = 30,
):
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
        sinkhorn_iters=cfg.get("sinkhorn_iters", 20),
        sinkhorn_tau=cfg.get("sinkhorn_tau", 1.0),
        mhc_residual_identity_mix=cfg.get("mhc_residual_identity_mix", False),
        mhc_residual_alpha=cfg.get("mhc_residual_alpha", 0.01),
        routing_granularity=cfg.get("routing_granularity", None),
        routing_bottleneck_dim=cfg.get("routing_bottleneck_dim", None),
        full_transport_routing=cfg.get("full_transport_routing", False),
        sinkhorn_checkpoint=sk_ckpt,
    )

    device = "cuda"
    model = GPT(model_config)
    model.to(device)

    optimizer = model.configure_optimizers(
        weight_decay=cfg.get("weight_decay", 0.1),
        learning_rate=cfg.get("learning_rate", 6e-4),
        betas=(cfg.get("beta1", 0.9), cfg.get("beta2", 0.95)),
        device_type="cuda",
    )

    ctx = torch.amp.autocast(device_type="cuda", dtype=ptdtype)

    def train_step():
        x = torch.randint(0, 50304, (batch_size, block_size), device=device)
        y = torch.randint(0, 50304, (batch_size, block_size), device=device)
        optimizer.zero_grad(set_to_none=True)
        with ctx:
            _, loss = model(x, y)
        loss.backward()
        optimizer.step()

    # ---- warmup outside profiler (triggers lazy allocs, Triton compilation) ----
    for _ in range(WARMUP_ITERS):
        train_step()
    torch.cuda.synchronize()

    # ---- profile ----
    name = os.path.basename(config_path).replace("train_", "").replace(".py", "")
    total_params = sum(p.numel() for p in model.parameters())

    print(f"Config:     {name}")
    print(f"Params:     {total_params:,}")
    print(f"Batch:      {batch_size} x {block_size}")
    print(f"Dtype:      {dtype_str}")
    print(f"Profiling {ACTIVE_ITERS} steps...")
    print()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=0, warmup=1, active=ACTIVE_ITERS, repeat=1),
        record_shapes=True,
        with_stack=True,
        profile_memory=True,
    ) as prof:
        for _ in range(1 + ACTIVE_ITERS):  # 1 warmup + active
            train_step()
            prof.step()

    # ---- print top ops by GPU time ----
    print(prof.key_averages().table(sort_by=sort_by, row_limit=top_n))

    # ---- export Chrome trace ----
    if trace_path is None:
        trace_path = f"trace_{name}.json"
    prof.export_chrome_trace(trace_path)
    print(f"\nChrome trace saved to: {trace_path}")
    print("Open in chrome://tracing or https://ui.perfetto.dev")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to config file")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--sinkhorn_checkpoint", action="store_true", default=None,
                        help="Force sinkhorn gradient checkpointing on")
    parser.add_argument("--sort", default="cuda_time_total",
                        help="Sort column (default: cuda_time_total)")
    parser.add_argument("--trace", default=None,
                        help="Output trace JSON path (default: trace_<config>.json)")
    parser.add_argument("--top", type=int, default=30,
                        help="Number of top ops to show (default: 30)")
    args = parser.parse_args()

    profile_step(args.config, args.batch_size, args.sinkhorn_checkpoint,
                 args.sort, args.trace, args.top)
