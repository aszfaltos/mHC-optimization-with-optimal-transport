"""Analyze H^res routing dynamics across training checkpoints.

Loads periodic checkpoints from an out_dir and plots how routing params
evolve over training: alpha_res, phi_res norms, static H^res structure,
and entropy/sparsity of the static coupling.

Usage:
    uv run python analyze_routing.py out-L1-m4-s42
    uv run python analyze_routing.py out-L1-m4-s42 --layers 0 11 23  # specific layers
"""
import argparse
import os
import re
import sys

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from hyper_connections.hyper_connections import sinkhorn_log


def find_checkpoints(out_dir):
    """Find all checkpoints and return sorted by iter number."""
    ckpts = []
    for fname in os.listdir(out_dir):
        if not fname.endswith(".pt"):
            continue
        path = os.path.join(out_dir, fname)
        # Extract iter from filename: ckpt_5000.pt -> 5000, ckpt.pt -> best, ckpt_final.pt -> final
        m = re.match(r"ckpt_(\d+)\.pt", fname)
        if m:
            ckpts.append((int(m.group(1)), path, fname))
        elif fname == "ckpt_final.pt":
            ckpts.append((float("inf"), path, fname))
        elif fname == "ckpt.pt":
            ckpts.append((-1, path, fname))  # best val, sort first
    ckpts.sort(key=lambda x: x[0])
    return ckpts


def extract_routing_params(state_dict):
    """Extract per-layer routing params from model state dict."""
    layers = {}
    for key, val in state_dict.items():
        # Pattern: transformer.h.{idx}.hc_{attn,mlp}.{param}
        m = re.match(r"transformer\.h\.(\d+)\.hc_(attn|mlp)\.(.+)", key)
        if not m:
            continue
        block_idx, sub, param = int(m.group(1)), m.group(2), m.group(3)
        layer_key = f"L{block_idx}_{sub}"
        if layer_key not in layers:
            layers[layer_key] = {}
        layers[layer_key][param] = val.cpu()
    return layers


def compute_static_hres(H_res_logits, sinkhorn_iters=20, tau=1.0):
    """Compute static H^res = sinkhorn(H_res_logits) (no dynamic input)."""
    with torch.no_grad():
        return sinkhorn_log(H_res_logits.unsqueeze(0), sinkhorn_iters, tau).squeeze(0)


def analyze_checkpoint(state_dict):
    """Compute per-layer routing stats from a checkpoint."""
    layers = extract_routing_params(state_dict)
    stats = {}
    for layer_key, params in sorted(layers.items()):
        s = {}
        # alpha_res
        if "alpha_res" in params:
            s["alpha_res"] = params["alpha_res"].item()

        # H_res_logits stats
        if "H_res_logits" in params:
            logits = params["H_res_logits"]
            s["H_res_logits_std"] = logits.std().item()
            s["H_res_logits_range"] = (logits.max() - logits.min()).item()

            # Static H^res (what routing looks like without any input-dependent signal)
            H = compute_static_hres(logits)
            s["static_entropy"] = -(H.clamp(min=1e-8) * H.clamp(min=1e-8).log()).sum().item()
            s["static_sparsity"] = (H < 0.01).float().mean().item()
            s["static_max"] = H.max().item()
            s["static_diag_mean"] = H.diag().mean().item()
            m = H.shape[0]
            s["static_diag_dominance"] = H.diag().mean().item() * m  # >1 means diagonal-heavy

        # phi_res or compress_res+to_logits_res norms
        if "phi_res" in params:
            s["phi_res_norm"] = params["phi_res"].norm().item()
            s["phi_res_max"] = params["phi_res"].abs().max().item()
        if "compress_res" in params:
            s["compress_res_norm"] = params["compress_res"].norm().item()
        if "to_logits_res" in params:
            s["to_logits_res_norm"] = params["to_logits_res"].norm().item()

        # Dynamic signal magnitude: alpha_res * phi_res_norm gives rough scale
        if "alpha_res" in s and "phi_res_norm" in s:
            s["dynamic_scale"] = s["alpha_res"] * s["phi_res_norm"]

        stats[layer_key] = s
    return stats


def plot_dynamics(all_stats, iters, layer_filter, out_path):
    """Plot routing param evolution across checkpoints."""
    # Collect all layer keys
    all_layers = set()
    for stats in all_stats:
        all_layers.update(stats.keys())
    all_layers = sorted(all_layers)

    if layer_filter:
        # Filter to specific block indices
        all_layers = [l for l in all_layers if any(f"L{i}_" in l for i in layer_filter)]

    if not all_layers:
        print("No layers found!")
        return

    # Use only attn layers for cleaner plots (mlp mirrors attn)
    attn_layers = [l for l in all_layers if "_attn" in l]
    if not attn_layers:
        attn_layers = all_layers

    fig = plt.figure(figsize=(22, 18))
    fig.suptitle(f"Routing Parameter Dynamics ({len(iters)} checkpoints)", fontsize=16, y=0.98)
    gs = gridspec.GridSpec(3, 3, hspace=0.35, wspace=0.3)

    def get_series(layer, key):
        return [s.get(layer, {}).get(key, np.nan) for s in all_stats]

    # Color by layer depth
    colors = plt.cm.viridis(np.linspace(0, 1, len(attn_layers)))

    # 1. alpha_res over training
    ax = fig.add_subplot(gs[0, 0])
    for layer, color in zip(attn_layers, colors):
        vals = get_series(layer, "alpha_res")
        if not all(np.isnan(v) for v in vals):
            ax.plot(iters, vals, "o-", color=color, markersize=3, label=layer)
    ax.set_xlabel("iter")
    ax.set_ylabel("alpha_res")
    ax.set_title("alpha_res (dynamic scaling)")
    ax.legend(fontsize=5, ncol=2, loc="best")

    # 2. phi_res norm over training
    ax = fig.add_subplot(gs[0, 1])
    for layer, color in zip(attn_layers, colors):
        vals = get_series(layer, "phi_res_norm")
        if not all(np.isnan(v) for v in vals):
            ax.plot(iters, vals, "o-", color=color, markersize=3, label=layer)
    ax.set_xlabel("iter")
    ax.set_ylabel("||phi_res||")
    ax.set_title("phi_res norm (projection learned)")

    # 3. Dynamic scale (alpha * phi_norm)
    ax = fig.add_subplot(gs[0, 2])
    for layer, color in zip(attn_layers, colors):
        vals = get_series(layer, "dynamic_scale")
        if not all(np.isnan(v) for v in vals):
            ax.plot(iters, vals, "o-", color=color, markersize=3, label=layer)
    ax.set_xlabel("iter")
    ax.set_ylabel("alpha * ||phi||")
    ax.set_title("Dynamic Signal Scale")

    # 4. Static H^res entropy
    ax = fig.add_subplot(gs[1, 0])
    for layer, color in zip(attn_layers, colors):
        vals = get_series(layer, "static_entropy")
        if not all(np.isnan(v) for v in vals):
            ax.plot(iters, vals, "o-", color=color, markersize=3, label=layer)
    ax.set_xlabel("iter")
    ax.set_ylabel("entropy")
    ax.set_title("Static H^res Entropy")

    # 5. Static H^res sparsity
    ax = fig.add_subplot(gs[1, 1])
    for layer, color in zip(attn_layers, colors):
        vals = get_series(layer, "static_sparsity")
        if not all(np.isnan(v) for v in vals):
            ax.plot(iters, vals, "o-", color=color, markersize=3, label=layer)
    ax.set_xlabel("iter")
    ax.set_ylabel("frac < 0.01")
    ax.set_title("Static H^res Sparsity")

    # 6. Static diagonal dominance
    ax = fig.add_subplot(gs[1, 2])
    for layer, color in zip(attn_layers, colors):
        vals = get_series(layer, "static_diag_dominance")
        if not all(np.isnan(v) for v in vals):
            ax.plot(iters, vals, "o-", color=color, markersize=3, label=layer)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, label="uniform")
    ax.set_xlabel("iter")
    ax.set_ylabel("diag_mean * m")
    ax.set_title("Diagonal Dominance (1=uniform, >1=identity-like)")

    # 7. H_res_logits std
    ax = fig.add_subplot(gs[2, 0])
    for layer, color in zip(attn_layers, colors):
        vals = get_series(layer, "H_res_logits_std")
        if not all(np.isnan(v) for v in vals):
            ax.plot(iters, vals, "o-", color=color, markersize=3, label=layer)
    ax.set_xlabel("iter")
    ax.set_ylabel("std")
    ax.set_title("H_res_logits Std (bias learning)")

    # 8-9. Heatmap of static H^res for first and last checkpoint, one layer
    sample_layer = attn_layers[len(attn_layers) // 2]  # middle layer
    for plot_idx, ckpt_idx, title in [(1, 0, "First ckpt"), (2, -1, "Last ckpt")]:
        ax = fig.add_subplot(gs[2, plot_idx])
        params = all_stats[ckpt_idx].get(sample_layer, {})
        # Re-extract H_res_logits from the checkpoint to compute H
        layer_params = extract_routing_params(
            torch.load(ckpt_files[ckpt_idx], map_location="cpu", weights_only=False).get("model", {})
        )
        lp = layer_params.get(sample_layer, {})
        if "H_res_logits" in lp:
            H = compute_static_hres(lp["H_res_logits"])
            im = ax.imshow(H.numpy(), cmap="Blues", vmin=0)
            ax.set_title(f"Static H^res: {sample_layer}\n{title} (iter {iters[ckpt_idx]})")
            fig.colorbar(im, ax=ax, shrink=0.8)
        else:
            ax.text(0.5, 0.5, "No H_res_logits", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"Static H^res: {title}")

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze routing dynamics from checkpoints")
    parser.add_argument("out_dir", help="Checkpoint directory (e.g. out-L1-m4-s42)")
    parser.add_argument("--layers", type=int, nargs="*", default=None,
                        help="Block indices to plot (default: all). E.g. --layers 0 11 23")
    parser.add_argument("--output", default=None, help="Output PNG path")
    args = parser.parse_args()

    ckpts = find_checkpoints(args.out_dir)
    if not ckpts:
        print(f"No checkpoints found in {args.out_dir}")
        sys.exit(1)

    print(f"Found {len(ckpts)} checkpoints:")
    for sort_key, path, fname in ckpts:
        print(f"  {fname}")

    # Load and analyze each
    all_stats = []
    iters = []
    ckpt_files = []
    for sort_key, path, fname in ckpts:
        print(f"Analyzing {fname}...")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        iter_num = ckpt.get("iter_num", sort_key if sort_key != float("inf") else -1)
        model_sd = ckpt.get("model", ckpt)
        stats = analyze_checkpoint(model_sd)
        all_stats.append(stats)
        iters.append(iter_num)
        ckpt_files.append(path)

        # Print summary for first layer
        first_layer = sorted(stats.keys())[0] if stats else None
        if first_layer:
            s = stats[first_layer]
            print(f"  {first_layer}: alpha={s.get('alpha_res', '?'):.4f}, "
                  f"phi_norm={s.get('phi_res_norm', s.get('compress_res_norm', '?')):.4f}, "
                  f"diag_dom={s.get('static_diag_dominance', '?'):.3f}, "
                  f"entropy={s.get('static_entropy', '?'):.3f}")

    out_path = args.output or os.path.join(args.out_dir, "routing_dynamics.png")
    plot_dynamics(all_stats, iters, args.layers, out_path)
