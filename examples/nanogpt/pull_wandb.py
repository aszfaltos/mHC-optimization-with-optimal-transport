"""Pull wandb data and plot HC routing dynamics over training.

Usage:
    uv run python pull_wandb.py                          # auto-find L1_m4 run
    uv run python pull_wandb.py --run-name L1_m8_s42     # specific run
    uv run python pull_wandb.py --run-id abc123           # by wandb run id
"""
import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import wandb


def find_run(api, project, run_name=None, run_id=None):
    if run_id:
        return api.run(f"{project}/{run_id}")
    runs = api.runs(project, filters={"display_name": {"$regex": run_name or "L1_m4"}})
    if not runs:
        raise ValueError(f"No runs matching '{run_name}' in {project}")
    # Pick the longest run (most iters)
    best = max(runs, key=lambda r: r.summary.get("_step", 0))
    print(f"Found: {best.name} ({best.id}), {best.summary.get('_step', '?')} steps, state={best.state}")
    return best


def pull_scalars(run):
    history = run.scan_history(
        keys=[
            "train/loss", "val/loss", "train/loss_eval", "train/lr",
            "train/grad_norm", "perf/tok_per_sec",
            "perf/max_mem_allocated_mb", "perf/max_mem_reserved_mb",
            "tokens/seen", "perf/elapsed_s",
        ],
    )
    rows = list(history)
    df = pd.DataFrame(rows)
    if "_step" in df.columns:
        df = df.set_index("_step").sort_index()
    return df


def pull_tables(run, key):
    """Pull wandb Table logged at `key` across all steps.

    Tables are stored as artifacts (run_table type). Each eval step creates
    a new version of the artifact. We iterate artifact versions to get the
    table data at each step.
    """
    all_rows = []

    # Method 1: Try artifact-based tables (how wandb actually stores them)
    try:
        # Find the artifact collection for this key
        # Artifact name pattern: run-{run_id}-{key_sanitized}
        key_sanitized = key.replace("/", "")  # "hc/layer_cosine" -> "hclayer_cosine"
        artifact_name = f"run-{run.id}-{key_sanitized}"
        api = run.client
        project_path = f"{run.entity}/{run.project}"

        for version_idx in range(200):  # safety limit
            try:
                art = api.artifact(f"{project_path}/{artifact_name}:v{version_idx}", type="run_table")
                table = art.get(f"{key_sanitized}.table.json")
                if table is None:
                    # Try alternate name
                    entries = list(art.manifest.entries.keys())
                    if entries:
                        table = art.get(entries[0])
                if table is not None:
                    # wandb.Table object
                    cols = table.columns
                    for row_data in table.data:
                        r = dict(zip(cols, row_data))
                        r["_step"] = version_idx  # version index maps to eval step
                        all_rows.append(r)
            except wandb.errors.CommError:
                break
            except Exception:
                break

        if all_rows:
            # Fix step numbers: map version indices to actual iters using history
            history = list(run.scan_history(keys=[key], min_step=0))
            step_map = {i: row.get("_step", i) for i, row in enumerate(history)}
            for r in all_rows:
                r["_step"] = step_map.get(r["_step"], r["_step"])
            return pd.DataFrame(all_rows)
    except Exception as e:
        print(f"  Artifact method failed for {key}: {e}")

    # Method 2: Fallback to inline table scan
    history = run.scan_history(keys=[key])
    for row in history:
        step = row.get("_step", None)
        table_data = row.get(key)
        if table_data is None:
            continue
        if hasattr(table_data, "get"):
            cols = table_data.get("columns", [])
            data = table_data.get("data", [])
            for entry in data:
                r = dict(zip(cols, entry))
                r["_step"] = step
                all_rows.append(r)
    return pd.DataFrame(all_rows)


def plot_routing_dynamics(scalars, layer_cosine_df, layer_stats_df, run_name, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(f"HC Routing Dynamics: {run_name}", fontsize=16, y=0.98)
    gs = gridspec.GridSpec(3, 3, hspace=0.35, wspace=0.3)

    # 1. Train/val loss
    ax = fig.add_subplot(gs[0, 0])
    if "train/loss" in scalars.columns:
        s = scalars["train/loss"].dropna()
        # Downsample for smoothness
        ax.plot(s.index, s.values, alpha=0.2, color="C0", linewidth=0.5)
        ax.plot(s.rolling(50, min_periods=1).mean(), color="C0", label="train")
    if "val/loss" in scalars.columns:
        s = scalars["val/loss"].dropna()
        ax.plot(s.index, s.values, "o-", color="C1", markersize=3, label="val")
    ax.set_xlabel("iter")
    ax.set_ylabel("loss")
    ax.set_title("Loss")
    ax.legend()

    # 2. Learning rate
    ax = fig.add_subplot(gs[0, 1])
    if "train/lr" in scalars.columns:
        s = scalars["train/lr"].dropna()
        ax.plot(s.index, s.values, color="C2")
    ax.set_xlabel("iter")
    ax.set_ylabel("lr")
    ax.set_title("Learning Rate")

    # 3. Grad norm
    ax = fig.add_subplot(gs[0, 2])
    if "train/grad_norm" in scalars.columns:
        s = scalars["train/grad_norm"].dropna()
        ax.plot(s.index, s.values, alpha=0.2, color="C3", linewidth=0.5)
        ax.plot(s.rolling(50, min_periods=1).mean(), color="C3")
    ax.set_xlabel("iter")
    ax.set_ylabel("grad norm")
    ax.set_title("Gradient Norm")

    # 4. Layer cosine similarity heatmap
    ax = fig.add_subplot(gs[1, :])
    # Support both old schema ("layer") and new ("block")
    block_col = "block" if "block" in layer_cosine_df.columns else "layer"
    if not layer_cosine_df.empty and block_col in layer_cosine_df.columns:
        steps = sorted(layer_cosine_df["_step"].unique())
        layers = sorted(layer_cosine_df[block_col].unique())
        matrix = np.full((len(layers), len(steps)), np.nan)
        step_to_idx = {s: i for i, s in enumerate(steps)}
        layer_to_idx = {l: i for i, l in enumerate(layers)}
        for _, row in layer_cosine_df.iterrows():
            li = layer_to_idx.get(row[block_col])
            si = step_to_idx.get(row["_step"])
            if li is not None and si is not None:
                matrix[li, si] = row["cosine"]
        im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn_r", vmin=0.8, vmax=1.0,
                        interpolation="nearest")
        ax.set_xlabel("eval step")
        ax.set_ylabel("block")
        ax.set_xticks(range(len(steps)))
        ax.set_xticklabels(steps, rotation=45, fontsize=7)
        ax.set_yticks(range(0, len(layers), max(1, len(layers)//10)))
        ax.set_title("Inter-layer Cosine Similarity (lower = more routing)")
        fig.colorbar(im, ax=ax, shrink=0.6)
    else:
        ax.text(0.5, 0.5, "No layer_cosine data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Inter-layer Cosine Similarity")

    # 5-7. H^res stats over training
    stat_keys = [
        ("h_res_entropy", "H^res Entropy (high=uniform)", "C4"),
        ("h_res_sparsity", "H^res Sparsity (frac < 0.01)", "C5"),
        ("h_res_row_sum", "H^res Row Sum (should be ~1)", "C6"),
    ]
    for col_idx, (stat_key, title, color) in enumerate(stat_keys):
        ax = fig.add_subplot(gs[2, col_idx])
        if not layer_stats_df.empty and stat_key in layer_stats_df.columns:
            steps = sorted(layer_stats_df["_step"].unique())
            # Plot mean across all layers (both attn and mlp) + shade min/max
            means, mins, maxs = [], [], []
            for step in steps:
                vals = layer_stats_df[layer_stats_df["_step"] == step][stat_key].dropna()
                if len(vals) > 0:
                    means.append(vals.mean())
                    mins.append(vals.min())
                    maxs.append(vals.max())
                else:
                    means.append(np.nan)
                    mins.append(np.nan)
                    maxs.append(np.nan)
            ax.plot(steps, means, color=color, label="mean")
            ax.fill_between(steps, mins, maxs, alpha=0.2, color=color, label="min-max")
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, f"No {stat_key} data", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("iter")
        ax.set_title(title)

    path = os.path.join(out_dir, f"{run_name}_dynamics.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # Also save raw data
    scalars.to_csv(os.path.join(out_dir, f"{run_name}_scalars.csv"))
    if not layer_cosine_df.empty:
        layer_cosine_df.to_csv(os.path.join(out_dir, f"{run_name}_layer_cosine.csv"), index=False)
    if not layer_stats_df.empty:
        layer_stats_df.to_csv(os.path.join(out_dir, f"{run_name}_layer_stats.csv"), index=False)
    print(f"CSVs saved to {out_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Pull wandb data and plot HC routing dynamics")
    parser.add_argument("--project", default="fine-grained-routing")
    parser.add_argument("--run-name", default=None, help="Regex to match run name (default: L1_m4)")
    parser.add_argument("--run-id", default=None, help="Exact wandb run ID")
    parser.add_argument("--out-dir", default="wandb_plots", help="Output directory")
    args = parser.parse_args()

    api = wandb.Api()
    run = find_run(api, args.project, args.run_name, args.run_id)

    print("Pulling scalars...")
    scalars = pull_scalars(run)
    print(f"  {len(scalars)} rows, columns: {list(scalars.columns)}")

    print("Pulling layer cosine tables...")
    layer_cosine_df = pull_tables(run, "hc/layer_cosine")
    print(f"  {len(layer_cosine_df)} rows")

    print("Pulling layer stats tables...")
    layer_stats_df = pull_tables(run, "hc/layer_stats")
    print(f"  {len(layer_stats_df)} rows")

    plot_routing_dynamics(scalars, layer_cosine_df, layer_stats_df, run.name, args.out_dir)


if __name__ == "__main__":
    main()
