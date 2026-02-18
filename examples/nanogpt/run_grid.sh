#!/bin/bash
set -euo pipefail

# Fine-grained routing experiment grid
# Distributes experiments across 4x 3090 GPUs (no DDP, independent runs)
#
# Features:
#   - Auto-resume: if out_dir has a checkpoint, passes --resume_from to continue
#   - Skip finished: if ckpt_final.pt exists, the job is skipped entirely
#   - Greedy scheduling: starts a new job as soon as any GPU becomes free
#
# Usage:
#   bash run_grid.sh             # run Session 1 (L0 + L1 + L2)
#   bash run_grid.sh L0          # run only L0 baselines
#   bash run_grid.sh L1          # run only L1 routing sweep
#   bash run_grid.sh L2          # run only L2 bottleneck sweep
#   bash run_grid.sh session2    # run Session 2 (L3 + L4, fill in best params first)
#   bash run_grid.sh L1 --resume # continue finished runs (e.g. after bumping max_iters)

cd "$(dirname "$0")"

SEEDS_BASELINE=(42 137)
SEEDS=(42 137 256)
NUM_GPUS=4
RESUME=false

# Parse args: positional filter + optional --resume
FILTER="session1"
for arg in "$@"; do
    case "$arg" in
        --resume) RESUME=true ;;
        *) FILTER="$arg" ;;
    esac
done

# --- Greedy GPU scheduler ---
# Tracks one PID per GPU slot. Launches next job on first free slot.

declare -a GPU_PIDS
for ((i=0; i<NUM_GPUS; i++)); do GPU_PIDS[$i]=0; done

wait_for_free_gpu() {
    # Poll until a GPU slot is free (its process finished)
    while true; do
        for ((i=0; i<NUM_GPUS; i++)); do
            local pid=${GPU_PIDS[$i]}
            if [ "$pid" -eq 0 ] || ! kill -0 "$pid" 2>/dev/null; then
                # Slot free — reap it (ignore exit status of individual jobs)
                if [ "$pid" -ne 0 ]; then
                    wait "$pid" 2>/dev/null || true
                fi
                GPU_PIDS[$i]=0
                FREE_GPU=$i
                return
            fi
        done
        sleep 2
    done
}

wait_all_gpus() {
    for ((i=0; i<NUM_GPUS; i++)); do
        local pid=${GPU_PIDS[$i]}
        if [ "$pid" -ne 0 ]; then
            wait "$pid" 2>/dev/null || true
            GPU_PIDS[$i]=0
        fi
    done
}

run_queue() {
    local -a cmds=("$@")
    for cmd in "${cmds[@]}"; do
        wait_for_free_gpu
        echo "[GPU $FREE_GPU] $cmd"
        CUDA_VISIBLE_DEVICES=$FREE_GPU bash -c "$cmd" &
        GPU_PIDS[$FREE_GPU]=$!
    done
    wait_all_gpus
}

# --- Resume / skip logic ---
# Wraps a train command: skip if done, add --resume_from if checkpoint exists.

make_cmd() {
    local base_cmd="$1"
    local out_dir="$2"

    # Has final checkpoint
    if [ -f "${out_dir}/ckpt_final.pt" ]; then
        if [ "$RESUME" = true ]; then
            echo "${base_cmd} --resume_from=${out_dir}"
        else
            echo ""
        fi
        return
    fi

    # Has intermediate checkpoint — always resume
    if [ -f "${out_dir}/ckpt.pt" ]; then
        echo "${base_cmd} --resume_from=${out_dir}"
        return
    fi

    # Fresh run
    echo "${base_cmd}"
}

# --- Level 0: Scaling baselines (original unmodified code) ---

build_L0() {
    local -n _jobs=$1
    for config in baseline_32M baseline_50M baseline_100M mhc_original; do
        for seed in "${SEEDS_BASELINE[@]}"; do
            local out_dir="out-L0-${config}-s${seed}"
            local cmd
            cmd=$(make_cmd \
                "uv run python train.py config/train_${config}.py --seed=$seed --wandb_run_name=L0_${config}_s${seed} --out_dir=${out_dir}" \
                "${out_dir}")
            [ -n "$cmd" ] && _jobs+=("$cmd") || echo "[SKIP] ${out_dir} (already finished)"
        done
    done
}

# --- Level 1: Routing sweep (n=1, no bottleneck, m=4,8,16,32) ---

build_L1() {
    local -n _jobs=$1
    for m in 4 8 16 32; do
        for seed in "${SEEDS[@]}"; do
            local out_dir="out-L1-m${m}-s${seed}"
            local cmd
            cmd=$(make_cmd \
                "uv run python train.py config/train_L1_m${m}.py --seed=$seed --wandb_run_name=L1_m${m}_s${seed} --out_dir=${out_dir}" \
                "${out_dir}")
            [ -n "$cmd" ] && _jobs+=("$cmd") || echo "[SKIP] ${out_dir} (already finished)"
        done
    done
}

# --- Level 2: Bottleneck sweep (n=1, m=16 fixed, d=4,8,16,32,64,256) ---

build_L2() {
    local -n _jobs=$1
    for d in 4 8 16 32 64 256; do
        for seed in "${SEEDS[@]}"; do
            local out_dir="out-L2-m16-d${d}-s${seed}"
            local cmd
            cmd=$(make_cmd \
                "uv run python train.py config/train_L2_m16_d${d}.py --seed=$seed --wandb_run_name=L2_m16_d${d}_s${seed} --out_dir=${out_dir}" \
                "${out_dir}")
            [ -n "$cmd" ] && _jobs+=("$cmd") || echo "[SKIP] ${out_dir} (already finished)"
        done
    done
}

# --- Session 2: L3 + L4 (fill in best params after analyzing L0-L2) ---
# Level 3: Capacity x granularity (sweep n x m with best d)
# Level 4: Full transport cycle (top configs from L3)
# UPDATE these variables from L1/L2 results before running:

# BEST_M=16       # from L1
# DOUBLE_BEST_M=32
# BEST_D=32       # from L2

build_session2() {
    local -n _jobs=$1
    echo "WARNING: Session 2 not yet configured. Update BEST_M, BEST_D in run_grid.sh first."
    echo "After analyzing L0-L2 results, create L3/L4 configs and update this function."
    return 1
}

case "$FILTER" in
    L0)
        JOBS=()
        build_L0 JOBS
        echo "L0: ${#JOBS[@]} jobs across $NUM_GPUS GPUs"
        run_queue "${JOBS[@]}"
        ;;
    L1)
        JOBS=()
        build_L1 JOBS
        echo "L1: ${#JOBS[@]} jobs across $NUM_GPUS GPUs"
        run_queue "${JOBS[@]}"
        ;;
    L2)
        JOBS=()
        build_L2 JOBS
        echo "L2: ${#JOBS[@]} jobs across $NUM_GPUS GPUs"
        run_queue "${JOBS[@]}"
        ;;
    session1)
        JOBS=()
        build_L0 JOBS
        build_L1 JOBS
        build_L2 JOBS
        echo "Session 1 (L0+L1+L2): ${#JOBS[@]} jobs across $NUM_GPUS GPUs"
        run_queue "${JOBS[@]}"
        echo "Session 1 complete. Analyze results, then run session2."
        ;;
    session2)
        JOBS=()
        build_session2 JOBS
        echo "Session 2 (L3+L4): ${#JOBS[@]} jobs across $NUM_GPUS GPUs"
        run_queue "${JOBS[@]}"
        echo "All experiments complete."
        ;;
    *)
        echo "Usage: $0 [session1|session2|L0|L1|L2]"
        exit 1
        ;;
esac

echo ""
echo "Grid complete."
