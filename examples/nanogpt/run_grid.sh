#!/bin/bash
set -euo pipefail

# Fine-grained routing experiment grid
# Distributes experiments across 4x 3090 GPUs (no DDP, independent runs)
#
# Usage:
#   bash run_grid.sh             # run Session 1 (L0 + L1 + L2)
#   bash run_grid.sh L0          # run only L0 baselines
#   bash run_grid.sh L1          # run only L1 routing sweep
#   bash run_grid.sh L2          # run only L2 bottleneck sweep
#   bash run_grid.sh session2    # run Session 2 (L3 + L4, fill in best params first)

cd "$(dirname "$0")"

SEEDS_BASELINE=(42 137)
SEEDS=(42 137 256)
NUM_GPUS=4
FILTER="${1:-session1}"

run_queue() {
    local -a cmds=("$@")
    local gpu=0
    local count=0
    for cmd in "${cmds[@]}"; do
        echo "[GPU $gpu] $cmd"
        CUDA_VISIBLE_DEVICES=$gpu bash -c "$cmd" &
        gpu=$(( (gpu + 1) % NUM_GPUS ))
        count=$((count + 1))
        if [ $((count % NUM_GPUS)) -eq 0 ]; then
            wait
        fi
    done
    wait
}

# --- Level 0: Scaling baselines (original unmodified code) ---

build_L0() {
    local -n _jobs=$1
    for config in baseline_32M baseline_50M baseline_100M mhc_original; do
        for seed in "${SEEDS_BASELINE[@]}"; do
            _jobs+=("python train.py config/train_${config}.py --seed=$seed --wandb_run_name=L0_${config}_s${seed} --out_dir=out-L0-${config}-s${seed}")
        done
    done
}

# --- Level 1: Routing sweep (n=1, no bottleneck, m=4,8,16,32) ---

build_L1() {
    local -n _jobs=$1
    for m in 4 8 16 32; do
        for seed in "${SEEDS[@]}"; do
            _jobs+=("python train.py config/train_L1_m${m}.py --seed=$seed --wandb_run_name=L1_m${m}_s${seed} --out_dir=out-L1-m${m}-s${seed}")
        done
    done
}

# --- Level 2: Bottleneck sweep (n=1, m=16 fixed, d=4,8,16,32,64,256) ---

build_L2() {
    local -n _jobs=$1
    for d in 4 8 16 32 64 256; do
        for seed in "${SEEDS[@]}"; do
            _jobs+=("python train.py config/train_L2_m16_d${d}.py --seed=$seed --wandb_run_name=L2_m16_d${d}_s${seed} --out_dir=out-L2-m16-d${d}-s${seed}")
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
