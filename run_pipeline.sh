#!/usr/bin/env bash
#
# End-to-end SpikeRNN pipeline:
#   1. Train a rate RNN on a cognitive task
#   2. Optimize the scaling factor via grid search
#   3. Evaluate the spiking network
#
# Usage:
#   ./run_pipeline.sh                  # defaults: go-nogo task, N=200
#   ./run_pipeline.sh --task xor       # run XOR task instead
#   ./run_pipeline.sh --task mante     # run Mante task instead
#
set -euo pipefail

# ── Configurable parameters ──────────────────────────────────────────
TASK="${1:---task}"
if [[ "$TASK" == "--task" ]]; then
    TASK="${2:-go-nogo}"
fi

N=200
P_INH=0.20
P_REC=0.20
SOM_N=0
GAIN=1.5
ACT=sigmoid
LOSS_FN=l2
DECAY_TAUS="4 20"
N_TRAIN_TRIALS=5000
N_EVAL_TRIALS=100
SCALING_FACTORS="20:76:5"
EVAL_AMP_THRESH=0.7
GPU=0
GPU_FRAC=0.4

# ── Task-specific settings ───────────────────────────────────────────
# Defaults match _get_default_task_settings in eval_tasks.py
case "${TASK}" in
    go-nogo)
        T=200; STIM_ON=30; STIM_DUR=20; DELAY="" ;;
    xor)
        T=300; STIM_ON=50; STIM_DUR=50; DELAY=20 ;;
    mante)
        T=300; STIM_ON=50; STIM_DUR=100; DELAY="" ;;
    *)
        echo "ERROR: Unknown task '${TASK}'. Choose from: go-nogo, xor, mante" >&2
        exit 1 ;;
esac

# Directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}"

# ── Derived paths ────────────────────────────────────────────────────
# Rate training saves models under <output_dir>/models/<task>/
MODEL_DIR="${OUTPUT_DIR}/models/${TASK}"

echo "=============================================="
echo "  SpikeRNN End-to-End Pipeline"
echo "  Task:       ${TASK}"
echo "  N:          ${N}"
echo "  T:          ${T}"
echo "  stim_on:    ${STIM_ON}"
echo "  stim_dur:   ${STIM_DUR}"
echo "  delay:      ${DELAY:-n/a}"
echo "  eval_thresh:${EVAL_AMP_THRESH}"
echo "=============================================="

# ── Step 1: Train the rate RNN ───────────────────────────────────────
echo ""
echo "[Step 1/3] Training rate RNN on '${TASK}' task..."
echo "----------------------------------------------"

cd "${SCRIPT_DIR}/rate"

python main.py \
    --gpu "${GPU}" \
    --gpu_frac "${GPU_FRAC}" \
    --n_trials "${N_TRAIN_TRIALS}" \
    --mode train \
    --output_dir "${OUTPUT_DIR}" \
    --N "${N}" \
    --gain "${GAIN}" \
    --P_inh "${P_INH}" \
    --P_rec "${P_REC}" \
    --som_N "${SOM_N}" \
    --apply_dale True \
    --task "${TASK}" \
    --act "${ACT}" \
    --loss_fn "${LOSS_FN}" \
    --decay_taus ${DECAY_TAUS}

# Find the .mat model file produced by training
MODEL_PATH=$(find "${MODEL_DIR}" -name "*.mat" -type f | head -1)
if [[ -z "${MODEL_PATH}" ]]; then
    echo "ERROR: No .mat file found in ${MODEL_DIR}" >&2
    exit 1
fi
echo "Trained model saved to: ${MODEL_PATH}"

cd "${SCRIPT_DIR}"

# ── Step 2: Optimize scaling factor via grid search ──────────────────
# Convert task name: rate uses "go-nogo" but lambda_grid_search CLI also expects hyphenated form
echo ""
echo "[Step 2/3] Optimizing scaling factor for '${TASK}'..."
echo "----------------------------------------------"

# Build optional args for task-specific settings
TASK_ARGS="--T ${T} --stim_on ${STIM_ON} --stim_dur ${STIM_DUR} --eval_amp_thresh ${EVAL_AMP_THRESH}"
if [[ -n "${DELAY}" ]]; then
    TASK_ARGS="${TASK_ARGS} --delay ${DELAY}"
fi

python -m spiking.lambda_grid_search \
    --model_path "${MODEL_PATH}" \
    --task_name "${TASK}" \
    --n_trials "${N_EVAL_TRIALS}" \
    --scaling_factors "${SCALING_FACTORS}" \
    ${TASK_ARGS}

# ── Step 3: Evaluate the spiking network ─────────────────────────────
# eval_tasks uses underscore format (go_nogo)
TASK_UNDERSCORE="${TASK//-/_}"

echo ""
echo "[Step 3/3] Evaluating spiking network on '${TASK_UNDERSCORE}'..."
echo "----------------------------------------------"

# Build optional args
EVAL_ARGS="--T ${T} --stim_on ${STIM_ON} --stim_dur ${STIM_DUR}"
if [[ -n "${DELAY}" ]]; then
    EVAL_ARGS="${EVAL_ARGS} --delay ${DELAY}"
fi

python -m spiking.eval_tasks \
    --task "${TASK_UNDERSCORE}" \
    --model_path "${MODEL_PATH}" \
    --n_trials "${N_EVAL_TRIALS}" \
    ${EVAL_ARGS}

echo ""
echo "=============================================="
echo "  Pipeline complete!"
echo "  Model:   ${MODEL_PATH}"
echo "  Task:    ${TASK}"
echo "=============================================="
