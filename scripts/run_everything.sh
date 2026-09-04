#!/bin/bash
set -e

MODELS=${MODELS:-"bit-ladder"}
DATASETS=${DATASETS:-"harmbench xstest"}
SUBSET=${SUBSET:-""}
THREADS=${THREADS:-""}
EVICT=${EVICT:-""}
RESUME=${RESUME:-""}
BATCH=${BATCH:-""}
LATENCY_REPEATS=${LATENCY_REPEATS:-""}
# Skip the pip step by default on a pod: run_everything.sh must never
# reinstall llama-cpp-python, because requirements.txt cannot know which
# backend build is correct and a reinstall would replace a CUDA build with
# a CPU one mid-experiment.  Set INSTALL=1 for a first local run.
INSTALL=${INSTALL:-""}
VENV=${VENV:-".venv"}

# Default to full offload wherever a GPU exists; 0 forces CPU.
if [ -z "$GPU_LAYERS" ]; then
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
        GPU_LAYERS="-1"
    elif [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
        GPU_LAYERS="-1"
    else
        GPU_LAYERS="0"
    fi
fi

echo "=========================================================="
echo " Quantized Guardrail Pipeline"
echo " models:   $MODELS"
echo " datasets: $DATASETS"
echo " threads:  ${THREADS:-auto}   gpu_layers: $GPU_LAYERS"
echo "=========================================================="

echo ""
echo "[1/5] Environment"
if [ -d "$VENV" ]; then
    source "$VENV/bin/activate"
fi
if [ -n "$INSTALL" ]; then
    if [ ! -d "$VENV" ]; then
        python3 -m venv "$VENV"
        source "$VENV/bin/activate"
    fi
    pip install -q -r requirements.txt
    bash scripts/install_engine.sh
else
    echo "  skipping pip (set INSTALL=1 to install deps and the matching engine build)"
fi
python -c "
from evaluation.hardware import env_fingerprint, fingerprint_hash
fp = env_fingerprint($GPU_LAYERS)
print(f\"  backend={fp['backend']} device={fp.get('gpu_name') or fp['processor']} env={fingerprint_hash(fp)}\")
"

echo ""
echo "[2/5] Datasets"
python scripts/download_datasets.py --datasets $DATASETS

echo ""
echo "[3/5] Gate B verification"
FIRST_MODEL=$(echo $MODELS | awk '{print $1}')
if [ "$FIRST_MODEL" = "bit-ladder" ]; then FIRST_MODEL="q4"; fi
python scripts/verify_scorer.py --model "$FIRST_MODEL" --dataset xstest --n 40 \
    --n-gpu-layers "$GPU_LAYERS" ${THREADS:+--n-threads $THREADS} || {
    echo "Gate B FAILED - stopping. Fix the scorer before running the full sweep."
    exit 1
}

echo ""
echo "[4/5] Inference"
ARGS="--phase 0 --models $MODELS --datasets $DATASETS --n-gpu-layers $GPU_LAYERS --skip-analysis"
if [ -n "$SUBSET" ]; then ARGS="$ARGS --subset $SUBSET"; fi
if [ -n "$THREADS" ]; then ARGS="$ARGS --n-threads $THREADS"; fi
if [ -n "$EVICT" ]; then ARGS="$ARGS --evict"; fi
if [ -n "$RESUME" ]; then ARGS="$ARGS --resume"; fi
if [ -n "$BATCH" ]; then ARGS="$ARGS --n-batch $BATCH"; fi
if [ -n "$LATENCY_REPEATS" ]; then ARGS="$ARGS --latency-repeats $LATENCY_REPEATS"; fi
python scripts/run_phase.py $ARGS

echo ""
echo "[5/5] Analysis"
python evaluation/analyze.py

echo "=========================================================="
echo " Done."
echo " predictions: results/predictions/"
echo " tables:      results/tables/"
echo " figures:     results/figures/"
echo " gates:       results/tables/gates.json"
echo " hardware:    results/tables/environments.csv"
echo "=========================================================="
