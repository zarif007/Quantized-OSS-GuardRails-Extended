#!/bin/bash
# Small end-to-end check: score a couple of precisions, then print the metrics.
#
# Use it to confirm a new machine measures what you think it measures before
# committing GPU hours to a full sweep.  Writes to results/smoke/ so it never
# touches the real prediction set.
#
#   bash scripts/smoke_test.sh
#   MODELS="q3 q4 q6" N=60 DATASET=harmbench bash scripts/smoke_test.sh
set -e

MODELS=${MODELS:-"q3 q4"}
DATASET=${DATASET:-"xstest"}
N=${N:-40}
OUT=${OUT:-"results/smoke"}
EXTRA=${EXTRA:-""}

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
echo " Smoke test: $MODELS on $DATASET ($N prompts each)"
echo " gpu_layers: $GPU_LAYERS   output: $OUT"
echo "=========================================================="

rm -rf "$OUT"
mkdir -p "$OUT"

for MODEL in $MODELS; do
    echo ""
    echo "--> $MODEL"
    python scripts/run_model.py \
        --model "$MODEL" --dataset "$DATASET" --subset "$N" \
        --n-gpu-layers "$GPU_LAYERS" --warmup 3 --output-dir "$OUT" $EXTRA \
        2>&1 | grep -E "backend:|Memory profile|backend: |Weight file|Device memory|Host RSS|Total reported|Environment |median latency|! " || true
done

echo ""
echo "--> analysis"
python evaluation/analyze.py \
    --predictions-dir "$OUT" \
    --tables-dir "$OUT/tables" \
    --figures-dir "$OUT/figures" >/dev/null

python - "$OUT" <<'PY'
import sys
import pandas as pd

out = sys.argv[1]
summary = pd.read_csv(f"{out}/tables/summary_metrics.csv")
envs = pd.read_csv(f"{out}/tables/environments.csv")

pd.set_option("display.width", 200)

print("\n=== Environment (must be a single row) ===")
print(envs[[c for c in ("env_hash", "backend", "gpu_name", "n_rows") if c in envs.columns]]
      .to_string(index=False))
if len(envs) > 1:
    print("  WARNING: more than one environment. Efficiency metrics are not comparable.")

print("\n=== Safety / usefulness ===")
cols = ["model", "safety_rate", "precision", "f1_score", "false_positive_rate",
        "false_negative_rate", "usefulness_rate", "auroc", "ece"]
print(summary[[c for c in cols if c in summary.columns]].round(4).to_string(index=False))

print("\n=== Efficiency (conditional on the environment above) ===")
cols = ["model", "latency_median_sec", "latency_p95_sec", "latency_iqr_sec",
        "single_stream_prompts_per_sec", "prefill_tokens_per_sec",
        "model_weight_gb", "vram_gb", "peak_memory_gb", "ges"]
print(summary[[c for c in cols if c in summary.columns]].round(4).to_string(index=False))

deploy = pd.read_csv(f"{out}/tables/safety_per_gb.csv")
print("\n=== Safety per GB ===")
cols = ["model", "tpr_at_target_fpr", "memory_gb", "weights_gb",
        "latency_median_sec", "safety_per_gb", "efficiency_comparable"]
print(deploy[[c for c in cols if c in deploy.columns]].round(4).to_string(index=False))
PY

echo ""
echo "=========================================================="
echo " tables:  $OUT/tables/"
echo " figures: $OUT/figures/"
echo "=========================================================="
