#!/bin/bash
# One-shot bootstrap for a RunPod (or any CUDA) pod.
#
# The experiment's efficiency metrics are only comparable across precisions if
# every run happens in one fixed environment.  This script pins that
# environment and prints the fingerprint that analyze.py later verifies.
#
#   VOLUME=/workspace bash scripts/setup_runpod.sh
set -e

VOLUME=${VOLUME:-/workspace}
REQUIRED_GB=${REQUIRED_GB:-200}

echo "=========================================================="
echo " RunPod bootstrap"
echo " persistent volume: $VOLUME"
echo "=========================================================="

if [ ! -d "$VOLUME" ]; then
    echo "ERROR: $VOLUME does not exist. Attach a network volume and set VOLUME."
    exit 1
fi

# Model weights must live on the persistent volume: container disk is wiped
# when the pod stops, and the full model set is ~145 GB of GGUF plus ~32 GB of
# safetensors for the layer sweep.
export HF_HOME="$VOLUME/hf"
export HF_HUB_ENABLE_HF_TRANSFER=1
# GGUF weights are fetched with an explicit cache_dir, which takes precedence
# over HF_HOME, so they need their own variable or ~145 GB lands on the
# container disk and is lost when the pod stops.
export MODEL_WEIGHTS_DIR="$VOLUME/weights"
mkdir -p "$HF_HOME" "$MODEL_WEIGHTS_DIR"
{
  echo "export HF_HOME=$VOLUME/hf"
  echo "export HF_HUB_ENABLE_HF_TRANSFER=1"
  echo "export MODEL_WEIGHTS_DIR=$VOLUME/weights"
} >> ~/.bashrc

FREE_GB=$(df -BG "$VOLUME" | awk 'NR==2 {gsub("G","",$4); print $4}')
echo ""
echo "Free space on $VOLUME: ${FREE_GB} GB (need ~${REQUIRED_GB} GB for the full set)"
if [ "$FREE_GB" -lt "$REQUIRED_GB" ]; then
    echo "  WARNING: tight. Use --evict on run_phase.py, or run fewer models per pass."
fi

echo ""
echo "[1/4] Python dependencies"
pip install -q --upgrade pip
pip install -q -r requirements.txt
pip install -q hf_transfer

echo ""
echo "[2/4] Inference engine"
bash scripts/install_engine.sh

echo ""
echo "[3/4] PyTorch dependencies (Phase 6 layer sweep only)"
if [ -n "$WITH_MECHANISM" ]; then
    pip install -q -r requirements-mechanism.txt
else
    echo "  skipped (set WITH_MECHANISM=1 to install torch + transformers)"
fi

echo ""
echo "[4/4] Environment fingerprint"
python - <<'PY'
import json
from evaluation.hardware import env_fingerprint, fingerprint_hash

fp = env_fingerprint(-1)
print(json.dumps(fp, indent=2, default=str))
print(f"\nenv_hash: {fingerprint_hash(fp)}")
print("Every prediction file must carry this hash. analyze.py refuses to treat")
print("latency/throughput/memory as comparable across differing hashes.")
PY

echo ""
echo "=========================================================="
echo " Ready."
echo ""
echo " Gated models (Phase 6 only) need a token:"
echo "   huggingface-cli login      # or: export HF_TOKEN=hf_..."
echo ""
echo " Then:"
echo "   python scripts/prefetch_models.py --models bit-ladder"
echo "   python scripts/download_datasets.py --core"
echo "   python scripts/verify_scorer.py --model q4 --dataset xstest --n 40"
echo "   MODELS=bit-ladder bash scripts/run_everything.sh"
echo "=========================================================="
