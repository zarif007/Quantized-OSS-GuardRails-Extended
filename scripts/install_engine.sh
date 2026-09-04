#!/bin/bash
# Install the llama-cpp-python build that matches this machine's backend.
#
# This is separate from requirements.txt because the wheel is backend-specific:
# `pip install llama-cpp-python` produces a CPU-only build, and on a rented GPU
# pod that build runs the whole experiment on the host CPU without any error.
set -e

VERSION=${LLAMA_CPP_VERSION:-0.3.16}
FORCE=${FORCE:-""}

detect_backend() {
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
        echo "cuda"
    elif [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
        echo "metal"
    else
        echo "cpu"
    fi
}

BACKEND=${BACKEND:-$(detect_backend)}
echo "Detected backend: $BACKEND"

PIP_FLAGS="--no-cache-dir"
if [ -n "$FORCE" ]; then PIP_FLAGS="$PIP_FLAGS --force-reinstall"; fi

case "$BACKEND" in
    cuda)
        # CUDA 12.4 wheels; change cu124 to match the pod image if needed.
        CUDA_TAG=${CUDA_TAG:-cu124}
        echo "Installing prebuilt CUDA wheel ($CUDA_TAG)..."
        if ! pip install $PIP_FLAGS "llama-cpp-python==$VERSION" \
            --extra-index-url "https://abetlen.github.io/llama-cpp-python/whl/$CUDA_TAG"; then
            echo "Prebuilt wheel unavailable; compiling from source (10-20 min)..."
            CMAKE_ARGS="-DGGML_CUDA=on" pip install $PIP_FLAGS --no-binary llama-cpp-python \
                "llama-cpp-python==$VERSION"
        fi
        ;;
    metal)
        echo "Installing Metal build..."
        CMAKE_ARGS="-DGGML_METAL=on" pip install $PIP_FLAGS "llama-cpp-python==$VERSION"
        ;;
    *)
        echo "Installing CPU build..."
        pip install $PIP_FLAGS "llama-cpp-python==$VERSION"
        ;;
esac

echo ""
echo "Verifying GPU offload support..."
python - <<'PY'
import sys
import llama_cpp

supported = llama_cpp.llama_supports_gpu_offload()
print(f"  llama-cpp-python {llama_cpp.__version__}")
print(f"  llama_supports_gpu_offload(): {supported}")
sys.exit(0)
PY

if [ "$BACKEND" = "cuda" ]; then
    python -c "
import llama_cpp, sys
if not llama_cpp.llama_supports_gpu_offload():
    print('')
    print('ERROR: the installed build has no GPU support. Retry with:')
    print('  FORCE=1 BACKEND=cuda bash scripts/install_engine.sh')
    sys.exit(1)
print('')
print('CUDA build verified.')
"
fi
