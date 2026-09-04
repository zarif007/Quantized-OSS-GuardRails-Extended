#!/bin/bash
set -e

LLAMA_CPP_DIR=${LLAMA_CPP_DIR:-"third_party/llama.cpp"}
JOBS=${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}

echo "Building llama.cpp tools into $LLAMA_CPP_DIR"
echo "Required for: llama-quantize (mixed precision), llama-imatrix (calibration experiment)"

if [ ! -d "$LLAMA_CPP_DIR" ]; then
    mkdir -p "$(dirname "$LLAMA_CPP_DIR")"
    git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_CPP_DIR"
else
    git -C "$LLAMA_CPP_DIR" pull --ff-only || echo "  (skipping pull)"
fi

cmake -S "$LLAMA_CPP_DIR" -B "$LLAMA_CPP_DIR/build" -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
cmake --build "$LLAMA_CPP_DIR/build" --config Release -j "$JOBS" \
      --target llama-quantize llama-imatrix llama-cli

echo ""
echo "Built binaries:"
find "$LLAMA_CPP_DIR/build" -name "llama-quantize" -o -name "llama-imatrix" | sed 's/^/  /'
echo ""
echo "Export these before running mixed-precision or imatrix scripts:"
echo "  export LLAMA_QUANTIZE=$LLAMA_CPP_DIR/build/bin/llama-quantize"
echo "  export LLAMA_IMATRIX=$LLAMA_CPP_DIR/build/bin/llama-imatrix"
