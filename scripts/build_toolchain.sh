#!/bin/bash
set -e

LLAMA_CPP_DIR=${LLAMA_CPP_DIR:-"third_party/llama.cpp"}
JOBS=${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}

echo "Building llama.cpp tools into $LLAMA_CPP_DIR"
echo "Required for: llama-quantize (mixed precision), llama-imatrix (calibration experiment)"

REPO_URL=${REPO_URL:-https://github.com/ggml-org/llama.cpp}
CLONE_RETRIES=${CLONE_RETRIES:-3}

# A directory that exists but is not a valid git repo is a leftover from an
# interrupted clone.  Without this, the else-branch below runs `git pull` on
# it, fails, and the build then compiles nothing.
if [ -d "$LLAMA_CPP_DIR" ] && ! git -C "$LLAMA_CPP_DIR" rev-parse --git-dir >/dev/null 2>&1; then
    echo "Removing incomplete checkout at $LLAMA_CPP_DIR"
    rm -rf "$LLAMA_CPP_DIR"
fi

if [ ! -d "$LLAMA_CPP_DIR" ]; then
    mkdir -p "$(dirname "$LLAMA_CPP_DIR")"
    # Shallow clones of this repo over HTTP/2 fail intermittently with
    # "stream not closed cleanly"; retry, then fall back to HTTP/1.1.
    cloned=""
    for attempt in $(seq 1 "$CLONE_RETRIES"); do
        echo "Clone attempt $attempt/$CLONE_RETRIES"
        if git clone --depth 1 "$REPO_URL" "$LLAMA_CPP_DIR"; then
            cloned=1
            break
        fi
        rm -rf "$LLAMA_CPP_DIR"
        echo "  clone failed; retrying with HTTP/1.1"
        if git -c http.version=HTTP/1.1 clone --depth 1 "$REPO_URL" "$LLAMA_CPP_DIR"; then
            cloned=1
            break
        fi
        rm -rf "$LLAMA_CPP_DIR"
        sleep 5
    done
    if [ -z "$cloned" ]; then
        echo "ERROR: could not clone $REPO_URL after $CLONE_RETRIES attempts."
        echo "Check network access, or clone manually into $LLAMA_CPP_DIR and re-run."
        exit 1
    fi
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
