#!/usr/bin/env bash
set -euo pipefail

# ----------------------------
# compiler checks
# ----------------------------
max_major=11
check_compiler() {
  local bin="$1"
  local max="$2"

  if ! command -v "$bin" >/dev/null 2>&1; then
    echo "ERROR: $bin not found in PATH" >&2
    exit 1
  fi

  local major
  major="$("$bin" -dumpversion | cut -d. -f1)"
  if [[ "$major" -gt "$max" ]]; then
    echo "ERROR: $bin version $major > $max (unsupported)" >&2
    echo "       Please use $bin-$max or lower." >&2
    exit 1
  fi

  echo "OK: $bin version $major ≤ $max"
}

check_compiler gcc "$max_major"
check_compiler g++ "$max_major"
# don't hard-require a standalone `c++` binary; we'll force CXX=g++ for builds.

# ----------------------------
# env / paths
# ----------------------------
ENV_NAME="oldfast"          # keep your name, but this will now match target-style layout
PY_VER="3.8"

CONDA_BIN="$(command -v conda)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUN_IN_ENV() { "$CONDA_BIN" run -n "$ENV_NAME" "$@"; }

# ----------------------------
# create env (pin numpy<2 from day 0)
# ----------------------------
"$CONDA_BIN" create -n "$ENV_NAME" -y python="$PY_VER" "numpy<2" pip

# Constraints applied to ALL pip installs (prevents numpy2 and also prevents pip from swapping torch away)
CONSTRAINTS_FILE="$SCRIPT_DIR/pip-constraints-${ENV_NAME}.txt"
cat > "$CONSTRAINTS_FILE" <<'EOF'
numpy<2
torch==2.0.1
torchvision==0.15.2
torchmetrics==1.2.0
torch-ema==0.3
EOF

# Basic build tooling (helps when compiling tinycudann / other extensions)
RUN_IN_ENV "$CONDA_BIN" install -y -n "$ENV_NAME" -c conda-forge cmake ninja

# ----------------------------
# Install PyTorch the "target" way: conda pytorch + pytorch-cuda=11.7
# ----------------------------
# This is the key change that makes `conda list | fgrep cuda` show cuda-* runtime packages.
RUN_IN_ENV "$CONDA_BIN" install -y -n "$ENV_NAME" \
  -c pytorch -c nvidia \
  pytorch==2.0.1 torchvision==0.15.2 pytorch-cuda=11.7

# sanity print
RUN_IN_ENV python -c "import numpy as np, torch; print('numpy', np.__version__, 'torch', torch.__version__)"

# ----------------------------
# tiny-cuda-nn / tinycudann
# ----------------------------
# To match your target's `tinycudann 1.6`, you MUST pin to the same tag/commit you used before.
# If you don't know it, start by trying a tag name you believe exists, or paste a commit hash.
TCNN_REF="${TCNN_REF:-v1.6}"   # change to a known-good tag/commit if needed

RUN_IN_ENV bash -lc '
  set -euo pipefail
  export CC="$(command -v gcc)"
  export CXX="$(command -v g++)"
  export CUDAHOSTCXX="$CXX"
  python -m pip install -U -c "'"$CONSTRAINTS_FILE"'" pip setuptools wheel

  # IMPORTANT: --no-build-isolation so the build env can import torch
  python -m pip install -v --no-build-isolation -c "'"$CONSTRAINTS_FILE"'" \
    "git+https://github.com/NVlabs/tiny-cuda-nn@'"$TCNN_REF"'#subdirectory=bindings/torch"
'

# ----------------------------
# build your local extensions
# ----------------------------
cd "$SCRIPT_DIR/../P/Polattngp5Berkeley/tngp1/raymarching/"
RUN_IN_ENV bash -lc 'export CC=gcc CXX=g++; python setup.py install'

cd "$SCRIPT_DIR/../versions/codes_py/extern_cuda/nerfacc_pytorch_v053/"
RUN_IN_ENV bash -lc 'export CC=gcc CXX=g++; python setup.py install'

cd "$SCRIPT_DIR/"

# ----------------------------
# Python deps (pin to match target where relevant)
# ----------------------------
RUN_IN_ENV python -m pip install -c "$CONSTRAINTS_FILE" torch-ema==0.3 torchmetrics==1.2.0
RUN_IN_ENV python -m pip install -c "$CONSTRAINTS_FILE" opt_einsum lpips kornia
RUN_IN_ENV python -m pip install -c "$CONSTRAINTS_FILE" opencv-python scikit-image scikit-video
RUN_IN_ENV python -m pip install -c "$CONSTRAINTS_FILE" matplotlib plyfile trimesh

RUN_IN_ENV python -m pip install -c "$CONSTRAINTS_FILE" easydict prettytable dominate

# ----------------------------
# Final diagnostics (compare against your target)
# ----------------------------
echo "===== pip torch/cuda ====="
RUN_IN_ENV pip list | egrep -i 'torch|tinycudann' || true

echo "===== conda torch/cuda ====="
RUN_IN_ENV conda list | egrep -i 'pytorch|torchvision|pytorch-cuda|cuda-|tinycudann|numpy' || true

echo "Done. Environment '$ENV_NAME' created (conda-first PyTorch/CUDA, numpy<2 enforced)."
