#!/bin/bash
#
# Set the path
ENV_NAME=newslow
ANACONDA_PATH=$(conda info --base)

PIP_BIN=$ANACONDA_PATH/envs/$ENV_NAME/bin/pip
CONDA_BIN=$(which conda)
PY_BIN=$ANACONDA_PATH/envs/$ENV_NAME/bin/python

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

$CONDA_BIN create -n $ENV_NAME python=3 -y
$PIP_BIN install torch torchvision --index-url https://download.pytorch.org/whl/cu130
$PIP_BIN install -v --no-build-isolation git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
cd $SCRIPT_DIR/../P/Polattngp5Berkeley/tngp1/raymarching/
$PY_BIN setup.py install
cd $SCRIPT_DIR/../versions/codes_py/extern_cuda/nerfacc_pytorch_v053/
$PY_BIN setup.py install
cd $SCRIPT_DIR/
$PIP_BIN install torch-ema torchmetrics
$PIP_BIN install opt_einsum
$PIP_BIN install lpips
$PIP_BIN install kornia
$PIP_BIN install opencv-python scikit-image scikit-video
$PIP_BIN install matplotlib plyfile trimesh PyMCubes
$PIP_BIN install easydict prettytable dominate
