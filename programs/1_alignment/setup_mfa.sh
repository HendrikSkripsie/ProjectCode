#!/usr/bin/env bash
# One-time setup: Montreal Forced Aligner in a conda environment + the British English models.
# Needs Miniconda/Miniforge (https://github.com/conda-forge/miniforge) installed first.
set -euo pipefail

ENV_NAME=aligner

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Install Miniforge first:"
  echo "  wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
  echo "  bash Miniforge3-Linux-x86_64.sh"
  exit 1
fi

# MFA must come from conda-forge (it ships the compiled Kaldi + OpenFst binaries; pip alone is not enough)
conda create -y -n "$ENV_NAME" -c conda-forge montreal-forced-aligner pandas openpyxl

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

mfa version
python "$(dirname "$0")/zerosyl_align.py" setup

echo
echo "Done. Before each use run:  conda activate $ENV_NAME"
