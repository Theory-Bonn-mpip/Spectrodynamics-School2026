#!/usr/bin/env bash
# Master setup for the Spectrodynamics 2026 Tutorial II.
#
#   ./setup.sh
#
# does everything needed before the session:
#   1. creates the conda environment "Tutorial_II" from environment.yml
#   2. converts the tutorial script(s) to Jupyter notebooks
#   3. downloads the MACE models (~47 MB) and the Part III data (~1 GB)
#
# Safe to re-run: an existing environment is kept, notebooks are simply
# regenerated, and already-downloaded files are skipped.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
ENV_NAME=Tutorial_II

echo "=== 1/3  Conda environment '${ENV_NAME}' ==="
if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found. Install Miniconda/Anaconda first:"
    echo "       https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    # An env can be left half-built if a previous setup.sh was interrupted —
    # accept it only if the key packages actually import.
    if conda run -n "${ENV_NAME}" python -c "import ase, ipi, mace, chemiscope" >/dev/null 2>&1; then
        echo "Environment '${ENV_NAME}' already exists and works — keeping it."
    else
        echo "ERROR: environment '${ENV_NAME}' exists but is incomplete"
        echo "       (probably from an interrupted setup). Remove it and re-run:"
        echo "           conda env remove -n ${ENV_NAME}"
        echo "           ./setup.sh"
        exit 1
    fi
else
    conda env create -n "${ENV_NAME}" -f environment.yml
fi

echo
echo "=== 2/3  Generating the Jupyter notebooks ==="
bash generate_notebooks.sh

echo
echo "=== 3/3  Downloading the MACE models and the Part III data (~1 GB) ==="
bash download_all.sh

echo
echo "Setup complete. To start the tutorial:"
echo "    conda activate ${ENV_NAME}"
echo "    jupyter-lab tutorial_II.ipynb"
echo
echo "(Parts I and II are in tutorial_II.ipynb; the optional Part III on"
echo " sum-frequency generation is the separate notebook 3_vsfg.ipynb.)"
