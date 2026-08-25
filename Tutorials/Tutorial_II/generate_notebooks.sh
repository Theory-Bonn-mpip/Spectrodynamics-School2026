#!/usr/bin/env bash
shopt -s nullglob   # no numbered extra notebooks is fine
# Convert the tutorial scripts (sphinx-gallery / RST format) to Jupyter
# notebooks.
#
# Usage:
#   ./generate_notebooks.sh                # convert tutorial_II.py + [0-9]_*.py
#   ./generate_notebooks.sh input.py [output.ipynb]   # convert a single file

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONVERTER="${SCRIPT_DIR}/scripts/convert_to_notebook.py"

if [[ $# -gt 0 ]]; then
    INPUT="$1"
    OUTPUT="${2:-${INPUT%.py}.ipynb}"
    python3 "$CONVERTER" "$INPUT" "$OUTPUT"
else
    for f in "${SCRIPT_DIR}"/tutorial_II.py "${SCRIPT_DIR}"/[0-9]_*.py; do
        python3 "$CONVERTER" "$f" "${f%.py}.ipynb"
    done
fi
