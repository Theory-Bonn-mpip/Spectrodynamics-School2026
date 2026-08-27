#!/usr/bin/env bash
# Download the MACE models used throughout the tutorial from the Keeper
# (MPDL) public share into MODELS/.
#
# Run this ONCE, before the tutorial session (~47 MB):
#   ./download_models.sh
#
# It can be called from any working directory (and through a symlink): the
# models always land in the MODELS/ folder of the repository this script
# belongs to.
#
# Models that are already present with the correct size are skipped, so it
# is safe to re-run.

set -uo pipefail

# Directory of this script, whatever the caller's working directory is.
SELF="${BASH_SOURCE[0]}"
[[ -L "$SELF" ]] && SELF="$(readlink -f "$SELF" 2>/dev/null || echo "$SELF")"
REPO_DIR="$(cd "$(dirname "$SELF")" && pwd)"

source "${REPO_DIR}/scripts/keeper_download.sh"

fetch_folders "MODELS"

check_required download_models.sh \
    MODELS/MACE_MLIP.model \
    MODELS/MACE-MDP.model || exit 1

echo "MACE models complete."
