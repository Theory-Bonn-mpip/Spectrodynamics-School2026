#!/usr/bin/env bash
# Download the MACE models used throughout the tutorial from the Keeper
# (MPDL) public share into MODELS/.
#
# Run this ONCE, before the tutorial session (~47 MB):
#   ./download_models.sh
#
# Models that are already present with the correct size are skipped, so it
# is safe to re-run.

set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${REPO_DIR}/scripts/keeper_download.sh"

fetch_folders "MODELS"

check_required download_models.sh \
    MODELS/MACE_MLIP.model \
    MODELS/MACE-MDP.model || exit 1

echo "MACE models complete."
