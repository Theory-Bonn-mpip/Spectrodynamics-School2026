#!/usr/bin/env bash
# Download everything the tutorial needs from the Keeper (MPDL) public
# share: the MACE models (~47 MB) and the Part III trajectory data (~1 GB).
#
#   ./download_all.sh
#
# It can be called from any working directory (and through a symlink): the
# data always lands in the repository this script belongs to.
#
# Both steps skip files that are already present with the correct size, so
# this is safe to re-run. They can also be run separately:
#   ./download_models.sh
#   ./download_trajectories.sh

set -uo pipefail

# Directory of this script, whatever the caller's working directory is.
SELF="${BASH_SOURCE[0]}"
[[ -L "$SELF" ]] && SELF="$(readlink -f "$SELF" 2>/dev/null || echo "$SELF")"
REPO_DIR="$(cd "$(dirname "$SELF")" && pwd)"

echo "=== 1/2  MACE models (~47 MB) ==="
bash "${REPO_DIR}/download_models.sh" || exit 1

echo
echo "=== 2/2  Part III trajectory data (~1 GB) ==="
bash "${REPO_DIR}/download_trajectories.sh" || exit 1
