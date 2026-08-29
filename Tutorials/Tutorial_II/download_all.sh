#!/usr/bin/env bash
# Download everything the tutorial needs from the Keeper (MPDL) public
# share: the MACE models (~47 MB) and the Part III trajectory data (~1 GB).
#
#   ./download_all.sh
#
# Both steps skip files that are already present with the correct size, so
# this is safe to re-run. They can also be run separately:
#   ./download_models.sh
#   ./download_trajectories.sh

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

echo "=== 1/2  MACE models (~47 MB) ==="
bash download_models.sh || exit 1

echo
echo "=== 2/2  Part III trajectory data (~1 GB) ==="
bash download_trajectories.sh || exit 1
