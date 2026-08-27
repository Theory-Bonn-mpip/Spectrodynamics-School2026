#!/usr/bin/env bash
# Download the pre-computed trajectory data for Part III from the Keeper
# (MPDL) public share into part_iii/trajectory_files.
#
# Run this ONCE, before the tutorial session (~1 GB):
#   ./download_trajectories.sh
#
# It can be called from any working directory (and through a symlink): the
# data always lands in the repository this script belongs to.
#
# Files that are already present with the correct size are skipped, and
# interrupted downloads are resumed, so it is safe to re-run.

set -uo pipefail

# Directory of this script, whatever the caller's working directory is.
SELF="${BASH_SOURCE[0]}"
[[ -L "$SELF" ]] && SELF="$(readlink -f "$SELF" 2>/dev/null || echo "$SELF")"
REPO_DIR="$(cd "$(dirname "$SELF")" && pwd)"

source "${REPO_DIR}/scripts/keeper_download.sh"

fetch_folders "part_iii/trajectory_files"

SLAB=part_iii/trajectory_files/slab_lx10_ly10_lz100_n48_01
check_required download_trajectories.sh \
    ${SLAB}/traj.xyz \
    ${SLAB}/h2o.dipole_0 \
    ${SLAB}/h2o.polarizability_0 \
    ${SLAB}/h2o.atomic_charges_0 \
    ${SLAB}/h2o.atomic_dipoles_0 \
    ${SLAB}/h2o.atomic_polarizabilities_0 || exit 1

echo "Part III trajectory data complete."
