#!/usr/bin/env bash
# Download the pre-computed trajectory data for Part III from the Keeper
# (MPDL) public share into part_iii/trajectory_files.
#
# Run this ONCE, before the tutorial session (~1 GB):
#   ./download_trajectories.sh
#
# Files that are already present with the correct size are skipped, and
# interrupted downloads are resumed, so it is safe to re-run.

set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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
