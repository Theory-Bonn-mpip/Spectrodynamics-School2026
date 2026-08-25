#!/usr/bin/env bash
# Density and dipole-orientation profiles of the 48-water slab along z
# (all 25 000 frames = 50 ps of the provided trajectory). Takes about
# two minutes.
cd "$(dirname "$0")"
TRAJ=../trajectory_files/slab_lx10_ly10_lz100_n48_01/traj.xyz

python3 ../../scripts/analysis/slab_profiles.py -f $TRAJ -cell 10 10 100 -max 25000 -dz 0.1 -out slab
