#!/usr/bin/env bash
# IR spectrum of the 16-water NVE run of Exercise 2 with MACE-MDP dipoles
# (mdp.dipole_0 written by the replay run of run_ex3.sh). The replay is initialized with
# init.xyz, which is also the first frame of the trajectory, so its record
# appears twice, hence -skip 1. The reference files in
# reference_results/ were obtained with the same command on the first 50 ps
# of the provided trajectories (-max 25000 -lag 1).
cd "$(dirname "$0")"

python3 ../../scripts/analysis/ir_raman.py -dip mdp.dipole_0 -skip 1 -dt 0.002 -lag 0.5 -max 500 -corr fft -out mdp
