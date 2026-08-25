#!/usr/bin/env bash
# Raman spectra of the 16-water NVE run of Exercise 2 with MACE-MDP
# polarizabilities. The polarizabilities were written by the replay run of
# Exercise 3 (mdp.polarizability_0), so we only copy the file here and run
# the analysis; -pol_type both gives the isotropic and anisotropic spectra.
# Replay runs write the first step twice, hence -skip 1. The reference files
# in reference_results/ were obtained with the same command on the first
# 50 ps of the reference trajectories (-max 25000 -lag 1).
cd "$(dirname "$0")"

cp ../excercise_3/mdp.polarizability_0 .
python3 ../../scripts/analysis/ir_raman.py -pol mdp.polarizability_0 -skip 1 -pol_type both -dt 0.002 -lag 0.5 -max 500 -corr fft -out mdp
