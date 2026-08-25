#!/usr/bin/env bash
# IR spectrum of the 16-water NVE run of Exercise 2 with SPC/E point charges.
# Step 1 builds the total dipole from the positions (one line per frame),
# step 2 computes its derivative autocorrelation function and the spectrum.
# The reference files in reference_results/ were obtained with the same
# commands on the first 50 ps of the provided trajectories
# (-max 25000 -lag 1).
cd "$(dirname "$0")"

python3 ../../scripts/analysis/spce_dipole.py -f ../excercise_2/nvt2.pos_0.extxyz -max 500 -out dipole_spce.dat
python3 ../../scripts/analysis/ir_raman.py -dip dipole_spce.dat -dt 0.002 -lag 0.5 -max 500 -corr fft -out spce
