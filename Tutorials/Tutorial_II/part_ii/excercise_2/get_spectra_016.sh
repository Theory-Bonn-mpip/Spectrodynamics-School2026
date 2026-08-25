#!/usr/bin/env bash
# VDOS of the 16-water NVE trajectory produced by run_ex2.sh.
# Positions are written every 4 steps of 0.5 fs -> 0.002 ps between frames;
# 2000 steps give 500 frames (1 ps), so the correlation lag is limited to
# 0.5 ps. The converged reference spectra in reference_results/ were
# obtained with the same command on much longer trajectories (up to 500 ps),
# changing only -max and -lag; see the README of that folder.
cd "$(dirname "$0")"
l1=7.822
l2=7.822
l3=7.822

python3 ../../scripts/analysis/vdos.py -f nvt2.pos_0.extxyz -dt 0.002 -lag 0.5 -max 500 -cell ${l1} ${l2} ${l3} -corr fft
