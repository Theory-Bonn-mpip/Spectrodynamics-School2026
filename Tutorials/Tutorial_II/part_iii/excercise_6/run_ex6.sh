#!/usr/bin/env bash
# Evaluate the MACE-MDP atomic charges, dipoles and polarizabilities along
# the first 2000 frames of the slab trajectory with an i-PI "replay" run
# (same idea as run_ex3.sh of Exercise 3). Output files: mdp.dipole_0,
# mdp.polarizability_0, mdp.atomic_charges_0, mdp.atomic_dipoles_0,
# mdp.atomic_polarizabilities_0 (i-PI), ipi.log and mdp_0.log (screen
# output). Returns when done.
set -e
cd "$(dirname "$0")"
TRAJ=../trajectory_files/slab_lx10_ly10_lz100_n48_01/traj.xyz
command -v i-pi >/dev/null || { echo "i-pi not found: activate the tutorial environment first"; exit 1; }
[ -f $TRAJ ] || { echo "slab trajectory not found: $TRAJ"; exit 1; }
bash clean.sh
rm -f /tmp/ipi_mdp-ex6            # stale socket of an interrupted run
export PYTHONUNBUFFERED=1
i-pi input_mdp.xml > ipi.log 2>&1 &
sleep 10
python3 run-mace-mdp_ex6.py > mdp_0.log 2>&1
wait
