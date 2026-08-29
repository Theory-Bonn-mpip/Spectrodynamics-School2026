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

# i-PI and the client talk through /tmp/ipi_<address>, which is shared by
# everyone on the machine, so we give this folder its own address: on the
# school's cloud machines several students run the same exercise at the same
# time, and a fixed address would make the second one fail to start.
ADDR="mdp-ex6_$(printf '%s' "$PWD" | cksum | cut -d' ' -f1)"
SOCK="/tmp/ipi_${ADDR}"
rm -f "$SOCK"                     # socket left behind by an interrupted run
trap 'rm -f "$SOCK"' EXIT         # and do not leave one ourselves
sed "s|<address>.*</address>|<address>${ADDR}</address>|" input_mdp.xml > .input_run.xml
echo "i-PI socket: ${SOCK}"

export PYTHONUNBUFFERED=1
export IPI_ADDRESS="$ADDR"        # picked up by run-mace-mdp_ex6.py
i-pi .input_run.xml > ipi.log 2>&1 &
sleep 10
python3 run-mace-mdp_ex6.py > mdp_0.log 2>&1
wait
