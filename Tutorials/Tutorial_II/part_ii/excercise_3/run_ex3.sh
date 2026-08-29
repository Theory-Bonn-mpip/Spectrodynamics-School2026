#!/usr/bin/env bash
# Evaluate the MACE-MDP dipoles and polarizabilities along the trajectory of
# Exercise 2 with an i-PI "replay" run: i-PI re-reads the trajectory frame
# by frame and asks the client for the dipole and polarizability instead of
# energies and forces. Output files: mdp.dipole_0 and mdp.polarizability_0
# (i-PI), ipi.log and mdp_0.log (screen output). Returns when done.
set -e
cd "$(dirname "$0")"
command -v i-pi >/dev/null || { echo "i-pi not found: activate the tutorial environment first"; exit 1; }
[ -f ../excercise_2/nvt2.pos_0.extxyz ] || { echo "run Exercise 2 first: ../excercise_2/nvt2.pos_0.extxyz not found"; exit 1; }
bash clean.sh

# i-PI and the client talk through /tmp/ipi_<address>, which is shared by
# everyone on the machine, so we give this folder its own address: on the
# school's cloud machines several students run the same exercise at the same
# time, and a fixed address would make the second one fail to start.
ADDR="mdp-ex3_$(printf '%s' "$PWD" | cksum | cut -d' ' -f1)"
SOCK="/tmp/ipi_${ADDR}"
rm -f "$SOCK"                   # socket left behind by an interrupted run
trap 'rm -f "$SOCK"' EXIT       # and do not leave one ourselves
sed "s|<address>.*</address>|<address>${ADDR}</address>|" input_mdp.xml > .input_run.xml
echo "i-PI socket: ${SOCK}"

export PYTHONUNBUFFERED=1
export IPI_ADDRESS="$ADDR"      # picked up by run-mace-mdp_ex3.py
i-pi .input_run.xml > ipi.log 2>&1 &
sleep 10
python3 run-mace-mdp_ex3.py > mdp_0.log 2>&1
wait
