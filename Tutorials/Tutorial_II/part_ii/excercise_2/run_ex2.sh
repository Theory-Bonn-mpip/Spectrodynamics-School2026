#!/usr/bin/env bash
# Launch the NVE simulation of Exercise 2: the i-PI server and one MACE
# client, both inside this folder. Output files: nvt2.* (i-PI), ipi.log and
# mace_0.log (screen output of the two programs). Returns when the run is
# complete.
set -e
cd "$(dirname "$0")"
command -v i-pi >/dev/null || { echo "i-pi not found: activate the tutorial environment first"; exit 1; }
bash clean.sh
export PYTHONUNBUFFERED=1       # unbuffered output -> logs update in real time
i-pi input.xml > ipi.log 2>&1 &
sleep 10                        # give i-PI time to open the socket
python3 run-ase_ex2.py > mace_0.log 2>&1
wait
