#!/usr/bin/env bash
# Launch the NVE simulation of Exercise 2: the i-PI server and one MACE
# client, both inside this folder. Output files: nvt2.* (i-PI), ipi.log and
# mace_0.log (screen output of the two programs). Returns when the run is
# complete.
set -e
cd "$(dirname "$0")"
command -v i-pi >/dev/null || { echo "i-pi not found: activate the tutorial environment first"; exit 1; }
bash clean.sh

# i-PI and the client talk through /tmp/ipi_<address>, which is shared by
# everyone on the machine, so we give this folder its own address: on the
# school's cloud machines several students run the same exercise at the same
# time, and a fixed address would make the second one fail to start.
ADDR="mace-ex2_$(printf '%s' "$PWD" | cksum | cut -d' ' -f1)"
SOCK="/tmp/ipi_${ADDR}"
rm -f "$SOCK"                   # socket left behind by an interrupted run
trap 'rm -f "$SOCK"' EXIT       # and do not leave one ourselves
sed "s|<address>.*</address>|<address>${ADDR}</address>|" input.xml > .input_run.xml
echo "i-PI socket: ${SOCK}"

export PYTHONUNBUFFERED=1       # unbuffered output -> logs update in real time
export IPI_ADDRESS="$ADDR"      # picked up by run-ase_ex2.py
i-pi .input_run.xml > ipi.log 2>&1 &
sleep 10                        # give i-PI time to open the socket
python3 run-ase_ex2.py > mace_0.log 2>&1
wait
