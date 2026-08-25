#!/usr/bin/env bash
# Im chi(2)_xxz of the 48-water slab with the ssVVCF bond model, from all
# 25 000 frames (50 ps) of the provided trajectory. The surface window
# (-zref1 4 -zref2 5, measured from the slab centre) and the lag are those
# of the reference results; -rc 1.0 keeps the self (intramolecular) terms
# only; -non_condon adds the frequency-dependent transition moments as an
# extra column of ssVVCF_ImChi2.dat. Takes about eight minutes.
cd "$(dirname "$0")"
TRAJ=../trajectory_files/slab_lx10_ly10_lz100_n48_01/traj.xyz

python3 ../../scripts/analysis/ssvvcf_ml.py -f $TRAJ -cell 10 10 100 -dt 0.002 -lag 1 -max 25000 \
        -zref1 4 -zref2 5 -nmode 1 -rc 1.0 -non_condon
