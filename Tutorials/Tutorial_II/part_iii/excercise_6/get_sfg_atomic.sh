#!/usr/bin/env bash
# Im chi(2)_xxz of the slab from the MACE-MDP atomic charges, dipoles and
# polarizabilities provided with the trajectory (all 25 000 frames =
# 50 ps; one record per frame, so no -skip is needed). The surface window
# (-zref1 4 -zref2 5, measured from the slab centre) and the lag are those
# of the reference results; -rcut 4.0 includes the intra- and
# intermolecular cross terms up to the first solvation shell. Takes about
# ten minutes.
cd "$(dirname "$0")"
DATA=../trajectory_files/slab_lx10_ly10_lz100_n48_01
TRAJ=$DATA/traj.xyz

python3 ../../scripts/analysis/sfg_atomic.py -f $TRAJ \
        -atq $DATA/h2o.atomic_charges_0 -atmu $DATA/h2o.atomic_dipoles_0 -atpol $DATA/h2o.atomic_polarizabilities_0 \
        -cell 10 10 100 -dt 0.002 -lag 1 -max 25000 \
        -zref1 4 -zref2 5 -nmode 1 -rcut 4.0 -chi xxz -prefix mdp_
