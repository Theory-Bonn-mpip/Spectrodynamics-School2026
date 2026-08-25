# Trajectory files for Part III (distributed separately, not in git)

`slab_lx10_ly10_lz100_n48_01/`: water/air interface, 48 H2O in a
10 × 10 × 100 Å cell (≈ 15 Å of water, vacuum on both sides), NVE with the
MACE potential of Part I, 50 ps = 25 001 frames every 2 fs:

- `traj.xyz` — positions (extxyz, whole along z, x/y unwrapped)
- `h2o.atomic_charges_0`, `h2o.atomic_dipoles_0`, `h2o.atomic_polarizabilities_0`
  — MACE-MDP per-atom charges (e), dipoles (e·bohr) and polarizabilities
  (bohr³), one record per frame (step 1 … 25001 ↔ frame 0 … 25000)
- `h2o.dipole_0`, `h2o.polarizability_0` — the corresponding totals
- `input.xml`, `run-mace-mdp.py` — the replay run that produced them
  (full 50 ps on a GPU cluster)

Fetch the data with `./download_trajectories.sh` from the tutorial root.
