# Reference results for Exercise 6

Converged result for the 48-water slab (10 × 10 × 100 Å cell, frames every
2 fs), averaged over 1 ns of dynamics — twenty times the 50 ps trajectory
analysed in the exercise. Computed with `sfg_atomic.py` from the MACE-MDP
atomic charges, dipoles and polarizabilities, with the parameters of
`../get_sfg_atomic.sh` (`-dt 0.002 -lag 1 -zref1 4 -zref2 5 -nmode 1
-rcut 4.0 -chi xxz`).

| file | content |
|---|---|
| `slab-048_1ns_SFG_ImChi2_xxz_rc4.0.dat` | Im χ⁽²⁾_xxz from the atomic decomposition |
