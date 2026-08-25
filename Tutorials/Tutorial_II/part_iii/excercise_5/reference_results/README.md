# Reference results for Exercise 5

Converged results for the 48-water slab (10 × 10 × 100 Å cell, frames
every 2 fs), averaged over 1 ns of dynamics — twenty times the 50 ps
trajectory analysed in the exercise. Computed with `slab_profiles.py` /
`ssvvcf_ml.py` with the parameters of `../get_profiles.sh` /
`../get_ssvvcf.sh` (`-dz 0.1`; `-dt 0.002 -lag 1 -zref1 4 -zref2 5
-nmode 1 -non_condon`).

| file | content |
|---|---|
| `slab-048_1ns_profiles.dat` | density (O, H; Å⁻³) and dipole-orientation profiles |
| `slab-048_1ns_ssVVCF_ImChi2_rc1.0.dat` | Im χ⁽²⁾_xxz, ssVVCF bond model, `-rc 1.0` (self terms); column 3 = non-Condon corrected |
| `slab-048_1ns_ssVVCF_ImChi2_rc4.0.dat` | the same with `-rc 4.0` (cross terms up to the first solvation shell); plotted in 6c |
