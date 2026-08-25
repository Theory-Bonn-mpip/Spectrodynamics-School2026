# Reference data for Exercise 4 (Raman spectra of bulk water)

Two columns: wavenumber in cm⁻¹ and intensity in arbitrary units
(normalize before comparing different sources); lines starting with `#`
are comments.

## Computed in this tutorial

| file | content |
|---|---|
| `Raman-iso_n128_500ps_tau1.0.dat`, `Raman-aniso_n128_500ps_tau1.0.dat` | isotropic / anisotropic Raman spectra of 128 H₂O from 500 ps of trajectory, Eqs. (4)–(5) of the tutorial (no frequency prefactors, see Appendix A.5) |

MACE-MDP polarizabilities, correlation lag 1 ps, windowed cosine transform
(`ir_raman.py -pol … -pol_type both`), from a long NVE run on a GPU cluster
that is not distributed. The isotropic component converges slowly, hence
the long trajectory and the large box.

## Experiment

| file | content |
|---|---|
| `experiment_Raman-iso.dat`, `experiment_Raman-aniso.dat` | experimental isotropic / anisotropic Raman spectra over the full range, normalized to max = 1 |
| `{iso,aniso}_{low,high}_freq.dat` | the raw digitized pieces they are built from |

Ben-Amotz group (L. Streacker and D. Ben-Amotz), as displayed in Fig. 4 of
O. Marsalek and T. E. Markland, *J. Phys. Chem. Lett.* **8**, 1545 (2017),
https://doi.org/10.1021/acs.jpclett.7b00391. They are reported in the same
convention as the computed spectra above, so Exercise 4b compares the two
directly, each normalized to its maximum, with no frequency-dependent
prefactor applied to either.

The two `experiment_Raman-*.dat` files are built from the four raw pieces:
the low-frequency pieces are digitized from the magnified sub-panels of the
figure and are divided by 50 (isotropic) and 10 (anisotropic) before being
joined to the high-frequency pieces at ≈ 2350 cm⁻¹; the joined points are
sorted, near-duplicates (< 2 cm⁻¹ apart) averaged, PCHIP-interpolated to
1000 points, clipped at 0 and normalized to max = 1.
