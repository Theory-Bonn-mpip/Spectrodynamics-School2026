# Reference data for Exercise 3 (IR spectra of bulk water)

Two columns: wavenumber in cm⁻¹ and intensity in arbitrary units
(normalize before comparing different sources). The computed spectra are
*extensive* — divide by the number of molecules before comparing different
system sizes — and all use a correlation lag of 1 ps and the windowed
cosine transform of `scripts/analysis/ir_raman.py` (see
`scripts/analysis/README.md`).

## MACE-MDP dipoles (Exercise 3b)

| file | content |
|---|---|
| `IR-mdp_n016_{050,100,500}ps_tau1.0.dat` | 16 H₂O (L = 7.822 Å) from 50, 100 and 500 ps of trajectory — convergence with the simulation time |
| `IR-mdp_n{016,032,064,128}_500ps_tau1.0.dat` | 500 ps for the four system sizes — convergence with the system size |

Provided by the author from long NVE runs on a GPU cluster; those
trajectories are not distributed.

## SPC/E point-charge dipoles (Exercise 3a)

| file | content |
|---|---|
| `IR-spce_n016_{050,100,500}ps_tau1.0.dat` | 16 H₂O from 50, 100 and 500 ps of trajectory |

Only the 16-water system: Exercise 3a no longer shows a system-size
comparison (that discussion happens once, in 3b).

## Experiment

| file | content |
|---|---|
| `IR_raw.dat` | experimental IR spectrum of liquid H₂O, 138 digitized points, plotted as delivered (the notebook only normalizes it to its maximum) |

Primary source J. E. Bertie and Z. Lan, *Appl. Spectrosc.* **50**, 1047
(1996), as displayed in Fig. 3 of O. Marsalek and T. E. Markland,
*J. Phys. Chem. Lett.* **8**, 1545 (2017),
https://doi.org/10.1021/acs.jpclett.7b00391.
