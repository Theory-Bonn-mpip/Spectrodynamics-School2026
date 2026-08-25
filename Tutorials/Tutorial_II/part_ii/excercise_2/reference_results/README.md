# Reference data for Exercise 2 (VDOS of bulk water)

All files were produced with `scripts/analysis/vdos.py` (`-dt 0.002
-corr fft`, half-Hann window, the default) from NVE trajectories of bulk
water at ~300 K with the MACE MLIP, frames written every 2 fs.

Columns of the spectra: **wavenumber (cm⁻¹), total, O, H**; all of them share
the same frequency grid (Δν = 4.07 cm⁻¹). The spectra are *extensive*
(a sum over all atoms), so divide by the number of molecules before
comparing different system sizes.

## Spectra

| file | content |
|---|---|
| `Cvv_spectrum_n016_{050,100,500}ps_tau1.0.dat` | 16 H₂O (L = 7.822 Å), lag 1 ps, from 50, 100 and 500 ps of trajectory — convergence with the simulation time (2c) |
| `Cvv_spectrum_n016_500ps_tau{0.1,0.5,1.0,2.0}.dat` | 16 H₂O, 500 ps, correlation lags of 0.1, 0.5, 1 and 2 ps — convergence with the correlation length (2d) |
| `Cvv_spectrum_n{016,032,064,128}_500ps_tau1.0.dat` | 500 ps, lag 1 ps, for 16, 32, 64 and 128 H₂O — convergence with the system size (2e) |

`Cvv_spectrum_n016_500ps_tau1.0.dat` is the same file in all three sets.
The long (≥ 50 ps) trajectories behind these spectra were run on a GPU
cluster and are not distributed — they are much too large.

## Correlation functions

| file | content |
|---|---|
| `Cvv_n016_500ps_tau{0.1,0.5,1.0,2.0}.dat` | velocity autocorrelation function of the 16-water system from the same 500 ps trajectory, truncated at lags of 0.1, 0.5, 1 and 2 ps: time (fs), total, O, H (plotted in 2d) |

These are the *unwindowed* correlation functions, i.e. the same function
cut at four different lengths; `vdos.py` applies the half-Hann window
internally before transforming.
