# Analysis scripts

Self-contained Python (numpy only) tools that go from a trajectory or
property time series to a vibrational spectrum. They are written for
**clarity over efficiency**: everything is loaded into memory and the
correlation functions are accumulated with explicit loops that read like
the textbook formulas (Appendix A.2 of the tutorial).

| script | input | output |
|---|---|---|
| `vdos.py` | xyz trajectory (positions or velocities) | velocity ACF (total + per element) and VDOS |
| `spce_dipole.py` | xyz trajectory of water | total dipole from SPC/E point charges (input for `ir_raman.py -dip`) |
| `ir_raman.py` | total dipole and/or polarizability time series | IR and Raman (iso/aniso) correlation functions and spectra |
| `slab_profiles.py` | xyz trajectory of a water slab | density and dipole-orientation profiles along z |
| `ssvvcf_ml.py` | xyz trajectory of a water slab | SFG Im χ⁽²⁾ from the ssVVCF bond model (`-non_condon` adds the frequency-dependent transition moments) |
| `sfg_atomic.py` | slab trajectory + per-atom charges, dipoles, polarizabilities | SFG Im χ⁽²⁾ from ML atomic multipoles |

Run any script with `-h` for the options, and read the docstring at the
top of each file for what exactly is computed, the input formats and the
output files.

## Conventions shared by all scripts

- `-dt` and `-lag` are given in **ps**; positions in Å; orthorhombic cells
  only. Property files are plain ascii (lines starting with `#` are
  skipped, so raw i-PI extras files can be read directly; i-PI *replay*
  runs write step 0 twice — use `-skip 1` in `ir_raman.py`/`sfg_atomic.py` for their extras).
- Velocities are backward finite differences of consecutive frames,
  minimum-imaged across periodic boundaries.
- Every frame serves once as a time origin; `lag/dt` lags are resolved.
- Spectra are **cosine transforms** of the (windowed) correlation function
  (trapezoidal rule: the t = 0 term has half weight; time in **fs**, so a
  spectrum has the units of the correlation function times fs and O(1)
  magnitudes), evaluated with a zero-padded real FFT by default; `-ft direct` gives the
  explicit textbook sum (same numbers, slow). `vdos.py` and `ir_raman.py`
  also accept `-corr fft` to evaluate the correlation function itself via
  the Wiener–Khinchin theorem instead of the explicit double loop.
- IR/Raman/VDOS correlate time **derivatives** (Eq. 2 of the tutorial);
  the SFG scripts correlate μ̇ and α̇ and divide the cosine transform by ω,
  with an overall sign `+1` in both routes — only the *relative* sign of
  the spectral features is physical.

## Examples

```bash
python3 vdos.py -f pos.xyz -dt 0.0005 -lag 0.5 -max 2000 -cell 9.856 9.856 9.856
python3 ir_raman.py -dip dipole.dat -pol polarizability.dat -dt 0.0005 -lag 0.5 -max 10000 -pol_type both
python3 slab_profiles.py -f slab.xyz -cell 10 10 100 -max 25000 -dz 0.2
python3 ssvvcf_ml.py -f slab.xyz -cell 10 10 100 -dt 0.002 -lag 1.0 -max 25000 -zref1 4 -zref2 8 -non_condon
python3 sfg_atomic.py -f slab.xyz -atq q.dat -atmu mu.dat -atpol alpha.dat -cell 10 10 100 -dt 0.002 -lag 1.0 -max 25000 -zref1 4 -zref2 8 -chi xxz
```
