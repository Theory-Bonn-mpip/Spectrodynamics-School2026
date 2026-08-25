# Spectrodynamics 2026 — Tutorial II: Vibrational spectroscopy from simulations

IR, Raman, and SFG spectra of water from molecular dynamics, using
[i-PI](http://ipi-code.org) and [MACE](https://github.com/ACEsuit/mace)
machine-learning potentials.

## Setup (once, before the session)

```bash
./setup.sh
```

Creates the conda environment `Tutorial_II`, generates the notebooks, and
downloads the MACE models (~47 MB) and the Part III trajectory data
(~1 GB). Safe to re-run. See `INSTALLING.rst` for the individual steps,
or `setup_for_humans.sh` for the same steps written out and explained.

## Run

```bash
conda activate Tutorial_II
jupyter notebook tutorial_II.ipynb
```

Questions: Yair Litman <litmany@mpip-mainz.mpg.de>
