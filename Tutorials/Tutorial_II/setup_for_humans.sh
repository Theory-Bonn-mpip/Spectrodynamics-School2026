#!/usr/bin/env bash
# ============================================================================
#  What ./setup.sh does, spelled out in plain English.
# ============================================================================
#
#  setup.sh does four things for you in one go. This file does exactly the
#  same, but with every command written out and explained, so you can read
#  what is going to happen -- or copy the commands one by one into your own
#  terminal instead of running this script.
#
#  You can simply run it:
#      bash setup_for_humans.sh
#
#  It is safe to run twice: nothing is overwritten and files that are
#  already downloaded are not downloaded again.
#
#  Total: about 1 GB of downloads and, the first time, several minutes of
#  installing packages.
# ============================================================================

# Work in the folder where this script lives, so that all the paths below
# ("environment.yml", "tutorial_II.py", ...) are found.
cd "$(dirname "$0")"


# ----------------------------------------------------------------------------
# STEP 1 of 4 -- Install the software
# ----------------------------------------------------------------------------
# Everything the tutorial needs (i-PI, MACE, ASE, jupyter, matplotlib, ...) is
# listed in the file environment.yml. The command below reads that list and
# installs all of it into a new, separate conda environment called
# "Tutorial_II", so nothing on your computer is touched. This is the slow
# step: it downloads and compiles packages and can take several minutes.
#
# You need conda (Miniconda or Anaconda) for this. If the "conda" command is
# not found, install Miniconda first: https://docs.conda.io
#
# If you have already created the environment before, conda would stop with
# an error, so we simply skip the step in that case.

echo "=== STEP 1 of 4: installing the software into the conda environment 'Tutorial_II' ==="

if conda env list | grep -q '^Tutorial_II '; then
    echo "The environment 'Tutorial_II' already exists - skipping this step."
else
    conda env create -n Tutorial_II -f environment.yml
fi


# ----------------------------------------------------------------------------
# STEP 2 of 4 -- Turn the tutorial text into Jupyter notebooks
# ----------------------------------------------------------------------------
# The tutorial is written as ordinary Python files (tutorial_II.py for Parts I
# and II, 3_vsfg.py for the optional Part III). The small converter below
# turns each of them into the .ipynb notebook you will actually open. This
# takes a second, and it is what ./generate_notebooks.sh does.
#
# If you ever mess up a notebook, just run these two lines again: they
# rebuild it from the .py file.

echo
echo "=== STEP 2 of 4: creating the notebooks ==="

python3 scripts/convert_to_notebook.py tutorial_II.py tutorial_II.ipynb
python3 scripts/convert_to_notebook.py 3_vsfg.py     3_vsfg.ipynb


# ----------------------------------------------------------------------------
# STEP 3 of 4 -- Download the two MACE models (about 47 MB)
# ----------------------------------------------------------------------------
# The tutorial uses two machine-learned models, which are too big to keep in
# the repository:
#   MODELS/MACE_MLIP.model  - gives the forces that drive the simulations
#   MODELS/MACE-MDP.model   - gives dipoles and polarizabilities (the spectra)
#
# The command below fetches both from our Keeper share into the MODELS
# folder. Files that are already there are left alone.

echo
echo "=== STEP 3 of 4: downloading the MACE models (~47 MB) ==="

./download_models.sh


# ----------------------------------------------------------------------------
# STEP 4 of 4 -- Download the simulation data for Part III (about 1 GB)
# ----------------------------------------------------------------------------
# Part III analyses a 50 ps simulation of a water surface that was run for
# you on a computer cluster (running it yourself would take days). The
# command below downloads that trajectory, and the machine-learned dipoles
# and polarizabilities that go with it, into part_iii/trajectory_files.
#
# This is the big one, roughly 1 GB, so it needs a few minutes on a good
# connection. If it is interrupted, just run it again: it continues where it
# stopped and skips whatever is already complete.
#
# Parts I and II do NOT need this data. If you are short of time or disk
# space and only plan to do Parts I and II, you can skip this step.

echo
echo "=== STEP 4 of 4: downloading the Part III simulation data (~1 GB) ==="

./download_trajectories.sh


# ----------------------------------------------------------------------------
# All done -- how to start the tutorial
# ----------------------------------------------------------------------------
# Two commands, every time you want to work on the tutorial. The first one
# switches on the software installed in step 1; the second opens the
# notebook in your browser.

echo
echo "Setup finished. To start the tutorial, type these two commands:"
echo
echo "    conda activate Tutorial_II"
echo "    jupyter-lab tutorial_II.ipynb"
echo
echo "Parts I and II are in tutorial_II.ipynb."
echo "The optional Part III is the separate notebook 3_vsfg.ipynb."
