"""

Vibrational spectroscopy of water from machine-learned molecular dynamics
*************************************************************************

:Authors: Yair Litman `@litman90 <https://github.com/litman90/>`_

This tutorial was prepared for the *Spectrodynamics 2026* summer school. It
shows how to compute vibrational spectra (IR, Raman, and sum-frequency
generation) of liquid water and the water/air interface from molecular
dynamics simulations, using `i-PI <http://ipi-code.org>`_ for the nuclear
dynamics and `MACE <https://github.com/ACEsuit/mace>`_ machine-learning
models for the interatomic potential and, later, for the dipoles and
polarizabilities.

**How to use this tutorial.** Participants come from diverse backgrounds,
so the main text is kept as light as possible and contains everything you
need to follow the exercises. Deeper derivations and technical details
are collected in the Appendix at the end, and the answers to the questions
posed along the way in a separate section after it. Feel free to skip the
Appendix on a first pass and come back to it later, or to dive in right
away -- the idea is that everyone can go through the tutorial at their own
pace and complete it within the allocated session.

This notebook is organized in two parts:

- **Part I -- Introduction**

  - **Exercise 1**: Introduction to ``i-PI`` and ``MACE``: running NVT
    molecular dynamics of bulk water.


- **Part II -- IR and Raman spectroscopy of bulk water**

  - **Exercise 2**: Vibrational density of states.
  - **Exercise 3**: IR spectra, with increasingly accurate approximations
    for the dipole moment.
  - **Exercise 4**: Raman spectra.

A third part, **optional**, lives in its own notebook, ``3_vsfg.ipynb``:

- **Part III -- Vibrational sum-frequency generation (VSFG) spectroscopy
  of the water/air interface**

  - **Exercise 5**: VSFG spectra from surface-specific velocity
    correlation functions.
  - **Exercise 6**: VSFG spectra from machine-learned atomic dipoles and
    polarizabilities.

It builds on Parts I and II, so go through them first; it needs the
interface trajectory downloaded by ``./download_trajectories.sh``.

.. warning::

   **Run the code cell below first!** It imports the packages used by every
   other cell of this notebook. If you later see an error such as
   ``NameError: name 'np' is not defined``, this cell was skipped -- go
   back and execute it.
"""
import os
import re
import subprocess
import time

import chemiscope
import ipi
from ase.io import read
import matplotlib.pyplot as plt
import numpy as np


# %%
# Part I: Introduction
# ********************
#
# In this first part we set up and run a classical NVT molecular dynamics
# simulation of bulk liquid water (32 molecules in a periodic cubic box) with
# ``i-PI`` driving the nuclei and a MACE machine-learned potential providing
# the forces. Along the way we introduce the two codes, the structure of an
# i-PI input file, and the basic checks one should always perform on an MD
# run. This is the kind of simulation from which the trajectories analysed
# in Parts II and III are generated.

# %%
# Exercise 1: Introduction to i-PI and MACE
# =========================================

# %%
# 1a) i-PI: a molecular dynamics engine
# -------------------------------------
#
# Most atomistic simulations rest on the Born--Oppenheimer approximation,
# which exploits the large mass difference between electrons and nuclei to
# decouple their degrees of freedom: for each nuclear configuration the
# electrons are assumed to remain in their ground state, defining a potential
# energy surface on which the nuclei move. This separation splits the
# simulation into two independent tasks -- evaluating energies and forces for
# a given configuration, and using those forces to evolve the nuclei.
#
# `i-PI <http://ipi-code.org>`_ exploits this separation and implements it as
# a *client--server* paradigm. i-PI is a *force engine*: acting as the
# server, it is in command of the evolution of the nuclear positions, while
# one or more instances of a client code handle the evaluation of energies,
# forces, and possibly other properties for individual configurations.
# Designed to be universal, i-PI's architecture is agnostic to the identity
# of the force providers, with a general and flexible backend that
# accommodates a wide range of client codes.
#
# In this tutorial the client is a MACE machine-learning potential, but the
# same input would run with electronic structure codes such as FHI-aims,
# CP2K, Quantum ESPRESSO or DFTB+, or with force-field engines such as
# LAMMPS, either directly or through the ASE interface.
#
# .. figure:: images/ipi_scheme.png
#    :align: center
#    :width: 90%
#
#    Figure 1: The i-PI client--server architecture: i-PI evolves the nuclei and sends
#    positions and cell to the client, which returns energies, forces,
#    stresses and, optionally, extra properties.
#
# This modularity has practical advantages that we will exploit throughout
# the tutorial:
#
# - *One implementation, many force providers.* An advanced sampling method
#   implemented once in i-PI (classical MD, path integral MD, replica
#   exchange, geometry optimization, ...) is immediately available to every
#   client code, and different clients can be compared within an identical
#   simulation setup.
# - *Reusable clients.* The same MACE client can be reused, unchanged, for
#   any other kind of simulation i-PI offers -- geometry optimization,
#   path integral molecular dynamics, ... -- because the exchanged data
#   (positions in, forces out) do not change.
# - *Composable forces.* Several clients can be combined, e.g. to sum force
#   components computed at different levels of theory, or to parallelize the
#   evaluation over multiple client instances connected to the same server.
#
# Communication
# ~~~~~~~~~~~~~
#
# Communication between i-PI and the clients happens through sockets, either
# UNIX-domain sockets (fast, on a single machine) or INET/TCP-IP sockets
# (across nodes, or even different HPC facilities). At each MD step i-PI
# sends the nuclear positions and cell to the connected clients, receives
# energies, forces and stresses back, and uses them to propagate the nuclei.
# Only this minimal information is exchanged, which is what keeps the two
# sides completely independent programs: each has its own input file, the
# client must simply be initialized with a consistent set of atoms, and you
# can kill and restart a client without corrupting the dynamics.


# %%
# 1b) MACE: machine-learned interatomic potentials
# ------------------------------------------------
#
# `MACE <https://github.com/ACEsuit/mace>`_ is a machine-learning
# architecture for predicting many-body atomic interactions (Batatia et al.,
# *NeurIPS* 2022). It belongs to the family of *equivariant message-passing*
# neural networks: each atom is described by features that transform
# consistently under rotations, and these features are refined in a few
# rounds of "messages" exchanged between neighbouring atoms. The distinctive
# ingredient of MACE is that each message is built from *higher-order*
# (many-body) combinations of the neighbour features, which makes the model
# both accurate and fast, typically requiring only two message-passing
# layers. Like most machine-learned interatomic potentials (MLIPs), MACE
# writes the total energy as a sum of atomic contributions that depend on
# the local environment within a cutoff radius; forces are obtained as exact
# analytical gradients of the energy, so energy conservation is built in.
#
# In this tutorial we use two pre-trained MACE models, both downloaded
# into the ``MODELS/`` folder by ``./download_models.sh`` (or ``setup.sh``):
#
# - ``MACE_MLIP.model``: an interatomic potential for water trained on
#   revPBE-D3(0) reference data computed with FHI-aims. It provides the
#   energies and forces that drive the molecular dynamics.
# - ``MACE-MDP.model``: a general dipole and polarizability model built on
#   the same equivariant architecture (Gönnheimer et al., 2026). It does not
#   predict energies or forces; instead it returns atomic charges, dipoles
#   and polarizabilities, which we will use in Exercises 3--4 to compute IR
#   and Raman spectra, and again in the optional Part III for the
#   surface-specific SFG response.
#
# .. note::
#
#    Training MLIPs is outside the scope of this tutorial. Interested readers
#    can find documentation, tutorials and foundation models on the
#    `MACE documentation pages <https://mace-docs.readthedocs.io>`_ and the
#    `MACE GitHub repository <https://github.com/ACEsuit/mace>`_.


# %%
# 1c) Anatomy of an i-PI input file
# ---------------------------------
#
# An i-PI calculation is specified by a single XML file. XML is a simple
# hierarchical format, where every field has the form
# ``<tag_name attribute="value"> data </tag_name>`` and fields can be nested.
# The i-PI input is built from a handful of top-level blocks:
#
# - ``<output>``: which properties and trajectories to write, and how often.
# - ``<ffsocket>`` (or other ``<ff...>`` blocks): the *force field*, i.e.
#   the connection to the client that computes energies and forces.
# - ``<system>``: the physical system and what to do with it. Inside this
#   block we have:
#
#   - ``<initialize>``: the starting structure and velocities;
#   - ``<forces>``: which force fields act on the system;
#   - ``<ensemble>``: the thermodynamic state (temperature, pressure);
#   - ``<motion>``: the type of calculation (dynamics, geometry
#     optimization, ...) and its parameters.
#
# Internally i-PI works in atomic units, but any input or output field can
# be tagged with explicit units, e.g. ``units='femtosecond'`` for an input
# value or ``time{picosecond}`` for an output property; values are converted
# automatically.
#
# The folder structure of this tutorial mirrors its organization: each part
# has its own folder (``part_i``, ``part_ii``, ``part_iii``) containing one
# sub-folder per exercise with the i-PI input, the client script and the
# starting structure; all output files are written there. Two of them also
# have a ``trajectory_files`` folder with pre-computed data: ``part_i``,
# the longer reference run we compare with in 1f, and ``part_iii``, the
# trajectory of the water/air interface used by the optional Part III
# notebook. Let's have a look at the input file we will use:

workdir = "part_i/excercise_1"

# Open and read the XML file
with open(f"{workdir}/input.xml", "r") as file:
    xml_content = file.read()
print(xml_content)

# %%
# Going through the file from top to bottom:
#
# - ``<output prefix='simulation'>``: every output file will be named
#   ``simulation.*``. The ``<properties>`` line requests a table with the
#   step number, the time, the conserved quantity, the temperature and the
#   potential energy at every step (``stride='1'``); the ``<trajectory>``
#   line writes the positions every 2 steps (i.e. every 1 fs) in
#   extended-xyz format (``format='ase'``); and ``<checkpoint>``
#   periodically saves a restart file.
# - ``<total_steps>``: the length of the run, here only 300 steps.
# - ``<ffsocket name='mace' mode='unix'>``: i-PI will open a UNIX-domain
#   socket with the address ``h2o-mace_ex1`` and wait for clients to
#   connect. The client must use exactly the same address and mode.
# - ``<initialize nbeads='1'>``: a single replica of the system, i.e.
#   classical MD (path integral simulations would set ``nbeads > 1``).
#   The structure is read from ``init.xyz`` -- 16 water molecules
#   (48 atoms) in a cubic box of 7.82 Å, the same system we will use in
#   Part II -- and the initial velocities are drawn from a
#   Maxwell--Boltzmann distribution at 300 K.
# - ``<forces>``: the forces on the system come from the ``mace`` force
#   field defined above.
# - ``<ensemble>``: the target temperature, 300 K.
# - ``<motion mode='dynamics'>`` / ``<dynamics mode='nvt'>``: constant-volume,
#   constant-temperature dynamics with a time step of 0.5 fs, using a
#   Langevin thermostat (``pile_l``; for a single bead this is a standard
#   white-noise Langevin thermostat acting on each atom) with a relaxation
#   time of 200 fs.
#
# .. note::
#
#    This run is only a **demonstration** of the machinery: 300 steps of
#    0.5 fs cover 0.15 ps, enough to see i-PI and the client talk to each
#    other in a couple of minutes, but far too short to converge any
#    property -- let alone a vibrational spectrum, which needs tens of
#    picoseconds. Production runs simply use a much larger
#    ``<total_steps>``. In the rest of the tutorial we keep running short
#    trajectories ourselves and compare what we get with **converged
#    reference results** obtained on a GPU cluster, which are provided with
#    each exercise; already in this exercise we compare our short run with
#    a longer reference one (see 1f).


# %%
# 1d) The client: running MACE through ASE
# ----------------------------------------
#
# The client side is a short Python script, ``run-ase_ex1.py``, which uses
# the `Atomic Simulation Environment (ASE) <https://wiki.fysik.dtu.dk/ase>`_
# to connect a MACE calculator to i-PI:

with open(f"{workdir}/run-ase_ex1.py", "r") as file:
    print(file.read())

# %%
# The script does three things:
#
# 1. It reads the starting structure (``init.xyz``, an extended-xyz file
#    that also carries the cell and periodic boundary conditions) into an
#    ASE ``Atoms`` object. The atoms must be consistent with those in the
#    i-PI input (same number, order and species), since i-PI only sends
#    positions and cell.
# 2. It attaches a MACE calculator to the atoms: ``MACECalculator`` loads
#    our water potential, ``MODELS/MACE_MLIP.model`` (see 1b), from disk.
#    ``device='cpu'`` runs the model on the CPU; on a machine with a GPU,
#    use the commented-out line with ``device='cuda'`` instead, which is
#    considerably faster. ``default_dtype='float32'`` selects single
#    precision, which is sufficient for MD and roughly twice as fast as
#    double precision.
# 3. It creates an ASE ``SocketClient`` with the same UNIX-socket address as
#    the ``<ffsocket>`` block, and calls ``client.run(atoms)``. From then on
#    the script simply waits for positions from i-PI, returns energies and
#    forces, and exits when i-PI closes the connection.
#
# Any code with an ASE calculator can be turned into an i-PI client in
# exactly the same way -- this is the "reusable clients" advantage mentioned
# above.


# %%
# 1e) Running an NVT simulation of bulk water with MACE
# -----------------------------------------------------
#
# ``i-PI`` and the MLIP client are two completely independent programs, each
# with its own input, connected only through the socket: the address and
# mode set in ``run-ase_ex1.py`` must match the ``<ffsocket>`` block of the
# i-PI input. Here we start them from the notebook with ``subprocess``, so
# that everything stays in one place, but you can equally open two terminals
# and type the commands shown at the top of the cell below yourself.

# The same thing on the command line:
#
#     cd part_i/excercise_1
#     i-pi input.xml > ipi.log &
#     sleep 5
#     python3 run-ase_ex1.py > mace_0.log &

# clean.sh removes the output of previous runs from the exercise folder, so
# that every execution starts from a clean state (comment it out if you want
# to keep an earlier trajectory!)
subprocess.run(["bash", "clean.sh"], cwd=workdir, check=True)

# PYTHONUNBUFFERED=1 (the equivalent of `python -u`) makes the log files
# update in real time instead of only when the process ends
env = dict(os.environ, PYTHONUNBUFFERED="1")

# cwd=workdir runs i-PI inside the exercise folder, so all its output files
# are written there; what it would print to the screen goes to ipi.log
ipi_log = open(f"{workdir}/ipi.log", "w")
ipi_process = subprocess.Popen(
    ["i-pi", "input.xml"],
    cwd=workdir,
    env=env,
    stdout=ipi_log,
    stderr=subprocess.STDOUT,
)
time.sleep(5)  # wait for i-PI to start and open the socket

# %%
# Launch the MACE client(s): the force evaluation can be parallelized simply
# by connecting more clients to the same socket.

n_clients = 1
mace_logs = [open(f"{workdir}/mace_{i}.log", "w") for i in range(n_clients)]
mace_process = [
    subprocess.Popen(
        ["python3", "run-ase_ex1.py"],
        cwd=workdir,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    for log in mace_logs
]
time.sleep(5)  # wait for few seconds to allow mace launch and connect to i-pi

# %%
# While the simulation runs, expected duration 3-4 min, we can peek at the end of the ``i-PI`` log to
# check the progress.
# Note: you can re-run this cell as often as you like:

with open(f"{workdir}/ipi.log") as f:
    print("".join(f.readlines()[-15:]))


# %%
# 1f) Monitoring the simulation
# -----------------------------
#
# After (or while) the simulation runs, we should check that it is behaving
# sensibly before trusting any property computed from it. The property file
# ``simulation.out`` is the first place to look. Three things to watch for:
#
# - **Thermalization (equilibration).** The starting structure and the
#   randomly drawn initial velocities are not, in general, representative of
#   the equilibrium ensemble at the target temperature. During the first part
#   of the run the system relaxes towards it: the potential energy drifts
#   until it settles around a stationary value, after which it only
#   *fluctuates*. This initial segment must be discarded when computing
#   averages or correlation functions. How long it takes depends on the
#   system and on how far from equilibrium one started; for liquid water
#   started from a reasonable configuration it is typically a picosecond or
#   less, but always check rather than assume.
#
# - **The conserved quantity.** In the NVE ensemble the total energy
#   :math:`K + V` is conserved. In an NVT simulation the thermostat
#   continuously exchanges energy with the system, so :math:`K + V` is *not*
#   conserved; i-PI instead reports a *conserved quantity* that also
#   accounts for the energy exchanged with the thermostat. This quantity
#   should show no systematic drift and should fluctuate much less than the
#   potential energy itself (its residual fluctuations come from the finite
#   time step). A drifting conserved quantity signals a too-large time step,
#   inconsistent forces, or a problem in the client.
#
# - **The temperature.** The instantaneous temperature should fluctuate
#   around the target value. For a small system the fluctuations are large:
#   the relative standard deviation of the instantaneous temperature in the
#   canonical ensemble is :math:`\sqrt{2/(3N)}`, i.e. about 12% (roughly
#   :math:`\pm 35` K) for our :math:`N = 48` atoms. **Why?** Try to derive
#   this before looking at the derivation in Appendix A.1. What
#   matters is that the *average* over the equilibrated part of the
#   trajectory matches the target, and that the fluctuations are of the
#   expected size.
#
# Our 0.15 ps demo is much too short to judge any of this, so we look at the
# same quantities in a longer **reference run** of the same system, provided
# in ``part_i/trajectory_files/ref_simulation.out``: the very same input,
# but 10 000 steps (5 ps). Its final configuration is the ``init.xyz`` our
# run starts from (with freshly drawn velocities), so the two panels below
# are consecutive pieces of the same trajectory.
#
# Note: you can re-run the cells below while the simulation is running.

# read the output of our run and of the reference run
output_data, output_desc = ipi.read_output(f"{workdir}/simulation.out")
ref_data, ref_desc = ipi.read_output("part_i/trajectory_files/ref_simulation.out")

# The reference file was written without unit tags in its <properties> line,
# so (as always when units are not given explicitly) its energies are in
# atomic units; we convert them to eV to compare with our run.
HARTREE_TO_EV = 27.211386
for key in ("potential", "conserved"):
    ref_data[key] = ref_data[key] * HARTREE_TO_EV

runs = [(output_data, "our run (0.15 ps)"), (ref_data, "reference run (5 ps)")]

# plot potential energy and conserved quantity
fig, axes = plt.subplots(1, 2, figsize=(9, 3), sharey=True, constrained_layout=True)
for ax, (data, title) in zip(axes, runs):
    ax.plot(
        data["time"],
        data["potential"] - data["potential"][0],
        "b-",
        label="Potential, $V$",
    )
    ax.plot(
        data["time"],
        data["conserved"] - data["conserved"][0],
        "r-",
        label="Conserved quantity",
    )
    ax.set_title(title)
    ax.set_xlabel(r"$t$ / ps")
axes[0].set_ylabel(r"energy / eV")
axes[0].legend()
plt.show()

# %%
# Now the temperature, compared with the target value set in the
# ``<ensemble>`` block of the input. We discard the first ``t_eq``
# picoseconds as equilibration when computing the average:

T_target = 300.0  # K, as in <ensemble> of input.xml
t_eq = 0.00  # ps, equilibration time discarded from the average. Adjust this value

# compare the fluctuations with the canonical-ensemble prediction
natoms = len(read(f"{workdir}/init.xyz"))
T_std_canonical = np.sqrt(2.0 / (3.0 * natoms)) * T_target
print(f"Expected standard deviation for N = {natoms} atoms: {T_std_canonical:.1f} K")

fig, axes = plt.subplots(1, 2, figsize=(9, 3), sharey=True, constrained_layout=True)
for ax, (data, title) in zip(axes, runs):
    mask = data["time"] > t_eq
    T_avg = data["temperature"][mask].mean()
    T_std = data["temperature"][mask].std()
    print(f"{title}: average temperature after {t_eq} ps = {T_avg:.1f} +/- {T_std:.1f} K")

    ax.plot(data["time"], data["temperature"], "k-", lw=0.8, label="instantaneous")
    ax.axhline(T_target, color="r", ls="--", label=f"target ({T_target:.0f} K)")
    ax.axhline(T_avg, color="b", ls=":", label=f"average ({T_avg:.0f} K)")
    ax.axvline(t_eq, color="gray", lw=0.5)
    ax.set_title(title)
    ax.set_xlabel(r"$t$ / ps")
axes[0].set_ylabel(r"$T$ / K")
axes[0].legend(fontsize=8)
plt.show()

# %%
# **Questions** (answers at the end of the notebook):
#
# 1. In the reference run, which quantity shows a clear relaxation towards
#    equilibrium, the potential energy or the temperature, and over what
#    time? How much of the trajectory would you discard?
# 2. Is the amplitude of the conserved-quantity fluctuations small compared
#    to those of the potential energy?
# 3. Does the average temperature agree with the target within its
#    statistical uncertainty? Compare the two panels: how meaningful is the
#    average over our 0.15 ps run?
# 4. The initial velocities were drawn at 300 K, yet the temperature at step
#    0 is not 300 K. Why? 


# %%
# 1g) Visualizing the trajectory
# ------------------------------
#
# `chemiscope <https://chemiscope.org>`_ is an interactive viewer designed
# to explore collections of atomic structures together with their
# properties. It runs directly inside the notebook: the structure panel
# lets you rotate, zoom, display the unit cell and step through the frames
# of a trajectory, and in its full *map* mode a second panel shows a
# scatter plot of per-structure (or per-atom) properties linked to the
# structures, which is convenient for navigating large datasets. Here we
# only use the structure panel to look at the trajectory. We drop the first
# frame, where the atoms are still at their initial positions:

traj_data = ipi.read_trajectory(f"{workdir}/simulation.pos_0.extxyz")[1:]

chemiscope.show(
    traj_data,
    mode="structure",
    settings=chemiscope.quick_settings(structure_settings={"unitCell": True}),
)

# %%
# Use the slider (or the play button) to move along the trajectory, and
# observe the fast O--H stretching and bending motions within the molecules
# superimposed on the much slower reorientation and diffusion of the
# molecules and the continuous rearrangement of the hydrogen-bond network.
# These are the motions whose spectral fingerprints we will compute in the
# next exercises.

# %%
# Part II: IR and Raman spectroscopy of bulk water
# ************************************************
#
# In this part we compute the infrared (IR) and Raman spectra of bulk liquid
# water from molecular dynamics trajectories. Running trajectories long
# enough to converge vibrational spectra takes more time than we have in
# this tutorial, so we will run short ones -- long enough to see the whole
# machinery work -- and compare them with **converged reference spectra**
# computed with exactly the same tools from much longer runs (50 to 500 ps)
# on a GPU cluster. They are provided in the ``reference_results`` folder
# of each exercise, together with a README that documents where each file
# comes from.
#
# Within linear response theory, the IR absorption spectrum is given by the
# Fourier transform of the time-correlation function of the *time
# derivative* of the total dipole moment :math:`\bar{\mu}(t)` of the
# simulation cell (see e.g. M. E. Tuckerman, *Statistical Mechanics: Theory
# and Molecular Simulation*, Oxford University Press (2010), Chapter 14),
#
# .. math::
#
#    I^{\mathrm{IR}}(\omega) \propto
#    \int_{-\infty}^{\infty} \mathrm{d}t\, e^{-i\omega t}\,
#    \bigl\langle \dot{\bar{\mu}}(t) \cdot \dot{\bar{\mu}}(0) \bigr\rangle
#    \tag{1}
#
# where the dot denotes the time derivative and
# :math:`\langle \cdots \rangle` a thermal average in the canonical
# ensemble. Eq. (1) is the form implemented in
# ``scripts/analysis/ir_raman.py``. It is equivalent to the more familiar
# expression that correlates the dipole itself, at the price of a factor
# :math:`\omega^2`, but is better behaved numerically under periodic
# boundary conditions -- Appendix A.3 shows both the equivalence and the
# reason, and Appendix A.2 explains how correlation functions are evaluated
# in practice. The Raman spectra are obtained in the same way from the
# polarizability tensor instead of the dipole (Exercise 4).
#
# In practice we therefore need two ingredients: (i) a trajectory that
# samples the correct ensemble, and (ii) a model for the dipole moment (or
# polarizability) along the trajectory. Part II explores ingredient (ii)
# with a hierarchy of approximations of increasing accuracy:
#
# - Exercise 2: the vibrational density of states (VDOS) as a quick,
#   dipole-free estimate;
# - Exercise 3: the IR spectrum from the dipole autocorrelation function,
#   first with fixed SPC/E point charges and then with machine-learned
#   (MACE-MDP) dipoles;
# - Exercise 4: the Raman spectrum (isotropic and anisotropic) using
#   machine-learned polarizabilities.
#
# The analysis scripts live in ``scripts/analysis/`` and are self-contained
# Python scripts; run any of them with ``-h`` for the full list of options.
#
# %%
# Exercise 2: Vibrational density of states
# =========================================
#
# The **vibrational density of states** (VDOS) is the cheapest spectrum one
# can extract from a trajectory: the Fourier transform of the velocity
# autocorrelation function of the nuclei,
#
# .. math::
#
#    I^{\mathrm{vdos}}(\omega) \propto \int_0^{\infty} \mathrm{d}t\,
#    e^{-i\omega t} \sum_{i}^{3N}
#    \bigl\langle v_i(t)\, v_i(0) \bigr\rangle
#
# where :math:`v_i` are the :math:`3N` Cartesian velocity components. It is
# exactly what the IR spectrum, Eq. (1), reduces to if the dipole is assumed
# to be a *linear* function of the atomic displacements with
# configuration-independent derivatives -- the "electrical harmonicity", or
# Condon, approximation -- so that the transition dipoles factor out of the
# ensemble average and leave only the velocity correlations. Appendix A.4
# derives this and explains what the cross terms :math:`i \neq j`, dropped
# in the definition above, would contribute.
#
# The VDOS is much cheaper to evaluate -- no dipoles or polarizabilities are
# needed -- and it normally converges faster. Peak *positions* are largely
# preserved, but for liquid water the *intensities* are a poor approximation,
# because the transition dipoles depend strongly on the local
# hydrogen-bonding environment (strong electrical anharmonicity / non-Condon
# effects; see Schmidt, Corcelli and Skinner, *J. Chem. Phys.* **123**,
# 044513 (2005), and Auer and Skinner, *J. Chem. Phys.* **128**, 224511
# (2008)). Collective and intermolecular intensity features are lost.
#
# 2a) The simulation
# ------------------
#
# Exercise 2 needs a short trajectory of its own. Its input is the same kind
# of i-PI input as in Exercise 1, with one important difference in the
# ``<motion>`` block, and a detail in the ``<output>`` block that matters
# for the spectrum:

workdir_ex2 = "part_ii/excercise_2"

with open(f"{workdir_ex2}/input.xml") as f:
    xml_ex2 = f.read()
print(re.search(r"<motion.*?</motion>", xml_ex2, re.DOTALL).group(0))
print(re.search(r"<trajectory.*?</trajectory>", xml_ex2).group(0))

# %%
# The dynamics is run in the **NVE** ensemble -- no thermostat. A thermostat
# is needed to *sample* the canonical ensemble (Exercise 1), but it also
# perturbs the dynamics: a Langevin thermostat, for instance, adds friction
# and random kicks that broaden the vibrational lines (see Question 5 of
# Exercise 1). Time correlation functions are therefore computed from NVE
# trajectories, started from configurations (and velocities) drawn from an
# equilibrated NVT run; ``init.xyz`` is such a snapshot. A common
# compromise is to keep a *weak* thermostat, i.e. one with a long
# relaxation time ``tau`` (several picoseconds), which helps the sampling
# over a long run while disturbing the dynamics only negligibly.
#
# The run is short: 2000 steps of 0.5 fs, i.e. 1 ps, and the positions are
# written every 4 steps, i.e. every 2 fs -- we come back to this choice
# below. The converged reference results we will compare with were
# obtained on a GPU cluster with exactly the same input, only much longer:
# 100 000 to 1 000 000 steps, i.e. 50 to 500 ps. Those trajectories are far
# too large to distribute, so what is provided with each exercise is the
# analysis of them -- the spectra in ``reference_results``.
#
# .. note::
#
#    **Optional homework.** Add a thermostat to ``input.xml`` (copy the
#    ``<thermostat>`` block of Exercise 1 into the ``<dynamics>`` block and
#    change its mode to ``nvt``), run the simulation with a short (10 fs)
#    and with a long (5 ps) ``tau``, compute the VDOS as below and compare
#    with the NVE result.
#
# We are ready to run it. The script ``run_ex2.sh`` in the exercise folder
# does what we did by hand in Exercise 1: it cleans the folder, starts i-PI
# and then the MACE client, redirecting their output to ``ipi.log`` and
# ``mace_0.log``. It takes a few minutes, and runs in the background so that
# we can keep reading.

ex2_process = subprocess.Popen(["bash", "run_ex2.sh"], cwd=workdir_ex2) # (expected duration 13-16 min)

# %%
# While it runs we can peek at the i-PI log (re-run this cell as often as
# you like):

with open(f"{workdir_ex2}/ipi.log") as f:
    print("".join(f.readlines()[-10:]))

# %%
# Wait for the simulation to finish before computing the spectrum:

# run_ex2.sh returns when i-PI and the client have finished, so this call
# blocks until the trajectory is complete (skip the cell and come back
# later if you prefer to keep reading)
ex2_process.wait()

# %%
# 2b) Computing the VDOS
# ----------------------
#
# We compute the VDOS with ``scripts/analysis/vdos.py``, which derives the
# velocities from the positions by finite differences (or reads them
# directly with ``-vel``). The command for our run is
#
#   ``python3 vdos.py -f nvt2.pos_0.extxyz -dt 0.002 -lag 0.5 -max 500 -cell 7.822 7.822 7.822 -corr fft``
#
# flag by flag:
#
# - ``-f`` the trajectory to analyse, here the positions written by the run
#   we just launched;
# - ``-cell`` the (orthorhombic) box lengths in Å, needed to apply the
#   minimum-image convention when the velocities are obtained from the
#   positions;
# - ``-max`` how many frames to read, here all 500 of them;
# - ``-corr fft`` evaluates the correlation function with the FFT-based
#   method mentioned in Appendix A.2 (the default, ``direct``, is the
#   explicit double loop over time origins and lags);
# - ``-dt`` and ``-lag`` deserve a longer comment.
#
# - ``-dt`` is the time between *written* frames, here 4 steps of 0.5 fs =
#   2 fs. This sampling interval sets the highest frequency the spectrum
#   can contain, :math:`1/(2c\,\Delta t)` (the Nyquist frequency), which
#   for 2 fs is about 8300 cm :math:`^{-1}` -- comfortably above the O--H
#   stretch. Writing every step would only produce ten times more data for
#   no gain; 2 fs is the standard choice for water, and one can push it to
#   4 fs (Nyquist at 4200 cm :math:`^{-1}`) when disk space matters.
# - ``-lag`` is the length :math:`\tau_{\max}` of the correlation
#   function. It sets the frequency resolution
#   (:math:`\approx 1/(c\,\tau_{\max})`) and must be long enough for the
#   correlation function to have decayed; but every lag has to be averaged
#   over many time origins, which requires a trajectory much longer than
#   the lag (Appendix A.2).
#
# Before the transform, the correlation function is multiplied by a window
# that goes smoothly from 1 at :math:`t = 0` to 0 at :math:`t = \tau_{\max}`
# (the descending half of a Hann window). This ensures that the function
# being transformed vanishes at the end of the interval: an abrupt
# truncation would otherwise produce oscillatory artifacts ("ringing") in
# the spectrum.
#
# The command is stored in ``get_spectra_016.sh`` so that we can run it in
# one line. It writes ``Cvv.dat`` (the velocity autocorrelation function,
# total and per element) and ``Cvv_spectrum.dat`` (the VDOS). The reference
# spectra in ``reference_results`` were produced with the very same command
# on much longer trajectories of the same systems (up to 500 ps), only
# changing ``-max`` and ``-lag``; their provenance is documented in the
# README of that folder. Run ``vdos.py -h`` in a terminal for the remaining
# options.

subprocess.run(["bash", "get_spectra_016.sh"], cwd=workdir_ex2, check=True)

# %%
# 2c) Convergence with the simulation time
# ----------------------------------------
#
# Our own run is 1 ps long. To see how far that is from a converged
# spectrum, the folder ``reference_results`` contains the VDOS of the same
# 16-water system computed from the first 50 ps, the first 100 ps and the
# full 500 ps of a much longer NVE trajectory (the trajectory itself is far
# too big to distribute; the README in that folder lists every reference
# file). All of them use the same correlation lag of 1 ps:

ref_ex2 = f"{workdir_ex2}/reference_results"
vdos_short = np.loadtxt(f"{workdir_ex2}/Cvv_spectrum.dat")

fig, ax = plt.subplots(1, 1, figsize=(5.5, 3.2), constrained_layout=True)
ax.plot(vdos_short[:, 0], vdos_short[:, 1], "r-", lw=1, label="our run (1 ps)")
for length, style in [("050ps", "b-"), ("100ps", "g-"), ("500ps", "k-")]:
    ref = np.loadtxt(f"{ref_ex2}/Cvv_spectrum_n016_{length}_tau1.0.dat")
    ax.plot(ref[:, 0], ref[:, 1], style, lw=1.2, label=f"{int(length[:3])} ps")
ax.set_xlim(0, 4000)
ax.set_xlabel(r"$\omega$ / cm$^{-1}$")
ax.set_ylabel("VDOS (arb. units)")
ax.legend()
plt.show()

# %%
# Three bands are visible in the converged spectrum, which are the
# fingerprint of liquid water:
#
# - the broad *librational* band below about 1000 cm :math:`^{-1}`
#   (hindered rotations of the molecules in the hydrogen-bond network);
# - the H--O--H *bending* mode near 1650 cm :math:`^{-1}`;
# - the O--H *stretching* band between roughly 3000 and 3700
#   cm :math:`^{-1}`, whose width reflects the distribution of
#   hydrogen-bonding environments.
#
# The 1 ps spectrum shows the same bands at the same positions, but it is
# far from converged: the band shapes are distorted by statistical noise,
# since only 250 time origins contribute to each lag of the correlation
# function, and the short lag of 0.5 ps limits the resolution to about
# 70 cm :math:`^{-1}`.
#
# Even 50 or 100 ps is not the end of the story. The *integrated*
# intensity of each band changes by only a few per cent between 50 and
# 500 ps -- the total is fixed by the average kinetic energy of the atoms
# -- but the peak heights still move by 8--12 %, and the top of the very
# flat librational band wanders by some tens of cm :math:`^{-1}`.
# This is the central practical lesson of Part II: spectra are statistical
# quantities, they converge slowly with the length of the trajectory, and
# the *shape* of a band converges much more slowly than its area.

# %%
# 2d) Convergence with the correlation length
# -------------------------------------------
#
# How long does the lag need to be? Long enough for the correlation
# function to have decayed. The reference folder contains the velocity
# autocorrelation function of the 500 ps trajectory computed with lags of
# 0.1, 0.5, 1 and 2 ps. Let's plot them, with a zoom on the region beyond
# 0.1 ps:

fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), constrained_layout=True)
for tau, lw in [("2.0", 2.0), ("1.0", 1.4), ("0.5", 0.8), ("0.1", 0.6)]:
    cvv_ref = np.loadtxt(f"{ref_ex2}/Cvv_n016_500ps_tau{tau}.dat")
    for ax in axes:
        ax.plot(cvv_ref[:, 0] / 1000, cvv_ref[:, 1] / cvv_ref[0, 1],
                lw=lw, label=f"lag {tau} ps")
for ax in axes:
    ax.axhline(0.0, color="gray", lw=0.5)
    ax.set_xlabel(r"$t$ / ps")
axes[0].set_ylabel(r"$C_{vv}(t)\,/\,C_{vv}(0)$")
axes[0].legend()
axes[1].set_xlim(0.1, 2.0)
axes[1].set_ylim(-0.1, 0.1)
axes[1].set_title("zoom")
plt.show()

# %%
# The four curves coincide where they overlap -- they are the same
# function, computed from the same trajectory, and differ only in how far
# they extend. The fast oscillations are the intramolecular vibrations;
# their envelope decays within a few hundred femtoseconds: it is at 9 % of
# :math:`C_{vv}(0)` after 0.1 ps, at 1.5 % after 0.5 ps and below 1 % after
# 1 ps (right panel, note the vertical scale). Truncating the function at
# 0.1, 0.5, 1 or 2 ps therefore throws away very little of the signal --
# but it does set the frequency resolution :math:`1/(c\,\tau_{\max})`,
# which for these four lags is 330, 67, 33 and 17 cm :math:`^{-1}`. Here
# are the corresponding spectra:

fig, ax = plt.subplots(1, 1, figsize=(5.5, 3.2), constrained_layout=True)
for tau in ("0.1", "0.5", "1.0", "2.0"):
    ref = np.loadtxt(f"{ref_ex2}/Cvv_spectrum_n016_500ps_tau{tau}.dat")
    ax.plot(ref[:, 0], ref[:, 1], lw=1, label=f"lag {tau} ps")
ax.set_xlim(0, 4000)
ax.set_xlabel(r"$\omega$ / cm$^{-1}$")
ax.set_ylabel("VDOS (arb. units)")
ax.legend()
plt.show()

# %%
# What matters is how the resolution compares with the *intrinsic* width of
# each feature:
#
# - with a lag of 0.1 ps the resolution (330 cm :math:`^{-1}`) is worse
#   than the width of every band: the bending peak is smeared to a third of
#   its height and the two broad bands are visibly washed out;
# - the broad librational and stretching bands, which are hundreds of
#   cm :math:`^{-1}` wide, are already converged with a lag of 0.5 ps;
# - the narrow bending peak is the slowest: its height still grows by 14 %
#   from 0.5 to 1 ps and by another 4 % from 1 to 2 ps.
#
# A longer lag is not free, though: with a trajectory of fixed length,
# doubling the lag halves the number of time origins available for the
# longest lags, so the tail of the correlation function -- and with it the
# spectrum -- gets noisier. A lag of 1 ps is a good compromise for liquid
# water, and it is what all the other reference spectra of Part II use.

# %%
# 2e) Convergence with the system size
# ------------------------------------
#
# The reference folder also contains spectra for 16, 32, 64 and 128 water
# molecules, each from 500 ps of trajectory with a lag of 1 ps. Since the
# VDOS is a sum over all atoms, we divide each spectrum by its number of
# molecules before comparing them:

fig, ax = plt.subplots(1, 1, figsize=(5.5, 3.2), constrained_layout=True)
for n_water in [16, 32, 64, 128]:
    ref = np.loadtxt(f"{ref_ex2}/Cvv_spectrum_n{n_water:03d}_500ps_tau1.0.dat")
    ax.plot(ref[:, 0], ref[:, 1] / n_water, lw=1, label=f"{n_water} H$_2$O")
ax.set_xlim(0, 4000)
ax.set_xlabel(r"$\omega$ / cm$^{-1}$")
ax.set_ylabel("VDOS per molecule (arb. units)")
ax.legend()
plt.show()

# %%
# The per-molecule spectra of 32, 64 and 128 molecules fall on top of each
# other, so for a local property such as the VDOS a box of 32 molecules is
# already converged. The 16-molecule box, however, stands out:
#
# - its librational band is more intense;
# - its stretching band is more intense and sits a few tens of
#   cm :math:`^{-1}` lower;
# - below about 300 cm :math:`^{-1}` -- the region of intermolecular
#   translations -- it has *less* intensity than the larger boxes;
# - the intramolecular bend is unaffected.
#
# In a 7.8 Å box each molecule interacts with its own periodic images,
# which constrains the hydrogen-bond network and leaves no room for
# long-wavelength collective motions -- exactly what the librational and
# low-frequency bands, and the position of the stretching band, report on.
# The purely intramolecular bend does not care. Keep this in mind: the
# 16-molecule system is convenient for a tutorial, but a production
# calculation needs at least 32 molecules.

# %%
# **Questions** (answers at the end of the notebook):
#
# 1. Why is the trajectory run in the NVE ensemble, and why does it still
#    represent a canonical (NVT) average?
# 2. The VDOS has no intensity information. Which bands do you expect to be
#    most different in the IR spectrum of water, and why? (We will check in
#    Exercise 3.)


# %%
# Exercise 3: IR spectra of bulk water
# ====================================
#
# We now go beyond the VDOS and compute the IR spectrum from the dipole
# autocorrelation function introduced at the beginning of Part II, with two
# models of increasing sophistication for the dipole moment. Both are
# evaluated along the trajectory produced in Exercise 2.

workdir_ex3 = "part_ii/excercise_3"

# %%
# 3a) IR spectrum with SPC/E point charges
# ----------------------------------------
#
# The simplest dipole model beyond the VDOS assigns *fixed point charges*
# to each atom, as in the SPC/E water model
# (:math:`q_\mathrm{O} = -0.8476\,e`, :math:`q_\mathrm{H} = +0.4238\,e`),
# so that the dipole of the simulation cell is
#
# .. math::
#
#    \bar{\mu}(t) = \sum_i q_i \, \mathbf{r}_i(t)
#    \tag{2}
#
# What such a model captures, and what it misses:
#
# - Each molecule carries a permanent dipole (2.35 D for SPC/E), so the
#   dipole of the cell changes when molecules *rotate* (librations) and
#   when their geometry changes (bending and stretching). All three bands
#   of the VDOS therefore acquire an IR intensity.
# - The charges never change: there is no electronic *polarization* of a
#   molecule by its neighbours and no *charge transfer* along hydrogen
#   bonds. In real water these effects make the transition dipole of an
#   O--H stretch several times larger than in the gas phase, and strongly
#   dependent on its hydrogen-bonding environment (the non-Condon effects
#   of Exercise 2). With fixed charges the dipole derivative of a bond is a
#   constant, so the stretching band comes out far too weak relative to
#   the other bands, and its shape is wrong.
#
# (The classic mixed quantum/classical calculations of Skinner and
# co-workers -- Auer, Kumar, Schmidt and Skinner, *Proc. Natl. Acad. Sci.
# USA* **104**, 14215 (2007); Auer and Skinner, *J. Chem. Phys.* **128**,
# 224511 (2008) -- also run the dynamics with the SPC/E model, but obtain
# the OH stretch frequencies and transition dipoles from *spectroscopic
# maps* of the local electric field, precisely to capture the environment
# dependence that fixed charges alone cannot.)
#
# **Building the dipole.** The script ``scripts/analysis/spce_dipole.py``
# evaluates Eq. (2) along a trajectory. Under periodic boundary conditions
# :math:`\sum_i q_i \mathbf{r}_i` depends on where the box is cut, so the
# script sums *molecular* dipoles instead, placing each hydrogen next to
# its own oxygen with the minimum-image convention: every molecule is
# neutral, so its dipole is well defined and the total does not depend on
# the origin. The dipole is written in atomic units, the units of the
# machine-learned dipoles of the next section. We call it as
#
#   ``python3 spce_dipole.py -f ../excercise_2/nvt2.pos_0.extxyz -max 500 -out dipole_spce.dat``
#
# i.e. the trajectory of Exercise 2 (``-f``), its 500 frames (``-max``), and
# the name of the file to write (``-out``), which holds one dipole vector
# per frame.
#
# **Computing the IR spectrum.** ``scripts/analysis/ir_raman.py`` takes that
# dipole time series and does the rest:
#
#   ``python3 ir_raman.py -dip dipole_spce.dat -dt 0.002 -lag 0.5 -max 500 -corr fft -out spce``
#
# where ``-dip`` is the dipole file just written, and ``-dt``, ``-lag``,
# ``-max`` and ``-corr`` have exactly the same meaning as in ``vdos.py``.
# The main points:
#
# - It correlates the *time derivative* of the dipole, Eq. (1), obtained by
#   finite differences of consecutive frames -- the same idea as the
#   velocities in ``vdos.py``.
# - The same half-Hann window as in ``vdos.py`` is applied to the
#   correlation function before the cosine transform.
# - ``-out`` sets the prefix of the output files: ``<prefix>-dip.dat`` (the
#   correlation function), ``<prefix>-dip-win.dat`` (windowed) and
#   ``<prefix>-dip-spectrum.dat`` (the IR spectrum). The same script will
#   give the Raman spectra from a polarizability file (``-pol``) in
#   Exercise 4.
#
# Both commands are stored in ``get_spectra_spce_016.sh``; the reference
# spectra were produced with the same commands on much longer trajectories
# of the same system, changing only ``-max`` and ``-lag``:

subprocess.run(["bash", "get_spectra_spce_016.sh"], cwd=workdir_ex3, check=True)

# %%
# Plot the SPC/E IR spectrum of the 1 ps run together with the references
# for the same system, from 50, 100 and 500 ps of trajectory:

ref_ex3 = f"{workdir_ex3}/reference_results"
ir_short = np.loadtxt(f"{workdir_ex3}/spce-dip-spectrum.dat")
ir_ref16 = np.loadtxt(f"{ref_ex3}/IR-spce_n016_500ps_tau1.0.dat")

fig, ax = plt.subplots(1, 1, figsize=(5.5, 3.2), constrained_layout=True)
ax.plot(ir_short[:, 0], ir_short[:, 1], "r-", lw=1, label="2000 steps (1 ps)")
for length, style in [("050ps", "b-"), ("100ps", "g-"), ("500ps", "k-")]:
    ref = np.loadtxt(f"{ref_ex3}/IR-spce_n016_{length}_tau1.0.dat")
    ax.plot(ref[:, 0], ref[:, 1], style, lw=1.2, label=f"{int(length[:3])} ps")
ax.set_xlim(0, 4000)
ax.set_xlabel(r"$\omega$ / cm$^{-1}$")
ax.set_ylabel("IR intensity (arb. units)")
ax.legend()
plt.show()

# %%
# As for the VDOS, the 1 ps spectrum is noisy and under-resolved, but the
# bands are where the converged spectrum has them, and the three long
# references differ by only about 10 % in the peak heights -- the *total*
# intensity is fixed by :math:`\langle \dot{\bar\mu}^2 \rangle` and is
# identical to 0.3 % in the three of them. We will look at the convergence
# with the system size in Exercise 3b, with the more realistic dipoles.
#
# Now let's compare the SPC/E IR spectrum with the VDOS of Exercise 2
# for the same 16-water system, each normalized to its maximum:

vdos_ref16 = np.loadtxt(f"{ref_ex2}/Cvv_spectrum_n016_500ps_tau1.0.dat")

fig, ax = plt.subplots(1, 1, figsize=(5.5, 3.2), constrained_layout=True)
ax.plot(vdos_ref16[:, 0], vdos_ref16[:, 1] / vdos_ref16[:, 1].max(), "k-", lw=1, label="VDOS")
ax.plot(ir_ref16[:, 0], ir_ref16[:, 1] / ir_ref16[:, 1].max(), "b-", lw=1, label="IR, SPC/E")
ax.set_xlim(0, 4000)
ax.set_xlabel(r"$\omega$ / cm$^{-1}$")
ax.set_ylabel("normalized intensity")
ax.legend()
plt.show()

# %%
# The differences between the two curves are the transition dipoles at
# work:
#
# - Below about 300 cm :math:`^{-1}` the VDOS has considerable intensity
#   (translations of whole molecules and hydrogen-bond stretching), but
#   the IR spectrum has almost none: a neutral molecule moving rigidly
#   does not change the dipole of the cell.
# - The librational band keeps a large intensity, since rotating a
#   permanent dipole changes it strongly, and the bending band is the
#   most intense feature of the SPC/E spectrum.
# - The stretching band carries, if anything, slightly *less* relative
#   weight than in the VDOS. In real water it is *by far* the most intense
#   band, because the transition dipole of a hydrogen-bonded O--H is
#   strongly enhanced by polarization and charge transfer -- effects that
#   fixed charges cannot describe. This is what the next section
#   addresses.
#
# We will compare with experiment at the end of Exercise 3b, once we have
# a better dipole model.
#
# **Questions** (answers at the end of the notebook):
#
# 1. Why does the script sum molecular dipoles instead of evaluating
#    :math:`\sum_i q_i \mathbf{r}_i` directly over all atoms of the cell?
# 2. Which band of the SPC/E IR spectrum do you expect to change most when
#    the fixed charges are replaced by a model that includes polarization
#    and charge transfer, and in which direction?


# %%
# 3b) IR spectrum with machine-learned dipoles
# --------------------------------------------
#
# Instead of fixed charges, we now use a machine-learning model trained to
# predict the environment-dependent dipole moment. Specifically, we use
# **MACE-MDP** (`Gönnheimer, et al., ChemRxiv (2026)
# <https://doi.org/10.26434/chemrxiv.15000716>`_), a general dipole and
# polarizability model built on the equivariant MACE architecture (the
# ``AtomicDielectricMACE`` class of the `MACE package
# <https://github.com/ACEsuit/mace>`_). Unlike the MLIP that drives the
# dynamics, this model does not predict energies and forces: it outputs, for
# each atom :math:`i`, a scalar partial charge :math:`q_i`, an atomic dipole
# vector :math:`\boldsymbol{\mu}_i`, and atomic (isotropic + anisotropic)
# polarizability contributions. The total dipole of the cell is assembled as
#
# .. math::
#
#    \boldsymbol{\mu} = \sum_i \left( \boldsymbol{\mu}_i +
#    q_i\, \mathbf{r}_i \right)
#
# i.e. a local atomic dipole plus a charge-times-position term that carries
# the non-local (charge-transfer) part of the response. Because the dipole is
# an environment-dependent, *nonlinear* function of the atomic positions,
# this model captures the polarization and charge-redistribution effects that
# fixed point charges miss, and gives quantitatively meaningful IR
# intensities.
#
# MACE-MDP is trained on the SPICE-:math:`\alpha` dataset (~1.6 million
# charge-neutral organic structures, including a dedicated water subset) with
# dipoles and polarizabilities computed at the
# :math:`\omega`\ B97M-D3(BJ)/def2-TZVPPD level of theory. 
#
# Replaying the trajectory
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# To get these dipoles we let i-PI *replay* the trajectory of Exercise 2:
# it re-reads the stored frames one by one and asks a MACE-MDP client for
# the dipole and the polarizability of each, instead of energies and forces.
# The whole run is wrapped in ``run_ex3.sh``; let's first look at the i-PI
# input it uses, ``input_mdp.xml``:

with open(f"{workdir_ex3}/input_mdp.xml") as f:
    xml_ex3 = f.read()
print('...')
print(re.search(r"<output.*?</output>", xml_ex3, re.DOTALL).group(0))
print('...')
print(re.search(r"<motion.*?</motion>", xml_ex3, re.DOTALL).group(0))
print('...')

# %%
# - ``<motion mode='replay'>``: instead of integrating the equations of
#   motion, i-PI reads the positions frame by frame from the trajectory
#   file and sends them to the client, exactly as it would during dynamics.
#   Replaying a trajectory is the standard way to evaluate an expensive or
#   additional property *after* the dynamics has been generated.
# - ``<trajectory ... extra_type='dipole'> extras``: besides energies and
#   forces, a client can return arbitrary *extra* quantities; here i-PI
#   writes the dipole and the polarizability received at every step to
#   ``mdp.dipole_0`` and ``mdp.polarizability_0``.
#
# The client, ``run-mace-mdp_ex3.py``, differs from the one of Exercise 1
# in the model it loads and in what it sends back:

with open(f"{workdir_ex3}/run-mace-mdp_ex3.py") as f:
    print(f.read())

# %%
# - ``MACECalculator(..., model_type='DipolePolarizabilityMACE')`` loads
#   ``MODELS/MACE-MDP.model``, which predicts the dipole and polarizability
#   of a configuration (and no energies or forces).
# - ``MDP_SocketClient`` (``scripts/socketIO_mdp.py``) is a socket client
#   that sends zero energies and forces -- irrelevant in a replay -- and
#   packs the dipole (``has_dipole``) and polarizability
#   (``has_polarizability``) into the extras, converted to atomic units.
#
# Let's launch the replay. It evaluates the model on 500 frames, which takes
# a few minutes on a CPU (seconds on a GPU):

ex3_process = subprocess.Popen(["bash", "run_ex3.sh"], cwd=workdir_ex3) # (expected running time 8-10 min)

# %%
# While it runs we can peek at the i-PI log (re-run as often as you like):

with open(f"{workdir_ex3}/ipi.log") as f:
    print("".join(f.readlines()[-10:]))

# %%
# Wait for the replay to finish:

# run_ex3.sh returns when i-PI has replayed the whole trajectory, so this
# call blocks until mdp.dipole_0 and mdp.polarizability_0 are complete
ex3_process.wait()

# %%
# The output files
# ~~~~~~~~~~~~~~~~
#
# The dipole and the polarizability are written as plain text, one block
# per step:

with open(f"{workdir_ex3}/mdp.dipole_0") as f:
    print("".join(f.readlines()[:6]))
with open(f"{workdir_ex3}/mdp.polarizability_0") as f:
    print("".join(f.readlines()[:2]))

# %%
# - Each step is a ``#EXTRAS`` comment line followed by the three
#   components of the total dipole (atomic units, :math:`e\,a_0`) or the
#   nine components of the polarizability tensor (row-major
#   :math:`3 \times 3`, :math:`a_0^3`).
# - A replay run evaluates the first frame twice (step 0 appears two
#   times), so the first entry must be skipped: ``ir_raman.py -skip 1``.
#
# Computing the IR spectrum
# ~~~~~~~~~~~~~~~~~~~~~~~~~
#
# ``ir_raman.py`` reads the dipole file directly (the ``#`` lines are
# ignored), so the command is the one of Exercise 3a with a different input
# file and the extra ``-skip``:
#
#   ``python3 ir_raman.py -dip mdp.dipole_0 -skip 1 -dt 0.002 -lag 0.5 -max 500 -corr fft -out mdp``
#
# It is stored in ``get_spectra_mdp_016.sh``:

subprocess.run(["bash", "get_spectra_mdp_016.sh"], cwd=workdir_ex3, check=True)

# %%
# Convergence with the simulation time
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# As in Exercise 2, first our 1 ps run against the long references of the
# same system, here 50, 100 and 500 ps:

ir_mdp_short = np.loadtxt(f"{workdir_ex3}/mdp-dip-spectrum.dat")
ir_mdp_ref16 = np.loadtxt(f"{ref_ex3}/IR-mdp_n016_500ps_tau1.0.dat")

fig, ax = plt.subplots(1, 1, figsize=(5.5, 3.2), constrained_layout=True)
ax.plot(ir_mdp_short[:, 0], ir_mdp_short[:, 1], "r-", lw=1, label="2000 steps (1 ps)")
for length, style in [("050ps", "b-"), ("100ps", "g-"), ("500ps", "k-")]:
    ref = np.loadtxt(f"{ref_ex3}/IR-mdp_n016_{length}_tau1.0.dat")
    ax.plot(ref[:, 0], ref[:, 1], style, lw=1.2, label=f"{int(length[:3])} ps")
ax.set_xlim(0, 4000)
ax.set_xlabel(r"$\omega$ / cm$^{-1}$")
ax.set_ylabel("IR intensity (arb. units)")
ax.legend()
plt.show()

# %%
# Convergence with the system size
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Then the system-size comparison, per molecule, from the 500 ps runs:

fig, ax = plt.subplots(1, 1, figsize=(5.5, 3.2), constrained_layout=True)
for n_water in [16, 32, 64, 128]:
    ref = np.loadtxt(f"{ref_ex3}/IR-mdp_n{n_water:03d}_500ps_tau1.0.dat")
    ax.plot(ref[:, 0], ref[:, 1] / n_water, lw=1, label=f"{n_water} H$_2$O")
ax.set_xlim(0, 4000)
ax.set_xlabel(r"$\omega$ / cm$^{-1}$")
ax.set_ylabel("IR intensity per molecule (arb. units)")
ax.legend()
plt.show()

# %%
# The 1 ps spectrum is, as always, noisy, but it already shows the main
# result of this exercise: with the machine-learned dipoles the stretching
# band dominates the spectrum. Between 50 and 500 ps the band *areas* move
# by only a few per cent, while the height of the stretching peak still
# grows by about 15 % -- the same lesson as in Exercise 2, that band shapes
# converge more slowly than band areas.
#
# The system-size comparison shows a finite-size effect that the VDOS only
# hinted at, and it is *not* converged at 128 molecules:
#
# - the stretching band blue-shifts monotonically with the size of the box,
#   from 3424 cm :math:`^{-1}` for 16 molecules to 3473 cm :math:`^{-1}`
#   for 128;
# - its per-molecule intensity rises by 13 % from 16 to 32 molecules, is
#   unchanged from 32 to 64, and rises by another 9 % from 64 to 128;
# - the 16-water box also has the weakest librational band and, as in
#   Exercise 2, too little intensity below 300 cm :math:`^{-1}`.
#
# The total dipole of a box is a *collective* quantity, a sum over
# molecules whose orientations stay correlated over several hydrogen-bond
# lengths, so the IR spectrum is far more sensitive to the box size than
# the VDOS of Exercise 2, which is a sum over single-atom velocities. Keep
# this in mind when you see "converged" spectra in the literature.

# %%
# Comparison with experiment
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Finally, the comparison we have been working towards: the two dipole
# models side by side with the experimental IR spectrum of liquid water
# (``reference_results/IR_raw.dat``, digitized from the literature -- see
# the ``README.md`` there for the source), all normalized to their
# maximum:

ir_spce_ref16 = np.loadtxt(f"{ref_ex3}/IR-spce_n016_500ps_tau1.0.dat")
ir_exp = np.loadtxt(f"{ref_ex3}/IR_raw.dat")

fig, ax = plt.subplots(1, 1, figsize=(5.5, 3.2), constrained_layout=True)
ax.plot(ir_spce_ref16[:, 0], ir_spce_ref16[:, 1] / ir_spce_ref16[:, 1].max(), "b-", lw=1, label="SPC/E dipoles (500 ps)")
ax.plot(ir_mdp_ref16[:, 0], ir_mdp_ref16[:, 1] / ir_mdp_ref16[:, 1].max(), "g-", lw=1, label="MACE-MDP dipoles (500 ps)")
ax.plot(ir_exp[:, 0], ir_exp[:, 1] / ir_exp[:, 1].max(), "k-", lw=1, label="experiment")
ax.set_xlim(0, 4000)
ax.set_xlabel(r"$\omega$ / cm$^{-1}$")
ax.set_ylabel("normalized intensity")
ax.legend()
plt.show()

# %%
# - **Intensities.** With the machine-learned dipoles the relative
#   intensities of the three bands are essentially those of the
#   experiment: the stretching band dominates, the bend is weak, and the
#   librational band is in between -- whereas the SPC/E spectrum has the
#   bend as its strongest feature. Polarization and charge transfer along
#   the hydrogen bonds, learned by MACE-MDP from the electronic structure,
#   are what give the O--H stretch its large transition dipole.
#
# **Questions** (answers at the end of the notebook):
#
# 3. Why did we evaluate the dipoles by *replaying* the trajectory of
#    Exercise 2 rather than running a new simulation with the MACE-MDP
#    model as the client?
# 4. The MACE-MDP and SPC/E spectra come from the *same* trajectory. Which
#    of the two conclusions -- about band positions and about band
#    intensities -- follows from this, and what would you need to change to
#    improve the other?


# %%
# Exercise 4: Raman spectra of bulk water
# =======================================
#
# The Raman spectra are obtained in the same way as the IR spectrum, from
# the time-correlation functions of the *polarizability* tensor
# :math:`\bar{\bar{\alpha}}(t)` of the cell. We already have it: the replay
# run of Exercise 3 asked the MACE-MDP model for the polarizability along
# with the dipole, and wrote it to ``mdp.polarizability_0``. The tensor is
# split into its isotropic and anisotropic parts,
#
# .. math::
#
#    a = \tfrac{1}{3}\mathrm{Tr}\,\bar{\bar{\alpha}}, \qquad
#    \bar{\bar{\beta}} = \bar{\bar{\alpha}} - a\,\mathbf{1}
#    \tag{3}
#
# which give rise to the isotropic and anisotropic Raman spectra,
#
# .. math::
#
#    I^{\mathrm{iso}}(\omega) \propto \int_{-\infty}^{\infty} \mathrm{d}t\,
#    e^{-i\omega t}\, \bigl\langle \dot{a}(t)\, \dot{a}(0) \bigr\rangle
#    \tag{4}
#
# .. math::
#
#    I^{\mathrm{aniso}}(\omega) \propto \int_{-\infty}^{\infty} \mathrm{d}t\,
#    e^{-i\omega t}\, \Bigl\langle \mathrm{Tr}\bigl[
#    \dot{\bar{\bar{\beta}}}(t)\, \dot{\bar{\bar{\beta}}}(0) \bigr]
#    \Bigr\rangle
#    \tag{5}
#
# This is exactly what ``scripts/analysis/ir_raman.py -pol`` computes
# (``-pol_type iso``, ``aniso`` or ``both``). The two components
# correspond to different scattering geometries and obey different
# selection rules:
#
# - The *isotropic* spectrum is what remains of the polarized (VV)
#   spectrum after subtracting 4/3 of the depolarized (VH) one; it probes
#   vibrations that change the *mean* polarizability, essentially the
#   symmetric O--H stretch, and is insensitive to reorientation.
# - The *anisotropic* spectrum is the depolarized (VH) spectrum; it probes
#   changes of the *shape* of the polarizability tensor and is therefore
#   sensitive to molecular reorientation (librations) as well as to the
#   stretching modes. The ratio of the two is the depolarization ratio
#   (for the theory and a detailed interpretation of the IR, VV and VH
#   spectra of water see Auer and Skinner, *J. Chem. Phys.* **128**, 224511
#   (2008)).

# %%
# 4a) Computing the Raman spectra
# -------------------------------
#
# The same script computes the Raman spectra, reading a polarizability file
# instead of a dipole one:
#
#   ``python3 ir_raman.py -pol mdp.polarizability_0 -skip 1 -pol_type both -dt 0.002 -lag 0.5 -max 500 -corr fft -out mdp``
#
# with ``-pol`` the polarizabilities written by the replay run of
# Exercise 3, ``-pol_type both`` asking for the isotropic *and* the
# anisotropic spectrum, and everything else as in Exercise 3b. The script
# ``get_spectra_016.sh`` of the exercise folder copies the polarizability
# file over and runs it:

workdir_ex4 = "part_ii/excercise_4"

subprocess.run(["bash", "get_spectra_016.sh"], cwd=workdir_ex4, check=True)

# %%
# The output files follow the same pattern as for the IR spectrum, with
# one set per component: ``mdp-iso-spectrum.dat`` and
# ``mdp-aniso-spectrum.dat`` (plus the correlation functions).
#
# The Raman spectra have to be converged with the simulation time and with
# the system size exactly as the IR spectrum of Exercise 3, and we do not
# repeat those checks here. One thing is worth knowing: the *isotropic*
# component is by far the slowest to converge. The isotropic
# polarizability :math:`a` is a single scalar for the whole box, so its
# correlation function has far fewer independent contributions than a sum
# over atoms or tensor components. The 1 ps spectra we have just computed
# are therefore very noisy, and the comparison below uses a well converged
# reference: 128 water molecules and 500 ps.
#
# Because the Raman bands below 2000 cm :math:`^{-1}` are much weaker than
# the stretching band, we show every spectrum in two panels -- the
# low-frequency region magnified by a factor that is printed in the panel,
# and the stretching region on its own scale -- for the isotropic (top)
# and anisotropic (bottom) components. The small function below does that
# for any set of curves:


def plot_raman(curves, ylabel):
    """curves: {"iso": [(nu, intensity, style, label), ...], "aniso": [...]}.
    Left column: 0-2000 cm^-1 magnified by a round factor chosen from the
    first curve; right column: 2500-4000 cm^-1."""
    fig, axes = plt.subplots(2, 2, figsize=(9, 5.5), constrained_layout=True)
    for row, comp in enumerate(["iso", "aniso"]):
        nu0, i0 = curves[comp][0][0], curves[comp][0][1]
        ratio = i0[(nu0 > 2500) & (nu0 < 4000)].max() / i0[(nu0 > 200) & (nu0 < 2000)].max()
        factor = [1, 2, 5, 10, 20, 50, 100][np.argmin(np.abs(np.log([1, 2, 5, 10, 20, 50, 100]) - np.log(ratio)))]
        top = 0.0
        for nu, intensity, style, label in curves[comp]:
            axes[row, 0].plot(nu, intensity * factor, style, lw=1, label=label)
            axes[row, 1].plot(nu, intensity, style, lw=1, label=label)
            top = max(top, (intensity * factor)[(nu > 200) & (nu < 2000)].max())
        axes[row, 0].set_xlim(0, 2000)
        axes[row, 0].set_ylim(0, 1.2 * top)
        axes[row, 1].set_xlim(2500, 4000)
        axes[row, 0].set_title(f"{comp}, low frequency (x{factor})", fontsize=10)
        axes[row, 1].set_title(f"{comp}, O-H stretch", fontsize=10)
        axes[row, 0].set_ylabel(ylabel)
        axes[row, 1].legend(fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel(r"$\omega$ / cm$^{-1}$")
    plt.show()

# %%
# 4b) Comparison with experiment
# ------------------------------
#
# The experimental spectra in ``reference_results`` (see its ``README.md``)
# were digitized from Fig. 4 of Marsalek and Markland. They are reported
# in the same convention as Eqs. (4)-(5), so no frequency-dependent
# prefactor has to be applied to either of them; both are simply
# normalized to their maximum:

ref_ex4 = f"{workdir_ex4}/reference_results"

curves = {}
for comp in ["iso", "aniso"]:
    ref = np.loadtxt(f"{ref_ex4}/Raman-{comp}_n128_500ps_tau1.0.dat")
    exp = np.loadtxt(f"{ref_ex4}/experiment_Raman-{comp}.dat")
    curves[comp] = [(ref[:, 0], ref[:, 1] / ref[:, 1].max(), "g-", "MACE-MDP (128 H$_2$O, 500 ps)"),
                    (exp[:, 0], exp[:, 1] / exp[:, 1].max(), "k-", "experiment")]
plot_raman(curves, "normalized intensity")

# %%
# - Both components reproduce the experimental band shapes well: the
#   isotropic spectrum is a single stretching band, and the anisotropic one
#   is dominated by the stretching band with weak librational and bending
#   features. The isotropic band is broader on its low-frequency side,
#   where the strongly hydrogen-bonded O--H groups absorb.
# - The *relative* intensities come out right: with both curves normalized
#   to their maximum, the weight of the whole 200-2000 cm :math:`^{-1}`
#   region relative to the stretching band matches the experiment to 1 %
#   for the anisotropic component and to about 30 % for the isotropic one,
#   which is remarkable given that no adjustable factor is involved.
# - The band *positions* are less good, and in a revealing way. The
#   anisotropic maximum is almost exact (3465 cm :math:`^{-1}` against
#   3476 in the experiment), but the isotropic one is blue-shifted by more
#   than 100 cm :math:`^{-1}` (3367 against 3256). In the experiment the
#   two components peak 220 cm :math:`^{-1}` apart, in our spectra only
#   100: the model underestimates how differently the two components weigh
#   the hydrogen-bonding environments. The isotropic spectrum is the one
#   that reports on the strongly hydrogen-bonded, low-frequency side of
#   the stretching band -- the same region that classical nuclei and the
#   potential energy surface describe least well.
# - Below about 300 cm :math:`^{-1}` our anisotropic spectrum has clearly
#   too little intensity: the hydrogen-bond stretching band near
#   180 cm :math:`^{-1}` is present but much weaker than measured. This is
#   also the region where the digitized experimental curves are least
#   reliable, so it should not be over-interpreted.
# %%
# Take-home messages of Part II
# -----------------------------
#
# - The band *positions* are set by the potential energy surface and by
#   the nuclear dynamics; they are the same in the VDOS, IR and Raman
#   spectra computed from one trajectory, and are not changed by the dipole
#   or polarizability model.
# - The band *intensities* are set by the dipole (IR) and polarizability
#   (Raman) surfaces. Fixed point charges are qualitatively wrong for
#   water; environment-dependent machine-learned models give intensities
#   close to experiment.
# - The VDOS is a cheap first look at *where* the bands are; the spectra
#   are statistical quantities that converge slowly with the trajectory
#   length, and collective quantities (total dipole, polarizability)
#   converge more slowly than atomic velocities.
# - Simulations are cheap to re-analyse: once a trajectory exists, any
#   property model can be evaluated along it by replaying it.

# %%
# What comes next: VSFG spectroscopy of the water/air interface (optional)
# ------------------------------------------------------------------------
#
# Everything so far was about *bulk* liquid water, where every molecule
# sees the same average environment. The optional Part III of this
# tutorial, in the separate notebook ``3_vsfg.ipynb``, turns to the
# water/air interface and to **vibrational sum-frequency generation**
# (VSFG), a spectroscopy that is blind to the bulk: within the dipole
# approximation its signal vanishes in a centrosymmetric medium, so only
# the few molecular layers where inversion symmetry is broken contribute.
#
# It uses the same concepts as Part II -- MACE-MDP and correlation
# functions of the same kind -- applied to a slab of water, and runs no
# simulations, only analyses (about 20 minutes of computing in total):
#
# - **Exercise 5** computes the spectrum from surface-specific
#   velocity-velocity correlation functions, which need nothing but the
#   trajectory;
# - **Exercise 6** computes it from the machine-learned atomic charges,
#   dipoles and polarizabilities of every molecule.
#
# Open ``3_vsfg.ipynb`` whenever you like -- during the session if you get
# here early, or afterwards. It needs the interface trajectory that
# ``./download_trajectories.sh`` downloads into
# ``part_iii/trajectory_files``.

# %%
# Appendix
# ========
#
# Footnotes and derivations referred to in the text, collected here to keep
# the exercises uncluttered.
#
# A.1 Temperature fluctuations in the canonical ensemble
# ------------------------------------------------------
#
# *Why is the relative standard deviation of the instantaneous temperature
# equal to* :math:`\sqrt{2/(3N)}` *(Exercise 1f)?*
#
# In the canonical ensemble the Boltzmann factor
# :math:`e^{-\beta \sum_i p_i^2/2m_i}` factorizes, so the :math:`f = 3N`
# momentum components are independent Gaussian random variables with
# :math:`\langle p_i^2 \rangle = m_i k_B T`. Writing
# :math:`p_i = \sqrt{m_i k_B T}\, z_i` with :math:`z_i` a standard normal
# variable (zero mean, unit variance), the kinetic energy becomes
#
# .. math::
#
#    K = \sum_{i=1}^{f} \frac{p_i^2}{2 m_i}
#      = \frac{k_B T}{2} \sum_{i=1}^{f} z_i^2
#
# A sum of squares of :math:`f` independent standard normal variables is,
# *by definition*, a :math:`\chi^2` random variable with :math:`f` degrees
# of freedom, :math:`\chi^2_f = \sum_{i=1}^f z_i^2`, whose mean and variance
# are :math:`\langle \chi^2_f \rangle = f` and
# :math:`\mathrm{Var}(\chi^2_f) = 2f`. (The latter follows from
# :math:`\langle z^4 \rangle = 3` for a Gaussian, so that
# :math:`\mathrm{Var}(z^2) = \langle z^4 \rangle - \langle z^2 \rangle^2 = 2`
# for each term, and variances of independent terms add.) Therefore
#
# .. math::
#
#    \langle K \rangle = \frac{f}{2} k_B T, \qquad
#    \mathrm{Var}(K) = \left(\frac{k_B T}{2}\right)^2 2f
#                     = \frac{f}{2} (k_B T)^2
#
# and the relative fluctuation of the kinetic energy is
#
# .. math::
#
#    \frac{\sigma_K}{\langle K \rangle}
#    = \frac{\sqrt{f/2}\; k_B T}{(f/2)\, k_B T}
#    = \sqrt{\frac{2}{f}} = \sqrt{\frac{2}{3N}}
#
# Since the instantaneous temperature is *defined* as
# :math:`T_{\mathrm{inst}} = 2K/(f k_B)`, i.e. proportional to :math:`K`,
# it has exactly the same relative fluctuation. For :math:`N = 48` atoms
# this gives :math:`\sqrt{2/144} = 0.118`, i.e. about 35 K at 300 K.
#
# Two remarks. (i) i-PI may subtract the centre-of-mass degrees of freedom,
# :math:`f = 3N - 3`, which changes the number by less than 1%. (ii) The
# result holds for the canonical ensemble, which is what a Langevin
# thermostat samples; in the microcanonical (NVE) ensemble the kinetic
# energy fluctuations are smaller, because :math:`K` and :math:`V` are
# anticorrelated by energy conservation.
#
# A.2 How time correlation functions are computed in practice
# -----------------------------------------------------------
#
# All the spectra in this tutorial are Fourier transforms of equilibrium
# time correlation functions of the form
# :math:`C_{AB}(\tau) = \langle a(0)\, b(\tau) \rangle`. The thermal
# average is, formally, an average over initial conditions drawn from the
# equilibrium (canonical) distribution, each propagated for a time
# :math:`\tau`. Following Tuckerman (*Statistical Mechanics: Theory and
# Molecular Simulation*, Sec. 13.4), there are two ways of turning this
# into a recipe for a simulation (see the figure below).
#
# **Direct method.** Sample :math:`K` independent configurations
# :math:`x^{(\lambda)}`, :math:`\lambda = 1, \ldots, K`, from the
# equilibrium distribution (e.g. snapshots of a long thermostatted run),
# use each as the initial condition of a short trajectory of :math:`M`
# steps, and average the products at equal lag:
#
# .. math::
#
#    C_{AB}(n\Delta t) = \frac{1}{K} \sum_{\lambda=1}^{K}
#    a\bigl(x^{(\lambda)}_0\bigr)\, b\bigl(x^{(\lambda)}_{n\Delta t}\bigr),
#    \qquad n = 0, \ldots, M
#
# This is rigorous, but each trajectory contributes a single term to each
# lag, so many trajectories are needed to converge the average.
#
# **Single-trajectory method.** If the trajectory is much longer than the
# correlation time, points separated by more than the correlation time are
# effectively independent samples of the equilibrium distribution. Every
# point of a single long trajectory can then serve both as a *time origin*
# and as a *time-evolved point*, and the correlation function is obtained
# by averaging over all origins:
#
# .. math::
#
#    C_{AB}(n\Delta t) = \frac{1}{M - n} \sum_{m=1}^{M-n}
#    a\bigl(x_{m\Delta t}\bigr)\, b\bigl(x_{(m+n)\Delta t}\bigr),
#    \qquad n = 0, \ldots, n_{\max}
#
# This uses the data far more efficiently and is what all the analysis
# scripts of this tutorial do (every frame serves once as :math:`t = 0`, and
# the number of lags :math:`n_{\max}` is set with the ``-lag`` option).
# Note that the number of origins available, :math:`M - n`, decreases with
# the lag, so the statistics degrade at long lags: the trajectory must be
# much longer than the longest lag one wants to resolve, and the lag in
# turn must be long enough for the correlation function to have decayed.
# It rests on two assumptions: that the system is large enough for the
# microcanonical and canonical ensembles to be equivalent (so that a
# thermostat is not needed *during* the correlation window, or perturbs
# the dynamics only weakly), and that the dynamics is ergodic.
#
# .. figure:: images/acf_scheme.png
#    :align: center
#    :width: 95%
#
#    Figure A.1: Evaluation of a time correlation function (a) from
#    :math:`K` independent trajectories, each contributing one product per
#    lag (brackets connect the points that are multiplied; solid: lag
#    :math:`\Delta t`, dashed: lags :math:`2\Delta t`, :math:`3\Delta t`),
#    and (b) from a single trajectory, where every point is a time origin
#    (products at lag :math:`\Delta t` above the axis, at lag
#    :math:`2\Delta t` below). All brackets of the same lag are averaged.
#
# For very long correlation times the double loop over origins and lags
# (cost :math:`\propto M^2`) can be replaced by a fast-Fourier-transform
# evaluation based on the Wiener--Khinchin theorem (cost
# :math:`\propto M \log M`). The scripts here keep the explicit sum for
# clarity and use the FFT only for the final cosine transform to the
# frequency domain.
#
#
# A.3 Equivalence of the dipole and dipole-derivative forms of the IR spectrum
# ----------------------------------------------------------------------------
#
# *Part II computes the IR spectrum from the time derivative of the dipole,
# Eq. (1), while textbooks usually write it as the transform of the dipole
# autocorrelation function itself,*
#
# .. math::
#
#    I^{\mathrm{IR}}(\omega) \propto \omega^{2}
#    \int_{-\infty}^{\infty} \mathrm{d}t\, e^{-i\omega t}\,
#    \bigl\langle \bar{\mu}(t) \cdot \bar{\mu}(0) \bigr\rangle
#
# *Why are the two identical, and what do the scripts actually compute?*
#
# Write :math:`C(t) = \langle \bar{\mu}(t) \cdot \bar{\mu}(0) \rangle`.
# Equilibrium correlation functions are *stationary*: they depend only on
# the time difference, so :math:`\langle \bar{\mu}(t+s) \cdot \bar{\mu}(s) \rangle = C(t)`
# for any :math:`s`. Differentiating this identity with respect to
# :math:`s` gives
# :math:`\langle \dot{\bar{\mu}}(t+s) \cdot \bar{\mu}(s) \rangle + \langle \bar{\mu}(t+s) \cdot \dot{\bar{\mu}}(s) \rangle = 0`,
# i.e. :math:`\langle \bar{\mu}(t) \cdot \dot{\bar{\mu}}(0) \rangle = -\langle \dot{\bar{\mu}}(t) \cdot \bar{\mu}(0) \rangle = -\dot{C}(t)`.
# Differentiating once more with respect to :math:`t`,
#
# .. math::
#
#    \bigl\langle \dot{\bar{\mu}}(t) \cdot \dot{\bar{\mu}}(0) \bigr\rangle
#    = -\frac{\mathrm{d}^2 C(t)}{\mathrm{d}t^2}
#
# Inserting this into Eq. (1) and integrating by parts twice (the boundary
# terms vanish because :math:`C(t)` and :math:`\dot{C}(t)` decay to zero
# at :math:`t \to \pm\infty`),
#
# .. math::
#
#    -\int_{-\infty}^{\infty} \mathrm{d}t\, e^{-i\omega t}\,
#    \frac{\mathrm{d}^2 C}{\mathrm{d}t^2}
#    = -\int_{-\infty}^{\infty} \mathrm{d}t\,
#    \frac{\mathrm{d}^2 e^{-i\omega t}}{\mathrm{d}t^2}\, C(t)
#    = \omega^2 \int_{-\infty}^{\infty} \mathrm{d}t\, e^{-i\omega t}\, C(t)
#
# which is the :math:`\omega^2` form above. The two are therefore the same
# spectrum, and Eq. (1) is the one implemented in the analysis scripts:
# under periodic boundary conditions the absolute dipole of the cell is
# defined only up to a "quantum" of polarization, whereas its time
# derivative is continuous and well defined.
#
# Finally, a classical equilibrium correlation function is *even* in time,
# :math:`C(-t) = \langle \bar{\mu}(-t) \cdot \bar{\mu}(0) \rangle = \langle \bar{\mu}(0) \cdot \bar{\mu}(t) \rangle = C(t)`
# (stationarity again), so the two-sided Fourier transform reduces to a
# cosine transform over positive times,
#
# .. math::
#
#    \int_{-\infty}^{\infty} \mathrm{d}t\, e^{-i\omega t}\, C(t)
#    = 2 \int_{0}^{\infty} \mathrm{d}t\, \cos(\omega t)\, C(t)
#
# This is what the analysis scripts compute: the correlation function is
# accumulated for positive lags only (Appendix A.2) and then
# cosine-transformed.

# %%
# A.4 The VDOS as the Condon limit of the IR spectrum
# ---------------------------------------------------
#
# *Where does the velocity autocorrelation function of Exercise 2 come from,
# and what is lost on the way?*
#
# Following Haggard, Litman and Althorpe
# (`J. Chem. Phys. 164, 144120 (2026)
# <https://doi.org/10.1063/5.0325115>`_), we can make the connection between
# the IR spectrum and the underlying nuclear dynamics explicit by applying
# the chain rule to the dipole time derivative,
# :math:`\dot{\bar{\mu}}(t) = \sum_i^{3N} \frac{\partial
# \bar{\mu}(\mathbf{r}(t))}{\partial r_i} v_i(t)`, where the sums run over
# the :math:`3N` Cartesian nuclear coordinates :math:`r_i` and
# :math:`v_i = \dot{r}_i` are the corresponding velocities. The IR spectrum
# then becomes
#
# .. math::
#
#    I^{\mathrm{IR}}(\omega) \propto \int_0^{\infty} \mathrm{d}t\,
#    e^{-i\omega t} \sum_{i,j}^{3N} \left\langle
#    \frac{\partial \bar{\mu}(\mathbf{r}(t))}{\partial r_i}\,
#    \frac{\partial \bar{\mu}(\mathbf{r}(0))}{\partial r_j}\,
#    v_i(t)\, v_j(0) \right\rangle
#
# where it becomes apparent that the transition dipole moments
# :math:`\partial \bar{\mu} / \partial r_i` *modulate the intensity* of the
# "pure" vibrational signal given by the vibrational density of states
# (VDOS),
#
# .. math::
#
#    I^{\mathrm{vdos}}(\omega) \propto \int_0^{\infty} \mathrm{d}t\,
#    e^{-i\omega t} \sum_{i,j}^{3N}
#    \bigl\langle v_i(t)\, v_j(0) \bigr\rangle
#
# The VDOS is thus what the IR spectrum reduces to if the dipole is assumed
# to be a *linear* function of the atomic displacements with
# configuration-independent derivatives (the "electrical harmonicity", or
# Condon, approximation): the constant transition dipoles then factor out of
# the ensemble average, leaving only the velocity correlations. Note also
# that the most common definition of the VDOS drops the cross terms
# :math:`i \neq j` and keeps only the velocity *autocorrelation* functions.
#
#
# %%
# References
# ----------
#
# - M. E. Tuckerman, *Statistical Mechanics: Theory and Molecular
#   Simulation*, Oxford University Press, Oxford (2010).
# - C. Haggard, Y. Litman, and S. C. Althorpe, "Infrared and Raman
#   perspectives on vibrational coupling in liquid water",
#   *J. Chem. Phys.* **164**, 144120 (2026).
#   `DOI:10.1063/5.0325115 <https://doi.org/10.1063/5.0325115>`_
# - O. Marsalek and T. E. Markland, "Quantum Dynamics and Spectroscopy of
#   Ab Initio Liquid Water: The Interplay of Nuclear and Electronic Quantum
#   Effects", *J. Phys. Chem. Lett.* **8**, 1545-1551 (2017).
#   `DOI:10.1021/acs.jpclett.7b00391 <https://doi.org/10.1021/acs.jpclett.7b00391>`_
#   (the IR and Raman expressions are given in Sec. 7 of the Supporting
#   Information)
# - B. Auer, R. Kumar, J. R. Schmidt, and J. L. Skinner, "Hydrogen bonding
#   and Raman, IR, and 2D-IR spectroscopy of dilute HOD in liquid D2O",
#   *Proc. Natl. Acad. Sci. USA* **104**, 14215-14220 (2007).
#   `DOI:10.1073/pnas.0701482104 <https://doi.org/10.1073/pnas.0701482104>`_
# - B. M. Auer and J. L. Skinner, "IR and Raman spectra of liquid water:
#   Theory and interpretation", *J. Chem. Phys.* **128**, 224511 (2008).
#   `DOI:10.1063/1.2925258 <https://doi.org/10.1063/1.2925258>`_
# - J. R. Schmidt, S. A. Corcelli, and J. L. Skinner, "Pronounced non-Condon
#   effects in the ultrafast infrared spectroscopy of water",
#   *J. Chem. Phys.* **123**, 044513 (2005).
#   `DOI:10.1063/1.1961472 <https://doi.org/10.1063/1.1961472>`_
# - The MACE-MDP models used here are described in Litman et al.,
#   *in preparation* (see ``MODELS/README.md``).
# %%
# Answers to the questions
# ========================
#
# Exercise 1
# ----------
#
# 1. In this particular case it is the *temperature*: the reference run
#    starts from an equilibrated liquid configuration, so its potential
#    energy fluctuates around a stationary value from the very beginning,
#    but the velocities were freshly drawn and the instantaneous temperature
#    starts about 30 K below the target and is pulled up by the thermostat
#    over roughly a picosecond -- a few relaxation times :math:`\tau`. The
#    fluctuations are so large that this is easier to see in a running
#    average than by eye: the mean temperature is 272 K over the first
#    0.25 ps, 297 K between 0.5 and 1 ps, and 311 K afterwards. One would
#    discard that first picosecond. Starting instead from a poorly
#    relaxed structure, the potential energy would show an equally clear
#    drift; which quantity relaxes visibly depends on how the run was
#    started, which is why one always looks at both.
# 2. Yes. The potential energy fluctuates by several tenths of an eV (it
#    exchanges energy with the kinetic energy and, through the thermostat,
#    with the bath), whereas the conserved quantity -- kinetic plus potential
#    energy plus the energy exchanged with the thermostat -- only shows the
#    small, non-drifting fluctuations caused by the finite integration time
#    step. If it drifts, the time step is too large or the forces are
#    inconsistent.
# 3. It should. With the thermostat the equilibrated trajectory samples the
#    canonical ensemble at 300 K; the standard deviation of the
#    instantaneous temperature is about 35 K for 48 atoms (Appendix A.1),
#    and the statistical error of the average is much smaller than that,
#    since it decreases with the number of *independent* samples in the
#    trajectory. The temperature decorrelates on the scale of the thermostat
#    relaxation time, so the 5 ps reference run contains of the order of 25
#    independent samples and its average carries an uncertainty of several
#    kelvin -- an average some 10 K above the target is still compatible
#    with it. Our 0.15 ps run, on the other hand, is shorter than a single
#    relaxation time: its "average" is one sample of a broad distribution
#    and says nothing about whether the simulation is at the right
#    temperature.
# 4. The ``<velocities mode='thermal'>`` line does not *set* the temperature
#    to 300 K; it draws each momentum component at random from the
#    Maxwell--Boltzmann distribution at 300 K. The instantaneous temperature
#    of a single draw is therefore itself a random variable, with the same
#    :math:`\pm 35` K spread derived in Appendix A.1, so a starting value
#    20--30 K away from the target is perfectly normal. Because the random
#    seed is fixed in ``<prng>``, every run draws exactly the same velocities
#    and starts from the same value; change the seed and the starting
#    temperature will move. The thermostat then brings the average to the
#    target within a few relaxation times.
#
# Exercise 2
# ----------
#
# 1. A thermostat perturbs the dynamics (friction and random forces in a
#    Langevin thermostat), which broadens and can shift the vibrational
#    lines. NVE dynamics conserves the true Hamiltonian flow. The average
#    over the canonical ensemble is still obtained because the initial
#    condition is drawn from an equilibrated NVT run (and, for a large
#    enough system, a single NVE trajectory samples a microcanonical shell
#    whose averages coincide with the canonical ones -- the ensemble
#    equivalence assumed in Appendix A.2). In practice one averages over
#    several NVE runs started from different NVT snapshots.
# 2. The VDOS weights every motion by its kinetic energy, whereas the IR
#    spectrum weights it by the square of the transition dipole. For water
#    the stretching and librational bands carry by far the largest dipole
#    derivatives (strongly enhanced by hydrogen bonding), so their relative
#    intensity will increase; the bend is weaker, and the relative intensity
#    *within* the stretching band shifts towards its low-frequency,
#    strongly hydrogen-bonded side (non-Condon effects).
#
# Exercise 3
# ----------
#
# 1. Under periodic boundary conditions the atoms are wrapped into the box
#    independently of each other, so a molecule can be cut by the boundary
#    with its oxygen on one side and a hydrogen on the other; the bare sum
#    :math:`\sum_i q_i \mathbf{r}_i` then jumps by :math:`q_\mathrm{H} L`
#    every time an atom crosses a face. Placing each hydrogen next to its
#    own oxygen by the minimum-image convention reconstructs intact,
#    neutral molecules, whose dipoles are well defined and
#    origin-independent; their sum is then a smooth function of time, as
#    required to take its derivative.
# 2. The O--H stretching band. Its intensity should increase several-fold
#    relative to the bending and librational bands, and its shape should
#    change, with more weight on its low-frequency, strongly
#    hydrogen-bonded side: polarization and charge transfer along a
#    hydrogen bond enhance the transition dipole of the donating O--H, the
#    more so the stronger (and lower in frequency) the bond.
# 3. The dipole is a function of the positions only, so it can be
#    evaluated on any trajectory after the fact: the dynamics does not
#    depend on which dipole model we use, only the spectrum does. Replaying
#    also makes the comparison clean -- the SPC/E and MACE-MDP spectra
#    differ *only* through the dipole model. A new simulation would in
#    addition require a model that provides forces, which MACE-MDP does
#    not.
# 4. Because the trajectory is the same, the band *positions* are
#    identical in the two spectra: they are set by the potential energy
#    surface (the MACE potential) and by the classical treatment of the
#    nuclei. Only the band *intensities* change, and these are what the
#    MACE-MDP model fixes. To move the positions one would need a better
#    potential and/or the inclusion of nuclear quantum effects, not a
#    better dipole.
#
