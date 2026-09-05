"""

Part III: Vibrational sum-frequency generation spectroscopy of the water/air interface
**************************************************************************************

:Authors: Yair Litman `@litman90 <https://github.com/litman90/>`_
:Version: 1.1

**This notebook is the optional third part of the tutorial** *Vibrational
spectroscopy of water from machine-learned molecular dynamics*
(``tutorial_II.ipynb``). Parts I and II of that notebook introduce i-PI,
MACE and the machinery of time-correlation functions and compute the IR and
Raman spectra of bulk water; here we move to an *interface*, and to a
spectroscopy that sees only the interface. The two notebooks are separate
files: you can go through this one right after Part II, or leave it for
later.

**Before you start.**

- Work through Parts I and II of ``tutorial_II.ipynb``, at least up to
  Exercise 3: we reuse the i-PI replay machinery, the MACE-MDP model and
  the analysis conventions introduced there without repeating them.
- Make sure the trajectory of the water/air interface is in
  ``part_iii/trajectory_files`` -- ``./download_trajectories.sh`` puts it
  there (about 1 GB).
- Use the same conda environment, ``Tutorial_II``.

This part contains two exercises:

- **Exercise 5**: VSFG spectra from surface-specific velocity correlation
  functions.
- **Exercise 6**: VSFG spectra from machine-learned atomic dipoles and
  polarizabilities.

Unlike Parts I and II, this part runs **no simulations**: everything that
needs one (the slab trajectory itself and the machine-learned atomic
quantities of Exercise 6) was computed beforehand and is part of the
download. The i-PI inputs used to produce these data are provided as well
-- with the downloaded trajectory and in ``part_iii/excercise_6`` -- in
case you want to reproduce them yourself. The notebook only runs analyses
-- about 20 minutes of computing in total, and each step states its
expected time.

As in Parts I and II, the derivations are collected in the Appendix at the
end, and the answers to the questions in the section after it.

.. warning::

   **Run the code cell below first!** It imports the packages used by every
   other cell of this notebook. If you later see an error such as
   ``NameError: name 'np' is not defined``, this cell was skipped -- go
   back and execute it.
"""
import json
import os
import re
import subprocess

import chemiscope
from ase.io import read
import matplotlib.pyplot as plt
import numpy as np

# JupyterLab 4 sometimes leaves cells blank with its default virtualized
# rendering; this makes sure every cell is always drawn. Takes effect the
# next time the page is (re)loaded.
_settings = os.path.join(os.path.expanduser("~"), ".jupyter", "lab",
                         "user-settings", "@jupyterlab", "notebook-extension")
_tracker = os.path.join(_settings, "tracker.jupyterlab-settings")
try:
    with open(_tracker) as _f:
        _cfg = json.load(_f)
except (FileNotFoundError, ValueError):
    _cfg = None if os.path.exists(_tracker) else {}
if _cfg is not None and _cfg.get("windowingMode") != "none":
    _cfg["windowingMode"] = "none"
    os.makedirs(_settings, exist_ok=True)
    with open(_tracker, "w") as _f:
        json.dump(_cfg, _f, indent=2)
    print("Notebook display settings updated -- if cells ever appear blank,"
          " reload the browser tab once.")

# %%
# Vibrational sum-frequency generation (VSFG) spectroscopy is a second-order
# nonlinear technique that is *surface specific*: within the dipole
# approximation the signal vanishes in centrosymmetric bulk media, so only
# the few molecular layers where inversion symmetry is broken contribute. In
# this part we compute the VSFG spectrum of the water/air interface from a
# slab simulation (a pre-computed 50 ps trajectory of a small slab is
# provided in ``part_iii/trajectory_files``), using:
#
# - Exercise 5: an approach based on the velocities of the OH bonds, which
#   needs nothing but the trajectory;
# - Exercise 6: a fully atomistic route based on the machine-learned atomic
#   charges, dipoles and polarizabilities of the MACE-MDP model.
#
# The analysis scripts live in ``scripts/analysis/`` (``slab_profiles.py``,
# ``ssvvcf_ml.py`` and ``sfg_atomic.py``); run them with ``-h`` for the full
# list of options.
#
# Three analyses are run along the way: the density and orientation
# profiles of the slab (about 2 minutes), the velocity-based spectrum
# (about 8 minutes) and the atomic-decomposition spectrum (about 10 minutes) --
# roughly 20 minutes of computing in total, each launched where the text
# introduces it.

# %%
# Exercise 5: VSFG spectra from velocity correlation functions
# ============================================================
#
# Theory: SFG and the second-order susceptibility
# -----------------------------------------------
#
# The SFG response is governed by the resonant part of the second-order
# susceptibility, which in the classical time-correlation formalism reads
# (A. Morita, *Theory of Sum Frequency Generation Spectroscopy*, Springer
# (2018); see also Khatib et al., *Sci. Rep.* **6**, 24287 (2016), and
# Khatib and Sulpizi, *J. Phys. Chem. Lett.* **8**, 1310 (2017))
#
# .. math::
#
#    \chi^{(2),R}_{\zeta\eta\kappa}(\omega_{\mathrm{IR}}) =
#    \frac{i}{k_B T\, \omega_{\mathrm{IR}}}
#    \int_0^{\infty} \mathrm{d}t\, e^{i\omega_{\mathrm{IR}} t}\,
#    \bigl\langle \dot{{\alpha}}_{\zeta\eta}(t)\,
#    \dot{\mu}_{\kappa}(0) \bigr\rangle
#    \tag{1}
#
# where :math:`\zeta\eta\kappa` are the polarizations of the SFG, visible and
# IR beams, :math:`\omega_{\mathrm{IR}}` is the IR frequency,
# :math:`\alpha` and :math:`\mu` are the *total*
# polarizability tensor and dipole moment of the system as in Part II, the
# dot denotes the time derivative, and :math:`\langle \cdots \rangle` a
# canonical ensemble average. The :math:`i` in the prefactor means that the
# *imaginary* part of :math:`\chi^{(2)}` is simply the cosine transform of
# the correlation function divided by :math:`\omega_{\mathrm{IR}}` -- the
# quantity the scripts of this part compute.
# In the common ssp polarization combination the probed element is
# :math:`\chi^{(2)}_{xxz}` (equivalently the average of :math:`xxz` and
# :math:`yyz` for an isotropic surface, with :math:`z` the surface normal),
# and its imaginary part is the quantity directly measured by
# heterodyne-detected (phase-resolved) experiments. Note the family
# resemblance with the IR and Raman expressions of Part II: a
# *cross*-correlation of the
# polarizability and dipole derivatives replaces the autocorrelations of
# the IR and Raman spectra, and the sign of Im :math:`\chi^{(2)}` carries
# information on the *orientation* of the vibrating groups.
#
# Surface specificity follows from symmetry: the correlation function couples
# a rank-2 (even) and a rank-1 (odd) response, so
# :math:`\chi^{(2)}` vanishes in any centrosymmetric medium -- in the bulk
# liquid the contributions of oppositely oriented molecules cancel, and only
# the few layers where inversion symmetry is broken contribute. Can you
# prove it? (Question 1.)

# %%
# 5a) The slab
# ------------
#
# An interface is commonly simulated as a *slab*: a film of liquid with vacuum on
# both sides, periodic in all three directions, so that the system has two
# equivalent water/air interfaces with the surface normal along :math:`z`.
# Let's look at the first frame of the provided trajectory with chemiscope
# (tick "unit cell" if needed, and rotate the box):

workdir_ex5 = "part_iii/excercise_5"

slab_geometry = read(f"{workdir_ex5}/geo.xyz")
chemiscope.show(
    [slab_geometry],
    mode="structure",
    settings=chemiscope.quick_settings(structure_settings={"unitCell": True}),
)

# %%
# - 48 water molecules (144 atoms) in a :math:`10 \times 10 \times 100` Å
#   cell: about 15 Å of liquid, and 85 Å of vacuum that separates the film
#   from its periodic images along :math:`z`. The cross-section is small
#   (which keeps the exercise cheap), the vacuum gap is generous.
# - The trajectory is 50 ps of NVE dynamics with the MACE potential of
#   Part I (frames every 2 fs, as in Part II), started from an equilibrated
#   NVT configuration; the folder also contains the MACE-MDP atomic
#   quantities for the whole run, used in Exercise 6.
# - Converged reference results (from 1 ns of dynamics of the same slab,
#   i.e. 20 times our trajectory) are provided in the
#   ``reference_results`` folders of both exercises for comparison.
#
# 5b) Density and orientation profiles
# ------------------------------------
#
# Before computing a spectrum it is worth looking at the structure of the
# interface. Two profiles along the surface normal summarize it: the
# density of oxygen atoms, :math:`\rho_{\mathrm{O}}(z)`, and the average
# orientation of the molecular dipole,
# :math:`\langle \cos\theta \rangle (z)`, with :math:`\theta` the angle
# between the dipole (the H--O--H bisector, pointing from the O towards the
# hydrogens) and the :math:`+z` axis. The script ``slab_profiles.py``
# computes both, after moving the centre of mass of the slab to
# :math:`z = 0` in every frame. The command for our slab, stored in
# ``get_profiles.sh``, is
#
#   ``python3 slab_profiles.py -f $TRAJ -cell 10 10 100 -max 25000 -dz 0.1 -out slab``
#
# with ``-f`` the slab trajectory, ``-cell`` its box (10 × 10 × 100 Å),
# ``-max`` all 25 000 frames (50 ps), ``-dz`` the width of the histogram
# bins in Å and ``-out`` the prefix of the output file,
# ``slab_profiles.dat``. It takes about two minutes:

subprocess.run(["bash", "get_profiles.sh"], cwd=workdir_ex5, check=True)

# %%
# We compare our 50 ps profiles with the converged reference computed from
# 1 ns of dynamics of the same slab (``slab-048_1ns_profiles.dat``, same
# format). The orientation profile is averaged over 1 Å blocks, weighting
# each bin by its number of molecules; towards the outer edge of the
# interface it becomes noise -- the outermost bins contain only a handful
# of molecules:
rho_bulk = 0.0334  # oxygen number density of liquid water, Angstrom^-3


def orientation_1A(z, cos_theta, rho):
    """Weighted average of <cos theta> over 1 A blocks (weights: the density,
    i.e. the number of molecules of each bin); blocks with less than 2 % of
    the bulk density are hidden."""
    n = int(round(1.0 / (z[1] - z[0])))
    nblock = len(z) // n
    z_b = z[:nblock * n].reshape(nblock, n).mean(axis=1)
    w = rho[:nblock * n].reshape(nblock, n)
    c = cos_theta[:nblock * n].reshape(nblock, n)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos_b = np.sum(c * w, axis=1) / np.sum(w, axis=1)
    return z_b, np.where(w.mean(axis=1) > 0.02 * rho_bulk, cos_b, np.nan)


ours = np.loadtxt(f"{workdir_ex5}/slab_profiles.dat")
ref_1ns = np.loadtxt(f"{workdir_ex5}/reference_results/slab-048_1ns_profiles.dat")
fig, (ax_rho, ax_cos) = plt.subplots(2, 1, figsize=(8, 5), sharex=True,
                                     constrained_layout=True)
ax_rho.plot(ours[:, 0], ours[:, 1], "r-", lw=1.5, label="50 ps (ours)")
ax_cos.plot(*orientation_1A(ours[:, 0], ours[:, 3], ours[:, 1]), "r.-", lw=1.5)
ax_rho.plot(ref_1ns[:, 0], ref_1ns[:, 1], "k-", lw=1, label="1 ns, reference")
ax_cos.plot(*orientation_1A(ref_1ns[:, 0], ref_1ns[:, 3], ref_1ns[:, 1]),
            "k.-", lw=1)
ax_rho.axhline(rho_bulk, color="gray", lw=0.5, ls="--")
ax_rho.set_ylabel(r"$\rho_\mathrm{O}(z)$ / Å$^{-3}$")
ax_rho.legend(fontsize=8)
ax_cos.axhline(0, color="gray", lw=0.5)
ax_cos.set_xlabel("z / Å")
ax_cos.set_ylabel(r"$\langle \cos\theta \rangle$")
ax_cos.set_xlim(-15, 15)
plt.show()

# %%
# - The interface is sharp on the molecular scale: the density drops from
#   its central value to zero over about 3 Å on each side. The *Gibbs
#   dividing surface* -- where the density has fallen to half the bulk
#   value -- sits at :math:`|z| \approx 7.4` Å in the converged profile.
# - Our film is only about five molecular layers thick, and it shows: the
#   density oscillates (each surface induces a layering that has not
#   decayed when it meets the one of the other surface), so a film this
#   thin has no truly bulk-like centre -- a limitation to keep in mind for
#   the rest of this part.
# - Comparing the two curves shows what 50 ps can and cannot do: the
#   position and width of the interface are already right, but the layering
#   oscillations are exaggerated and asymmetric -- with 20 times more
#   statistics the 1 ns profile is smoother and symmetric between the two
#   surfaces.
# - :math:`\langle \cos\theta \rangle` vanishes where the film is
#   bulk-like and is largest in the outermost layers: the molecules at the
#   surface are oriented, with the dipole pointing slightly *into* the
#   liquid on average.
# - The two surfaces are mirror images: the profile at the bottom is the
#   negative of the profile at the top. This is exactly the asymmetry
#   that SFG measures -- and the reason why the two surfaces of a slab
#   must be treated with care (next section).

# %%
# 5c) SFG from velocity-velocity correlation functions
# ----------------------------------------------------
#
# Evaluating the correlation function of Eq. (1) requires the total dipole
# and polarizability at every step -- expensive if they come from first
# principles. Khatib and Sulpizi (*J. Phys. Chem. Lett.* **8**, 1310
# (2017)) showed that in the O--H stretching region it can be built from
# the *velocities* of the OH bonds alone, weighted by parametrized
# transition dipole and polarizability derivatives. A closely related
# formulation, the *surface-specific velocity-velocity correlation
# function* (ssVVCF), was derived by Ohto, Usui, Hasegawa, Bonn and Nagata
# (*J. Chem. Phys.* **143**, 124702 (2015)); we adopt their acronym for the
# script and output files of this exercise.
#
# The model combines two approximations in one equation. The total time
# derivatives are written as sums over the two OH bonds of every molecule
# (*molecular decomposition*), and each bond contribution is linearized in
# the OH bond length :math:`r` -- the bond elongations are small, and the
# stretch is much faster than the molecular reorientation (*linear
# approximation*):
#
# .. math::
#
#    \dot{\mu} \simeq \sum_{i=1}^{N_{\mathrm{mol}}}
#    \sum_{\mathrm{bond}}
#    \frac{\partial \mu^{\,\mathrm{bond}}}{\partial r}\,
#    \dot{r}_{i,\mathrm{bond}}, \qquad
#    \dot{\alpha} \simeq \sum_{i=1}^{N_{\mathrm{mol}}}
#    \sum_{\mathrm{bond}}
#    \frac{\partial \alpha^{\,\mathrm{bond}}}{\partial r}\,
#    \dot{r}_{i,\mathrm{bond}}
#    \tag{2}
#
# where the inner sum runs over the two OH bonds of molecule :math:`i`,
# :math:`\dot{r}_{i,\mathrm{bond}}` is the velocity projected on the bond
# axis, and the derivatives are *constants*, evaluated in the frame of the
# bond. The bond frame and the lab frame are related by rotations, left
# implicit in Eq. (2); the explicit rotation matrices, and the
# approximations behind them, are written out in Appendix A.1.
#
# The constant bond-frame derivatives were parametrized from maximally
# localized Wannier centers by finite differences (stretching the OH bond
# by :math:`\pm 0.02` Å); for liquid water Khatib et al. obtained
# :math:`\partial \mu_z / \partial r = 2.1` D Å :math:`^{-1}` along the bond
# and :math:`\partial \alpha_{xx} / \partial r = 0.40`,
# :math:`\partial \alpha_{yy} / \partial r = 0.53`,
# :math:`\partial \alpha_{zz} / \partial r = 1.56` Å :math:`^{2}`.
#
# Inserting Eq. (2) into Eq. (1) turns the dipole-polarizability
# correlation into a double sum over pairs of OH bonds, which splits into
# three kinds of terms (Eq. (3) of Khatib and Sulpizi; the polarization
# indices are suppressed on the right-hand side):
#
# .. math::
#
#    \bigl\langle \dot{\alpha}_{\zeta\eta}(t)\, \dot{\mu}_{\kappa}(0)
#    \bigr\rangle
#    = \sum_{i}^{N_{\mathrm{mol}}} \sum_{b}
#      \bigl\langle \dot{\alpha}_{i,b}(t)\, \dot{\mu}_{i,b}(0) \bigr\rangle
#    + \sum_{i}^{N_{\mathrm{mol}}} \sum_{b}
#      \bigl\langle \dot{\alpha}_{i,b}(t)\, \dot{\mu}_{i,-b}(0) \bigr\rangle
#    + \sum_{i \neq j}^{N_{\mathrm{mol}}} \sum_{b,b'}
#      \bigl\langle \dot{\alpha}_{i,b}(t)\, \dot{\mu}_{j,b'}(0) \bigr\rangle
#    \tag{3}
#
# where :math:`b` runs over the two OH bonds of a molecule and :math:`-b`
# denotes the other bond of the same molecule. The first sum collects the
# *self* terms (a bond with itself), the second the *intramolecular* terms
# (the two bonds of the same molecule) and the third the *intermolecular*
# terms (bonds of different molecules). The ``-rc`` cutoff of the analysis
# script selects how many of the cross terms are kept (5d).

# %%
# The surface window
# ~~~~~~~~~~~~~~~~~~
#
# In a slab the *total* correlation function of Eq. (1) is useless as it
# stands, for two reasons that the profiles of 5b make obvious:
#
# - the two surfaces are mirror images, so their contributions to
#   :math:`\chi^{(2)}_{xxz}` have opposite signs and cancel exactly on
#   average;
# - the molecules in the middle of the slab are bulk-like: they contribute
#   nothing on average, but they add statistical noise.
#
# Both are handled by a *window function* :math:`w(z)` that multiplies the
# dipole contribution of every bond (or atom, in Exercise 6) according to
# the position :math:`z` of its oxygen relative to the slab centre. With
# :math:`z_1 =` ``zref1`` and :math:`z_2 =` ``zref2``,
#
# .. math::
#
#    w(z) = \mathrm{sign}(z) \times
#    \begin{cases}
#    0 & |z| \le z_1 \\[2pt]
#    \sin\left( \dfrac{\pi}{2}\, \dfrac{|z| - z_1}{z_2 - z_1} \right)
#      & z_1 < |z| < z_2 \\[6pt]
#    1 & |z| \ge z_2
#    \end{cases}
#
# .. figure:: images/window_scheme.png
#    :align: center
#    :width: 85%
#
#    Figure 2: The surface window :math:`w(z)` for ``-zref1 4 -zref2 5``
#    on top of the converged density profile of our slab: the bulk-like
#    centre is removed, each surface is switched on smoothly, and the
#    bottom surface enters with opposite sign.
#
# The bulk-like region is *removed* (:math:`w = 0`), and each surface is
# switched on smoothly by the sine ramp. Inverting
# the sign of the bottom half *undoes the mirror symmetry*, so that the two
# surfaces add coherently and the statistics is doubled. For our 15 Å
# thick slab we use ``-zref1 4 -zref2 5``: the window is fully on from
# 5 Å outwards, i.e. it keeps the outer molecular layers -- the Gibbs
# dividing surfaces sit at :math:`|z| \approx 7.4` Å (5b) -- and removes
# the innermost, most bulk-like part of the film. ``-nmode`` lets you
# select the top or the bottom surface only, or a window of :math:`+1`
# everywhere (``-nmode 4``, see Question 2). Since only the product of the
# dipole and polarizability signs is physical, both scripts of this part
# use an overall sign convention that makes the free-OH peak positive, as
# in experiment.

# %%
# 5d) Computing the spectrum
# --------------------------
#
# The script ``ssvvcf_ml.py`` implements this bond model. We run it as
#
#   ``python3 ssvvcf_ml.py -f $TRAJ -cell 10 10 100 -dt 0.002 -lag 1 -max 25000 -zref1 4 -zref2 5 -nmode 1 -rc 1.0 -non_condon``
#
# where, besides the familiar ``-f``, ``-cell``, ``-dt``, ``-lag`` and
# ``-max`` (all 25 000 frames, i.e. the full 50 ps; a lag of 1 ps is
# enough for the broad stretching band):
#
# - ``-zref1`` and ``-zref2`` are the two edges of the surface window
#   :math:`w(z)` of Fig. 2, and ``-nmode 1`` selects the antisymmetric
#   window that adds the two surfaces coherently;
# - ``-rc`` is the O--O cutoff for the cross terms of 5c. With ``-rc 1.0``
#   (no two oxygens come closer than about 2.5 Å) only the self terms
#   survive; a larger value adds the intra- and intermolecular pairs up to
#   that distance (Khatib and Sulpizi restrict them to the first solvation
#   shell, 4 Å -- we return to this at the end of Exercise 6);
# - ``-non_condon`` adds a second, non-Condon spectrum as an extra column
#   of the output (explained at the end of this section).
#
# The command is stored in ``get_ssvvcf.sh`` and takes about eight
# minutes:

subprocess.run(["bash", "get_ssvvcf.sh"], cwd=workdir_ex5, check=True)

# %%
# The output follows the pattern of Part II: ``ssVVCF_cf.dat`` (the
# correlation function), ``ssVVCF_wcf.dat`` (windowed in time) and
# ``ssVVCF_ImChi2.dat`` with Im :math:`\chi^{(2)}_{xxz}` computed with the
# constant bond derivatives of Eq. (2) (column 2) and with a *non-Condon*
# correction (column 3, explained at the end of this section).
# Let's plot the Condon spectrum together with the converged reference of
# the same analysis over 1 ns, both normalized to the free-OH peak (the
# absolute scale depends on the surface area and on the parametrization):


def normalized_to_free_oh(nu, spectrum):
    """Divide by the maximum of the spectrum between 3500 and 4000 cm^-1."""
    return spectrum / spectrum[(nu > 3500) & (nu < 4000)].max()


def stretch_axes(ax):
    """Axis settings shared by the plots of the stretching region (the 1/omega
    factor makes every curve diverge at very low frequency, so the y range
    must be fixed by hand)."""
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlim(2800, 4000)
    ax.set_ylim(-2.2, 1.3)
    ax.set_xlabel(r"$\omega$ / cm$^{-1}$")


sfg_ssvvcf = np.loadtxt(f"{workdir_ex5}/ssVVCF_ImChi2.dat")
sfg_ssvvcf_ref = np.loadtxt(f"{workdir_ex5}/reference_results/slab-048_1ns_ssVVCF_ImChi2_rc1.0.dat")

fig, ax = plt.subplots(1, 1, figsize=(6, 3.4), constrained_layout=True)
ax.plot(sfg_ssvvcf[:, 0], normalized_to_free_oh(sfg_ssvvcf[:, 0], sfg_ssvvcf[:, 1]), "r-", lw=1.5, label="50 ps (ours)")
ax.plot(sfg_ssvvcf_ref[:, 0], normalized_to_free_oh(sfg_ssvvcf_ref[:, 0], sfg_ssvvcf_ref[:, 1]), "k-", lw=1, label="1 ns, reference")
stretch_axes(ax)
ax.set_ylabel(r"Im $\chi^{(2)}_{xxz}$ (normalized)")
ax.legend(fontsize=8)
plt.show()

# %%
# - The spectrum has the two hallmarks of the water/air interface: a sharp
#   **positive** peak near 3770 cm :math:`^{-1}` from the free OH groups
#   that point into the vapour, and a broad **negative** band between
#   about 3200 and 3600 cm :math:`^{-1}` from the hydrogen-bonded OH groups,
#   which on average point into the liquid. The sign is the orientation:
#   opposite signs mean opposite average directions of the OH transition
#   dipole along the surface normal.
# - Only the stretching region is meaningful in this model: the bond
#   velocities carry no information on bending or libration.
# - With the self terms only, the ssVVCF is an average over 96 OH
#   oscillators, so 50 ps already give the right band positions. The
#   *relative* intensity of the two features is what converges slowly:
#   ours overestimates the depth of the hydrogen-bonded band by about a
#   quarter of the free-OH peak. **No such calculation is finished
#   without convergence tests** -- with respect to the trajectory length
#   and the analysis parameters (we come back to this at the end of
#   Exercise 6).
#
# Non-Condon effects
# ~~~~~~~~~~~~~~~~~~
#
# The bond derivatives of Eq. (2) are constants (the Condon approximation).
# In reality they depend strongly on the hydrogen-bonding environment of
# each OH group: the transition dipole of an OH oscillator absorbing at
# 3200 cm :math:`^{-1}` (strongly hydrogen-bonded) is several times larger
# than that of a free OH at 3700 cm :math:`^{-1}` (compare the SPC/E and
# MACE-MDP spectra of Exercise 3). Ohto et al. (*J. Chem. Phys.* **143**,
# 124702 (2015)) include this dependence
# through the spectroscopic maps of Skinner and co-workers (Auer et al.,
# *Proc. Natl. Acad. Sci. USA* **104**, 14215 (2007); Auer and Skinner,
# *J. Chem. Phys.* **128**, 224511 (2008)), which relate the transition
# moments of an OH oscillator to its frequency; the spectrum is simply
# multiplied by the frequency-dependent factor
#
# .. math::
#
#    \mu'(\omega)\, \alpha'(\omega) = \left( 1.377 +
#    \frac{53.03\,(3737 - \omega)}{6932.2} \right)
#    \left( 1.271 + \frac{6.287\,(3737 - \omega)}{6932.2} \right)
#    \tag{4}
#
# with :math:`\omega` in cm :math:`^{-1}` and 3737 cm :math:`^{-1}` the
# gas-phase OH frequency. Before applying it, let's look at the factor
# itself in the stretching region:

nu_nc = np.linspace(2800, 3800, 200)
x_nc = (3737.0 - nu_nc) / 6932.2
factor_nc = (1.377 + 53.03 * x_nc) * (1.271 + 6.287 * x_nc)
fig, ax = plt.subplots(1, 1, figsize=(6, 2.6), constrained_layout=True)
ax.plot(nu_nc, factor_nc, "C1-", lw=1.5)
ax.axhline(1, color="gray", lw=0.5, ls="--")
ax.set_xlim(2800, 3800)
ax.set_xlabel(r"$\omega$ / cm$^{-1}$")
ax.set_ylabel(r"$\mu'(\omega)\, \alpha'(\omega)$")
plt.show()

# %%
# The factor is positive across the whole band and grows steeply towards
# the red: about 1.4 at the free-OH peak, 6 at 3400 cm :math:`^{-1}` and
# almost 10 at 3200 cm :math:`^{-1}` -- it rescales the spectrum without
# flipping any sign, and boosts exactly the strongly hydrogen-bonded,
# red-shifted oscillators. ``ssvvcf_ml.py -non_condon`` already wrote the
# corrected spectrum as the third column of ``ssVVCF_ImChi2.dat``, so
# taking the Condon approximation out is one plot away:

fig, ax = plt.subplots(1, 1, figsize=(6, 3.4), constrained_layout=True)
ax.plot(sfg_ssvvcf[:, 0], normalized_to_free_oh(sfg_ssvvcf[:, 0], sfg_ssvvcf[:, 1]), "C0-", lw=1, label="Condon")
ax.plot(sfg_ssvvcf[:, 0], normalized_to_free_oh(sfg_ssvvcf[:, 0], sfg_ssvvcf[:, 2]), "C1-", lw=1.5, label="non-Condon")
stretch_axes(ax)
ax.set_ylabel(r"Im $\chi^{(2)}_{xxz}$ (normalized)")
ax.legend(fontsize=8)
plt.show()

# %%
# The correction barely touches the free-OH peak -- an isolated OH
# vibrates close to the gas-phase frequency, where the maps are ~1 -- but
# it boosts the hydrogen-bonded band from about half of the free-OH peak
# to nearly twice it, and shifts its minimum from about 3510 to 3470
# cm :math:`^{-1}`: the strongly hydrogen-bonded, red-shifted oscillators
# are exactly the ones whose transition moments the maps enhance. Keep
# both curves in mind for the comparison with the parameter-free atomic
# route at the end of Exercise 6.
#
# **Questions** (answers at the end of the notebook):
#
# 1. Prove the statement of the theory section: :math:`\chi^{(2)}` vanishes
#    in any centrosymmetric medium. (Hint: start from the polarization
#    induced at second order, :math:`P^{(2)}_{\zeta} = \sum_{\eta\kappa}
#    \chi^{(2)}_{\zeta\eta\kappa} E_{\eta} E_{\kappa}`, and apply the
#    inversion operation.)
# 2. What would you expect to see with ``-nmode 4`` (window equal to +1
#    everywhere), and with ``-nmode 2`` (top surface only)? Try it if you
#    have time.

# %%
# Exercise 6: VSFG spectra from machine-learned atomic dipoles and polarizabilities
# =================================================================================
#
# The bond model of Exercise 5 is rather cheap and insightful, with many approximations.
# Exercise 3 taught us that the machine-learned dipoles
# of MACE-MDP get the intensities of liquid water right without any of
# this -- can we use them here? Not directly: the *total* dipole and
# polarizability of a slab, which is all we used in Part II, do not
# separate the two surfaces from each other and from the bulk (the window
# function of 5c must be applied *molecule by molecule*).
#
# What makes it possible is that MACE-MDP, like every atom-centred
# machine-learning model, builds the total quantities from *atomic*
# contributions: a charge :math:`q_i`, an atomic dipole
# :math:`\boldsymbol{\nu}_i` and an atomic polarizability
# :math:`\boldsymbol{\alpha}_i` for every atom, with
#
# .. math::
#
#    \mu = \sum_i \bigl( \boldsymbol{\nu}_i + q_i \mathbf{r}_i \bigr),
#    \qquad
#    \alpha = \sum_i \boldsymbol{\alpha}_i
#    \tag{5}
#
# This atomic decomposition is exactly what is needed to make Eq. (1)
# surface specific -- the idea of Litman et al. (*J. Phys. Chem. Lett.*
# **14**, 8175 (2023)), who used it to compute the converged SFG spectrum of the
# water/air interface fully from first principles. The atomic pieces are
# grouped into molecules (referencing the charge term to the oxygen of the
# molecule, so that each molecular dipole is origin independent, as in
# Exercise 3a), the surface window :math:`w(z_m)` multiplies each molecular
# dipole, and the windowed sums enter the correlation function:
#
# .. math::
#
#    \mathrm{Im}\,\chi^{(2)}_{xxz}(\omega) \propto \frac{1}{\omega}
#    \int_0^{\infty} \mathrm{d}t\, \cos(\omega t)\,
#    \Bigl\langle \dot{\tilde{\alpha}}_{xx}(t)\,
#    \dot{\tilde{\mu}}_{z}(0) \Bigr\rangle,
#    \qquad
#    \tilde{\mu} = \sum_m w(z_m)\, \tilde{\boldsymbol{\mu}}_m,
#    \quad
#    \tilde{\alpha} = \sum_m \tilde{\boldsymbol{\alpha}}_m
#    \tag{6}
#
# with :math:`\tilde{\boldsymbol{\mu}}_m = \sum_{i \in m} [\boldsymbol{\nu}_i
# + q_i (\mathbf{r}_i - \mathbf{r}_{\mathrm{O},m})]` and
# :math:`\tilde{\boldsymbol{\alpha}}_m = \sum_{i \in m}
# \boldsymbol{\alpha}_i`. No bond model, no parameters, and the
# environment dependence of the transition moments (the non-Condon
# effects) is included by construction. This is what
# ``scripts/analysis/sfg_atomic.py`` does; the price is that the atomic
# quantities must be evaluated along the whole trajectory. For our slab
# this was done beforehand -- they are part of the download -- but the
# setup is worth a look.
#
# 6a) The replay that produced the atomic quantities
# --------------------------------------------------
#
# As in Exercise 3b, the atomic quantities come from an i-PI *replay* run.
# The input is the one of Exercise 3b with three more extras:

workdir_ex6 = "part_iii/excercise_6"

with open(f"{workdir_ex6}/input_mdp.xml") as f:
    xml_ex6 = f.read()
print(re.search(r"<output.*?</output>", xml_ex6, re.DOTALL).group(0))

# %%
# and the client differs from that of Exercise 3b in one argument:
# ``has_atomic=True`` tells the socket client to also pack the atomic
# charges, dipoles and polarizabilities of Eq. (5) -- which the MACE
# calculator of the STREAM branch (see ``environment.yml``) returns
# alongside the totals -- into the extras:

with open(f"{workdir_ex6}/run-mace-mdp_ex6.py") as f:
    print(f.read())

# %%
# We do **not** launch this run here: with 144 atoms per frame the model
# takes about 2.5 s per frame on a laptop CPU, so even the 2000 frames the
# input above is restricted to would take an hour and a half, and the full
# 25 000 frames about 17 hours. (If you are curious, ``run_ex6.sh`` in the
# exercise folder launches exactly this input.) Instead we use the files
# produced by the same replay of the **whole** trajectory on a GPU
# cluster, which are part of the download:
# ``h2o.atomic_charges_0``, ``h2o.atomic_dipoles_0`` and
# ``h2o.atomic_polarizabilities_0`` in
# ``part_iii/trajectory_files/slab_lx10_ly10_lz100_n48_01/``.
#
# The atomic files have one record per frame, with one value per atom
# (charges), three per atom (dipoles) or nine per atom (polarizabilities),
# in atomic units and in the order of the atoms in the trajectory:

traj_dir = "part_iii/trajectory_files/slab_lx10_ly10_lz100_n48_01"

with open(f"{traj_dir}/h2o.atomic_charges_0") as f:
    print("".join(f.readline()[:100] + " ...\n" for _ in range(4)))

# %%
# 6b) Computing the spectrum
# --------------------------
#
# ``sfg_atomic.py`` builds the molecular dipoles and polarizabilities from
# these atomic pieces and correlates them. We call it as
#
#   ``python3 sfg_atomic.py -f $TRAJ -atq $DATA/h2o.atomic_charges_0 -atmu $DATA/h2o.atomic_dipoles_0 -atpol $DATA/h2o.atomic_polarizabilities_0 -cell 10 10 100 -dt 0.002 -lag 1 -max 25000 -zref1 4 -zref2 5 -nmode 1 -rcut 4.0 -chi xxz -prefix mdp_``
#
# The trajectory and the analysis flags are those of Exercise 5; the ones
# specific to this route are:
#
# - ``-atq``, ``-atmu`` and ``-atpol``: the three files of atomic charges,
#   dipoles and polarizabilities inspected in 6a (they carry one record
#   per frame, so no ``-skip`` is needed);
# - ``-rcut 4.0``: the O--O pair cutoff, as ``-rc`` in Exercise 5 -- here
#   we include the intra- and intermolecular cross terms up to the first
#   solvation shell from the start;
# - ``-chi xxz``: which tensor element of :math:`\chi^{(2)}` to compute;
# - ``-prefix``: the prefix of the output files, here ``mdp_``.
#
# The command is stored in ``get_sfg_atomic.sh`` and takes about ten
# minutes -- the longest analysis of this notebook. If it is taking much
# longer on your machine, you can abort it (interrupt the kernel) and
# analyse a shorter stretch of the trajectory instead: open
# ``part_iii/excercise_6/get_sfg_atomic.sh``, reduce ``-max 25000`` to,
# say, ``-max 10000`` (the first 20 ps -- the run time shrinks
# proportionally), and re-run this cell. The plotting cells below then
# work unchanged, just with noisier curves:

subprocess.run(["bash", "get_sfg_atomic.sh"], cwd=workdir_ex6, check=True)

# %%
# Our 50 ps result against the converged reference of the same slab and
# the same analysis parameters, averaged over 1 ns of dynamics. We show
# only the stretching region: at lower frequencies the SFG response of the
# bending band carries quadrupolar contributions that we do not discuss
# here, and the librational region is not meaningfully described at this
# level:

sfg_at = np.loadtxt(f"{workdir_ex6}/mdp_SFG_ImChi2_xxz.dat")
sfg_at_1ns = np.loadtxt(f"{workdir_ex6}/reference_results/slab-048_1ns_SFG_ImChi2_xxz_rc4.0.dat")

fig, ax = plt.subplots(1, 1, figsize=(6, 3.4), constrained_layout=True)
ax.plot(sfg_at[:, 0], sfg_at[:, 1], "r-", lw=1, label="50 ps (ours)")
ax.plot(sfg_at_1ns[:, 0], sfg_at_1ns[:, 1], "k-", lw=1.5, label="1 ns, reference")
ax.axhline(0, color="gray", lw=0.5)
ax.set_xlim(2700, 4000)
stretch = (sfg_at[:, 0] > 2700) & (sfg_at[:, 0] < 4000)
top = 1.15 * max(np.abs(sfg_at[stretch, 1]).max(),
                 np.abs(sfg_at_1ns[stretch, 1]).max())
ax.set_ylim(-top, top)
ax.set_xlabel(r"$\omega$ / cm$^{-1}$")
ax.set_ylabel(r"Im $\chi^{(2)}_{xxz}$ (arb. units)")
ax.legend(fontsize=8)
plt.show()

# %%
# - The spectrum shows the same positive free-OH peak (3770
#   cm :math:`^{-1}`) and negative hydrogen-bonded band as the
#   velocity-based spectra of Exercise 5, the latter now much stronger and
#   centred lower, near 3450 cm :math:`^{-1}`.
# - The free-OH peak of the atomic route also develops a shoulder around
#   3680-3700 cm :math:`^{-1}`, related to the intramolecular coupling
#   between the two OH oscillators of a molecule -- a feature the bond
#   model of Exercise 5 cannot capture.
# - Our 50 ps curve reproduces every feature and every sign, but it is
#   much noisier than its Exercise 5 counterpart: the SFG signal comes
#   from the few molecules in the interfacial layers only, a small
#   fraction of an already small system. The velocity-based approach
#   converges much faster -- particularly when only the self terms of the
#   correlation function are kept, as in 5d, where the average runs over
#   96 equivalent OH oscillators.
#
# **A word on convergence.** A publishable SFG calculation must be
# converged in three separate directions: the *slab thickness* (thick
# enough for a bulk-like centre -- ours, as 5b showed, barely has one),
# the *cross-section area* (the number of interfacial molecules sampled
# per frame) and the *simulation time*. For neat water, spectra like the
# ones above need of the order of 1--2 ns to converge -- twenty to forty
# times our trajectory. For interfaces that contain solutes the cost grows
# dramatically: on top of the vibrational statistics, the interfacial
# *distributions* of the solute molecules or ions must themselves
# converge, which requires much longer (and larger) simulations (see
# Litman et al., *Nat. Chem.* **16**, 644 (2024) for an example on
# electrolyte solutions).
#
# 6c) Bond model versus machine-learned atomic decomposition
# ----------------------------------------------------------
#
# Finally, the two routes side by side in the stretching region,
# normalized to the free-OH peak. We use the converged 1 ns references of
# both exercises, so that the differences we discuss are the *models* and
# not the statistics of 6b. For the bond model we show the non-Condon
# spectrum at both cutoffs: ``-rc 1.0`` (self terms only, as in 5d) and
# ``-rc 4.0`` (first solvation shell -- the spectrum you would get by
# re-running ``get_ssvvcf.sh`` with that cutoff):

sfg_ssvvcf_rc4 = np.loadtxt(f"{workdir_ex5}/reference_results/slab-048_1ns_ssVVCF_ImChi2_rc4.0.dat")

fig, ax = plt.subplots(1, 1, figsize=(6, 3.4), constrained_layout=True)
ax.plot(sfg_ssvvcf_ref[:, 0], normalized_to_free_oh(sfg_ssvvcf_ref[:, 0], sfg_ssvvcf_ref[:, 2]), "C0-", lw=1, label="ssVVCF, non-Condon, rc 1.0")
ax.plot(sfg_ssvvcf_rc4[:, 0], normalized_to_free_oh(sfg_ssvvcf_rc4[:, 0], sfg_ssvvcf_rc4[:, 2]), "C1-", lw=1, label="ssVVCF, non-Condon, rc 4.0")
ax.plot(sfg_at_1ns[:, 0], normalized_to_free_oh(sfg_at_1ns[:, 0], sfg_at_1ns[:, 1]), "k-", lw=1.5, label="MACE-MDP atomic")
stretch_axes(ax)
ax.set_ylabel(r"Im $\chi^{(2)}_{xxz}$ (normalized)")
ax.legend(fontsize=8)
plt.show()

# %%
# - All three agree on the signs of the two features -- positive free OH
#   near 3770 cm :math:`^{-1}`, negative hydrogen-bonded band -- and,
#   once normalized, roughly on the width of the band: they describe the
#   same dynamics, and the sign is geometry.
# - The  intermolecular cross terms (``-rc 4.0``) widens  the
#   hydrogen-bonded band of the bond model 
#   and red-shift its minimum.  
# - The machine-learned atomic quantities deepen the hydrogen-bonded band further,
#   and add the 3680-3700 cm :math:`^{-1}`
#   shoulder. This is what the ML model contains on top of the
#   empirical maps (intermolecular polarization, charge transfer and the
#   full environment dependence of the transition moments) -- the same
#   lesson as in Exercise 3, now at the interface.
#
# Take-home messages of Part III
# ------------------------------
#
# - SFG probes the few molecular layers where the inversion symmetry is
#   broken; the *sign* of Im :math:`\chi^{(2)}` gives the average
#   orientation of the vibrating groups along the surface normal.
# - A slab has two mirror-image surfaces: the surface window removes the
#   bulk and inverts the bottom surface so that both add coherently.
# - The ssVVCF bond model gives the stretching region from positions
#   alone, with fixed, parametrized transition moments; non-Condon
#   corrections and couplings between oscillators must be added by hand.
# - Machine-learned *atomic* dipoles and polarizabilities make the
#   parameter-free route of Part II surface specific -- at the cost of
#   evaluating them along the whole trajectory.
# - SFG spectra converge much more slowly than bulk spectra -- nanoseconds
#   rather than tens of picoseconds: only the interfacial molecules
#   contribute, and their signal is a difference of nearly cancelling
#   orientations. Window
#   parameters and coupling cutoffs must be tested as carefully as the
#   trajectory length.

# %%
# Appendix
# ========
#
# The derivation referred to in Exercise 5, collected here to keep the
# exercise uncluttered.
#
# A.1 The rotation matrices of the ssVVCF bond model
# --------------------------------------------------
#
# Eq. (2) of Exercise 5 hides the rotations that connect the frame of each
# OH bond to the laboratory frame. Written out, the bond contributions are
#
# .. math::
#
#    \dot{\mu}^{\,\mathrm{lab}}_{i,\mathrm{bond}} =
#    \mathbf{D}^{\mathsf{T}}_{i,\mathrm{bond}}
#    \left( \frac{\partial \mu^{\,\mathrm{bond}}}{\partial r}\,
#    \dot{r}_{i,\mathrm{bond}} \right)
#
# .. math::
#
#    \dot{\alpha}^{\,\mathrm{lab}}_{i,\mathrm{bond}} =
#    \mathbf{D}^{\mathsf{T}}_{i,\mathrm{bond}}
#    \left( \frac{\partial \alpha^{\,\mathrm{bond}}}{\partial r}\,
#    \dot{r}_{i,\mathrm{bond}} \right)
#    \mathbf{D}_{i,\mathrm{bond}}
#
# where :math:`\mathbf{D}_{i,\mathrm{bond}}` rotates from the lab frame to
# the frame of the bond: its :math:`z'` axis points along the OH bond and
# the transverse axes are fixed by the molecular plane, so the matrix is
# built at every frame from the positions of the three atoms of the
# molecule. (Khatib and Sulpizi write this same rotation as a product
# :math:`\mathbf{D}_{m,i}\mathbf{D}_{b,i,\mathrm{bond}}` of a
# molecule-to-lab and a bond-to-molecule matrix, with the molecular
# :math:`z`-axis along the H--O--H bisector -- see Sec. 2 of their SI; the
# single matrix used here is that product.) The polarizability, a rank-2
# tensor, is rotated from both sides. Two
# approximations lead from the exact time derivative to these expressions:
#
# - a first-order Taylor expansion of the bond dipole and polarizability
#   in the OH bond length :math:`r` around its equilibrium value, which
#   replaces the bond-frame quantities by the constant derivatives
#   :math:`\partial \mu^{\,\mathrm{bond}} / \partial r` and
#   :math:`\partial \alpha^{\,\mathrm{bond}} / \partial r`
#   times :math:`\dot{r}_{i,\mathrm{bond}}`;
# - the product rule applied to
#   :math:`\mathbf{D}^{\mathsf{T}}_{i,\mathrm{bond}}\,
#   \mu^{\,\mathrm{bond}}` also
#   produces a term with :math:`\dot{\mathbf{D}}_{i,\mathrm{bond}}`.
#   Because the OH stretch
#   (:math:`\sim 3400` cm :math:`^{-1}`) is much faster than the molecular
#   reorientation (libration, below :math:`\sim 1000` cm :math:`^{-1}`),
#   these rotational terms contribute only far below the stretching band
#   and are neglected.
#
# The scripts of Part III build :math:`\mathbf{D}_{i,\mathrm{bond}}` from
# the positions of the three atoms of
# each molecule at every frame, and take :math:`\dot{r}_{i,\mathrm{bond}}`
# from finite differences of consecutive frames.

# %%
# References
# ----------
#
# - N. Gönnheimer, K. Reuter, V. Kapil, and J. T. Margraf, "MACE-MDP: A
#   General Dipole and Polarizability Model for Organic Molecules and
#   Materials", ChemRxiv (2026).
#   `DOI:10.26434/chemrxiv.15000716/v2
#   <https://chemrxiv.org/doi/full/10.26434/chemrxiv.15000716/v2>`_
# - V. Kapil, D. P. Kovács, G. Csányi, and A. Michaelides, "First-principles
#   spectroscopy of aqueous interfaces using machine-learned electronic and
#   quantum nuclear effects", *Faraday Discuss.* **249**, 50-68 (2024).
#   `DOI:10.1039/D3FD00113J <https://doi.org/10.1039/D3FD00113J>`_
# - A. Morita, *Theory of Sum Frequency Generation Spectroscopy*,
#   Lecture Notes in Chemistry 97, Springer, Singapore (2018).
#   `DOI:10.1007/978-981-13-1607-4 <https://doi.org/10.1007/978-981-13-1607-4>`_
# - R. Khatib, E. H. G. Backus, M. Bonn, M.-J. Perez-Haro, M.-P. Gaigeot,
#   and M. Sulpizi, "Water orientation and hydrogen-bond structure at the
#   fluorite/water interface", *Sci. Rep.* **6**, 24287 (2016).
#   `DOI:10.1038/srep24287 <https://doi.org/10.1038/srep24287>`_
# - R. Khatib and M. Sulpizi, "Sum Frequency Generation Spectra from
#   Velocity-Velocity Correlation Functions", *J. Phys. Chem. Lett.* **8**,
#   1310-1314 (2017).
#   `DOI:10.1021/acs.jpclett.7b00207 <https://doi.org/10.1021/acs.jpclett.7b00207>`_
# - T. Ohto, K. Usui, T. Hasegawa, M. Bonn, and Y. Nagata, "Toward ab initio
#   molecular dynamics modeling for sum-frequency generation spectra; an
#   efficient algorithm based on surface-specific velocity-velocity
#   correlation function", *J. Chem. Phys.* **143**, 124702 (2015).
#   `DOI:10.1063/1.4931106 <https://doi.org/10.1063/1.4931106>`_
# - Y. Litman, J. Lan, Y. Nagata, and D. M. Wilkins, "Fully First-Principles
#   Surface Spectroscopy with Machine Learning", *J. Phys. Chem. Lett.*
#   **14**, 8175-8182 (2023).
#   `DOI:10.1021/acs.jpclett.3c01989 <https://doi.org/10.1021/acs.jpclett.3c01989>`_
# - Y. Litman, K.-Y. Chiang, T. Seki, Y. Nagata, and M. Bonn, "Surface
#   stratification determines the interfacial water structure of simple
#   electrolyte solutions", *Nat. Chem.* **16**, 644-650 (2024).
#   `DOI:10.1038/s41557-023-01416-6 <https://doi.org/10.1038/s41557-023-01416-6>`_
# - The MACE-MDP models used here and their SFG extension are described in
#   Litman et al., *in preparation* (see ``MODELS/README.md``).

# %%
# Answers to the questions
# ========================
#
# Exercise 5
# ----------
#
# 1. In a centrosymmetric medium, inverting the coordinate system leaves
#    the material -- and therefore its response tensors -- unchanged. Under
#    inversion all vectors flip sign: the fields become
#    :math:`-E_{\eta}, -E_{\kappa}` and the induced polarization becomes
#    :math:`-P^{(2)}_{\zeta}`, while :math:`\chi^{(2)}` (a material
#    property) stays the same. Applying the inversion to
#    :math:`P^{(2)}_{\zeta} = \sum_{\eta\kappa} \chi^{(2)}_{\zeta\eta\kappa}
#    E_{\eta} E_{\kappa}` therefore gives
#    :math:`-P^{(2)}_{\zeta} = \sum_{\eta\kappa}
#    \chi^{(2)}_{\zeta\eta\kappa} (-E_{\eta})(-E_{\kappa}) =
#    +\sum_{\eta\kappa} \chi^{(2)}_{\zeta\eta\kappa} E_{\eta} E_{\kappa}`:
#    the two signs are only compatible if
#    :math:`\chi^{(2)}_{\zeta\eta\kappa} = 0`. (The rank-2/rank-1 argument
#    of the main text says the same: the product of an even and an odd
#    response cannot survive inversion.)
# 2. With ``-nmode 4`` the window is +1 everywhere: the two surfaces enter
#    with opposite signs and cancel, and the bulk adds noise -- the spectrum
#    averages to zero (a useful sanity check of the surface specificity;
#    with a finite trajectory you get noise around zero). With ``-nmode 2``
#    only the top surface contributes: the same spectrum as with both
#    surfaces, but with half the signal and correspondingly larger
#    statistical noise. The bottom surface alone with a +1 window would
#    give the mirror image (all signs flipped) -- which is why the window
#    is -1 there.
