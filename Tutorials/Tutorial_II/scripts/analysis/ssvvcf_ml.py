#!/usr/bin/env python3
"""
ssvvcf_ml.py — resonant SFG response of a water slab from the
surface-specific velocity-velocity correlation function (ssVVCF).

Written with a focus on CLARITY OVER EFFICIENCY: the whole trajectory is
loaded into memory and every step is spelled out with explicit loops.

Method
------
Sum-frequency generation (SFG) is surface specific: the second-order
susceptibility chi(2) vanishes in a centrosymmetric bulk, so only the
interfacial molecules contribute. Instead of explicit dipole and
polarizability trajectories, the ssVVCF method models the TIME DERIVATIVES
of each OH bond's dipole and polarizability from the OH stretching
velocity, with fixed bond-frame derivatives taken from
Khatib et al., Sci. Rep. 6, 24287 (2016):

    dmu/dr   = (-0.15, 0, 2.0) D/Angstrom      (z' along the OH bond)
    dalpha/dr = diag(0.40, 0.53, 1.56) A^2/A   (H2O; molecular frame)

For every OH bond a molecular frame is built (z' along O->H, y' in the
molecular plane, x' = y' x z'), the bond stretching velocity v_z' is
extracted, and

    mu_dot    = R^T (dmu/dr    * v_z')          (lab frame, vector)
    alpha_dot = R^T (dalpha/dr * v_z') R        (lab frame, tensor)

where R is the lab->molecule rotation (rows = molecular axes). A smooth
SURFACE WINDOW factor w(z_O), +1 near the top interface and -1 near the
bottom one (so the two opposite-oriented interfaces add rather than
cancel), multiplies the dipole. The correlation function accumulated is

    cf(t) = sum over molecule pairs and bonds of
            w * [ mu_dot_z(0) (alpha_dot_xx + alpha_dot_yy)(t)
                + mu_dot_z(t) (alpha_dot_xx + alpha_dot_yy)(0) ]

over an O-O pair list within a cutoff -rc (the default 0.1 A keeps only
each molecule with itself, i.e. the purely intramolecular response; a large
-rc adds the cross terms). Because BOTH factors are time derivatives, the
imaginary part of chi(2) is the COSINE transform of the (windowed) cf
divided by the angular frequency:

    Im chi(2)(nu)  proportional to  +(1/omega) integral dt cf(t) cos(omega t)

The + sign is the analytic prefactor of
chi(2) = (i omega/kT) integral <alpha(t) mu(0)> exp(i omega t) dt after two
integrations by parts (only the RELATIVE sign of mu and alpha is physical);
it makes the free-OH peak of the water surface positive, as in experiment.
The transform is evaluated by default with a zero-padded real FFT (fast);
pass -ft direct for the explicit cosine sum on a fixed grid (the textbook
formula, slow).

Input
-----
An ascii xyz trajectory (Angstrom) of a WATER slab, whole along z (the
slab must not be split across the z boundary). Every frame is first
translated so that the geometric center of the O atoms sits at z = 0; the
surface window is measured from that center. Velocities are backward
finite differences of consecutive (centered) frames with minimum-image
folding, so wrapped coordinates in x/y are fine. Only H2O is supported:
the script stops on anything that is not a 2-hydrogen water.

Non-Condon correction (optional, -non_condon)
---------------------------------------------
The bond-frame derivatives above are constants (Condon approximation).
Ohto et al., J. Chem. Phys. 143, 124702 (2015) account for their
dependence on the hydrogen-bonding environment through the OH frequency,
multiplying the spectrum by the frequency-dependent transition moments of
the spectroscopic maps of Auer and Skinner (nu in cm^-1):

    mu'(nu)    = 1.377 + 53.03 (3737 - nu) / 6932.2
    alpha'(nu) = 1.271 + 6.287 (3737 - nu) / 6932.2

With -non_condon the product mu'(nu) alpha'(nu) is applied to Im chi(2)
and written as an extra column (the Condon result is kept).

Output (all names take the optional -prefix)
--------------------------------------------
  ssVVCF_cf.dat      time(fs), cf(t) / (number of processed frames)
  ssVVCF_wcf.dat     same, apodized with sin^2(pi(i-N)/2N) (1 at t=0, 0 at t=lag)
  ssVVCF_ImChi2.dat  wavenumber(cm^-1), Im chi(2) = +(1/omega) * cosine
                     transform of the windowed cf [, non-Condon corrected]

Example
-------
  python3 ssvvcf_ml.py -f traj.xyz -cell 10 10 100 -dt 0.002 -lag 1 \
                       -max 25000 -zref1 4 -zref2 5 -rc 1.0 -non_condon
"""

import argparse
import sys
import time

import numpy as np

C_CM_PER_FS = 2.99792458e-5   # speed of light in cm/fs (time unit of the transform)
D2EA = 0.2081943             # Debye -> e*Angstrom

# Bond-frame response derivatives for the OH bond of H2O (molecular frame,
# z' along the bond), Khatib et al., Sci. Rep. 6, 24287 (2016).
DMU_DR = np.array([-0.15, 0.0, 2.0]) * D2EA      # e (from D/Angstrom)
DALPHA_DR = np.diag([0.40, 0.53, 1.56])          # Angstrom^2 per Angstrom


def progress(done, total, label, t_start):
    """One-line progress indicator on stderr (percentage, rate, ETA)."""
    stride = max(1, total // 200)
    if done % stride and done != total:
        return
    elapsed = time.time() - t_start
    rate = done / elapsed if elapsed > 0 else 0.0
    eta = (total - done) / rate if rate > 0 else 0.0
    sys.stderr.write(f"\r# {label}: {done}/{total} ({100.0 * done / total:3.0f}%)"
                     f"  {rate:,.0f}/s  ETA {eta:4.0f} s ")
    if done == total:
        sys.stderr.write("\n")
    sys.stderr.flush()


def read_xyz_frames(filename, max_frames):
    """Read up to max_frames frames of an ascii xyz trajectory."""
    symbols = []
    frames = []
    t_start = time.time()
    with open(filename) as fh:
        while len(frames) < max_frames:
            natoms_line = fh.readline()
            if natoms_line.strip() == "":
                break
            natoms = int(natoms_line.split()[0])
            fh.readline()                       # comment line, ignored
            frame = np.empty((natoms, 3))
            frame_symbols = []
            for i in range(natoms):
                tokens = fh.readline().split()
                frame_symbols.append(tokens[0])
                frame[i, :] = [float(tokens[1]), float(tokens[2]), float(tokens[3])]
            if not symbols:
                symbols = frame_symbols
            elif len(frame_symbols) != len(symbols):
                sys.exit("Error: the number of atoms changed along the trajectory.")
            frames.append(frame)
            progress(len(frames), max_frames, f"reading {filename}", t_start)
    if not frames:
        sys.exit(f"Error: could not read any frame from {filename}")
    return symbols, np.array(frames)


def minimum_image(displacement, cell):
    """Fold displacement vector(s) to the minimum image (orthorhombic cell)."""
    return displacement - cell * np.rint(displacement / cell)


def cosine_transform(time_fs, data, numax, method="fft", dnu=5.0):
    """Cosine transform  spectrum(nu) = sum_t data(t) cos(omega t) * dt
    on a wavenumber grid in cm^-1 (omega = 2*pi*c*nu). The t = 0 term enters
    with half weight (trapezoidal rule): the one-sided integral is half of
    the two-sided transform, in which t = 0 is counted once. A full weight
    would add the constant dt*data(0)/2 to every frequency.

    method="fft" (default): evaluate the sum for every frequency at once
        with a real FFT — Re(rfft(x, n))[k] is exactly
        sum_t x_t cos(2*pi*k*t/n). The data is zero-padded so the
        frequency spacing is at most dnu cm^-1; padding is harmless (the
        windowed cf has decayed to zero by t = lag) and only interpolates
        the spectrum on a finer grid.
    method="direct": the same sum evaluated literally, one cosine at a
        time, on the fixed grid nu = 1, 1+dnu, ... — the textbook formula,
        transparent but O(N_nu * N_t) slow.

    The two methods give identical numbers at shared frequencies and
    differ only in their frequency grids. Returns (nu, spectrum).
    """
    t = time_fs                                  # fs (the spectra carry one
    dt = t[1] - t[0]                             # factor of fs from the dt)
    data = np.array(data, dtype=float, copy=True)
    data[0] *= 0.5                               # trapezoidal weight of t = 0
    if method == "fft":
        # smallest power of two whose FFT grid is at least dnu-fine
        n_min = max(len(t), int(np.ceil(1.0 / (C_CM_PER_FS * dnu * dt))))
        n_fft = 2 ** int(np.ceil(np.log2(n_min)))
        nu = np.fft.rfftfreq(n_fft, d=dt) / C_CM_PER_FS     # cm^-1
        spectrum = np.real(np.fft.rfft(data, n=n_fft, axis=0)) * dt
        keep = (nu > 0.0) & (nu < numax)     # drop nu = 0, like the direct grid
        return nu[keep], spectrum[keep]
    nu = np.arange(1.0, numax, dnu)              # cm^-1 (start at 1: avoid nu=0)
    omega = 2.0 * np.pi * C_CM_PER_FS * nu       # rad/fs
    spectrum = np.cos(np.outer(omega, t)) @ data * dt
    return nu, spectrum


def compute_topology(pos, indexes_O, indexes_H, cell, cutoff):
    """Assign every H to its nearest O (minimum-image distance).

    Returns a list of (O_index, H1_index, H2_index) per oxygen (water
    only). Raises ValueError if an H is farther than `cutoff` from every O,
    or if any O does not end up with exactly two hydrogens.
    """
    bonded_H = [[] for _ in indexes_O]
    posO = pos[indexes_O]                                    # (n_O, 3)
    for iH in indexes_H:                                     # in file order
        d = minimum_image(pos[iH] - posO, cell)              # H -> every O
        dist = np.linalg.norm(d, axis=1)
        closest = int(np.argmin(dist))
        if dist[closest] > cutoff:
            raise ValueError(
                f"H atom {iH} is {dist[closest]:.2f} A from the closest O "
                f"(> topology cutoff {cutoff} A)")
        bonded_H[closest].append(iH)
    for iO, hs in enumerate(bonded_H):
        if len(hs) != 2:
            raise ValueError(
                f"O atom {indexes_O[iO]} has {len(hs)} bonded H "
                "(only H2O is supported in this tutorial script)")
    return [(indexes_O[iO], hs[0], hs[1]) for iO, hs in enumerate(bonded_H)]


def window_factor(z, nmode, z_ref1, z_ref2):
    """Smooth surface window from the oxygen z (slab centered at z = 0).

    nmode 1: both interfaces (top +1, bottom -1 so the two opposite-oriented
             surfaces ADD in chi(2) instead of cancelling)
    nmode 2: top interface only;  nmode 3: bottom only;  nmode 4: bulk (+1
             everywhere — chi(2) then vanishes on average, a useful check).
    Between z_ref1 and z_ref2 the factor switches on smoothly with a sine.
    """
    if nmode == 4:
        return 1.0
    ramp = np.pi / 2.0 / (z_ref2 - z_ref1)
    if z > z_ref2:
        return 1.0 if nmode in (1, 2) else 0.0
    elif z > z_ref1:
        return np.sin(ramp * (z - z_ref1)) if nmode in (1, 2) else 0.0
    elif z * z < z_ref1 * z_ref1:                 # |z| < z_ref1: bulk region
        return 0.0
    elif -z_ref2 < z < -z_ref1:
        # note the argument is negative here: the factor ramps 0 -> -1
        return np.sin(ramp * (z + z_ref1)) if nmode in (1, 3) else 0.0
    elif z < -z_ref2:
        return -1.0 if nmode in (1, 3) else 0.0
    return 0.0                                    # |z| exactly = z_ref1


def bond_rates(rO, vO, rH, vH, rPartner):
    """mu_dot and alpha_dot of one OH bond in the lab frame.

    Builds the molecular frame (z' along O->H, y' in the molecular plane
    defined with the partner H, x' = y' x z'), extracts the bond stretching
    velocity v_z' = e_z' . (vH - vO) — only the stretch carries signal in
    this model — and rotates the bond-frame derivatives back to the lab:
    vectors with R^T, rank-2 tensors as R^T D R (R rows = molecular axes).
    All positions must already be minimum-imaged to the oxygen.
    """
    ez = rH - rO
    ez = ez / np.linalg.norm(ez)
    in_plane = rPartner - rO
    in_plane = in_plane - np.dot(in_plane, ez) * ez     # project out the bond axis
    ey = in_plane / np.linalg.norm(in_plane)
    ex = np.cross(ey, ez)
    ex = ex / np.linalg.norm(ex)
    R = np.array([ex, ey, ez])                          # lab -> molecule

    v_stretch = np.dot(ez, vH - vO)                     # = (R (vH-vO))_z'
    mu_dot = R.T @ (DMU_DR * v_stretch)
    alpha_dot = R.T @ (DALPHA_DR * v_stretch) @ R
    return mu_dot, alpha_dot


def non_condon_factor(nu):
    """Frequency-dependent transition dipole and polarizability of an OH
    oscillator (spectroscopic maps of Auer and Skinner as used by Ohto et
    al., J. Chem. Phys. 143, 124702 (2015)), relative to the gas-phase
    values; nu in cm^-1. Multiplies the Condon spectrum."""
    dmu = 1.377 + 53.03 * (3737.0 - nu) / 6932.2
    dalpha = 1.271 + 6.287 * (3737.0 - nu) / 6932.2
    return dmu * dalpha


def pair_list(posO, cell, cutoff):
    """O-O pairs (iO1 <= iO2) with minimum-image distance < cutoff.

    Self pairs (iO1 == iO2, distance 0) are always included: they carry the
    intramolecular response. Returns two integer arrays (first, second).
    """
    d = minimum_image(posO[:, None, :] - posO[None, :, :], cell)
    dist = np.linalg.norm(d, axis=2)
    iO1, iO2 = np.where(np.triu(dist < cutoff))         # upper triangle incl. diagonal
    return iO1, iO2


def main():
    t_run = time.time()
    parser = argparse.ArgumentParser(
        description="ssVVCF SFG response of a water slab (bond-model, xxz+yyz).",
        allow_abbrev=False)
    parser.add_argument("-f", required=True, metavar="FILE", dest="traj",
                        help="xyz position trajectory of the slab (Angstrom)")
    parser.add_argument("-cell", nargs=3, type=float, required=True,
                        metavar=("LX", "LY", "LZ"),
                        help="orthorhombic cell lengths in Angstrom")
    parser.add_argument("-dt", required=True, type=float, metavar="PS",
                        help="time step between frames, in ps")
    parser.add_argument("-lag", type=float, default=1.0, metavar="PS",
                        help="correlation length, in ps (default 1.0)")
    parser.add_argument("-max", required=True, type=int, metavar="N",
                        dest="nframes_max", help="maximum number of frames to use")
    parser.add_argument("-rc", "-rcut", type=float, default=0.1, metavar="A",
                        dest="rc", help="O-O pair-list cutoff in Angstrom (default 0.1: "
                             "self pairs / intramolecular only)")
    parser.add_argument("-zref1", type=float, default=2.0, metavar="A",
                        help="surface window: |z| below which the factor is 0")
    parser.add_argument("-zref2", type=float, default=20000.0, metavar="A",
                        help="surface window: |z| above which the factor is +-1")
    parser.add_argument("-nmode", type=int, default=1, choices=[1, 2, 3, 4],
                        help="1 both interfaces, 2 top, 3 bottom, 4 bulk")
    parser.add_argument("-nfreq_topo", type=int, default=1, metavar="N",
                        help="recompute the H->O topology every N frames")
    parser.add_argument("-numax", type=float, default=4000.0, metavar="CM1",
                        help="upper edge of the spectrum wavenumber grid")
    parser.add_argument("-ft", choices=["fft", "direct"], default="fft",
                        help="spectrum evaluation: zero-padded real FFT (default) or "
                             "the explicit cosine sum (textbook formula, slow)")
    parser.add_argument("-non_condon", action="store_true",
                        help="also write Im chi(2) multiplied by the frequency-dependent "
                             "transition moments mu'(nu) alpha'(nu) of Ohto et al.")
    parser.add_argument("-prefix", default="", metavar="STR",
                        help="prefix for the output file names")
    args = parser.parse_args()

    dt = 1000.0 * args.dt      # ps -> fs, everything internal is fs
    lag = 1000.0 * args.lag
    ncf_step = int(lag / dt)
    if args.zref1 > args.zref2:
        sys.exit("Error: -zref1 must not exceed -zref2.")
    if ncf_step < 10:
        sys.exit("Error: lag/dt must be at least 10.")
    if args.nframes_max < ncf_step:
        sys.exit("Error: -max must be at least lag/dt frames.")
    if args.nfreq_topo < 1:
        sys.exit("Error: -nfreq_topo must be >= 1.")
    cell = np.array(args.cell)

    # ------------------------------------------------------------------
    # Read the trajectory (one extra frame beyond -max is read; it only
    # affects the 1/nframes normalization below) and identify atoms.
    # ------------------------------------------------------------------
    symbols, coords = read_xyz_frames(args.traj, args.nframes_max + 1)
    nread = len(coords)
    indexes_O = [i for i, s in enumerate(symbols) if s == "O"]
    indexes_H = [i for i, s in enumerate(symbols) if s in ("H", "D")]
    unknown = {s for s in symbols if s not in ("O", "H", "D")}
    if unknown:
        sys.exit(f"Error: only water (O,H,D) is supported; found {sorted(unknown)}")
    if len(indexes_H) != 2 * len(indexes_O):
        sys.exit("Error: this is not pure water (need exactly 2 H per O).")
    n_O = len(indexes_O)
    print(f"# Read {nread} frames: {n_O} water molecules.")

    # Center every frame once: shift z so the geometric (unweighted) center
    # of the O atoms is at z = 0. All frames then share one origin, which
    # the surface window and the velocity finite differences both rely on.
    coords[:, :, 2] -= coords[:, indexes_O, 2].mean(axis=1)[:, None]

    # Velocities: backward minimum-image finite difference, Angstrom/fs.
    # Frame 0 has no velocity (it stays zero and contributes nothing).
    velocities = np.zeros_like(coords)
    velocities[1:] = minimum_image(coords[1:] - coords[:-1], cell) / dt

    # ------------------------------------------------------------------
    # Bond model, frame by frame. For every water molecule and each of its
    # two OH bonds we store:
    #   mu_z:   window factor * lab-frame mu_dot_z      (factor on the dipole
    #           ONLY — chi(2) needs the factor once per mu*alpha product)
    #   pol_xx, pol_yy: lab-frame alpha_dot_xx / _yy    (no window factor)
    # Frame 0 stays zero (no velocity yet). The H->O topology is recomputed
    # every -nfreq_topo frames; in between the previous one is kept.
    # ------------------------------------------------------------------
    mu_z = np.zeros((nread, n_O, 2))
    pol_xx = np.zeros((nread, n_O, 2))
    pol_yy = np.zeros((nread, n_O, 2))

    try:
        topology = compute_topology(coords[0], indexes_O, indexes_H, cell, 1.35)
    except ValueError as err:
        sys.exit(f"Error: cannot compute the topology of the first frame: {err}")

    t_start = time.time()
    for frame in range(1, nread):
        progress(frame, nread - 1, "bond model", t_start)
        # (frame + 1) is the 1-based frame counter.
        if (frame + 1) % args.nfreq_topo == 0:
            try:
                topology = compute_topology(coords[frame], indexes_O,
                                            indexes_H, cell, 1.35)
            except ValueError as err:
                print(f"# Warning: topology failed at frame {frame + 1} "
                      f"({err}); keeping the previous one.")
        pos, vel = coords[frame], velocities[frame]
        for iO, (iOa, iH1, iH2) in enumerate(topology):
            rO, vO = pos[iOa], vel[iOa]
            factor = window_factor(rO[2], args.nmode, args.zref1, args.zref2)
            # Use the H images closest to the O so molecules split across
            # the periodic boundary still get a correct bond frame.
            rH1 = rO + minimum_image(pos[iH1] - rO, cell)
            rH2 = rO + minimum_image(pos[iH2] - rO, cell)
            for b, (rH, vH, rPartner) in enumerate(
                    [(rH1, vel[iH1], rH2), (rH2, vel[iH2], rH1)]):
                mu_dot, alpha_dot = bond_rates(rO, vO, rH, vH, rPartner)
                mu_z[frame, iO, b] = factor * mu_dot[2]
                pol_xx[frame, iO, b] = alpha_dot[0, 0]
                pol_yy[frame, iO, b] = alpha_dot[1, 1]

    # ------------------------------------------------------------------
    # Accumulate the correlation function. Every frame serves once as the
    # time origin t = 0 and is correlated backward in time with the
    # preceding frames. The pair list is rebuilt on the origin
    # frame's geometry. For each pair the two bonds are correlated
    # bond-by-bond (bond 1 of molecule 1 with bond 1 of molecule 2, etc.),
    # in both pair directions; for self pairs the reverse direction would
    # duplicate the forward one and is skipped, so every ordered pair
    # enters exactly once.
    # ------------------------------------------------------------------
    cf = np.zeros(ncf_step)
    last_origin = min(nread, args.nframes_max)     # frames beyond -max: no origins
    t_start = time.time()
    for origin in range(last_origin):
        progress(origin + 1, last_origin, "correlating", t_start)
        iO1, iO2 = pair_list(coords[origin][indexes_O], cell, args.rc)
        nonself = iO1 != iO2
        for k in range(min(ncf_step - 1, origin) + 1):
            past = origin - k
            # forward direction: dipole of molecule 1, polarizability of molecule 2
            cf[k] += np.sum(
                mu_z[origin][iO1] * (pol_xx[past] + pol_yy[past])[iO2]
                + mu_z[past][iO1] * (pol_xx[origin] + pol_yy[origin])[iO2])
            # reverse direction (dipole of molecule 2), self pairs skipped
            cf[k] += np.sum(
                (mu_z[origin][iO2] * (pol_xx[past] + pol_yy[past])[iO1]
                 + mu_z[past][iO2] * (pol_xx[origin] + pol_yy[origin])[iO1])[nonself])

    # Normalize by the total number of processed frames (a single overall
    # scale, not per lag).
    cf /= nread

    # Apodization: sin^2(pi(i-N)/2N), ~1 at t=0 and exactly 0 at t=lag.
    i = np.arange(1, ncf_step + 1)
    window = np.sin(np.pi * (i - ncf_step) / (2.0 * ncf_step)) ** 2
    wcf = cf * window

    # ------------------------------------------------------------------
    # Write the correlation functions and the spectrum.
    # ------------------------------------------------------------------
    time_fs = dt * np.arange(ncf_step)
    name = args.prefix + "ssVVCF"
    np.savetxt(f"{name}_cf.dat", np.column_stack([time_fs, cf]),
               header="time (fs), cf")
    np.savetxt(f"{name}_wcf.dat", np.column_stack([time_fs, wcf]),
               header="time (fs), windowed cf")
    print(f"# Wrote {name}_cf.dat, {name}_wcf.dat")

    # Im chi(2): both cf factors are time derivatives, so the spectrum is
    # the cosine transform DIVIDED by omega, with sign +1 — the analytic
    # prefactor of chi(2) = (i omega/kT) FT<alpha(t)mu(0)> after two
    # integrations by parts; it makes the free-OH peak positive, the usual
    # experimental convention for water.
    nu, transform = cosine_transform(time_fs, wcf, args.numax, method=args.ft)
    omega = 2.0 * np.pi * C_CM_PER_FS * nu           # rad/fs
    spectrum = transform / omega
    columns = [nu, spectrum]
    header = ("wavenumber(cm^-1)   Im_chi2(arb) = +(1/omega) * cosine "
              "transform of the windowed cf")
    if args.non_condon:
        columns.append(spectrum * non_condon_factor(nu))
        header += "   Im_chi2 * mu'(nu) alpha'(nu) (non-Condon, Ohto et al.)"
    np.savetxt(f"{name}_ImChi2.dat", np.column_stack(columns), header=header)
    print(f"# Wrote {name}_ImChi2.dat")

    elapsed = int(round(time.time() - t_run))
    print(f"# Finished in {elapsed // 3600:02d}:{elapsed % 3600 // 60:02d}:{elapsed % 60:02d}")


if __name__ == "__main__":
    main()
