#!/usr/bin/env python3
"""
sfg_atomic.py — resonant SFG response of a water slab from per-atom
(atomic) charges, dipoles and polarizabilities.

Written with a focus on CLARITY OVER EFFICIENCY: all time series are
loaded into memory and every step is spelled out with explicit loops.

Method (response-sited, "G2" molecular gauge)
---------------------------------------------
A machine-learning model (or any partitioning scheme) provides, per frame
and per atom i: a charge q_i, an atomic dipole nu_i, and an atomic
polarizability alpha_i. These atomic responses are grouped into molecules
(each H is assigned to its nearest O) and reduced to per-molecule STATE
FUNCTIONS:

    mu~_m    = sum_{i in m} [ nu_i + q_i * (r_i - r_O,m) ]
    alpha~_m = sum_{i in m} alpha_i

The charge moment is referenced to the molecule's OWN oxygen ("G2" gauge):
this drops the monopole lever arm Q_m * (z_O - z_center) which would
otherwise dominate — and corrupt — the surface-windowed signal. r_i - r_O
is minimum-imaged, so wrapped hydrogens do not produce +-L jumps.

Both state functions are differentiated in time by a backward finite
difference; the smooth SURFACE WINDOW w(z_O - z_center) (+1 top interface,
-1 bottom, 0 in the bulk; z_center = mean O height of the SAME frame — the
frames themselves are never translated) multiplies the dipole rate only,
so it enters each mu*alpha product exactly once:

    mu_dot_m    = w * [mu~_m(t)    - mu~_m(t-dt)]    / dt
    alpha_dot_m =     [alpha~_m(t) - alpha~_m(t-dt)] / dt

The correlation function, per chi(2) polarization channel, is the SYMMETRIC
combination summed over the O-O pair list within -rcut (both pair
directions; self pairs once):

    cf(t) = < mu_dot_k(0) alpha_dot_ij(t) + mu_dot_k(t) alpha_dot_ij(0) >

with the channels (last chi index = IR index; symmetry-equivalent elements
averaged so all channels share one scale):

    xxz :  mu_dot_z * (alpha_dot_xx + alpha_dot_yy)/2
    zzz :  mu_dot_z *  alpha_dot_zz
    xzx :  [ mu_dot_x * alpha_dot_xz + mu_dot_y * alpha_dot_yz ] / 2

Each lag is normalized by its own number of time origins. Because BOTH
factors are time derivatives, Im chi(2) is the COSINE transform of the
windowed cf divided by omega, with sign +1 — the analytic prefactor of
chi(2) = (i omega/kT) FT<alpha(t)mu(0)> after two integrations by parts
(same convention as ssvvcf_ml.py; only the RELATIVE sign of mu and alpha
is physical). The transform is evaluated by default with a zero-padded
real FFT (fast); pass -ft direct for the explicit cosine sum on a fixed
grid (the textbook formula, slow).

Input (all ascii; a frame may span several lines; lines starting with '#'
------ such as raw i-PI '#EXTRAS(...)# Step: N' comments are skipped)
  -f     xyz trajectory of the slab (Angstrom), whole along z
  -atq   atomic charges,        natoms   values per frame (e)
  -atmu  atomic dipoles,        natoms*3 values per frame (atom-major)
  -atpol atomic polarizability, natoms*9 values per frame (atom-major,
         each atom's 3x3 tensor row-major: xx xy xz yx yy yz zx zy zz)
All three are required — the molecular dipole needs both the atomic point
dipoles and the charge moment. The absolute units of q/nu/alpha only set
the overall scale of the spectrum.

Output (per active channel <chi> in xxz, zzz, xzx)
------
  <prefix>SFG_cf_<chi>.dat      time(fs), cf(t)          (per-lag normalized)
  <prefix>SFG_wcf_<chi>.dat     same, apodized with sin^2(pi(i-N)/2N)
  <prefix>SFG_ImChi2_<chi>.dat  wavenumber(cm^-1), +(1/omega) * cosine
                                transform of the windowed cf

Example
-------
  python3 sfg_atomic.py -f traj.xyz -atq charges.dat -atmu atomic_dipoles.dat \
        -atpol atomic_pols.dat -cell 12 12 35 -dt 0.0005 -lag 0.01 -max 55 \
        -zref1 2 -zref2 8 -chi all
"""

import argparse
import sys
import time

import numpy as np

C_CM_PER_FS = 2.99792458e-5   # speed of light in cm/fs (time unit of the transform)

# chi(2) channels: (component of mu_dot, row of alpha_dot, channel, weight).
# alpha~ is reduced per molecule to 4 rows: (a_xx+a_yy)/2, a_zz, a_xz, a_yz.
# The xzx channel averages its two symmetry-equivalent terms (weight 1/2).
CHANNELS = ["xxz", "zzz", "xzx"]
TERMS = [(2, 0, "xxz", 1.0),   # mu_dot_z * (alpha_dot_xx+alpha_dot_yy)/2
         (2, 1, "zzz", 1.0),   # mu_dot_z *  alpha_dot_zz
         (0, 2, "xzx", 0.5),   # mu_dot_x *  alpha_dot_xz
         (1, 3, "xzx", 0.5)]   # mu_dot_y *  alpha_dot_yz


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
    if len(frames) < max_frames:
        sys.stderr.write("\n")             # close the progress line of a short file
    if not frames:
        sys.exit(f"Error: could not read any frame from {filename}")
    return symbols, np.array(frames)


def read_property_frames(filename, values_per_frame, max_frames, skip=0):
    """Read frames of `values_per_frame` whitespace-separated numbers.

    Lines starting with '#' (raw i-PI extras comments) and blank lines are
    skipped; a frame may span several lines. The first `skip` frames are
    discarded (i-PI replay runs write step 0 twice: use skip=1).
    """
    frames = []
    leftover = []
    skipped = 0
    t_start = time.time()
    with open(filename) as fh:
        for line in fh:
            stripped = line.strip()
            if stripped == "" or stripped.startswith("#"):
                continue
            leftover.extend(float(tok) for tok in stripped.split())
            while len(leftover) >= values_per_frame:
                if skipped < skip:
                    skipped += 1
                else:
                    frames.append(leftover[:values_per_frame])
                    progress(len(frames), max_frames, f"reading {filename}", t_start)
                leftover = leftover[values_per_frame:]
                if len(frames) >= max_frames:
                    return np.array(frames)
    sys.stderr.write("\n")                 # close the progress line of a short file
    if not frames:
        sys.exit(f"Error: could not read any frame from {filename}")
    return np.array(frames)


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

    Returns one list per oxygen: [O_index, H_index, ...] with 1 to 3
    hydrogens (water, but also OH-/H3O+ are legal here — the molecular
    grouping does not care, unlike a bond model). Raises ValueError if an
    H is farther than `cutoff` from every O, if an O ends up with no H, or
    with more than three.
    """
    bonded_H = [[] for _ in indexes_O]
    posO = pos[indexes_O]
    for iH in indexes_H:                                 # in file order
        d = minimum_image(pos[iH] - posO, cell)
        dist = np.linalg.norm(d, axis=1)
        closest = int(np.argmin(dist))
        if dist[closest] > cutoff:
            raise ValueError(
                f"H atom {iH} is {dist[closest]:.2f} A from the closest O "
                f"(> topology cutoff {cutoff} A)")
        if len(bonded_H[closest]) >= 3:
            raise ValueError(f"O atom {indexes_O[closest]} has more than 3 H")
        bonded_H[closest].append(iH)
    for iO, hs in enumerate(bonded_H):
        if not hs:
            raise ValueError(f"O atom {indexes_O[iO]} has no bonded H")
    return [[indexes_O[iO]] + hs for iO, hs in enumerate(bonded_H)]


def window_factor(z, nmode, z_ref1, z_ref2):
    """Smooth surface window from the oxygen z relative to the slab center.

    nmode 1: both interfaces (top +1, bottom -1 so the two opposite-oriented
    surfaces ADD in chi(2)); 2: top only; 3: bottom only; 4: bulk (+1
    everywhere). Between z_ref1 and z_ref2 the factor switches on smoothly
    with a sine.
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


def molecular_state_functions(pos, topology, at_q, at_nu, at_alpha, cell):
    """Per-molecule state functions mu~ and alpha~ (G2 gauge).

    mu~[m]    = sum over the molecule's atoms of  nu_i + q_i*(r_i - r_O)
                (r_i - r_O minimum-imaged; zero for the oxygen itself)
    alpha~[m] = per-molecule sums reduced to the 4 channel rows
                [(a_xx+a_yy)/2, a_zz, a_xz, a_yz]
    No window factor here: these are state functions; the window multiplies
    the time DERIVATIVE of mu~ later, once per molecule.
    """
    n_O = len(topology)
    mutilde = np.zeros((n_O, 3))
    altilde = np.zeros((n_O, 4))
    for iO, members in enumerate(topology):
        iOa = members[0]
        axx = ayy = azz = axz = ayz = 0.0
        for ia in members:                       # the O and its hydrogens
            disp = minimum_image(pos[ia] - pos[iOa], cell)
            mutilde[iO] += at_nu[ia] + at_q[ia] * disp
            alpha_i = at_alpha[ia]
            axx += alpha_i[0, 0]
            ayy += alpha_i[1, 1]
            azz += alpha_i[2, 2]
            axz += alpha_i[0, 2]
            ayz += alpha_i[1, 2]
        altilde[iO] = [0.5 * (axx + ayy), azz, axz, ayz]
    return mutilde, altilde


def pair_list(posO, cell, cutoff):
    """O-O pairs (iO1 <= iO2) with minimum-image distance < cutoff.

    Self pairs (distance 0) are always included: they carry the
    intramolecular response.
    """
    d = minimum_image(posO[:, None, :] - posO[None, :, :], cell)
    dist = np.linalg.norm(d, axis=2)
    iO1, iO2 = np.where(np.triu(dist < cutoff))   # upper triangle incl. diagonal
    return iO1, iO2


def main():
    t_run = time.time()
    parser = argparse.ArgumentParser(
        description="SFG response from atomic charges, dipoles and "
                    "polarizabilities (response-sited G2 mode).",
        allow_abbrev=False)
    parser.add_argument("-f", required=True, metavar="FILE", dest="traj",
                        help="xyz position trajectory of the slab (Angstrom)")
    parser.add_argument("-atq", required=True, metavar="FILE",
                        help="atomic charges, natoms values per frame")
    parser.add_argument("-atmu", required=True, metavar="FILE",
                        help="atomic dipoles, natoms*3 values per frame")
    parser.add_argument("-atpol", required=True, metavar="FILE",
                        help="atomic polarizabilities, natoms*9 values per frame")
    parser.add_argument("-cell", nargs=3, type=float, required=True,
                        metavar=("LX", "LY", "LZ"),
                        help="orthorhombic cell lengths in Angstrom")
    parser.add_argument("-dt", required=True, type=float, metavar="PS",
                        help="time step between frames, in ps")
    parser.add_argument("-lag", type=float, default=1.0, metavar="PS",
                        help="correlation length, in ps (default 1.0)")
    parser.add_argument("-max", required=True, type=int, metavar="N",
                        dest="nframes_max", help="maximum number of frames to use")
    parser.add_argument("-rcut", "-rc", type=float, default=1.5, metavar="A", dest="rcut",
                        help="O-O pair-list cutoff in Angstrom (default 1.5: "
                             "self pairs / intramolecular only)")
    parser.add_argument("-topocut", type=float, default=1.35, metavar="A",
                        help="H->O bonding cutoff in Angstrom (default 1.35)")
    parser.add_argument("-nfreq_topo", type=int, default=1, metavar="N",
                        help="recompute the H->O topology every N frames")
    parser.add_argument("-zref1", type=float, default=2.0, metavar="A",
                        help="surface window: |z| below which the factor is 0")
    parser.add_argument("-zref2", type=float, default=20000.0, metavar="A",
                        help="surface window: |z| above which the factor is +-1")
    parser.add_argument("-nmode", type=int, default=1, choices=[1, 2, 3, 4],
                        help="1 both interfaces, 2 top, 3 bottom, 4 bulk")
    parser.add_argument("-chi", default="xxz",
                        choices=["xxz", "yyz", "zzz", "xzx", "yzy", "all"],
                        help="chi(2) channel(s) to write (yyz=xxz and yzy=xzx "
                             "by C_inf_v symmetry; default xxz)")
    parser.add_argument("-skip", type=int, default=0, metavar="N",
                        help="skip the first N records of the property files "
                             "(1 for i-PI replay runs, which write step 0 twice)")
    parser.add_argument("-prefix", default="", metavar="STR",
                        help="prefix for the output file names")
    parser.add_argument("-numax", type=float, default=4000.0, metavar="CM1",
                        help="upper edge of the spectrum wavenumber grid")
    parser.add_argument("-ft", choices=["fft", "direct"], default="fft",
                        help="spectrum evaluation: zero-padded real FFT (default) or "
                             "the explicit cosine sum (textbook formula, slow)")
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
    # By symmetry only three independent channels exist; yyz/yzy are the
    # same elements as xxz/xzx (already averaged inside those channels).
    active = {"all": CHANNELS, "yyz": ["xxz"], "yzy": ["xzx"]}.get(
        args.chi, [args.chi])

    # ------------------------------------------------------------------
    # Read the four files, kept in sync frame by frame. One extra frame
    # beyond -max is read; it is never used as a time origin.
    # ------------------------------------------------------------------
    symbols, coords = read_xyz_frames(args.traj, args.nframes_max + 1)
    natoms = len(symbols)
    at_q = read_property_frames(args.atq, natoms, args.nframes_max + 1, args.skip)
    at_nu = read_property_frames(args.atmu, natoms * 3, args.nframes_max + 1, args.skip)
    at_alpha = read_property_frames(args.atpol, natoms * 9, args.nframes_max + 1, args.skip)
    nread = min(len(coords), len(at_q), len(at_nu), len(at_alpha))
    coords = coords[:nread]
    at_nu = at_nu[:nread].reshape(nread, natoms, 3)
    at_alpha = at_alpha[:nread].reshape(nread, natoms, 3, 3)

    indexes_O = [i for i, s in enumerate(symbols) if s == "O"]
    indexes_H = [i for i, s in enumerate(symbols) if s in ("H", "D")]
    unknown = {s for s in symbols if s not in ("O", "H", "D")}
    if unknown:
        sys.exit(f"Error: only water systems (O,H,D) are supported; "
                 f"found {sorted(unknown)}")
    n_O = len(indexes_O)
    print(f"# Read {nread} frames: {n_O} molecules ({natoms} atoms).")

    # ------------------------------------------------------------------
    # Per frame: topology (every -nfreq_topo frames), state functions
    # mu~/alpha~, and their windowed/plain backward finite differences.
    # Frames are never translated; the slab center (mean O height of the
    # CURRENT frame) only shifts the argument of the window factor.
    # Frame 0 only seeds the finite difference: its rates stay zero.
    # ------------------------------------------------------------------
    try:
        topology = compute_topology(coords[0], indexes_O, indexes_H,
                                    cell, args.topocut)
    except ValueError as err:
        sys.exit(f"Error: cannot compute the topology of the first frame: {err}")
    counts = [len(m) - 1 for m in topology]
    print(f"# Topology (first frame): {counts.count(2)} water, "
          f"{counts.count(1)} OH-, {counts.count(3)} H3O+")

    mu_mol = np.zeros((nread, n_O, 3))      # windowed mu~ rate
    adot_mol = np.zeros((nread, n_O, 4))    # alpha~ rate (4 channel rows)
    mutilde_prev, altilde_prev = molecular_state_functions(
        coords[0], topology, at_q[0], at_nu[0], at_alpha[0], cell)

    t_start = time.time()
    for frame in range(1, nread):
        progress(frame, nread - 1, "state functions", t_start)
        # (frame + 1) is the 1-based frame counter.
        if (frame + 1) % args.nfreq_topo == 0:
            try:
                topology = compute_topology(coords[frame], indexes_O,
                                            indexes_H, cell, args.topocut)
            except ValueError as err:
                print(f"# Warning: topology failed at frame {frame + 1} "
                      f"({err}); keeping the previous one.")
        mutilde, altilde = molecular_state_functions(
            coords[frame], topology, at_q[frame], at_nu[frame],
            at_alpha[frame], cell)

        # Backward finite differences of the state functions. The window
        # multiplies the dipole rate ONLY (once per mu*alpha product),
        # keyed on the current frame's oxygen height above the slab center.
        z_center = coords[frame][indexes_O, 2].mean()
        adot_mol[frame] = (altilde - altilde_prev) / dt
        for iO, members in enumerate(topology):
            z_O = coords[frame][members[0], 2] - z_center
            factor = window_factor(z_O, args.nmode, args.zref1, args.zref2)
            if factor != 0.0:
                mu_mol[frame, iO] = factor * (mutilde[iO] - mutilde_prev[iO]) / dt
        # The prev buffers are refreshed for ALL molecules (windowed or
        # not), so a molecule entering the window has a valid difference.
        mutilde_prev, altilde_prev = mutilde, altilde

    # ------------------------------------------------------------------
    # Accumulate the correlation functions. Every frame serves once as the
    # time origin t = 0 and is correlated backward in time with the
    # preceding frames; the pair list is rebuilt on the origin
    # frame's geometry. SYMMETRIC combination, both pair directions (the
    # reverse one skipped for self pairs so each ordered pair enters once);
    # each lag counts its own number of time origins (frame 0 contributes
    # zero, but is still counted).
    # ------------------------------------------------------------------
    cf = np.zeros((ncf_step, len(CHANNELS)))
    ncount_lag = np.zeros(ncf_step, dtype=int)
    last_origin = min(nread, args.nframes_max)    # frames beyond -max: no origins
    t_start = time.time()
    for origin in range(last_origin):
        progress(origin + 1, last_origin, "correlating", t_start)
        iO1, iO2 = pair_list(coords[origin][indexes_O], cell, args.rcut)
        nonself = iO1 != iO2
        for k in range(min(ncf_step - 1, origin) + 1):
            past = origin - k
            ncount_lag[k] += 1
            for idip, irow, chi, weight in TERMS:
                ichi = CHANNELS.index(chi)
                cf[k, ichi] += weight * np.sum(
                    mu_mol[origin][iO1, idip] * adot_mol[past][iO2, irow]
                    + mu_mol[past][iO1, idip] * adot_mol[origin][iO2, irow])
                cf[k, ichi] += weight * np.sum(
                    (mu_mol[origin][iO2, idip] * adot_mol[past][iO1, irow]
                     + mu_mol[past][iO2, idip] * adot_mol[origin][iO1, irow])[nonself])

    # Per-lag normalization: each lag was sampled by a different number of
    # time origins. No mean removal: both rates are ~zero-mean.
    valid = ncount_lag > 0
    cf[valid] = cf[valid] / ncount_lag[valid, None]

    # Apodization: sin^2(pi(i-N)/2N), ~1 at t=0 and exactly 0 at t=lag.
    i = np.arange(1, ncf_step + 1)
    window = np.sin(np.pi * (i - ncf_step) / (2.0 * ncf_step)) ** 2
    wcf = cf * window[:, None]

    # ------------------------------------------------------------------
    # Write correlation functions and spectra for the requested channels.
    # ------------------------------------------------------------------
    time_fs = dt * np.arange(ncf_step)
    for chi in active:
        ichi = CHANNELS.index(chi)
        head = (f"chi(2) element {chi} (last index = IR)\n"
                "atomic (response-sited G2) mode: mu~ = sum nu_i + q_i (r_i - r_O), "
                "alpha~ = sum alpha_i, finite differences in time\n")
        np.savetxt(f"{args.prefix}SFG_cf_{chi}.dat",
                   np.column_stack([time_fs, cf[:, ichi]]),
                   header=head + "time (fs), cf (per-lag normalized)")
        np.savetxt(f"{args.prefix}SFG_wcf_{chi}.dat",
                   np.column_stack([time_fs, wcf[:, ichi]]),
                   header=head + "time (fs), windowed cf")
        # Im chi(2): cosine transform / omega (both cf factors are time
        # derivatives). Sign +1 = the analytic prefactor of
        # chi(2) = (i omega/kT) FT<alpha(t)mu(0)>, same as ssvvcf_ml.py;
        # only the relative sign of mu and alpha is physical.
        nu, transform = cosine_transform(time_fs, wcf[:, ichi], args.numax,
                                         method=args.ft)
        omega = 2.0 * np.pi * C_CM_PER_FS * nu            # rad/fs
        spectrum = transform / omega
        np.savetxt(f"{args.prefix}SFG_ImChi2_{chi}.dat",
                   np.column_stack([nu, spectrum]),
                   header=head + "wavenumber(cm^-1)   Im_chi2(arb) = +(1/omega) * "
                                 "cosine transform of the windowed cf")
        print(f"# Wrote {args.prefix}SFG_cf_{chi}.dat, "
              f"{args.prefix}SFG_wcf_{chi}.dat, {args.prefix}SFG_ImChi2_{chi}.dat")

    elapsed = int(round(time.time() - t_run))
    print(f"# Finished in {elapsed // 3600:02d}:{elapsed % 3600 // 60:02d}:{elapsed % 60:02d}")


if __name__ == "__main__":
    main()
