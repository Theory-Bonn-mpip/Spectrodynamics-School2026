#!/usr/bin/env python3
"""
vdos.py — velocity autocorrelation function (VACF) and vibrational density
of states (VDOS) from a molecular-dynamics trajectory.

Written with a focus on CLARITY OVER EFFICIENCY: the whole trajectory is
loaded into memory and the correlation function is accumulated with
explicit loops that read like the textbook formula. Pass -corr fft to
evaluate the very same sums with a zero-padded FFT (Wiener-Khinchin
theorem) — identical numbers, orders of magnitude faster.

What is computed
----------------
The velocity autocorrelation function

    Cvv(t) = < v(0) . v(t) >

summed over all atoms (column "total") and also split per chemical element
(one extra column per element). The average < ... > runs over all time
origins: every frame with a valid velocity serves once as t = 0 and is
correlated backward in time with the preceding frames.

The VDOS is the cosine (real Fourier) transform of Cvv(t):

    VDOS(nu) = integral dt  Cvv(t) cos(2*pi*c*nu*t)

evaluated by default with a zero-padded real FFT (fast); pass -ft direct
for the explicit cosine sum on a fixed wavenumber grid (the textbook
formula, slow). Before the transform Cvv(t) is multiplied by a window
(the descending half of a Hann window: 1 at t = 0, 0 at t = lag) so that
it goes smoothly to zero at the last lag; this removes the ringing that
an abrupt truncation would produce. Pass -no_window to transform the
bare Cvv(t).


Input
-----
An ascii xyz trajectory with a constant number of atoms and constant atom
order:  natoms / comment line (ignored) / "symbol x y z" per atom.
Two modes:

  * positions (default): coordinates in Angstrom. Velocities are computed
    by backward finite difference  v(t) = [r(t) - r(t-dt)] / dt  with the
    displacement folded by the minimum-image convention (orthorhombic cell,
    -cell required) so that atoms wrapping across the periodic boundary do
    not produce spurious +-L/dt spikes. The first frame has no velocity.
  * -vel: the file already contains velocities (e.g. produced by i-PI or
    by i-PI); coordinates are used directly, no cell needed.

Units: the command line uses ps for -dt and -lag (converted internally to
fs); positions in Angstrom, so velocities and Cvv come out in Angstrom/fs
and (Angstrom/fs)^2.

Output
------
  Cvv.dat            columns: time(fs), total, one column per element
  Cvv_spectrum.dat   columns: wavenumber(cm^-1), cosine transform of the
                     windowed Cvv(t), same data columns

Example
-------
  python3 vdos.py -f pos.xyz      -dt 0.0005 -lag 0.5 -max 2000 -cell 12.4 12.4 12.4
  python3 vdos.py -f vel.xyz -vel -dt 0.0005 -lag 0.5 -max 1999
"""

import argparse
import sys
import time

import numpy as np

# Physical constants for the wavenumber axis of the spectrum.
C_CM_PER_FS = 2.99792458e-5   # speed of light in cm/fs (time unit of the transform)


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
    """Read up to max_frames frames of an ascii xyz trajectory.

    Returns (symbols, coords) where symbols is a list of length natoms and
    coords is an array of shape (nframes, natoms, 3). Only the element
    symbol and three numbers per line are used; any extra columns are
    ignored. The comment line of
    every frame is ignored.
    """
    symbols = []
    frames = []
    t_start = time.time()
    with open(filename) as fh:
        while len(frames) < max_frames:
            natoms_line = fh.readline()
            if natoms_line.strip() == "":       # end of file
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
    if 0 < len(frames) < max_frames:
        sys.stderr.write("\n")             # close the progress line of a short file
    if not frames:
        sys.exit(f"Error: could not read any frame from {filename}")
    return symbols, np.array(frames)


def hann_half_window(ncf_step):
    """Descending second half of a Hann window of length 2*ncf_step.

    w(j) = 0.5 - 0.5*cos(2*pi*(ncf_step + j)/(2*ncf_step - 1)),  j = 0..ncf_step-1
    so w ~ 1 at t = 0 and w = 0 exactly at the last lag
    (= np.hanning(2*ncf_step)[ncf_step:]).
    """
    j = np.arange(ncf_step)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * (ncf_step + j) / (2 * ncf_step - 1))


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
        correlation function has decayed — or been windowed — by t = lag)
        and only interpolates the spectrum on a finer grid.
    method="direct": the same sum evaluated literally, one cosine at a
        time, on the fixed grid nu = 1, 1+dnu, ... — the textbook formula,
        transparent but O(N_nu * N_t) slow.

    The two methods give identical numbers at shared frequencies and
    differ only in their frequency grids. `data` may be 1D (ntimes,) or
    2D (ntimes, ncolumns). Returns (nu, spectrum).
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


def correlate_fft(x, nlags, y=None):
    """cf[k] = sum_j x[j+k] * y[j],  k = 0..nlags-1  (Wiener-Khinchin).

    The explicit origin-by-lag double loop sums x(origin).y(origin-k) over
    every origin in a contiguous range — exactly the linear correlation of
    the two series restricted to that range. By the correlation theorem this
    is one product in Fourier space: irfft(rfft(x) * conj(rfft(y)))[k].
    Zero-padding to at least len(x) + nlags makes the circular correlation
    of the FFT equal to the linear sum, so the result matches the double
    loop to round-off (~1e-12 relative) at a tiny fraction of the cost.

    x, y: arrays (n,) or (n, ncols), y defaults to x (autocorrelation).
    Returns (nlags,) or (nlags, ncols) — one correlation per column; the
    caller sums columns (e.g. the three Cartesian components of one atom).
    """
    n = len(x)
    n_fft = 2 ** int(np.ceil(np.log2(n + nlags)))
    fx = np.fft.rfft(x, n=n_fft, axis=0)
    fy = fx if y is None else np.fft.rfft(y, n=n_fft, axis=0)
    return np.fft.irfft(fx * np.conj(fy), n=n_fft, axis=0)[:nlags]


def main():
    t_run = time.time()
    parser = argparse.ArgumentParser(
        description="VACF / VDOS from an xyz trajectory (positions or velocities).",
        allow_abbrev=False)
    parser.add_argument("-f", required=True, metavar="FILE", dest="traj",
                        help="xyz trajectory (positions, or velocities with -vel)")
    parser.add_argument("-dt", required=True, type=float, metavar="PS",
                        help="time step between frames, in ps")
    parser.add_argument("-lag", required=True, type=float, metavar="PS",
                        help="correlation length, in ps (ncf_step = lag/dt frames)")
    parser.add_argument("-max", required=True, type=int, metavar="N", dest="nframes_max",
                        help="maximum number of frames to use")
    parser.add_argument("-vel", action="store_true",
                        help="the file already contains velocities")
    parser.add_argument("-cell", nargs=3, type=float, metavar=("LX", "LY", "LZ"),
                        help="orthorhombic cell lengths in Angstrom "
                             "(required without -vel, for the minimum image)")
    parser.add_argument("-out", default="Cvv.dat", metavar="FILE",
                        help="output file name (default Cvv.dat)")
    parser.add_argument("-not_normalize", action="store_true",
                        help="write raw sums instead of dividing each lag by its "
                             "number of time origins")
    parser.add_argument("-no_window", action="store_true",
                        help="do not apply the half-Hann window to Cvv(t) before the transform")
    parser.add_argument("-numax", type=float, default=4000.0, metavar="CM1",
                        help="upper edge of the spectrum wavenumber grid (default 4000)")
    parser.add_argument("-ft", choices=["fft", "direct"], default="fft",
                        help="spectrum evaluation: zero-padded real FFT (default) or "
                             "the explicit cosine sum (textbook formula, slow)")
    parser.add_argument("-corr", choices=["direct", "fft"], default="direct",
                        help="correlation-function evaluation: explicit origin-by-lag "
                             "double loop (default, reads like the textbook formula) or "
                             "one zero-padded FFT per column via the Wiener-Khinchin "
                             "theorem (identical numbers, fast). Not to be confused "
                             "with -ft, which selects how the SPECTRUM transform of "
                             "the finished Cvv(t) is evaluated")
    args = parser.parse_args()

    # The command line uses ps (convenient for MD input files); everything
    # internal is in fs.
    dt = 1000.0 * args.dt      # fs
    lag = 1000.0 * args.lag    # fs
    ncf_step = int(lag / dt)   # number of correlation lags (t = 0 ... lag-dt)
    if ncf_step < 10:
        sys.exit("Error: lag/dt must be at least 10.")
    if args.nframes_max < ncf_step:
        sys.exit("Error: -max must be at least lag/dt frames.")
    if (not args.vel) and args.cell is None:
        sys.exit("Error: -cell is required in positions mode (minimum image).")

    # ------------------------------------------------------------------
    # Read the trajectory. One extra frame beyond -max is read; it is never
    # used as a time origin.
    # ------------------------------------------------------------------
    symbols, coords = read_xyz_frames(args.traj, args.nframes_max + 1)
    nread, natoms, _ = coords.shape
    print(f"# Read {nread} frames, {natoms} atoms.")

    # Chemical species, in order of first appearance in the first frame.
    species = list(dict.fromkeys(symbols))
    species_masks = [np.array([s == sp for s in symbols]) for sp in species]
    print(f"# Species: {' '.join(species)}")

    # ------------------------------------------------------------------
    # Velocities.
    #   -vel:      the file content already IS the velocity, valid from frame 0.
    #   positions: backward finite difference between consecutive frames,
    #              with each Cartesian component folded to its minimum image
    #              so a jump across the periodic boundary does not count as
    #              a huge displacement. Valid from frame 1 (needs 2 frames).
    # ------------------------------------------------------------------
    if args.vel:
        velocities = coords
        first_valid = 0
    else:
        cell = np.array(args.cell)                       # Angstrom
        velocities = np.zeros_like(coords)
        displacement = coords[1:] - coords[:-1]          # r(t) - r(t-dt)
        displacement -= cell * np.rint(displacement / cell)   # minimum image
        velocities[1:] = displacement / dt               # Angstrom/fs
        first_valid = 1

    # ------------------------------------------------------------------
    # Accumulate the correlation function.
    # Every frame with a valid velocity is used once as the time origin
    # (t = 0) and correlated backward with the frames before it. Only the
    # first -max frames serve as origins.
    # cf[column, k] accumulates sum over origins and atoms of v(0).v(k*dt),
    # where column 0 is the total and the rest are the per-element parts.
    # ------------------------------------------------------------------
    ncols = 1 + len(species)
    cf = np.zeros((ncols, ncf_step))
    ncount_lag = np.zeros(ncf_step, dtype=int)   # number of origins per lag

    last_origin = min(nread, args.nframes_max)   # frames beyond -max are not origins
    if args.corr == "fft":
        # The origins form the contiguous frame range [first_valid,
        # last_origin), each correlated backward against the same range, so
        # the double loop below is exactly the linear autocorrelation of
        # the trimmed velocity series — one FFT per atom component.
        trimmed = velocities[first_valid:last_origin]     # (n, natoms, 3)
        n = len(trimmed)
        per_col = correlate_fft(trimmed.reshape(n, natoms * 3), ncf_step)
        per_atom = per_col.reshape(ncf_step, natoms, 3).sum(axis=2)
        cf[0] = per_atom.sum(axis=1)
        for isp, mask in enumerate(species_masks):
            cf[1 + isp] = per_atom[:, mask].sum(axis=1)
        # Origins available at lag k: all but the first k trimmed frames.
        ncount_lag[:] = np.maximum(n - np.arange(ncf_step), 0)
    else:
        t_start = time.time()
        for origin in range(first_valid, last_origin):
            # Lags available at this origin: we can only look back to the first
            # valid velocity frame (and at most ncf_step-1 steps).
            max_lag = min(ncf_step - 1, origin - first_valid)
            for k in range(max_lag + 1):
                partner = origin - k
                # per-atom dot product v_origin . v_partner
                dot_per_atom = np.sum(velocities[origin] * velocities[partner], axis=1)
                cf[0, k] += dot_per_atom.sum()
                for isp, mask in enumerate(species_masks):
                    cf[1 + isp, k] += dot_per_atom[mask].sum()
                ncount_lag[k] += 1
            progress(origin + 1 - first_valid, last_origin - first_valid,
                     "correlating", t_start)

    # Divide each lag by its own number of time origins (a per-lag average;
    # long lags have fewer origins than short ones).
    result = cf.copy()
    if not args.not_normalize:
        valid = ncount_lag > 0
        result[:, valid] = cf[:, valid] / ncount_lag[valid]

    # ------------------------------------------------------------------
    # Write the correlation function and its spectrum.
    # ------------------------------------------------------------------
    time_fs = dt * np.arange(ncf_step)
    header = "time(fs)        total " + " ".join(species)
    np.savetxt(args.out, np.column_stack([time_fs, result.T]), header=header)
    print(f"# Wrote {args.out}")

    # VDOS = cosine transform of the windowed Cvv(t). The window takes the
    # correlation function smoothly to zero at the last lag; without it an
    # abrupt truncation produces ringing in the spectrum.
    cvv = result.T
    if not args.no_window:
        cvv = cvv * hann_half_window(ncf_step)[:, None]
    nu, spectrum = cosine_transform(time_fs, cvv, args.numax, method=args.ft)
    spec_name = (args.out[:-4] if args.out.endswith(".dat") else args.out) + "_spectrum.dat"
    np.savetxt(spec_name, np.column_stack([nu, spectrum]),
               header="wavenumber(cm^-1)  total " + " ".join(species))
    print(f"# Wrote {spec_name}")

    elapsed = int(round(time.time() - t_run))
    print(f"# Finished in {elapsed // 3600:02d}:{elapsed % 3600 // 60:02d}:{elapsed % 60:02d}")


if __name__ == "__main__":
    main()
