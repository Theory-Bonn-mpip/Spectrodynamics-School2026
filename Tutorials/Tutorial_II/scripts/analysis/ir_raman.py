#!/usr/bin/env python3
"""
ir_raman.py — IR and Raman spectra from total-dipole and total-polarizability
trajectories.

Written with a focus on CLARITY OVER EFFICIENCY: the property time series
are loaded fully into memory and the correlation functions are accumulated
with explicit loops that read like the textbook formula. Pass -corr fft to
evaluate the very same sums with a zero-padded FFT (Wiener-Khinchin
theorem) — identical numbers, orders of magnitude faster.

What is computed
----------------
Each frame the raw property is reduced to a "descriptor": its backward
finite-difference time derivative,

    mu_dot(t)    = [mu(t)    - mu(t-dt)]    / dt        (IR)
    alpha_dot(t) = [alpha(t) - alpha(t-dt)] / dt        (Raman)

and for Raman the derivative tensor is split into its rotational invariants,

    a    = tr(alpha_dot)/3            (isotropic part, a scalar)
    beta = alpha_dot - a * identity   (anisotropic part, traceless tensor).

Then the autocorrelation functions

    IR:          C(t) = < mu_dot(0) . mu_dot(t) >
    Raman iso:   C(t) = < a(0) a(t) >
    Raman aniso: C(t) = < tr[ beta(0) beta(t) ] >

are accumulated over all time origins with per-lag normalization (each lag
divided by its own number of origins). The means are removed for the IR and
iso channels by subtracting |<d>|^2 at the end (exact for an autocorrelation:
<(d-<d>)(0).(d-<d>)(t)> = <d(0).d(t)> - <d>.<d>); the aniso channel gets no
mean removal (beta of an isotropic liquid averages to zero).

The spectrum of each channel is the cosine transform of the windowed C(t) on
a wavenumber grid in cm^-1, evaluated by default with a zero-padded real FFT
(fast); pass -ft direct for the explicit cosine sum on a fixed grid (the
textbook formula, slow). The window is the descending second half of a Hann
window of length 2N (w ~ 1 at t = 0, w = 0 at t = lag), which removes the
truncation ringing a bare C(t) would produce.

Input
-----
Plain ascii, one frame after another, whitespace separated (a frame may span
several lines). Lines starting with '#' are skipped, so raw i-PI extras
files with their per-frame '#EXTRAS(...)# Step: N' comment lines can be
read directly.

  -dip FILE : 3 values per frame,  mu_x mu_y mu_z          (a.u., e*bohr)
  -pol FILE : 9 values per frame, the 3x3 tensor row-major:
              a_xx a_xy a_xz a_yx a_yy a_yz a_zx a_zy a_zz (a.u., bohr^3)

At least one of the two is required. -skip N discards the first N frames
of every file (i-PI replay runs write the first step twice, so use
-skip 1 for their extras files). Units only set the overall scale of
C(t); peak positions do not depend on them.

Output (per active channel: dip, iso, aniso)
------
  <prefix>-dip.dat           time(fs), C(t)
  <prefix>-dip-win.dat       time(fs), C(t) * window
  <prefix>-dip-spectrum.dat  wavenumber(cm^-1), cosine transform of C(t)*window

Example
-------
  python3 ir_raman.py -dip dipole.dat -pol polarizability.dat \
                      -dt 0.0005 -lag 0.5 -max 9471 -pol_type both
"""

import argparse
import sys
import time

import numpy as np

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


def read_property_frames(filename, values_per_frame, max_frames):
    """Read up to max_frames frames of `values_per_frame` numbers each.

    The reader consumes a plain stream of whitespace-separated numbers, so a
    frame may span one or several lines. Lines starting with '#' (e.g. the
    per-frame "#EXTRAS(...)# Step: N" comments of raw i-PI extras files) and
    blank lines are skipped. Returns an array (nframes, values_per_frame).
    """
    frames = []
    leftover = []
    t_start = time.time()
    with open(filename) as fh:
        for line in fh:
            stripped = line.strip()
            if stripped == "" or stripped.startswith("#"):
                continue
            leftover.extend(float(tok) for tok in stripped.split())
            while len(leftover) >= values_per_frame:
                frames.append(leftover[:values_per_frame])
                progress(len(frames), max_frames, f"reading {filename}", t_start)
                leftover = leftover[values_per_frame:]
                if len(frames) >= max_frames:
                    return np.array(frames)
    if frames:
        sys.stderr.write("\n")             # close the progress line of a short file
    if not frames:
        sys.exit(f"Error: could not read any frame from {filename}")
    return np.array(frames)


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
        windowed C(t) has decayed to zero by t = lag) and only
        interpolates the spectrum on a finer grid.
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
    caller sums columns (e.g. the tensor components of one descriptor).
    """
    n = len(x)
    n_fft = 2 ** int(np.ceil(np.log2(n + nlags)))
    fx = np.fft.rfft(x, n=n_fft, axis=0)
    fy = fx if y is None else np.fft.rfft(y, n=n_fft, axis=0)
    return np.fft.irfft(fx * np.conj(fy), n=n_fft, axis=0)[:nlags]


def accumulate_acf_fft(x, ncf_step, nframes_max, y=None):
    """FFT twin of accumulate_acf, via the Wiener-Khinchin theorem.

    x (and optionally y) hold the descriptors of frames 1..nread-1 as an
    array (nread-1, ncols) — row i is the descriptor of raw frame i+1, the
    flattened tensor components in the columns. Only rows belonging to the
    origin frames 1..min(nread, -max)-1 enter, exactly the origin set of
    the explicit loop, so the two paths agree to round-off.
    Returns (cf, ncount_lag).
    """
    last_origin = min(len(x) + 1, nframes_max)
    n = last_origin - 1                      # number of origin frames
    cf = correlate_fft(x[:n], ncf_step, None if y is None else y[:n]).sum(axis=1)
    ncount_lag = np.maximum(n - np.arange(ncf_step), 0)
    return cf, ncount_lag


def accumulate_acf(descriptors, correlate_pair, ncf_step, nframes_max,
                   label="correlating"):
    """Generic autocorrelation accumulation over all time origins.

    `descriptors` is a list indexed by raw-frame number; entry 0 is None
    (the finite difference needs two frames, so frame 0 has no descriptor).
    `correlate_pair(d0, dt_)` returns the scalar contribution of one
    (origin, lagged) descriptor pair.

    Every frame with a descriptor serves once as the time origin t = 0 and
    is correlated backward in time with the preceding frames. Frames beyond
    -max are not used as origins. Returns (cf, ncount_lag).
    """
    cf = np.zeros(ncf_step)
    ncount_lag = np.zeros(ncf_step, dtype=int)
    nread = len(descriptors)
    last_origin = min(nread, nframes_max)      # frames beyond -max: no origins
    t_start = time.time()
    for origin in range(1, last_origin):       # frame 0 has no descriptor
        max_lag = min(ncf_step - 1, origin - 1)
        for k in range(max_lag + 1):
            cf[k] += correlate_pair(descriptors[origin], descriptors[origin - k])
            ncount_lag[k] += 1
        progress(origin, last_origin - 1, label, t_start)
    return cf, ncount_lag


def write_channel(prefix, channel, time_fs, cf, ncount_lag, mean2,
                  normalize, numax, ft_method, comment):
    """Normalize, subtract the mean term, window, transform and write."""
    c = cf.copy()
    if normalize:
        valid = ncount_lag > 0
        c[valid] = cf[valid] / ncount_lag[valid]   # per-lag average
        c[valid] -= mean2                          # remove |<d>|^2 (0 for aniso)

    window = hann_half_window(len(c))
    nu, spectrum = cosine_transform(time_fs, c * window, numax, method=ft_method)

    np.savetxt(f"{prefix}-{channel}.dat", np.column_stack([time_fs, c]),
               header=f"{comment}\ntime(fs)        C(t)")
    np.savetxt(f"{prefix}-{channel}-win.dat", np.column_stack([time_fs, c * window]),
               header=f"{comment}\ntime(fs)        C(t)*window (half Hann)")
    np.savetxt(f"{prefix}-{channel}-spectrum.dat", np.column_stack([nu, spectrum]),
               header=f"{comment}\nwavenumber(cm^-1)   cosine transform of windowed C(t)")
    print(f"# Wrote {prefix}-{channel}.dat, {prefix}-{channel}-win.dat, "
          f"{prefix}-{channel}-spectrum.dat")


def main():
    t_run = time.time()
    parser = argparse.ArgumentParser(
        description="IR / Raman correlation functions and spectra from total "
                    "dipole and total polarizability trajectories.",
        allow_abbrev=False)
    parser.add_argument("-dip", metavar="FILE",
                        help="total dipole file, 3 values per frame (a.u.)")
    parser.add_argument("-pol", metavar="FILE",
                        help="total polarizability file, 9 values per frame, "
                             "row-major 3x3 (a.u.)")
    parser.add_argument("-dt", required=True, type=float, metavar="PS",
                        help="time step between frames, in ps")
    parser.add_argument("-lag", required=True, type=float, metavar="PS",
                        help="correlation length, in ps (ncf_step = lag/dt frames)")
    parser.add_argument("-skip", type=int, default=0, metavar="N",
                        help="skip the first N frames of every file (i-PI replay "
                             "runs write step 0 twice: use -skip 1)")
    parser.add_argument("-max", required=True, type=int, metavar="N", dest="nframes_max",
                        help="maximum number of frames to use")
    parser.add_argument("-pol_type", choices=["iso", "aniso", "both"], default="iso",
                        help="Raman channel(s) to compute (default iso)")
    parser.add_argument("-out", default="acf", metavar="PREFIX",
                        help="output file prefix (default acf)")
    parser.add_argument("-not_normalize", action="store_true",
                        help="write raw sums (no per-lag average, no mean removal)")
    parser.add_argument("-numax", type=float, default=4000.0, metavar="CM1",
                        help="upper edge of the spectrum wavenumber grid (default 4000)")
    parser.add_argument("-ft", choices=["fft", "direct"], default="fft",
                        help="spectrum evaluation: zero-padded real FFT (default) or "
                             "the explicit cosine sum (textbook formula, slow)")
    parser.add_argument("-corr", choices=["direct", "fft"], default="direct",
                        help="correlation-function evaluation: explicit origin-by-lag "
                             "double loop (default, reads like the textbook formula) or "
                             "one zero-padded FFT per component via the Wiener-Khinchin "
                             "theorem (identical numbers, fast). Not to be confused "
                             "with -ft, which selects how the SPECTRUM transform of "
                             "the finished C(t) is evaluated")
    args = parser.parse_args()

    if args.dip is None and args.pol is None:
        sys.exit("Error: at least one of -dip / -pol is required.")

    dt = 1000.0 * args.dt      # ps -> fs, everything internal is fs
    lag = 1000.0 * args.lag
    ncf_step = int(lag / dt)
    if ncf_step < 10:
        sys.exit("Error: lag/dt must be at least 10.")
    if args.nframes_max < ncf_step:
        sys.exit("Error: -max must be at least lag/dt frames.")

    do_iso = args.pol is not None and args.pol_type in ("iso", "both")
    do_aniso = args.pol is not None and args.pol_type in ("aniso", "both")
    normalize = not args.not_normalize

    # ------------------------------------------------------------------
    # Read the property files. One extra frame beyond -max is read; it
    # enters the running mean below but never serves as a time origin.
    # ------------------------------------------------------------------
    nread = args.nframes_max + 1
    dip = pol = None
    if args.dip is not None:
        dip = read_property_frames(args.dip, 3, nread + args.skip)[args.skip:]
        nread = min(nread, len(dip))
    if args.pol is not None:
        pol = read_property_frames(args.pol, 9, nread + args.skip)[args.skip:]
        nread = min(nread, len(pol))
    if dip is not None:
        dip = dip[:nread]
    if pol is not None:
        pol = pol[:nread]
    print(f"# Read {nread} frames.")
    if nread < 2:
        sys.exit("Error: the finite difference needs at least two frames.")

    time_fs = dt * np.arange(ncf_step)

    # ------------------------------------------------------------------
    # Descriptors: backward finite-difference derivative of each property.
    # Frame 0 has no descriptor (the difference needs two frames), which we
    # represent with a leading None. The mean of every descriptor is
    # accumulated over ALL of them, including the extra frame that is
    # never a time origin.
    # ------------------------------------------------------------------
    if dip is not None:
        mu_dot = (dip[1:] - dip[:-1]) / dt              # (nread-1, 3)
        mean_mu_dot = mu_dot.mean(axis=0)
        if args.corr == "fft":
            # <mu_dot(0).mu_dot(t)> = sum of the 3 component autocorrelations
            cf, ncount_lag = accumulate_acf_fft(mu_dot, ncf_step, args.nframes_max)
        else:
            descriptors = [None] + list(mu_dot)
            cf, ncount_lag = accumulate_acf(
                descriptors, lambda d0, dt_: np.dot(d0, dt_),
                ncf_step, args.nframes_max, label="correlating dip")
        write_channel(args.out, "dip", time_fs, cf, ncount_lag,
                      np.dot(mean_mu_dot, mean_mu_dot), normalize, args.numax,
                      args.ft, "IR  C(t) = <mu_dot(0).mu_dot(t)>  (dipole a.u.)")

    if pol is not None:
        alpha_dot = ((pol[1:] - pol[:-1]) / dt).reshape(-1, 3, 3)   # (nread-1, 3, 3)
        # Rotational invariants of the derivative tensor:
        a_iso = np.trace(alpha_dot, axis1=1, axis2=2) / 3.0         # scalar per frame
        beta = alpha_dot - a_iso[:, None, None] * np.eye(3)         # traceless part

        if do_iso:
            if args.corr == "fft":
                cf, ncount_lag = accumulate_acf_fft(
                    a_iso[:, None], ncf_step, args.nframes_max)
            else:
                descriptors = [None] + list(a_iso)
                cf, ncount_lag = accumulate_acf(
                    descriptors, lambda d0, dt_: d0 * dt_,
                    ncf_step, args.nframes_max, label="correlating iso")
            write_channel(args.out, "iso", time_fs, cf, ncount_lag,
                          a_iso.mean() ** 2, normalize, args.numax, args.ft,
                          "Raman iso  C(t) = <a_dot(0) a_dot(t)>,  a = tr(alpha)/3 (a.u.)")

        if do_aniso:
            # tr(beta(0) beta(t)) = sum_ab beta(0)_ab beta(t)_ba; beta is
            # symmetric for a physical polarizability, but we keep the
            # transpose to stay faithful to the definition.
            if args.corr == "fft":
                # sum_ab of the component cross-correlations beta_ab x beta_ba
                cf, ncount_lag = accumulate_acf_fft(
                    beta.reshape(-1, 9), ncf_step, args.nframes_max,
                    y=beta.transpose(0, 2, 1).reshape(-1, 9))
            else:
                descriptors = [None] + list(beta)
                cf, ncount_lag = accumulate_acf(
                    descriptors, lambda d0, dt_: np.sum(d0 * dt_.T),
                    ncf_step, args.nframes_max, label="correlating aniso")
            write_channel(args.out, "aniso", time_fs, cf, ncount_lag,
                          0.0, normalize, args.numax, args.ft,
                          "Raman aniso  C(t) = <tr(beta(0)beta(t))>, beta traceless "
                          "(a.u.); no mean removal")

    elapsed = int(round(time.time() - t_run))
    print(f"# Finished in {elapsed // 3600:02d}:{elapsed % 3600 // 60:02d}:{elapsed % 60:02d}")


if __name__ == "__main__":
    main()
