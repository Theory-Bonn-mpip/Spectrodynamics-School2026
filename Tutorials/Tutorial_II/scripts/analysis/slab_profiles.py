#!/usr/bin/env python3
"""
slab_profiles.py — density and orientation profiles of a water slab
along the surface normal z.

Written with a focus on CLARITY OVER EFFICIENCY: frames are read one at
a time and binned with a few explicit numpy calls.

Method
------
For every frame the slab is first centred: the centre of mass of the
oxygen atoms is moved to z = 0, computed with a circular (periodic) mean

    theta_i = 2 pi z_i / Lz
    z_c     = Lz / (2 pi) * atan2( sum_i m_i sin theta_i, sum_i m_i cos theta_i )

which is well defined even when the slab is split across the z boundary.
After the shift every atom is wrapped into [-Lz/2, Lz/2) and counted in
z bins of width -dz. Two profiles are accumulated:

* the number densities of the O and H atoms,

      rho_s(z) = N_s(z) / ( n_frames * Lx * Ly * dz )       [Angstrom^-3]

  (liquid water: rho_O = 0.0334 A^-3, i.e. 1 g/cm^3);

* the average orientation of the molecular dipole. Water molecules are
  identified by assigning every H to its nearest O (minimum image); the
  bisector d = (r_H1 - r_O) + (r_H2 - r_O) points along the dipole (from
  the O towards the H atoms), and with the surface normal n = +z

      cos theta = d_z / |d|

  is averaged over the molecules whose oxygen falls in each bin. cos theta
  > 0 means "H atoms pointing up". In the bulk <cos theta> vanishes; near
  the two interfaces it takes opposite signs (mirror images).

Input
-----
An ascii xyz trajectory (Angstrom) of a water slab with the surface
normal along z. The cell is read from an extxyz Lattice="..." entry or
given with -cell (orthorhombic only). Only H2O is supported.

Output
------
  <out>_profiles.dat   z(A)  rho_O(A^-3)  rho_H(A^-3)  <cos theta>  molecules/frame

Example
-------
  python3 slab_profiles.py -f traj.xyz -cell 10 10 100 -max 25000 -dz 0.2
"""

import argparse
import re
import sys
import time

import numpy as np

MASS = {"O": 15.999, "H": 1.008, "D": 2.014}


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


def count_xyz_frames(filename):
    """Count the frames of an xyz file from its line count (fast pre-pass,
    used only to show progress)."""
    with open(filename) as fh:
        natoms = int(fh.readline().split()[0])
    nlines = 0
    with open(filename, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            nlines += block.count(b"\n")
    return nlines // (natoms + 2)


def read_cell_from_comment(comment):
    """Return the diagonal of an extxyz Lattice="..." entry, or None."""
    match = re.search(r'Lattice="([^"]+)"', comment)
    if match is None:
        return None
    values = [float(tok) for tok in match.group(1).split()]
    if len(values) != 9:
        return None
    lattice = np.array(values).reshape(3, 3)
    off_diagonal = lattice - np.diag(np.diag(lattice))
    if np.abs(off_diagonal).max() > 1e-8:
        sys.exit("Error: only orthorhombic cells are supported.")
    return np.diag(lattice)


def iter_xyz_frames(filename, max_frames):
    """Yield (symbols, coords, comment) for up to max_frames frames, one at
    a time (constant memory footprint)."""
    nread = 0
    with open(filename) as fh:
        while nread < max_frames:
            natoms_line = fh.readline()
            if natoms_line.strip() == "":           # end of file
                return
            natoms = int(natoms_line.split()[0])
            comment = fh.readline()
            symbols = []
            coords = np.empty((natoms, 3))
            for i in range(natoms):
                tokens = fh.readline().split()
                symbols.append(tokens[0])
                coords[i] = [float(tokens[1]), float(tokens[2]), float(tokens[3])]
            nread += 1
            yield symbols, coords, comment


def minimum_image(vectors, cell):
    """Fold displacement vector(s) to the minimum image (orthorhombic cell)."""
    return vectors - cell * np.rint(vectors / cell)


def assign_hydrogens(symbols, coords, cell):
    """Assign every H to its nearest O; return (index_O, index_H1, index_H2)
    per molecule. Stops if a molecule does not end up with two hydrogens."""
    index_O = [i for i, s in enumerate(symbols) if s == "O"]
    index_H = [i for i, s in enumerate(symbols) if s in ("H", "D")]
    if not index_O or len(index_H) != 2 * len(index_O):
        sys.exit("Error: the trajectory must contain water only (two H per O).")
    bonded = [[] for _ in index_O]
    pos_O = coords[index_O]
    for iH in index_H:
        d = minimum_image(coords[iH] - pos_O, cell)
        nearest = int(np.argmin(np.sum(d * d, axis=1)))
        bonded[nearest].append(iH)
    molecules = []
    for iO, hydrogens in zip(index_O, bonded):
        if len(hydrogens) != 2:
            sys.exit(f"Error: oxygen {iO} has {len(hydrogens)} hydrogens, expected 2.")
        molecules.append((iO, hydrogens[0], hydrogens[1]))
    return molecules


def centre_slab(z, Lz, masses, is_O):
    """Shift z so that the oxygen centre of mass (circular mean) is at 0 and
    wrap into [-Lz/2, Lz/2)."""
    theta = 2.0 * np.pi * z[is_O] / Lz
    w = masses[is_O]
    z_c = Lz / (2.0 * np.pi) * np.arctan2(np.sum(w * np.sin(theta)), np.sum(w * np.cos(theta)))
    return (z - z_c + Lz / 2.0) % Lz - Lz / 2.0


def main():
    t_run = time.time()
    parser = argparse.ArgumentParser(
        description="Density and dipole-orientation profiles of a water slab along z.",
        allow_abbrev=False)
    parser.add_argument("-f", required=True, metavar="FILE", dest="traj",
                        help="xyz trajectory of the slab (Angstrom)")
    parser.add_argument("-cell", nargs=3, type=float, metavar=("LX", "LY", "LZ"),
                        help="orthorhombic cell lengths in Angstrom (default: from "
                             "the extxyz Lattice entry)")
    parser.add_argument("-dz", type=float, default=0.2, metavar="A",
                        help="bin width along z (default 0.2)")
    parser.add_argument("-max", type=int, default=10**9, metavar="N", dest="nframes_max",
                        help="maximum number of frames to use (default: all)")
    parser.add_argument("-nfreq_topo", type=int, default=1, metavar="N",
                        help="recompute the H->O assignment every N frames")
    parser.add_argument("-out", default="slab", metavar="PREFIX",
                        help="output prefix (default slab -> slab_profiles.dat)")
    args = parser.parse_args()

    cell = None if args.cell is None else np.array(args.cell, dtype=float)
    nframes_total = min(args.nframes_max, count_xyz_frames(args.traj))
    t_start = time.time()
    nframes = 0
    for symbols, coords, comment in iter_xyz_frames(args.traj, args.nframes_max):
        if nframes == 0:
            if cell is None:
                cell = read_cell_from_comment(comment)
            if cell is None:
                sys.exit("Error: no cell found in the trajectory; use -cell.")
            Lx, Ly, Lz = cell
            nbins = int(round(Lz / args.dz))
            edges = np.linspace(-Lz / 2.0, Lz / 2.0, nbins + 1)
            symbols = np.array(symbols)
            is_O = symbols == "O"
            masses = np.array([MASS.get(s, 1.0) for s in symbols])
            count_O = np.zeros(nbins)
            count_H = np.zeros(nbins)
            sum_cos = np.zeros(nbins)
            n_mol = np.zeros(nbins)
            print(f"# cell {Lx:.3f} x {Ly:.3f} x {Lz:.3f} A, {nbins} bins of {Lz / nbins:.3f} A")

        if nframes % args.nfreq_topo == 0:
            molecules = assign_hydrogens(symbols, coords, cell)

        z = centre_slab(coords[:, 2], Lz, masses, is_O)
        count_O += np.histogram(z[is_O], bins=edges)[0]
        count_H += np.histogram(z[~is_O], bins=edges)[0]

        for iO, iH1, iH2 in molecules:
            bisector = (minimum_image(coords[iH1] - coords[iO], cell)
                        + minimum_image(coords[iH2] - coords[iO], cell))
            cos_theta = bisector[2] / np.linalg.norm(bisector)
            ibin = min(int((z[iO] + Lz / 2.0) / (Lz / nbins)), nbins - 1)
            sum_cos[ibin] += cos_theta
            n_mol[ibin] += 1
        nframes += 1
        progress(nframes, nframes_total, f"reading {args.traj}", t_start)

    if nframes == 0:
        sys.exit("Error: no frames read.")

    dz = Lz / nbins
    z_centres = 0.5 * (edges[1:] + edges[:-1])
    norm = nframes * Lx * Ly * dz
    with np.errstate(invalid="ignore", divide="ignore"):
        avg_cos = np.where(n_mol > 0, sum_cos / n_mol, 0.0)
    np.savetxt(f"{args.out}_profiles.dat",
               np.column_stack([z_centres, count_O / norm, count_H / norm, avg_cos, n_mol / nframes]),
               header="z(A)   rho_O(A^-3)   rho_H(A^-3)   <cos theta>   molecules/frame",
               fmt="%12.5e")
    print(f"# Read {nframes} frames, {len(molecules)} molecules.")
    print(f"# Wrote {args.out}_profiles.dat")

    elapsed = int(round(time.time() - t_run))
    print(f"# Finished in {elapsed // 3600:02d}:{elapsed % 3600 // 60:02d}:{elapsed % 60:02d}")


if __name__ == "__main__":
    main()
