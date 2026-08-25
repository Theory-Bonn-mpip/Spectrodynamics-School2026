#!/usr/bin/env python3
"""
spce_dipole.py — total dipole moment of a water trajectory from fixed
SPC/E point charges.

What is computed
----------------
Every atom carries a fixed partial charge (SPC/E: q_O = -0.8476 e,
q_H = +0.4238 e). The dipole of each water molecule is

    mu_mol = q_O r_O + q_H r_H1 + q_H r_H2

where the hydrogen positions are taken relative to their own oxygen with
the minimum-image convention, r_H = r_O + mic(r_H - r_O). This makes the
molecular dipole independent of how the periodic box is cut through the
molecule, and since every molecule is neutral the total dipole

    mu = sum over molecules of mu_mol

does not depend on the choice of origin either. Hydrogens are assigned to
their nearest oxygen (minimum-image distance) on the first frame and the
assignment is kept for the whole trajectory.

Input
-----
An ascii xyz trajectory (Angstrom) of water, with constant atom order. If
the comment line of the frames carries an extxyz Lattice="..." entry the
orthorhombic cell is read from it; otherwise -cell is required.

Output
------
One line per frame with mu_x mu_y mu_z, preceded by '#' header lines —
directly readable by ir_raman.py -dip. Units: atomic units (e*bohr) by
default, or e*Angstrom with -units eang.

Example
-------
  python3 spce_dipole.py -f traj.xyz -max 25000 -out dipole_spce.dat
"""

import argparse
import re
import sys
import time

import numpy as np

BOHR_IN_ANGSTROM = 0.529177210903


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
    """Yield (symbols, coords, comment) for up to max_frames frames.

    Frames are read one at a time so that arbitrarily long trajectories
    can be processed with a constant memory footprint. Only the element
    symbol and three numbers per atom line are used.
    """
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
    index_H = [i for i, s in enumerate(symbols) if s == "H"]
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


def main():
    t_run = time.time()
    parser = argparse.ArgumentParser(
        description="Total dipole of a water trajectory from SPC/E point charges.")
    parser.add_argument("-f", dest="traj", required=True, metavar="FILE",
                        help="xyz trajectory of water (Angstrom)")
    parser.add_argument("-max", required=True, type=int, metavar="N", dest="nframes_max",
                        help="maximum number of frames to use")
    parser.add_argument("-cell", nargs=3, type=float, metavar=("LX", "LY", "LZ"),
                        help="orthorhombic cell lengths in Angstrom (read from the "
                             "extxyz Lattice entry if omitted)")
    parser.add_argument("-out", default="dipole_spce.dat", metavar="FILE",
                        help="output file (default dipole_spce.dat)")
    parser.add_argument("-units", choices=["au", "eang"], default="au",
                        help="dipole units: atomic units (default) or e*Angstrom")
    parser.add_argument("-q_O", type=float, default=-0.8476, help="oxygen charge in e (default -0.8476)")
    parser.add_argument("-q_H", type=float, default=0.4238, help="hydrogen charge in e (default 0.4238)")
    args = parser.parse_args()

    scale = 1.0 if args.units == "eang" else 1.0 / BOHR_IN_ANGSTROM
    unit_label = "e*Angstrom" if args.units == "eang" else "a.u. (e*bohr)"

    cell = np.array(args.cell) if args.cell is not None else None
    molecules = None
    dipoles = []
    nframes_total = min(args.nframes_max, count_xyz_frames(args.traj))
    t_start = time.time()
    for symbols, coords, comment in iter_xyz_frames(args.traj, args.nframes_max):
        if cell is None:
            cell = read_cell_from_comment(comment)
            if cell is None:
                sys.exit("Error: no Lattice entry in the trajectory; give -cell.")
        if molecules is None:
            molecules = assign_hydrogens(symbols, coords, cell)
            iO = np.array([m[0] for m in molecules])
            iH1 = np.array([m[1] for m in molecules])
            iH2 = np.array([m[2] for m in molecules])
        pos_O = coords[iO]
        # hydrogen positions relative to their own oxygen, minimum-imaged
        pos_H1 = pos_O + minimum_image(coords[iH1] - pos_O, cell)
        pos_H2 = pos_O + minimum_image(coords[iH2] - pos_O, cell)
        mu_mol = args.q_O * pos_O + args.q_H * (pos_H1 + pos_H2)   # (nmol, 3), e*Angstrom
        dipoles.append(mu_mol.sum(axis=0) * scale)
        progress(len(dipoles), nframes_total, f"reading {args.traj}", t_start)

    if not dipoles:
        sys.exit(f"Error: could not read any frame from {args.traj}")
    dipoles = np.array(dipoles)
    header = (f"total SPC/E dipole, {unit_label}; q_O = {args.q_O} e, q_H = {args.q_H} e; "
              f"{len(molecules)} molecules, {len(dipoles)} frames\nmu_x mu_y mu_z")
    np.savetxt(args.out, dipoles, header=header, fmt="%18.10e")
    print(f"# Read {len(dipoles)} frames of {len(molecules)} water molecules "
          f"(cell {cell[0]:.4f} {cell[1]:.4f} {cell[2]:.4f} Angstrom).")
    print(f"# Wrote {args.out}")

    elapsed = int(round(time.time() - t_run))
    print(f"# Finished in {elapsed // 3600:02d}:{elapsed % 3600 // 60:02d}:{elapsed % 60:02d}")


if __name__ == "__main__":
    main()
