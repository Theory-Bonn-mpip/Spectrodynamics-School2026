"""
Schematic: sampling a molecular dynamics trajectory and evaluating the
dipole moment mu and the polarizability alpha at each sampled frame.

Output: traj_sampling.pdf / traj_sampling.png
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrow, FancyArrowPatch

# ----------------------------------------------------------------------
# style
# ----------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "mathtext.fontset": "dejavusans",
    "pdf.fonttype": 42,   # editable text in Illustrator / Inkscape
    "ps.fonttype": 42,
})

C_O = "#E24B4A"      # oxygen
C_H = "#D3D1C7"      # hydrogen
C_BOND = "#5F5E5A"
C_DIP = "#7F77DD"    # dipole vector
C_AXIS = "#3D3D3A"
C_LEAD = "#B4B2A9"

R_O, R_H = 0.28, 0.18   # atom radii (data units)

# ----------------------------------------------------------------------
# geometry of the snapshots: (x, y) of O, then the two O-H bond angles
# measured from the +x axis (degrees), then the bond lengths.
# Jitter them a little so the molecule looks like it is vibrating/rotating.
# ----------------------------------------------------------------------
snapshots = [
    dict(pos=(1.0, 1.00), angles=(-142, -38), lengths=(0.62, 0.62), tilt=0),
    dict(pos=(3.0, 1.06), angles=(-150, -32), lengths=(0.66, 0.58), tilt=12),
    dict(pos=(5.0, 0.96), angles=(-133, -46), lengths=(0.58, 0.64), tilt=-11),
    dict(pos=(7.0, 1.03), angles=(-146, -28), lengths=(0.64, 0.60), tilt=9),
]

Y_AXIS = 0.0          # the time arrow
Y_LABEL = -0.42       # mu(t_i), alpha(t_i) labels


def draw_bond(ax, p0, p1, r0, r1):
    """Draw a bond that stops at the edge of both atoms."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    u = d / np.linalg.norm(d)
    a, b = p0 + u * r0, p1 - u * r1
    ax.plot([a[0], b[0]], [a[1], b[1]], color=C_BOND, lw=2.0,
            solid_capstyle="round", zorder=2)


def draw_molecule(ax, snap):
    """Water molecule: one O, two H, plus the dipole vector."""
    o = np.asarray(snap["pos"], float)
    hs = []
    for ang, L in zip(snap["angles"], snap["lengths"]):
        th = np.deg2rad(ang)
        hs.append(o + L * np.array([np.cos(th), np.sin(th)]))

    for h in hs:
        draw_bond(ax, o, h, R_O, R_H)
    for h in hs:
        ax.add_patch(Circle(h, R_H, facecolor=C_H, edgecolor=C_BOND,
                            lw=0.6, zorder=3))
    ax.add_patch(Circle(o, R_O, facecolor=C_O, edgecolor="#A32D2D",
                        lw=0.6, zorder=3))

    # dipole: along the HOH bisector, pointing away from the hydrogens
    bis = o - 0.5 * (hs[0] + hs[1])
    bis /= np.linalg.norm(bis)
    th = np.deg2rad(snap["tilt"])
    rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    bis = rot @ bis

    start = o + bis * (R_O + 0.02)
    end = start + bis * 0.55
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                                 mutation_scale=9, lw=1.6,
                                 color=C_DIP, zorder=4))


# ----------------------------------------------------------------------
# figure
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.6, 2.5))

# time arrow
ax.add_patch(FancyArrow(0.2, Y_AXIS, 7.9, 0.0, width=0.004,
                        head_width=0.11, head_length=0.16,
                        length_includes_head=True,
                        color=C_AXIS, zorder=2))
ax.text(8.15, Y_AXIS + 0.16, "time", ha="right", va="bottom",
        fontsize=9, color=C_AXIS)

for i, snap in enumerate(snapshots, start=1):
    x = snap["pos"][0]
    draw_molecule(ax, snap)

    # dashed leader from the snapshot down to the trajectory
    ax.plot([x, x], [snap["pos"][1] - 0.55, Y_AXIS + 0.10],
            ls=(0, (2, 2)), lw=0.8, color=C_LEAD, zorder=1)

    # tick on the trajectory
    ax.plot(x, Y_AXIS, marker="o", ms=5, mfc=C_DIP, mec="none", zorder=3)

    ax.text(x, Y_LABEL,
            rf"$\mu(t_{i})$,  $\alpha(t_{i})$",
            ha="center", va="center", fontsize=9.5, color=C_AXIS)

ax.text(4.0, -0.82, r"purple vector = dipole at each sampled frame",
        ha="center", va="center", fontsize=8, color="#73726C")

ax.set_xlim(0.0, 8.3)
ax.set_ylim(-1.05, 2.25)
ax.set_aspect("equal")
ax.axis("off")

fig.tight_layout(pad=0.2)
fig.savefig("traj_sampling.pdf", bbox_inches="tight", transparent=True)
fig.savefig("traj_sampling.png", dpi=400, bbox_inches="tight")
print("wrote traj_sampling.pdf and traj_sampling.png")
