"""
Venn diagram of a total Hamiltonian split into three subsystems and their
pairwise couplings:

    H = H_e + H_n + H_ph + H_e-n + H_e-ph + H_n-ph

Pure matplotlib (no matplotlib-venn), so every element is directly editable.
Everything you are likely to want to change lives in the CONFIG block below.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------

R = 1.0  # circle radius; all geometry is in units of R

# Circle centres. Move these to change the amount of overlap.
CENTERS = {
    "e":  (-0.538,  0.308),
    "n":  ( 0.538,  0.308),
    "ph": ( 0.000, -0.615),
}

COLORS = {
    "e":  "#7F77DD",   # purple
    "n":  "#1D9E75",   # teal
    "ph": "#D85A30",   # coral
}

FILL_ALPHA = 0.40
EDGE_WIDTH = 1.0

# Labels for the exclusive region of each circle: (math label, caption)
SOLO_LABELS = {
    "e":  (r"$\hat{H}_{e}$",  "electrons"),
    "n":  (r"$\hat{H}_{n}$",  "nuclei"),
    "ph": (r"$\hat{H}_{ph}$", "photons"),
}

# Labels for the pairwise overlaps, keyed by the pair of circles.
PAIR_LABELS = {
    ("e", "n"):  r"$\hat{H}_{e-n}$",
    ("e", "ph"): r"$\hat{H}_{e-ph}$",
    ("n", "ph"): r"$\hat{H}_{n-ph}$",
}

# Label for the triple overlap. Set to None to leave it blank.
TRIPLE_LABEL =None 

# Radial distance of each label from the centroid, in units of R.
# Increase SOLO_OFFSET to push the single-subsystem labels further outward.
SOLO_OFFSET = 1.30
PAIR_OFFSET = 0.65

FONTSIZE_MATH = 22
FONTSIZE_CAPTION = 15
FONTSIZE_TRIPLE = 10

FIGSIZE = (7.2, 6.0)
DPI = 200
OUTFILES = ["hamiltonian_venn.png", "hamiltonian_venn.pdf"]

# ----------------------------------------------------------------------
# GEOMETRY HELPERS
# ----------------------------------------------------------------------

C = {k: np.asarray(v, dtype=float) for k, v in CENTERS.items()}
CENTROID = np.mean(list(C.values()), axis=0)


def _outward(point, distance):
    """Push a point radially away from the centroid by `distance` * R."""
    d = np.asarray(point, dtype=float) - CENTROID
    norm = np.linalg.norm(d)
    if norm < 1e-9:
        return np.asarray(point, dtype=float)
    return CENTROID + d / norm * distance * R


def solo_position(key):
    """Label position inside the exclusive lune of one circle."""
    return _outward(C[key], SOLO_OFFSET)


def pair_position(a, b):
    """Label position inside the two-way overlap of circles a and b."""
    midpoint = 0.5 * (C[a] + C[b])
    return _outward(midpoint, PAIR_OFFSET)


# ----------------------------------------------------------------------
# PLOT
# ----------------------------------------------------------------------

def make_figure():
    fig, ax = plt.subplots(figsize=FIGSIZE)

    for key, center in C.items():
        ax.add_patch(Circle(center, R,
                            facecolor=COLORS[key], alpha=FILL_ALPHA,
                            edgecolor="none", zorder=1))
        ax.add_patch(Circle(center, R,
                            facecolor="none",
                            edgecolor=COLORS[key], linewidth=EDGE_WIDTH,
                            zorder=2))

    for key, (math_label, caption) in SOLO_LABELS.items():
        x, y = solo_position(key)
        ax.text(x, y + 0.06, math_label, ha="center", va="center",
                fontsize=FONTSIZE_MATH, zorder=3)
        ax.text(x, y - 0.13, caption, ha="center", va="center",
                fontsize=FONTSIZE_CAPTION, color="#555555", zorder=3)

    for (a, b), math_label in PAIR_LABELS.items():
        x, y = pair_position(a, b)
        ax.text(x, y, math_label, ha="center", va="center",
                fontsize=FONTSIZE_MATH, zorder=3)

    if TRIPLE_LABEL:
        ax.text(CENTROID[0], CENTROID[1], TRIPLE_LABEL,
                ha="center", va="center", fontsize=FONTSIZE_TRIPLE,
                color="#777777", style="italic", zorder=3)

    all_xy = np.array(list(C.values()))
    pad = 0.18 * R
    ax.set_xlim(all_xy[:, 0].min() - R - pad, all_xy[:, 0].max() + R + pad)
    ax.set_ylim(all_xy[:, 1].min() - R - pad, all_xy[:, 1].max() + R + pad)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.tight_layout()
    return fig, ax


if __name__ == "__main__":
    fig, ax = make_figure()
    for path in OUTFILES:
        fig.savefig(path, dpi=DPI, bbox_inches="tight", transparent=True)
    plt.show()
