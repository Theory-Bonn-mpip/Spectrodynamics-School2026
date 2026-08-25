#!/usr/bin/env python3
"""Draw images/acf_scheme.png: how time correlation functions are evaluated
(a) from K independent trajectories and (b) from a single trajectory using
every point as a time origin (after the discussion in Tuckerman, Statistical
Mechanics: Theory and Molecular Simulation, Sec. 13.4). Original drawing."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def bracket(ax, x0, x1, y, h=0.22, lw=1.3, color="k", ls="-"):
    """square bracket connecting points x0 and x1, height h above y (h<0: below)"""
    ax.plot([x0, x0, x1, x1], [y, y + h, y + h, y], color=color, lw=lw, ls=ls,
            solid_capstyle="round")


fig, (axa, axb) = plt.subplots(1, 2, figsize=(9, 3.6),
                               gridspec_kw={"width_ratios": [1, 1.15]})

# (a) direct method: K independent trajectories, each contributes once per lag
npts = 8
for label, y in [("$\\lambda = 1$", 3.0), ("$\\lambda = 2$", 2.0), ("$\\lambda = K$", 0.5)]:
    axa.annotate("", xy=(npts + 0.3, y), xytext=(-0.3, y),
                 arrowprops=dict(arrowstyle="->", lw=1.2))
    axa.plot(range(npts), [y] * npts, "ko", ms=5)
    axa.text(-0.6, y, label, ha="right", va="center", fontsize=11)
    axa.text(npts + 0.45, y, "$t$", va="center", fontsize=11)
    for n, h in zip([1, 2, 3], [0.18, 0.30, 0.42]):
        bracket(axa, 0, n, y + 0.04, h=h, color="k" if n == 1 else "0.35",
                ls="-" if n == 1 else (0, (2, 1.5)))
axa.text(1.2, 1.35, "$\\vdots$", fontsize=16, ha="center")
for n, lab in [(0, "0"), (2, "$2\\Delta t$"), (4, "$4\\Delta t$")]:
    axa.text(n, 0.1, lab, ha="center", va="top", fontsize=10)
axa.set_xlim(-2.6, npts + 1)
axa.set_ylim(-0.4, 3.7)
axa.axis("off")
axa.set_title("(a) direct method: $K$ independent trajectories", fontsize=11, loc="left")

# (b) single trajectory: every point is a time origin
y, npts = 1.8, 9
axb.annotate("", xy=(npts + 0.3, y), xytext=(-0.3, y),
             arrowprops=dict(arrowstyle="->", lw=1.2))
axb.plot(range(npts), [y] * npts, "ko", ms=5)
axb.text(npts + 0.45, y, "$t$", va="center", fontsize=11)
for m in range(0, 5):                                   # lag dt: above
    bracket(axb, m, m + 1, y + 0.04, h=0.18)
for m, off in zip(range(0, 4), [0.0, 0.08, 0.16, 0.24]):  # lag 2dt: below
    bracket(axb, m, m + 2, y - 0.04 - off, h=-0.18, color="0.35", ls=(0, (2, 1.5)))
for n, lab in [(0, "0"), (2, "$2\\Delta t$"), (4, "$4\\Delta t$")]:
    axb.text(n, 0.95, lab, ha="center", va="top", fontsize=10)
axb.text(5.3, y + 0.32, "lag $\\Delta t$", fontsize=10)
axb.text(6.6, y - 0.40, "lag $2\\Delta t$", fontsize=10, color="0.35")
axb.set_xlim(-1, npts + 1)
axb.set_ylim(0.4, 3.0)
axb.axis("off")
axb.set_title("(b) single trajectory: every point is a time origin", fontsize=11, loc="left")

fig.tight_layout()
fig.savefig("images/acf_scheme.png", dpi=200, bbox_inches="tight")
print("wrote images/acf_scheme.png")
