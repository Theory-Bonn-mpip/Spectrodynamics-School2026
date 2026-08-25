#!/usr/bin/env python3
"""Draw images/window_scheme.png: the surface window w(z) used by the SFG
scripts (nmode 1, -zref1 4 -zref2 5) on top of the converged density
profile of the 48-water slab, so its meaning is visible at a glance.
Original drawing."""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
PROFILE = HERE / "../../part_iii/excercise_5/reference_results/slab-048_1ns_profiles.dat"
OUT = HERE / "../../images/window_scheme.png"

ZREF1, ZREF2 = 4.0, 5.0
RHO_BULK = 0.0334


def window(z, z1=ZREF1, z2=ZREF2):
    """nmode 1 window of ssvvcf_ml.py / sfg_atomic.py."""
    ramp = np.pi / 2.0 / (z2 - z1)
    out = np.zeros_like(z)
    out[z >= z2] = 1.0
    out[z <= -z2] = -1.0
    up = (z > z1) & (z < z2)
    out[up] = np.sin(ramp * (z[up] - z1))
    down = (z > -z2) & (z < -z1)
    out[down] = np.sin(ramp * (z[down] + z1))
    return out


prof = np.loadtxt(PROFILE)
z_rho, rho = prof[:, 0], prof[:, 1]

fig, ax = plt.subplots(figsize=(7, 3.2), constrained_layout=True)
ax.fill_between(z_rho, rho / RHO_BULK, color="0.85", zorder=0,
                label=r"$\rho_\mathrm{O}(z)\,/\,\rho_\mathrm{bulk}$ (48 H$_2$O slab, 1 ns)")

z = np.linspace(-14, 14, 601)
ax.plot(z, window(z), "r-", lw=2, label=r"window $w(z)$  (-zref1 4 -zref2 5)")

for zr in (ZREF1, ZREF2):
    for s in (+1, -1):
        ax.axvline(s * zr, color="0.5", lw=0.7, ls=":")
ax.axhline(0, color="0.5", lw=0.5)
ax.text(0, -0.45, "bulk\nremoved", ha="center", va="center", fontsize=9)
ax.text(11, 1.06, "top surface: $+1$", ha="center", fontsize=9)
ax.text(-11, -1.18, "bottom surface: $-1$", ha="center", fontsize=9)
ax.annotate("sine ramp", xy=(4.6, 0.55), xytext=(10.5, 0.35), fontsize=9,
            color="r", ha="center", va="center",
            arrowprops=dict(arrowstyle="->", color="r", lw=0.9))
ax.set_xlabel(r"$z$ / Å  (0 = slab centre)")
ax.set_xlim(-14, 14)
ax.set_ylim(-1.45, 1.45)
ax.set_yticks([-1, 0, 1])
ax.legend(fontsize=8, loc="upper left")
fig.savefig(OUT, dpi=200)
print(f"wrote {OUT.resolve()}")
