#!/usr/bin/env python3
"""
Radar / spider chart comparing simulation methods against what the problem demands.

Axes (clockwise from top):
    1. accuracy elec. structure   force field -> multireference
    2. accuracy nuclei dynamics   classical -> semi-classical -> quantum
    3. electron dynamics          no -> yes
    4. time scale of the process  fs -> ns
    5. light-matter interaction   none -> explicit field / QED
    6. degrees of freedom         2-3 -> 10^5

Edit AXES and METHODS below; everything else adapts automatically.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import RegularPolygon

# --------------------------------------------------------------------------
# 1. AXES  (label, subtitle)  -- order = clockwise starting at 12 o'clock
# --------------------------------------------------------------------------
AXES = [
    ("accuracy elec. structure", "force field  →  multireference"),
    ("accuracy nuclei dynamics", "classical → semi-classical → quantum"),
    ("electron dynamics",        "no  →  yes"),
    ("time scale of the process", "fs  →  ns"),
    ("light-matter interaction", "implicit  →  explicit field"),
    ("degrees of freedom",       "2 − 3  →  10$^5$"),
]

# --------------------------------------------------------------------------
# 2. METHODS: name -> (values in [0, 1], colour, linestyle, linewidth)
#    Values follow the same order as AXES.
# --------------------------------------------------------------------------
METHODS = {
    "classical MD": dict(
        values=[0.35, 0.50, 1.00, 0.60, 0.02, 0.98],
        color="#27ae60", linestyle=(0, (6, 3)), linewidth=2.2, fill=0.13,
    ),
    "AIMD (DFT)": dict(
        values=[0.7, 0.50, 0.20, 0.90, 0.02, 0.20],
        color="#e8552d", linestyle=(0, (1.5, 2)), linewidth=2.4, fill=0.11,
    ),
    "MLIP": dict(
        values=[0.88, 0.50, 0.80, 0.92, 0.02, 0.70],
        color="#4a4a4a", linestyle=(0, (7, 5)), linewidth=1.6, fill=0.0,
    ),
}

N_RINGS = 5          # number of concentric grid polygons
LABEL_RADIUS = 1.16  # distance of the axis labels from the centre

# --------------------------------------------------------------------------
# 2b. FONT SIZES  -- tweak these to scale the text
#     FONT_SCALE multiplies every size below at once.
# --------------------------------------------------------------------------
FONT_SCALE          = 1.0   # global multiplier for all font sizes
FS_AXIS_TITLE       = 18 #12.5  # axis names around the chart
FS_AXIS_SUBTITLE    = 15 #9.5   # small grey range description under each axis name
FS_LEGEND           = 13 #11.5  # legend entries


def fs(size):
    """Apply the global FONT_SCALE to a base font size."""
    return size * FONT_SCALE


# --------------------------------------------------------------------------
# 3. Helper: polygon-shaped frame instead of the default circular one
# --------------------------------------------------------------------------
def polygon_spine(ax, n_vars, theta):
    """Replace the circular outer spine by an n-sided polygon."""
    ax.set_frame_on(False)
    # outer boundary
    outer = RegularPolygon((0.5, 0.5), n_vars, radius=0.5,
                           orientation=0, transform=ax.transAxes,
                           facecolor="none", edgecolor="#b8b8b8",
                           linewidth=1.0, zorder=0)
    ax.add_patch(outer)
    # inner rings
    closed = np.append(theta, theta[0])
    for r in np.linspace(1.0 / N_RINGS, 1.0, N_RINGS)[:-1]:
        ax.plot(closed, [r] * len(closed), color="#dcdcdc",
                linewidth=0.7, zorder=0)
    # radial spokes
    for t in theta:
        ax.plot([t, t], [0, 1], color="#dcdcdc", linewidth=0.7, zorder=0)


def make_radar(save_as=None, show=False):
    n = len(AXES)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)

    fig = plt.figure(figsize=(11.5, 8.0), dpi=150)
    # explicit axes rect keeps room for the outer labels and the legend
    ax = fig.add_axes([0.30, 0.22, 0.40, 0.66], polar=True)

    # 12 o'clock start, clockwise
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.spines["polar"].set_visible(False)

    polygon_spine(ax, n, theta)

    # ---- data ------------------------------------------------------------
    closed_theta = np.append(theta, theta[0])
    for name, cfg in METHODS.items():
        vals = np.append(cfg["values"], cfg["values"][0])
        ax.plot(closed_theta, vals, label=name,
                color=cfg["color"], linestyle=cfg["linestyle"],
                linewidth=cfg["linewidth"], zorder=3,
                solid_capstyle="round", dash_capstyle="round")
        if cfg["fill"]:
            ax.fill(closed_theta, vals, color=cfg["color"],
                    alpha=cfg["fill"], zorder=2, linewidth=0)

    # ---- axis labels ------------------------------------------------------
    for t, (title, subtitle) in zip(theta, AXES):
        # horizontal / vertical alignment depending on position on the circle
        x = np.sin(t)          # because of offset+direction, sin/cos swap
        y = np.cos(t)
        if abs(x) < 1e-3:
            ha = "center"
        elif x > 0:
            ha = "left"
        else:
            ha = "right"
        # anchor the pair radially, then stack title/subtitle in *points*
        # so they never overlap on the near-horizontal axes
        ax.annotate(title, xy=(t, LABEL_RADIUS),
                    xytext=(0, 3), textcoords="offset points",
                    ha=ha, va="bottom", fontsize=fs(FS_AXIS_TITLE),
                    color="#1a1a1a", zorder=5, annotation_clip=False)
        ax.annotate(subtitle, xy=(t, LABEL_RADIUS),
                    xytext=(0, -4), textcoords="offset points",
                    ha=ha, va="top", fontsize=fs(FS_AXIS_SUBTITLE),
                    color="#7a7a7a", zorder=5, annotation_clip=False)

    # ---- legend -----------------------------------------------------------
    handles, labels = ax.get_legend_handles_labels()
    leg = ax.legend(handles, labels, loc="upper center",
                    bbox_to_anchor=(0.5, -0.30), ncol=2,
                    frameon=False, fontsize=fs(FS_LEGEND),
                    handlelength=2.6, columnspacing=2.5, labelspacing=0.8)
    for txt in leg.get_texts():
        txt.set_color("#1a1a1a")

    if save_as:
        for path in (save_as if isinstance(save_as, (list, tuple)) else [save_as]):
            fig.savefig(path, bbox_inches="tight", facecolor="white")
            print(f"saved: {path}")
    if show:
        plt.show()
    return fig, ax


if __name__ == "__main__":
    make_radar(save_as=["radar_methods.png", "radar_methods.pdf"])
