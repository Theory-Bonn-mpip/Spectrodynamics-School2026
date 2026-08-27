"""
Trade-off schematic (empty frame):
    y = quality of the potential energy surface
    x = complexity of the spectroscopic observable
plus a dashed iso-cost line and the shaded region that stays affordable.

Everything you are likely to change lives in the CONFIG block below.
Nothing below the "DRAWING" banner contains a hard-coded font size or colour.

Output: tradeoff_frame.pdf / tradeoff_frame.png
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow, Polygon

# ======================================================================
# CONFIG - edit this block only
# ======================================================================

# ---------------------------------------------------------------- fonts
# One family for everything. Journals usually want the figure font to
# match the body text: "sans-serif" (Helvetica/Arial look) or "serif"
# (Times look). Set FONT_NAME to a font installed on your machine, or
# leave it as None to use matplotlib's default for that family.
FONT_FAMILY = "sans-serif"      # "sans-serif" or "serif"
FONT_NAME = None                # e.g. "Helvetica", "Arial", "Times New Roman"

# Font sizes in points. These are ABSOLUTE - they do not scale when you
# change FIG_W / FIG_H, so if you shrink the figure the text gets
# relatively bigger. Rule of thumb for a single-column journal figure:
# nothing below 7 pt in the final PDF.
FS_AXIS_TITLE = 12#9.0     # "quality of the PES", "complexity of the ..."
FS_TICK_MAIN = 11 #9.0      # "IR / Raman", "SFG", "2D IR-Raman"
FS_TICK_SUB = 9#8.5       # the correlation-function formulas under them
FS_YTICK = 11 #8.5          # "empirical force field", etc.
FS_ANNOT = 8.5          # anything you add on top later

# Font weights: "normal", "medium", "semibold", "bold".
FW_AXIS_TITLE = "normal"
FW_TICK_MAIN = "normal"

# Set to True to render all text with a real LaTeX installation instead
# of matplotlib's built-in mathtext (slower, but matches a LaTeX paper
# exactly). Requires latex + dvipng on the PATH.
USE_LATEX = False

# --------------------------------------------------------------- canvas
FIG_W, FIG_H = 6.3, 3.6    # inches
DPI_PNG = 400

# --------------------------------------------------------------- colours
C_AXIS = "#3D3D3A"      # axis arrows and axis titles
C_TICK = "#5F5E5A"      # tick marks and tick labels
C_BAND = "#1D9E75"      # dashed budget line and shaded region
BAND_ALPHA = 0.10       # transparency of the shaded region

# ---------------------------------------------------------- line weights
LW_LINE = 1.4           # dashed budget line
LW_TICK = 0.8           # tick marks
DASH = (0, (5, 3))      # dash pattern of the budget line

# -------------------------------------------------------------- geometry
XMAX, YMAX = 3.35, 3.35          # length of the two axis arrows
XTICKS = [0.75, 1.70, 2.65]      # the three spectroscopies
YTICKS = [0.60, 1.60, 2.60]      # the three levels of theory
LINE = [(0.00, 2.95), (3.15, 0.55)]   # end points of the dashed line

# Gaps between the axes and the text, in data units. Increase these if a
# larger font size makes labels collide with the axis arrows.
PAD_XTICK_MAIN = 0.16    # axis -> "IR / Raman"
PAD_XTICK_SUB = 0.38     # axis -> formula line
PAD_XTITLE = 0.82        # axis -> "complexity of the ..."
PAD_YTICK = 0.12         # axis -> "empirical force field"

# ---------------------------------------------------------------- labels
X_TITLE = ''# "complexity of the spectroscopic observable"
Y_TITLE = '' #"PES computational cost"
XLABELS = [
    ("IR / Raman",  r"$\langle\mu(0)\mu(t)\rangle$, $\langle\alpha(0)\alpha(t)\rangle$"),
    ("SFG",         r"$\langle\alpha(0)\mu(t)\rangle$"),
    ("2D IR-Raman", r"$R(t_1,t_2)$"),
]
YLABELS = [
    "empirical force field",
    "machine-learned potential",
    "on-the-fly electronic structure",
]

# ======================================================================
# DRAWING - no font sizes or colours below this line
# ======================================================================

rc = {
    "font.family": FONT_FAMILY,
    "font.size": FS_TICK_MAIN,
    "mathtext.fontset": "dejavusans" if FONT_FAMILY == "sans-serif" else "dejavuserif",
    "text.usetex": USE_LATEX,
    "pdf.fonttype": 42,     # keep text editable in Illustrator / Inkscape
    "ps.fonttype": 42,
}
if FONT_NAME is not None:
    rc[f"font.{FONT_FAMILY}"] = [FONT_NAME]
plt.rcParams.update(rc)

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

# affordable region: under the dashed line, above the x axis
(x0, y0), (x1, y1) = LINE
ax.add_patch(Polygon([(x0, y0), (x1, y1), (x1, 0.0), (x0, 0.0)],
                     closed=True, facecolor=C_BAND, alpha=BAND_ALPHA,
                     edgecolor="none", zorder=0))

# dashed budget line
ax.plot([x0, x1], [y0, y1], ls=DASH, lw=LW_LINE, color=C_BAND, zorder=1)

# axes drawn as arrows
head = dict(head_width=0.075, head_length=0.11, length_includes_head=True,
            color=C_AXIS, zorder=3)
ax.add_patch(FancyArrow(0, 0, XMAX, 0, width=0.003, **head))
ax.add_patch(FancyArrow(0, 0, 0, YMAX, width=0.003, **head))

# tick marks
for x in XTICKS:
    ax.plot([x, x], [0, -0.06], lw=LW_TICK, color=C_TICK, clip_on=False, zorder=3)
for y in YTICKS:
    ax.plot([0, -0.06], [y, y], lw=LW_TICK, color=C_TICK, clip_on=False, zorder=3)

# tick labels
for x, (name, formula) in zip(XTICKS, XLABELS):
    ax.text(x, -PAD_XTICK_MAIN, name, ha="center", va="top",
            fontsize=FS_TICK_MAIN, fontweight=FW_TICK_MAIN, color=C_AXIS)
    ax.text(x, -PAD_XTICK_SUB, formula, ha="center", va="top",
            fontsize=FS_TICK_SUB, color=C_TICK)
for y, name in zip(YTICKS, YLABELS):
    ax.text(-PAD_YTICK, y, name, ha="right", va="center",
            fontsize=FS_YTICK, color=C_TICK)

# axis titles
ax.text(XMAX / 2, -PAD_XTITLE, X_TITLE, ha="center", va="top",
        fontsize=FS_AXIS_TITLE, fontweight=FW_AXIS_TITLE, color=C_AXIS)
ax.text(-PAD_YTICK, YMAX, Y_TITLE, ha="right", va="center",
        fontsize=FS_AXIS_TITLE, fontweight=FW_AXIS_TITLE, color=C_AXIS)

ax.set_xlim(-0.1, XMAX + 0.15)
ax.set_ylim(-(PAD_XTITLE + 0.23), YMAX + 0.2)
ax.axis("off")

fig.tight_layout(pad=0.2)
fig.savefig("tradeoff_frame.pdf", bbox_inches="tight", transparent=True)
fig.savefig("tradeoff_frame.png", dpi=DPI_PNG, bbox_inches="tight")
print("wrote tradeoff_frame.pdf and tradeoff_frame.png")

# ======================================================================
# ANNOTATION EXAMPLES - uncomment, then re-run to draw on top
# ======================================================================
# filled marker (explicit mu, alpha surfaces):
#   ax.plot(1.70, 1.60, "o", ms=8, mfc=C_BAND, mec="#0F6E56", mew=0.6, zorder=4)
#
# open marker (properties from velocities / fixed derivatives):
#   ax.plot(1.70, 2.60, "o", ms=8, mfc="white", mec=C_BAND, mew=1.5, zorder=4)
#
# label above a marker:
#   ax.text(1.70, 2.78, "AIMD + velocity approach", ha="center",
#           fontsize=FS_ANNOT, color=C_AXIS)
#
# label riding on the dashed line:
#   ax.text(0.05, 3.02, "fixed computational budget",
#           fontsize=FS_ANNOT, color="#0F6E56")
