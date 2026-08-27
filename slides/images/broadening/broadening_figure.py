"""
Homogeneous vs. inhomogeneous line broadening — publication-style figure.

Panel a: two Lorentzian lines with equal integrated area. The width is set by
         the relaxation time T2 (FWHM = 1 / (pi * T2) in angular-frequency-free
         units), so faster relaxation gives a shorter, broader line.
Panel b: several narrow Lorentzians at slightly different centre frequencies.
         Their sum (and the Gaussian distribution of centres) produces a broad
         envelope that hides the true homogeneous width.

Run:  python broadening_figure.py
Out:  broadening.png, broadening.pdf, broadening.svg
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

# =============================================================================
# FONT SIZES  <-- edit these
# =============================================================================
FS_PANEL_TITLE = 20 #15.0   # "a) homogeneous broadening"
FS_CURVE_LABEL = 13 #13.0   # "long T2", "many shifted lines"
FS_CURVE_SUB = 11.0     # "slow relaxation", "narrow lines sum to ..."
FS_AXIS_LABEL = 11.0    # the "frequency" axis label
FS_ANNOTATION = 10.5    # FWHM annotation (only drawn if SHOW_FWHM = True)

FONT_FAMILY = "DejaVu Sans"   # e.g. "Arial", "Helvetica", "DejaVu Serif"
TITLE_WEIGHT = "normal"       # "normal" | "bold"
LABEL_WEIGHT = "bold"         # weight of the FS_CURVE_LABEL texts

# =============================================================================
# FIGURE SIZE / OUTPUT  <-- edit these
# =============================================================================
FIG_W, FIG_H = 9.0, 4.2   # inches
DPI = 300
OUT_BASENAME = "broadening"
OUT_FORMATS = ("png", "pdf", "svg")
TRANSPARENT = False

# =============================================================================
# COLOURS AND LINE WIDTHS  <-- edit these
# =============================================================================
COLOR_BROAD = "#D64A2B"    # observed / homogeneous lineshapes
COLOR_LINES = "#1D9E75"    # individual packets in panel b
COLOR_AXIS = "#4A4A48"
COLOR_TEXT = "#1A1A19"

LW_BROAD = 2.0
LW_LINES = 1.2
LW_AXIS = 1.0

FILL_ALPHA = 0.10          # 0 = no fill under the curves

# =============================================================================
# PHYSICS  <-- edit these
# =============================================================================
# Panel a: two homogeneous lines, same area, different T2.
T2_SLOW = 1.00             # long T2  -> narrow line   (arbitrary units)
T2_FAST = 0.16             # short T2 -> broad line
PEAK_AREA = 1.0            # integrated intensity, identical for both lines

# Panel b: inhomogeneous ensemble. The envelope is the actual sum of the drawn
# packets, so the two are guaranteed to be consistent.
N_PACKETS = 11             # number of individual spin packets drawn and summed
SPREAD_FWHM = 3.4          # FWHM of the Gaussian distribution of centre frequencies
COVERAGE_SIGMAS = 2.3      # how far out the packets are placed, in sigma
PACKET_FWHM_RATIO = 1.7    # packet FWHM / packet spacing
                           #   ~1.0 -> smooth envelope
                           #   <0.7 -> visibly scalloped sum (spikier lines)

SHOW_FWHM = False          # draw a double-headed FWHM arrow on each panel-a line

# =============================================================================
# TEXT  <-- edit these
# =============================================================================
TITLE_A = "homogeneous broadening"
TITLE_B = "inhomogeneous broadening"
LABEL_A1, SUB_A1 =  "slow relaxation", ''
LABEL_A2, SUB_A2 = "fast relaxation" ,''
LABEL_B, SUB_B = "large variety of local environments", "narrow(er) lines sum up to a broad envelope"
XLABEL = "frequency"

# =============================================================================
# Lineshape helpers
# =============================================================================


def lorentzian(x, x0, t2, area=1.0):
    """Area-normalised Lorentzian; FWHM = 1 / (pi * T2)."""
    hwhm = 1.0 / (2.0 * np.pi * t2)
    return area * hwhm / (np.pi * ((x - x0) ** 2 + hwhm**2))


def gaussian(x, x0, fwhm, area=1.0):
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return area / (sigma * np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * ((x - x0) / sigma) ** 2)


def style_axis(ax, xmax_text):
    """Bare frequency axis with an arrowhead, no ticks, no box."""
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(COLOR_AXIS)
    ax.spines["bottom"].set_linewidth(LW_AXIS)
    ax.xaxis.set_major_locator(NullLocator())
    ax.yaxis.set_major_locator(NullLocator())
    ax.plot(1, 0, marker=">", transform=ax.get_yaxis_transform(),
            clip_on=False, color=COLOR_AXIS, markersize=5)
    ax.text(xmax_text, -0.06, XLABEL, transform=ax.get_xaxis_transform(),
            ha="right", va="top", fontsize=FS_AXIS_LABEL, color=COLOR_AXIS)


def caption(ax, x, label, sub):
    """Two-line caption under a curve, in axes-fraction y."""
    ax.text(x, -0.20, label, transform=ax.get_xaxis_transform(), ha="center",
            va="top", fontsize=FS_CURVE_LABEL, fontweight=LABEL_WEIGHT,
            color=COLOR_TEXT)
    ax.text(x, -0.33, sub, transform=ax.get_xaxis_transform(), ha="center",
            va="top", fontsize=FS_CURVE_SUB, color=COLOR_AXIS)


def fwhm_marker(ax, x0, t2, peak):
    width = 1.0 / (np.pi * t2)
    y = peak / 2.0
    ax.annotate("", xy=(x0 - width / 2, y), xytext=(x0 + width / 2, y),
                arrowprops=dict(arrowstyle="<->", color=COLOR_AXIS, lw=0.9))
    ax.text(x0, y * 1.08, r"$1/\pi T_2$", ha="center", va="bottom",
            fontsize=FS_ANNOTATION, color=COLOR_AXIS)


# =============================================================================
# Figure
# =============================================================================


def make_figure():
    plt.rcParams["font.family"] = FONT_FAMILY
    plt.rcParams["svg.fonttype"] = "none"          # keep text editable in SVG
    plt.rcParams["pdf.fonttype"] = 42              # TrueType, editable in Illustrator

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(FIG_W, FIG_H), dpi=DPI)
    fig.subplots_adjust(left=0.03, right=0.97, top=0.86, bottom=0.26, wspace=0.14)

    # ---------------- panel a ----------------
    span = 3.0 / (np.pi * T2_FAST)
    c1, c2 = -span * 0.30, span * 0.30
    x = np.linspace(c1 - span * 0.45, c2 + span * 0.45, 4000)

    y1 = lorentzian(x, c1, T2_SLOW, PEAK_AREA)
    y2 = lorentzian(x, c2, T2_FAST, PEAK_AREA)

    for y in (y1, y2):
        ax_a.plot(x, y, color=COLOR_BROAD, lw=LW_BROAD, solid_joinstyle="round")
        if FILL_ALPHA:
            ax_a.fill_between(x, y, color=COLOR_BROAD, alpha=FILL_ALPHA, lw=0)

    if SHOW_FWHM:
        fwhm_marker(ax_a, c1, T2_SLOW, y1.max())
        fwhm_marker(ax_a, c2, T2_FAST, y2.max())

    ax_a.set_xlim(x[0], x[-1])
    ax_a.set_ylim(0, y1.max() * 1.12)
    ax_a.set_title(TITLE_A, fontsize=FS_PANEL_TITLE, fontweight=TITLE_WEIGHT,
                   color=COLOR_TEXT, pad=14)
    style_axis(ax_a, x[-1])
    caption(ax_a, c1, LABEL_A1, SUB_A1)
    caption(ax_a, c2, LABEL_A2, SUB_A2)

    # ---------------- panel b ----------------
    xb = np.linspace(-SPREAD_FWHM * 1.6, SPREAD_FWHM * 1.6, 6000)
    sigma = SPREAD_FWHM / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    centres = np.linspace(-COVERAGE_SIGMAS * sigma, COVERAGE_SIGMAS * sigma,
                          N_PACKETS)
    spacing = centres[1] - centres[0]

    # Each packet is one homogeneous line. Its area is the population sitting at
    # that centre frequency, read off the Gaussian distribution of centres.
    packet_fwhm = PACKET_FWHM_RATIO * spacing
    packet_t2 = 1.0 / (np.pi * packet_fwhm)
    weights = gaussian(centres, 0.0, SPREAD_FWHM) * spacing

    packets = np.array([lorentzian(xb, c, packet_t2, w)
                        for c, w in zip(centres, weights)])
    envelope = packets.sum(axis=0)      # the observed line IS the sum

    ax_b.plot(xb, envelope, color=COLOR_BROAD, lw=LW_BROAD, zorder=2)
    if FILL_ALPHA:
        ax_b.fill_between(xb, envelope, color=COLOR_BROAD, alpha=FILL_ALPHA,
                          lw=0, zorder=1)
    for p in packets:
        ax_b.plot(xb, p, color=COLOR_LINES, lw=LW_LINES, zorder=3)

    ax_b.set_xlim(xb[0], xb[-1])
    ax_b.set_ylim(0, envelope.max() * 1.12)
    ax_b.set_title(TITLE_B, fontsize=FS_PANEL_TITLE, fontweight=TITLE_WEIGHT,
                   color=COLOR_TEXT, pad=14)
    style_axis(ax_b, xb[-1])
    caption(ax_b, 0.0, LABEL_B, SUB_B)

    return fig


if __name__ == "__main__":
    fig = make_figure()
    for fmt in OUT_FORMATS:
        fig.savefig(f"{OUT_BASENAME}.{fmt}", dpi=DPI, transparent=TRANSPARENT,
                    bbox_inches="tight", pad_inches=0.12)
        print(f"wrote {OUT_BASENAME}.{fmt}")
    plt.close(fig)
