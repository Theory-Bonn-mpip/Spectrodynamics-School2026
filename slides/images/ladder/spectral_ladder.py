"""
Spectral-window figure for a single system (water) probed across the
electromagnetic spectrum.

One card per spectral window: what responds, the photon-energy range in two
units, and the associated observable. A full-width band underneath marks the
strong-coupling / cavity regime.

Pure matplotlib. Layout is computed from the column list, so adding or removing
a column re-spaces everything automatically.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle

# ----------------------------------------------------------------------
# COLOUR RAMPS  (fill, edge, title text, secondary text)
# ----------------------------------------------------------------------

RAMPS = {
    "red":    ("#FCEBEB", "#A32D2D", "#501313", "#A32D2D"),
    "amber":  ("#FAEEDA", "#854F0B", "#412402", "#854F0B"),
    "teal":   ("#E1F5EE", "#0F6E56", "#04342C", "#0F6E56"),
    "purple": ("#EEEDFE", "#534AB7", "#26215C", "#534AB7"),
    "gray":   ("#F1EFE8", "#5F5E5A", "#2C2C2A", "#5F5E5A"),
    "blue":   ("#E6F1FB", "#185FA5", "#042C53", "#185FA5"),
    "coral":  ("#FAECE7", "#993C1D", "#4A1B0C", "#993C1D"),
}

# ----------------------------------------------------------------------
# CONTENT
# ----------------------------------------------------------------------

MOLECULE_LABEL = ("O", "H", "H")  # set to None to hide the molecule sketch

COLUMNS = [
    dict(ramp="red",
         window="THz / far-IR",
         probe="H-bond network",
         energy="1–40 meV",
         native=r"10–300 cm$^{-1}$",
         signal="THz-Raman, 2D-THz"),
    dict(ramp="amber",
         window="Mid-IR / Raman",
         probe="O–H stretch, bend",
         energy="0.12–0.5 eV",
         native=r"1000–4000 cm$^{-1}$",
         signal="IR, Raman, SFG"),
    dict(ramp="teal",
         window="UV",
         probe="photodissociation",
         energy="7–11 eV",
         native="110–180 nm",
         signal="abs., fragments"),
    dict(ramp="purple",
         window="X-ray",
         probe="O 1s core level",
         energy="530–560 eV",
         native="O K-edge",
         signal="XAS, RIXS, TR-XAS"),
]

AXIS_LABEL = "photon energy (log scale)"

BAND = dict(ramp="gray",
            title="inside a cavity",
            subtitle="photons no longer a weak perturbation  ·  "
                     "Rabi splitting 10–500 meV")
# Set BAND = None to drop the strong-coupling band entirely.

# ----------------------------------------------------------------------
# LAYOUT  (canvas units; y increases downward)
# ----------------------------------------------------------------------

W, H = 680, 398
X_LEFT, X_RIGHT = 42, 636
GAP = 18

MOL_CX, MOL_CY = 340, 52     # oxygen centre
MOL_R_O, MOL_R_H = 15, 11
MOL_DX, MOL_DY = 32, 32      # H displacement from O

AXIS_TEXT_Y = 105
AXIS_LINE_Y = 120

CARD_TOP = 140
CARD_H = 152
CORNER = 12

Y_WINDOW = 166
Y_PROBE = 188
Y_RULE = 207
Y_ENERGY = 229
Y_NATIVE = 249
Y_SIGNAL = 272

BAND_TOP = 316
BAND_H = 62
Y_BAND_TITLE = 340
Y_BAND_SUB = 362

FS_WINDOW = 14# 11.5
FS_ENERGY = 13 #11.5
FS_SMALL = 12 #9.5

DPI = 300
OUTFILES = ["spectral_ladder.png", "spectral_ladder.pdf"]

# ----------------------------------------------------------------------
# DRAWING
# ----------------------------------------------------------------------


def _card(ax, x, y, w, h, ramp):
    fill, edge, _, _ = RAMPS[ramp]
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={CORNER}",
        facecolor=fill, edgecolor=edge, linewidth=0.8, zorder=1))


def _text(ax, x, y, s, size, color, weight="normal"):
    ax.text(x, y, s, ha="center", va="center", fontsize=size,
            color=color, fontweight=weight, zorder=3)


def draw_molecule(ax):
    if MOLECULE_LABEL is None:
        return
    o_label, h1_label, h2_label = MOLECULE_LABEL
    _, o_edge, o_text, _ = RAMPS["red"]
    _, h_edge, h_text, _ = RAMPS["gray"]

    for sign in (-1, 1):
        hx = MOL_CX + sign * MOL_DX
        hy = MOL_CY + MOL_DY
        ax.plot([MOL_CX + sign * MOL_DX * 0.33, hx - sign * MOL_DX * 0.24],
                [MOL_CY + MOL_DY * 0.33, hy - MOL_DY * 0.24],
                color="#3d3d3a", linewidth=1.6, solid_capstyle="round",
                zorder=1)

    ax.add_patch(Circle((MOL_CX, MOL_CY), MOL_R_O, facecolor=RAMPS["red"][0],
                        edgecolor=o_edge, linewidth=0.8, zorder=2))
    _text(ax, MOL_CX, MOL_CY, o_label, FS_WINDOW, o_text, "medium")

    for sign, lab in ((-1, h1_label), (1, h2_label)):
        hx = MOL_CX + sign * MOL_DX
        hy = MOL_CY + MOL_DY
        ax.add_patch(Circle((hx, hy), MOL_R_H, facecolor=RAMPS["gray"][0],
                            edgecolor=h_edge, linewidth=0.8, zorder=2))
        _text(ax, hx, hy, lab, FS_SMALL, h_text)


def draw_axis(ax):
    ax.text(X_LEFT, AXIS_TEXT_Y, AXIS_LABEL, ha="left", va="center",
            fontsize=FS_SMALL, color="#73726c", zorder=3)
    ax.annotate("", xy=(X_RIGHT, AXIS_LINE_Y), xytext=(X_LEFT, AXIS_LINE_Y),
                arrowprops=dict(arrowstyle="-|>", color="#73726c",
                                linewidth=1.0, shrinkA=0, shrinkB=0))


def draw_columns(ax):
    n = len(COLUMNS)
    width = (X_RIGHT - X_LEFT - GAP * (n - 1)) / n
    for i, col in enumerate(COLUMNS):
        x = X_LEFT + i * (width + GAP)
        cx = x + width / 2
        fill, edge, title_c, sub_c = RAMPS[col["ramp"]]

        _card(ax, x, CARD_TOP, width, CARD_H, col["ramp"])
        _text(ax, cx, Y_WINDOW, col["window"], FS_WINDOW, title_c, "medium")
        _text(ax, cx, Y_PROBE, col["probe"], FS_SMALL, sub_c)
        ax.plot([x + 14, x + width - 14], [Y_RULE, Y_RULE],
                color=edge, linewidth=0.6, alpha=0.45, zorder=2)
        _text(ax, cx, Y_ENERGY, col["energy"], FS_ENERGY, title_c, "medium")
        _text(ax, cx, Y_NATIVE, col["native"], FS_SMALL, sub_c)
        _text(ax, cx, Y_SIGNAL, col["signal"], FS_SMALL, sub_c)


def draw_band(ax):
    if BAND is None:
        return
    _, _, title_c, sub_c = RAMPS[BAND["ramp"]]
    _card(ax, X_LEFT, BAND_TOP, X_RIGHT - X_LEFT, BAND_H, BAND["ramp"])
    cx = (X_LEFT + X_RIGHT) / 2
    _text(ax, cx, Y_BAND_TITLE, BAND["title"], FS_WINDOW, title_c, "medium")
    _text(ax, cx, Y_BAND_SUB, BAND["subtitle"], FS_SMALL, sub_c)


def make_figure():
    fig, ax = plt.subplots(figsize=(W / 72, H / 72))
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)          # y increases downward
    ax.set_aspect("equal")
    ax.axis("off")

    draw_molecule(ax)
    draw_axis(ax)
    draw_columns(ax)
    draw_band(ax)

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig, ax


if __name__ == "__main__":
    fig, ax = make_figure()
    for path in OUTFILES:
        fig.savefig(path, dpi=DPI, bbox_inches="tight", transparent=True)
    plt.show()
