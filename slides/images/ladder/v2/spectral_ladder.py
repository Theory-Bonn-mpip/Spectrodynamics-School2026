"""
Spectral-window figure for a single system (water) probed across the
electromagnetic spectrum.

One card per spectral window: what responds, the photon-energy range in two
units, and the associated observable. The cavity / strong-coupling regime is
drawn as a container that ENCLOSES all four windows, since strong coupling is a
modifier that can be applied to any of them rather than a fifth window.

Card width is derived from real font metrics, so changing FS_* or the strings
re-sizes the cards (and the whole canvas) instead of overflowing them.

Pure matplotlib.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties

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

# ----------------------------------------------------------------------
# WHAT TO DRAW
# ----------------------------------------------------------------------

SHOW_MOLECULE = False    # ball-and-stick sketch above the axis
SHOW_CAVITY = True       # container enclosing all four windows

MOLECULE_LABEL = ("O", "H", "H")   # only used when SHOW_MOLECULE is True

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

# Content of the enclosing cavity container (drawn only if SHOW_CAVITY).
CAVITY = dict(ramp="gray",
              title="inside a cavity",
              subtitle="photons no longer a weak perturbation  ·  "
                       "Rabi splitting 10–500 meV")

CAVITY_MIRRORS = True          # thick bars on the container's left/right edges
MIRROR_WIDTH = 5.0

# ----------------------------------------------------------------------
# LAYOUT  (canvas units = points; y increases downward)
# ----------------------------------------------------------------------

MARGIN = 26                    # canvas edge -> cavity container
CAVITY_PAD_X = 20              # container edge -> first card
CARD_PAD_X = 15                # card edge -> text
GAP = 18                       # between cards
CORNER = 12

MOL_R_O, MOL_R_H = 15, 11
MOL_DX, MOL_DY = 32, 32
MOL_TOP = 22

PAD_AFTER_MOLECULE = 26
AXIS_TO_LINE = 15
LINE_TO_CAVITY = 24

DY_TITLE = 28                  # container top -> title row
TITLE_TO_CARDS = 24
CARD_H = 152
CARDS_TO_SUB = 26
SUB_TO_BOTTOM = 22

# Text rows, measured downward from the top of a card
DY_WINDOW = 26
DY_PROBE = 50
DY_RULE = 70
DY_ENERGY = 92
DY_NATIVE = 113
DY_SIGNAL = 136

FS_WINDOW = 14
FS_ENERGY = 13
FS_SMALL = 12

DPI = 300
OUTFILES = ["spectral_ladder.png", "spectral_ladder.pdf"]


# ----------------------------------------------------------------------
# TEXT METRICS
# ----------------------------------------------------------------------

def text_width(s, size, weight="normal"):
    """Width of a rendered string in points (handles mathtext)."""
    prop = FontProperties(size=size, weight=weight)
    return TextPath((0, 0), s, prop=prop).get_extents().width


def layout():
    """Derive card width, container size and canvas size from the content."""
    widest = 0.0
    for col in COLUMNS:
        widest = max(widest,
                     text_width(col["window"], FS_WINDOW, "medium"),
                     text_width(col["energy"], FS_ENERGY, "medium"),
                     text_width(col["probe"], FS_SMALL),
                     text_width(col["native"], FS_SMALL),
                     text_width(col["signal"], FS_SMALL))
    card_w = widest + 2 * CARD_PAD_X

    n = len(COLUMNS)
    inner = n * card_w + (n - 1) * GAP

    # Widen the cards if the container's own strings need more room.
    if SHOW_CAVITY:
        need = max(text_width(CAVITY["title"], FS_WINDOW, "medium"),
                   text_width(CAVITY["subtitle"], FS_SMALL)) + 2 * CAVITY_PAD_X
        if inner + 2 * CAVITY_PAD_X < need:
            card_w += (need - 2 * CAVITY_PAD_X - inner) / n
            inner = n * card_w + (n - 1) * GAP

    cav_w = inner + 2 * CAVITY_PAD_X
    W = cav_w + 2 * MARGIN

    if SHOW_MOLECULE:
        y = MOL_TOP + MOL_R_O * 2 + MOL_DY + PAD_AFTER_MOLECULE
    else:
        y = MOL_TOP
    axis_text = y
    axis_line = y + AXIS_TO_LINE
    cav_top = axis_line + LINE_TO_CAVITY

    if SHOW_CAVITY:
        cards_top = cav_top + DY_TITLE + TITLE_TO_CARDS
        sub_y = cards_top + CARD_H + CARDS_TO_SUB
        cav_bottom = sub_y + SUB_TO_BOTTOM
        H = cav_bottom + MARGIN
    else:
        cards_top = cav_top
        sub_y = cav_bottom = None
        H = cards_top + CARD_H + MARGIN

    return dict(W=W, H=H, card_w=card_w,
                cav_left=MARGIN, cav_right=MARGIN + cav_w,
                cav_top=cav_top, cav_bottom=cav_bottom,
                cards_left=MARGIN + CAVITY_PAD_X, cards_top=cards_top,
                axis_text=axis_text, axis_line=axis_line, sub_y=sub_y)


# ----------------------------------------------------------------------
# DRAWING
# ----------------------------------------------------------------------

def _box(ax, x, y, w, h, fill, edge, lw=0.8, z=1):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={CORNER}",
        facecolor=fill, edgecolor=edge, linewidth=lw, zorder=z))


def _text(ax, x, y, s, size, color, weight="normal"):
    ax.text(x, y, s, ha="center", va="center", fontsize=size,
            color=color, fontweight=weight, zorder=4)


def draw_cavity(ax, L):
    if not SHOW_CAVITY:
        return
    fill, edge, title_c, sub_c = RAMPS[CAVITY["ramp"]]
    w = L["cav_right"] - L["cav_left"]
    h = L["cav_bottom"] - L["cav_top"]
    _box(ax, L["cav_left"], L["cav_top"], w, h, fill, edge, lw=1.0, z=0)

    cx = (L["cav_left"] + L["cav_right"]) / 2
    _text(ax, cx, L["cav_top"] + DY_TITLE, CAVITY["title"],
          FS_WINDOW, title_c, "medium")
    _text(ax, cx, L["sub_y"], CAVITY["subtitle"], FS_SMALL, sub_c)

    if CAVITY_MIRRORS:
        y0 = L["cav_top"] + CORNER
        y1 = L["cav_bottom"] - CORNER
        for x in (L["cav_left"] + MIRROR_WIDTH / 2 + 2,
                  L["cav_right"] - MIRROR_WIDTH / 2 - 2):
            ax.plot([x, x], [y0, y1], color=edge, linewidth=MIRROR_WIDTH,
                    solid_capstyle="round", alpha=0.55, zorder=1)


def draw_molecule(ax, L):
    if not SHOW_MOLECULE:
        return
    o_label, h1_label, h2_label = MOLECULE_LABEL
    _, o_edge, o_text, _ = RAMPS["red"]
    _, h_edge, h_text, _ = RAMPS["gray"]
    cx = L["W"] / 2
    cy = MOL_TOP + MOL_R_O

    for sign in (-1, 1):
        ax.plot([cx + sign * MOL_DX * 0.33, cx + sign * MOL_DX * 0.76],
                [cy + MOL_DY * 0.33, cy + MOL_DY * 0.76],
                color="#3d3d3a", linewidth=1.6, solid_capstyle="round",
                zorder=1)

    ax.add_patch(Circle((cx, cy), MOL_R_O, facecolor=RAMPS["red"][0],
                        edgecolor=o_edge, linewidth=0.8, zorder=2))
    _text(ax, cx, cy, o_label, FS_WINDOW, o_text, "medium")

    for sign, lab in ((-1, h1_label), (1, h2_label)):
        hx, hy = cx + sign * MOL_DX, cy + MOL_DY
        ax.add_patch(Circle((hx, hy), MOL_R_H, facecolor=RAMPS["gray"][0],
                            edgecolor=h_edge, linewidth=0.8, zorder=2))
        _text(ax, hx, hy, lab, FS_SMALL, h_text)


def draw_axis(ax, L):
    ax.text(L["cav_left"], L["axis_text"], AXIS_LABEL, ha="left", va="center",
            fontsize=FS_SMALL, color="#73726c", zorder=4)
    ax.annotate("", xy=(L["cav_right"], L["axis_line"]),
                xytext=(L["cav_left"], L["axis_line"]),
                arrowprops=dict(arrowstyle="-|>", color="#73726c",
                                linewidth=1.0, shrinkA=0, shrinkB=0))


def draw_columns(ax, L):
    w = L["card_w"]
    top = L["cards_top"]
    for i, col in enumerate(COLUMNS):
        x = L["cards_left"] + i * (w + GAP)
        cx = x + w / 2
        fill, edge, title_c, sub_c = RAMPS[col["ramp"]]

        _box(ax, x, top, w, CARD_H, fill, edge, z=2)
        _text(ax, cx, top + DY_WINDOW, col["window"], FS_WINDOW, title_c, "medium")
        _text(ax, cx, top + DY_PROBE, col["probe"], FS_SMALL, sub_c)
        ax.plot([x + CARD_PAD_X, x + w - CARD_PAD_X], [top + DY_RULE] * 2,
                color=edge, linewidth=0.6, alpha=0.45, zorder=3)
        _text(ax, cx, top + DY_ENERGY, col["energy"], FS_ENERGY, title_c, "medium")
        _text(ax, cx, top + DY_NATIVE, col["native"], FS_SMALL, sub_c)
        _text(ax, cx, top + DY_SIGNAL, col["signal"], FS_SMALL, sub_c)


def make_figure():
    L = layout()
    fig, ax = plt.subplots(figsize=(L["W"] / 72, L["H"] / 72))
    ax.set_xlim(0, L["W"])
    ax.set_ylim(L["H"], 0)      # y increases downward
    ax.set_aspect("equal")
    ax.axis("off")

    draw_cavity(ax, L)
    draw_molecule(ax, L)
    draw_axis(ax, L)
    draw_columns(ax, L)

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig, ax


if __name__ == "__main__":
    fig, ax = make_figure()
    for path in OUTFILES:
        fig.savefig(path, dpi=DPI, bbox_inches="tight", transparent=True)
    plt.show()
