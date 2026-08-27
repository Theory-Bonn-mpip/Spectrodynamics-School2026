#!/usr/bin/env python3
"""
Render the light-matter interaction term and highlight / cancel its factors.

    H_int = -mu * E(t)

Same controls as hamiltonian_terms.py: every element can be framed and/or
marked, EQ_SIZE scales the whole figure proportionally, and the background is
transparent-ready. Styled to match the orange sans-serif slide look.

Keys (interactive):
    s   save PNG + PDF next to the script
    r   reset all terms to the state defined in TERMS
    q   quit
"""

import argparse
import copy
import os

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, FancyBboxPatch, Rectangle
from matplotlib.transforms import IdentityTransform

# --------------------------------------------------------------------------
# 1. CONTENT  --  edit this part
# --------------------------------------------------------------------------

LHS = r"H_{e-ph} + H_{n-ph} ="          # left-hand side
OPERATOR = r"*"             # symbol inserted between the terms
TRAILING = None             # e.g. r"\Rightarrow" to put an arrow at the end

# One entry per factor; each one is framed / marked on its own.
TERMS = [
    {"tex": r"-\mu",   "box": False, "mark": None},
    {"tex": r"E(t)",   "box": False, "mark": None},
]

# --------------------------------------------------------------------------
# 2. SIZE
# --------------------------------------------------------------------------

EQ_SIZE = 34.0        # <-- size of the whole figure. Change this one.

# Geometry at REF_SIZE, in points. With PROPORTIONAL = True everything is
# rescaled by EQ_SIZE / REF_SIZE, so frames, padding, gaps and line widths
# follow the equation. Set it to False to freeze the geometry instead.
PROPORTIONAL = True
REF_SIZE = 34.0

PAD_X = 10.0          # horizontal space between a term and its frame
PAD_Y = 8.0           # vertical space between a term and its frame
GAP = 4.0             # space between consecutive elements
BOX_HEIGHT = 62.0     # frame height; None = fit to the tallest term
BAND_PAD = 5.0        # extra height of the background band around the frames
CORNER = 6.0          # frame corner radius

FIGSIZE = None        # (width, height) in inches to pin the canvas,
                      # or None to auto-fit the equation

DPI = 150

# --------------------------------------------------------------------------
# 3. STYLE
# --------------------------------------------------------------------------

TEXT_COLOR = "#1A1A1A"      # equation colour ("#FFAB40" for the orange slide)
MARKED_TEXT_COLOR = "#8A8A8A"
BOX_COLOR = "#FFAB40"       # frame colour
BOX_LW = 2.2                # scaled with the equation when PROPORTIONAL
MARK_COLOR = "#D01A1A"      # red
MARK_LW = 5.0               # scaled with the equation when PROPORTIONAL
BAN_RADIUS = 27.0           # radius of the "forbidden" circle, in points
BAND_COLOR = None           # background band: None = no band,
BAND_EDGE = None            # or e.g. "#DCE7F0" with edge "#2E6DA4"
TRANSPARENT = False         # True -> save with a transparent background

# Upright sans-serif maths, to match the screenshot. For the LaTeX look of
# the other script use: fontset "cm", default "it", family "serif".
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
matplotlib.rcParams["mathtext.fontset"] = "dejavusans"
matplotlib.rcParams["mathtext.default"] = "regular"

# state cycle used when clicking a term
CYCLE = [
    {"box": False, "mark": None},
    {"box": True,  "mark": None},
    {"box": False, "mark": "ban"},
    {"box": False, "mark": "cross"},
]


# --------------------------------------------------------------------------
# 4. RENDERING
# --------------------------------------------------------------------------

def _k(eq_size):
    """Geometry scale factor for a given equation size."""
    return (eq_size / REF_SIZE) if PROPORTIONAL else 1.0


def _px(points, dpi, eq_size=None):
    """Points -> pixels, rescaled with the equation unless frozen."""
    k = 1.0 if eq_size is None else _k(eq_size)
    return points * k * dpi / 72.0


def _renderer(fig):
    try:
        return fig.canvas.get_renderer()
    except AttributeError:
        fig.canvas.draw()
        return fig.canvas.get_renderer()


def _measure(fig, renderer, tex, fontsize):
    """Return (width, ascent, descent) in pixels for a mathtext string."""
    probe = fig.text(0, 100, f"${tex}$", fontsize=fontsize,
                     ha="left", va="baseline")
    probe.set_transform(IdentityTransform())
    probe.set_position((0, 100))
    bb = probe.get_window_extent(renderer)
    probe.remove()
    return bb.width, bb.y1 - 100, 100 - bb.y0


def _sequence(terms):
    """Flatten the equation into (kind, tex, term_index) elements."""
    elements = [("plain", LHS, None)]
    for i, term in enumerate(terms):
        if i > 0:
            elements.append(("plain", OPERATOR, None))
        elements.append(("term", term["tex"], i))
    if TRAILING:
        elements.append(("arrow", TRAILING, None))
    return elements


def _add_text(fig, tex, x, y, fontsize, color):
    t = fig.text(0, 0, f"${tex}$", fontsize=fontsize, ha="left",
                 va="baseline", color=color)
    t.set_transform(IdentityTransform())
    t.set_position((x, y))
    return t


def _add_ban(fig, cx, cy, radius, k=1.0):
    """Red 'forbidden' sign: circle + slash."""
    circ = Ellipse((cx, cy), 2 * radius, 2 * radius, fill=False,
                   edgecolor=MARK_COLOR, linewidth=MARK_LW * k, zorder=5)
    circ.set_transform(IdentityTransform())
    fig.add_artist(circ)
    d = radius * 0.7071
    slash = Line2D([cx - d, cx + d], [cy + d, cy - d], color=MARK_COLOR,
                   linewidth=MARK_LW * k, solid_capstyle="butt", zorder=6)
    slash.set_transform(IdentityTransform())
    fig.add_artist(slash)


def _add_cross(fig, x0, x1, y0, y1, k=1.0):
    for xa, ya, xb, yb in ((x0, y0, x1, y1), (x0, y1, x1, y0)):
        ln = Line2D([xa, xb], [ya, yb], color=MARK_COLOR,
                    linewidth=MARK_LW * k, solid_capstyle="round", zorder=6)
        ln.set_transform(IdentityTransform())
        fig.add_artist(ln)


def _add_strike(fig, x0, x1, y, k=1.0):
    ln = Line2D([x0, x1], [y, y], color=MARK_COLOR, linewidth=MARK_LW * k,
                solid_capstyle="round", zorder=6)
    ln.set_transform(IdentityTransform())
    fig.add_artist(ln)


def draw(fig, terms, eq_size=None):
    """Clear the figure and lay the equation out in pixel coordinates.

    Returns a list of (term_index, x0, x1, y0, y1) hit boxes for click tests.
    """
    eq_size = EQ_SIZE if eq_size is None else eq_size
    fig.clear()
    renderer = _renderer(fig)
    dpi = fig.dpi
    W, H = fig.bbox.width, fig.bbox.height

    k = _k(eq_size)
    pad_x = _px(PAD_X, dpi, eq_size)
    pad_y = _px(PAD_Y, dpi, eq_size)
    gap = _px(GAP, dpi, eq_size)

    elements = _sequence(terms)
    sizes = [_measure(fig, renderer, tex, eq_size) for _, tex, _ in elements]
    asc = max(s[1] for s in sizes)
    desc = max(s[2] for s in sizes)

    box_h = (_px(BOX_HEIGHT, dpi, eq_size) if BOX_HEIGHT
             else asc + desc + 2 * pad_y)

    total = sum(w + (2 * pad_x if kind == "term" else 0)
                for (kind, _, _), (w, _, _) in zip(elements, sizes))
    total += gap * (len(elements) - 1)

    x = (W - total) / 2.0
    y_mid = H / 2.0
    baseline = y_mid - (asc - desc) / 2.0
    y_bot, y_top = y_mid - box_h / 2.0, y_mid + box_h / 2.0

    # --- background band ----------------------------------------------------
    if BAND_COLOR:
        band_pad = _px(BAND_PAD, dpi, eq_size)
        band = Rectangle((0, y_bot - band_pad), W, box_h + 2 * band_pad,
                         facecolor=BAND_COLOR, edgecolor=BAND_EDGE or "none",
                         linewidth=1.5 * k, zorder=0)
        band.set_transform(IdentityTransform())
        fig.add_artist(band)

    # --- place --------------------------------------------------------------
    hitboxes = []
    for (kind, tex, idx), (w, _, _) in zip(elements, sizes):
        if kind == "term":
            term = terms[idx]
            color = MARKED_TEXT_COLOR if term.get("mark") else TEXT_COLOR
            _add_text(fig, tex, x + pad_x, baseline, eq_size, color)

            if term.get("box"):
                patch = FancyBboxPatch(
                    (x + 0.5 * pad_x, y_bot), w + pad_x, box_h,
                    boxstyle="round,pad=0,rounding_size="
                             f"{_px(CORNER, dpi, eq_size)}",
                    fill=False, edgecolor=BOX_COLOR, linewidth=BOX_LW * k,
                    zorder=4, mutation_aspect=1.0)
                patch.set_transform(IdentityTransform())
                fig.add_artist(patch)

            mark = term.get("mark")
            cx = x + pad_x + w / 2.0
            if mark == "ban":
                _add_ban(fig, cx, y_mid, _px(BAN_RADIUS, dpi, eq_size), k)
            elif mark == "cross":
                _add_cross(fig, x + 0.4 * pad_x, x + 1.6 * pad_x + w,
                           y_bot + 0.15 * box_h, y_top - 0.15 * box_h, k)
            elif mark == "strike":
                _add_strike(fig, x + 0.4 * pad_x, x + 1.6 * pad_x + w,
                            y_mid, k)

            hitboxes.append((idx, x, x + w + 2 * pad_x, y_bot, y_top))
            x += w + 2 * pad_x + gap
        else:
            _add_text(fig, tex, x, baseline, eq_size, TEXT_COLOR)
            x += w + gap

    fig.canvas.draw_idle()
    return hitboxes


# --------------------------------------------------------------------------
# 5. FIGURE / INTERACTION
# --------------------------------------------------------------------------

def _content_size(terms, eq_size, dpi):
    """Measure the equation on a scratch figure -> (width_px, height_px)."""
    scratch = plt.figure(figsize=(1, 1), dpi=dpi)
    renderer = _renderer(scratch)
    pad_x = _px(PAD_X, dpi, eq_size)
    pad_y = _px(PAD_Y, dpi, eq_size)
    gap = _px(GAP, dpi, eq_size)

    elements = _sequence(terms)
    sizes = [_measure(scratch, renderer, tex, eq_size) for _, tex, _ in elements]
    plt.close(scratch)

    width = sum(w + (2 * pad_x if kind == "term" else 0)
                for (kind, _, _), (w, _, _) in zip(elements, sizes))
    width += gap * (len(elements) - 1)
    height = (_px(BOX_HEIGHT, dpi, eq_size) if BOX_HEIGHT else
              max(s[1] for s in sizes) + max(s[2] for s in sizes) + 2 * pad_y)
    return width, height


def make_figure(terms, eq_size=None, figsize=FIGSIZE, dpi=DPI):
    eq_size = EQ_SIZE if eq_size is None else eq_size
    if figsize is None:
        w_px, h_px = _content_size(terms, eq_size, dpi)
        margin_x = _px(24, dpi, eq_size)
        margin_y = _px((2 * BAND_PAD + 6) if BAND_COLOR else 10, dpi, eq_size)
        figsize = ((w_px + 2 * margin_x) / dpi, (h_px + 2 * margin_y) / dpi)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor("white")
    return fig, draw(fig, terms, eq_size)


def run_interactive(terms, eq_size=None, outstem="interaction"):
    state = copy.deepcopy(terms)
    initial = copy.deepcopy(terms)
    fig, boxes = make_figure(state, eq_size)
    holder = {"boxes": boxes}

    def refresh():
        holder["boxes"] = draw(fig, state, eq_size)

    def on_click(event):
        if event.x is None:
            return
        for idx, x0, x1, y0, y1 in holder["boxes"]:
            if x0 <= event.x <= x1 and y0 <= event.y <= y1:
                cur = {"box": state[idx].get("box", False),
                       "mark": state[idx].get("mark")}
                try:
                    nxt = CYCLE[(CYCLE.index(cur) + 1) % len(CYCLE)]
                except ValueError:
                    nxt = CYCLE[0]
                state[idx].update(nxt)
                refresh()
                break

    def on_key(event):
        if event.key == "s":
            for ext in ("png", "pdf"):
                path = f"{outstem}.{ext}"
                fig.savefig(path, dpi=fig.dpi, transparent=TRANSPARENT,
                            facecolor=fig.get_facecolor())
                print(f"saved {os.path.abspath(path)}")
        elif event.key == "r":
            for i, t in enumerate(initial):
                state[i] = copy.deepcopy(t)
            refresh()

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("resize_event", lambda e: refresh())
    print("Click a term to cycle: plain -> boxed -> banned -> crossed. "
          "Press 's' to save, 'r' to reset.")
    plt.show()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--output", help="save to this file instead of "
                    "opening a window (.png, .pdf, .svg ...)")
    ap.add_argument("--size", type=float, default=EQ_SIZE,
                    help="size of the figure in pt")
    ap.add_argument("--box", type=int, nargs="*", default=None, metavar="i",
                    help="indices (0-based) of terms to frame")
    ap.add_argument("--ban", type=int, nargs="*", default=None, metavar="i",
                    help="indices of terms to mark with the red 'no' sign")
    ap.add_argument("--cross", type=int, nargs="*", default=None, metavar="i",
                    help="indices of terms to mark with a red X")
    args = ap.parse_args()

    terms = copy.deepcopy(TERMS)
    if args.box is not None or args.ban is not None or args.cross is not None:
        for t in terms:
            t["box"], t["mark"] = False, None
        for i in args.box or []:
            terms[i]["box"] = True
        for i in args.ban or []:
            terms[i]["mark"] = "ban"
        for i in args.cross or []:
            terms[i]["mark"] = "cross"

    if args.output:
        matplotlib.use("Agg")
        fig, _ = make_figure(terms, args.size)
        fig.savefig(args.output, dpi=fig.dpi, transparent=TRANSPARENT,
                    facecolor=fig.get_facecolor())
        print(f"saved {os.path.abspath(args.output)}")
    else:
        run_interactive(terms, args.size)


if __name__ == "__main__":
    main()
