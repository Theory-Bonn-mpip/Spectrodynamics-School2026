#!/usr/bin/env python3
"""
Render a set of LaTeX equations as standalone images.

Plain black mathematics, no frames or highlighting. Each equation is written
once in the EQUATIONS dictionary below and can then be exported individually,
all at once, or stacked into a single figure.

Examples
--------
    python equations.py --list                  # show the available names
    python equations.py ir_spectrum             # open one in a window
    python equations.py ir_spectrum -o ir.pdf   # export one
    python equations.py --all --format pdf      # export every equation
    python equations.py --all --stack -o all.png
    python equations.py --all --size 26 --transparent --format png

Notes
-----
matplotlib's own mathtext engine is used by default, so nothing besides
matplotlib is required. If you have a LaTeX installation and want the real
thing (custom packages, exact spacing), run with --usetex.
"""

import argparse
import os

import matplotlib
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# 1. EQUATIONS  --  add or edit entries here
# --------------------------------------------------------------------------

EQUATIONS = {
    "ir_spectrum": r"""
        I_{\mathrm{IR}}(\omega) \propto \omega^{2}
        \int_{-\infty}^{\infty}\mathrm{d}t\;
        e^{-i\omega t}\,\left\langle \boldsymbol{\mu}(t)\,\boldsymbol{\mu}(0) \right\rangle
    """,

    "sfg_response": r"""
        \chi^{(2)}_{pqr}(\omega_{\mathrm{IR}}) =  
        \frac{-i}{k_B T\, \omega_{\mathrm{IR}}}
        \int_{0}^{\infty}\mathrm{d}t\,
        e^{i\omega_{\mathrm{IR}}t} 
        \langle \dot{\alpha}_{pq}(t)\,
        \dot{\mu}_{r}(0) \rangle 
    """,

    "response_I": r"""
        R^{(I)}(t_{2},t_{1}) \;\propto\;
        -\,\mathrm{tr}\{\boldsymbol{\mu}(t_{2})
        \left[\boldsymbol{\mu}(0),
        \left[\boldsymbol{\Pi}(-t_{1}),\rho_{\mathrm{eq}}\right]\right]\}
    """,

    "response_II": r"""
        R^{(II)}(t_{2},t_{1}) \;\propto\;
        -\,\mathrm{tr}\{\boldsymbol{\mu}(t_{2})
        \left[\boldsymbol{\Pi}(0),
        \left[\boldsymbol{\mu}(-t_{1}),\rho_{\mathrm{eq}}\right]\right]\}
    """,
}

# --------------------------------------------------------------------------
# 2. SETTINGS
# --------------------------------------------------------------------------

EQ_SIZE = 30.0          # size of the mathematics, in points
TEXT_COLOR = "#000000"
PAD = 0.12              # white space around the equation, in inches
LINE_SPACING = 0.55     # vertical gap between equations when stacked, inches
DPI = 300               # 300 is a good default for figures going into slides
TRANSPARENT = False     # True -> no white background in the saved file
USETEX = False          # True -> render through a real LaTeX installation

# "stix"  -> Times-like, and the only built-in set with bold capital Greek,
#            so \boldsymbol{\Pi} comes out bold.
# "cm"    -> Computer Modern, the classic LaTeX look, but capital Greek
#            cannot be emboldened; \boldsymbol{\Pi} silently stays light.
# Use --usetex for real LaTeX, where \boldsymbol works everywhere.
MATH_FONT = "stix"

matplotlib.rcParams["mathtext.fontset"] = MATH_FONT
matplotlib.rcParams["font.family"] = "serif"


# --------------------------------------------------------------------------
# 3. RENDERING
# --------------------------------------------------------------------------

def _clean(tex):
    """Collapse the triple-quoted source into a single-line math string."""
    return " ".join(tex.split())


def _renderer(fig):
    try:
        return fig.canvas.get_renderer()
    except AttributeError:
        fig.canvas.draw()
        return fig.canvas.get_renderer()


def _text_size_inches(tex, size, dpi):
    """Measure one equation on a scratch figure -> (width, height) in inches."""
    scratch = plt.figure(figsize=(1, 1), dpi=dpi)
    t = scratch.text(0, 0, f"${_clean(tex)}$", fontsize=size)
    bb = t.get_window_extent(_renderer(scratch))
    plt.close(scratch)
    return bb.width / dpi, bb.height / dpi


def make_figure(items, size=EQ_SIZE, dpi=DPI, stack=False):
    """Build a figure holding one equation, or several stacked vertically.

    `items` is a list of LaTeX strings.
    """
    sizes = [_text_size_inches(tex, size, dpi) for tex in items]
    width = max(w for w, _ in sizes) + 2 * PAD
    if stack:
        height = sum(h for _, h in sizes) + LINE_SPACING * (len(items) - 1) \
                 + 2 * PAD
    else:
        height = sizes[0][1] + 2 * PAD

    fig = plt.figure(figsize=(width, height), dpi=dpi)
    fig.patch.set_facecolor("white")

    if len(items) == 1:
        fig.text(0.5, 0.5, f"${_clean(items[0])}$", fontsize=size,
                 color=TEXT_COLOR, ha="center", va="center")
    else:
        # place from the top down, each equation centred horizontally
        y = height - PAD
        for tex, (_, h) in zip(items, sizes):
            fig.text(0.5, (y - h / 2) / height, f"${_clean(tex)}$",
                     fontsize=size, color=TEXT_COLOR, ha="center", va="center")
            y -= h + LINE_SPACING
    return fig


def save(fig, path, dpi=DPI):
    fig.savefig(path, dpi=dpi, transparent=TRANSPARENT,
                bbox_inches="tight", pad_inches=PAD,
                facecolor=fig.get_facecolor())
    print(f"saved {os.path.abspath(path)}")


# --------------------------------------------------------------------------
# 4. COMMAND LINE
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", nargs="*", help="equation name(s) to render; "
                    "omit together with --all to render everything")
    ap.add_argument("--all", action="store_true", help="render every equation")
    ap.add_argument("--list", action="store_true", help="list the names and exit")
    ap.add_argument("--stack", action="store_true",
                    help="put the selected equations in one figure")
    ap.add_argument("-o", "--output", help="output file; with several "
                    "equations and no --stack this is used as a name stem")
    ap.add_argument("--format", default="png",
                    help="extension used when -o is not given (png, pdf, svg)")
    ap.add_argument("--size", type=float, default=EQ_SIZE)
    ap.add_argument("--font", default=MATH_FONT,
                    choices=["stix", "cm", "dejavuserif", "stixsans"],
                    help="mathtext font set (stix keeps capital Greek bold)")
    ap.add_argument("--dpi", type=float, default=DPI)
    ap.add_argument("--transparent", action="store_true")
    ap.add_argument("--usetex", action="store_true",
                    help="render with a real LaTeX installation")
    args = ap.parse_args()

    if args.list:
        for name in EQUATIONS:
            print(name)
        return

    global TRANSPARENT
    TRANSPARENT = TRANSPARENT or args.transparent
    matplotlib.rcParams["mathtext.fontset"] = args.font
    if args.usetex or USETEX:
        matplotlib.rcParams["text.usetex"] = True
        matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{amsmath,bm}"

    names = args.name or list(EQUATIONS)
    if args.all:
        names = list(EQUATIONS)
    unknown = [n for n in names if n not in EQUATIONS]
    if unknown:
        raise SystemExit(f"unknown equation(s): {', '.join(unknown)}\n"
                         f"available: {', '.join(EQUATIONS)}")

    writing = args.output or args.all or len(names) > 1
    if writing:
        matplotlib.use("Agg")

    if args.stack or (args.output and len(names) == 1):
        fig = make_figure([EQUATIONS[n] for n in names], args.size, args.dpi,
                          stack=args.stack)
        out = args.output or f"{'_'.join(names)}.{args.format}"
        save(fig, out, args.dpi)
        plt.close(fig)
    elif writing:
        stem = os.path.splitext(args.output)[0] + "_" if args.output else ""
        for n in names:
            fig = make_figure([EQUATIONS[n]], args.size, args.dpi)
            save(fig, f"{stem}{n}.{args.format}", args.dpi)
            plt.close(fig)
    else:
        make_figure([EQUATIONS[n] for n in names], args.size, args.dpi)
        plt.show()


if __name__ == "__main__":
    main()
