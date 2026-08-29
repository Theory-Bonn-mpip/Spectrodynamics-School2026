#!/usr/bin/env python3
"""Convert a sphinx-gallery format Python script to a Jupyter notebook.

Handles RST-to-markdown conversion for: section headings (underlines
* = - ~ ^ map to heading levels 1-5), inline code/math/links,
figure, code-block, math, and note/warning directives.
"""

import json
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# RST → Markdown helpers
# ---------------------------------------------------------------------------

def _inline(text):
    """Convert RST inline markup to markdown."""
    # ``code`` → `code`
    text = re.sub(r'``([^`]+?)``', r'`\1`', text)
    # :math:`expr` → $expr$
    text = re.sub(r':math:`([^`]+?)`', r'$\1$', text)
    # `text <url>`_ → [text](url)
    text = re.sub(r'`([^<`]+?) <([^>]+?)>`_+', r'[\1](\2)', text)
    return text


# An RST link `text <url>`_ may be wrapped over at most this many lines.
MAX_LINK_LINES = 10

_LINK_WRAPPED = re.compile(
    # the text may hold MAX_LINK_LINES - 2 line breaks; the \s* before the URL
    # may add one more, for MAX_LINK_LINES lines in total
    r'`((?:[^`<>\n]*\n){0,%d}[^`<>\n]*?)\s*<([^>]+?)>`_+' % (MAX_LINK_LINES - 2)
)


def _join_link(match):
    """Collapse a link that was wrapped over several source lines."""
    text = ' '.join(match.group(1).split())
    url = ''.join(match.group(2).split())
    return '`%s <%s>`_' % (text, url)


def rst_to_md(rst_text):
    """Convert an RST text block to GitHub-flavored markdown."""
    # join RST links wrapped across line breaks: `text\n<url>`_ → `text <url>`_.
    # The link text may span up to MAX_LINK_LINES lines; the bound keeps a
    # stray backtick from swallowing a whole paragraph. All internal
    # whitespace is collapsed, and whitespace inside the URL is removed.
    rst_text = _LINK_WRAPPED.sub(_join_link, rst_text)
    # join :math:`...` spans wrapped across line breaks (the closing backtick
    # blocks the match, so completed spans are left alone); loop because each
    # pass only joins one break per span
    while True:
        joined = re.sub(r'(:math:`[^`\n]*)\n\s*', r'\1 ', rst_text)
        if joined == rst_text:
            break
        rst_text = joined
    lines = rst_text.split('\n')
    out = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # ---- section heading (title line followed by underline) ----
        if (i + 1 < len(lines)
                and line.strip()
                and lines[i + 1].strip()
                and len(set(lines[i + 1].strip())) == 1
                and lines[i + 1].strip()[0] in '*=-~^'
                and len(lines[i + 1].strip()) >= len(line.strip())):
            char = lines[i + 1].strip()[0]
            lvl = {'*': '#', '=': '##', '-': '###', '~': '####', '^': '#####'}[char]
            out.append(f'{lvl} {_inline(line.strip())}')
            i += 2
            continue

        # ---- .. figure:: ----
        if re.match(r'\s*\.\. figure::', line):
            figpath = line.split('.. figure::')[-1].strip()
            i += 1
            while i < len(lines) and re.match(r'\s+:\w+:', lines[i]):
                i += 1
            if i < len(lines) and not lines[i].strip():
                i += 1
            cap_parts = []
            while i < len(lines) and lines[i].strip():
                cap_parts.append(lines[i].strip())
                i += 1
            caption = _inline(' '.join(cap_parts))
            out.append(f'![{caption}]({figpath})')
            if caption:
                out.append('')
                out.append(f'*{caption}*')
            continue

        # ---- .. math:: ----
        if re.match(r'\s*\.\. math::', line):
            i += 1
            if i < len(lines) and not lines[i].strip():
                i += 1
            math_lines = []
            while i < len(lines) and (not lines[i].strip() or lines[i].startswith(' ')):
                math_lines.append(lines[i].strip())
                i += 1
            while math_lines and not math_lines[-1]:
                math_lines.pop()
            out.append('$$')
            out.extend(math_lines)
            out.append('$$')
            out.append('')
            continue

        # ---- .. note:: / .. warning:: ----
        m = re.match(r'\s*\.\. (note|warning)::\s*(.*)', line)
        if m:
            kind = m.group(1).capitalize()
            adm_lines = [m.group(2).strip()] if m.group(2).strip() else []
            i += 1
            while i < len(lines) and (not lines[i].strip() or lines[i].startswith(' ')):
                adm_lines.append(lines[i].strip())
                i += 1
            while adm_lines and not adm_lines[-1]:
                adm_lines.pop()
            out.append(f'> **{kind}:**')
            for al in adm_lines:
                out.append(f'> {_inline(al)}' if al else '>')
            out.append('')
            continue

        # ---- .. code-block:: ----
        if re.match(r'\s*\.\. code-block::', line):
            lang = line.split('.. code-block::')[-1].strip()
            i += 1
            if i < len(lines) and not lines[i].strip():
                i += 1
            # detect indentation from first non-blank code line
            indent = 0
            if i < len(lines) and lines[i].strip():
                indent = len(lines[i]) - len(lines[i].lstrip())
            code_lines = []
            while i < len(lines) and (not lines[i].strip() or lines[i].startswith(' ')):
                code_lines.append(lines[i][indent:] if lines[i].strip() else '')
                i += 1
            while code_lines and not code_lines[-1].strip():
                code_lines.pop()
            out.append(f'```{lang}')
            out.extend(code_lines)
            out.append('```')
            out.append('')
            continue

        # ---- RST field list  :Name: value ----
        # (negative lookahead so inline roles like :math:`x` are not matched)
        m = re.match(r':(\w[\w ]*?):(?!`)\s*(.*)', line.strip())
        if m:
            out.append(f'**{m.group(1)}:** {_inline(m.group(2))}')
            i += 1
            continue

        # ---- plain line ----
        out.append(_inline(line))
        i += 1

    return '\n'.join(out)


# ---------------------------------------------------------------------------
# Parser: sphinx-gallery .py → list of (type, content) cells
# ---------------------------------------------------------------------------

def parse_file(filepath):
    """Parse a sphinx-gallery .py file into (cell_type, source) pairs."""
    text = Path(filepath).read_text()
    cells = []

    # Module docstring → first markdown cell
    m = re.match(r'\s*"""([\s\S]*?)"""\s*\n', text)
    if m:
        cells.append(('markdown', rst_to_md(m.group(1).strip())))
        text = text[m.end():]

    # Split remaining content on `# %%` delimiters
    parts = re.split(r'(?m)^# %%[ \t]*\n', text)

    for part in parts:
        if not part.strip():
            continue

        lines = part.splitlines()
        md_lines = []
        code_start = 0

        # Leading lines that start with `# ` or are bare `#` → markdown block
        for j, ln in enumerate(lines):
            if ln == '#' or ln.startswith('# '):
                md_lines.append(ln[2:] if ln.startswith('# ') else '')
                code_start = j + 1
            else:
                break   # first non-comment line ends the markdown block

        md_text = '\n'.join(md_lines).strip()
        code_text = '\n'.join(lines[code_start:]).strip()

        if md_text:
            cells.append(('markdown', rst_to_md(md_text)))
        if code_text:
            cells.append(('code', code_text))

    return cells


# ---------------------------------------------------------------------------
# Notebook builder
# ---------------------------------------------------------------------------

def build_notebook(cells):
    """Return an nbformat 4 notebook dict."""
    nb_cells = []
    for idx, (kind, src) in enumerate(cells):
        if kind == 'markdown':
            nb_cells.append({
                "cell_type": "markdown",
                "id": f"cell-{idx:03d}",
                "metadata": {},
                "source": src,
            })
        else:
            nb_cells.append({
                "cell_type": "code",
                "id": f"cell-{idx:03d}",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": src,
            })

    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
                "version": "3.9.0",
            },
        },
        "cells": nb_cells,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tutorial_II.py")
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_suffix(".ipynb")

    cells = parse_file(src)
    nb = build_notebook(cells)

    dst.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + '\n')
    print(f"Wrote {len(cells)} cells → {dst}")
