"""Convert Python files with `# %%` cell markers to .ipynb notebooks.

Usage
-----
    python notebooks/build_notebooks.py

This reads every file matching `notebooks/*.py` and writes a sibling .ipynb.
Cells are split on lines starting with `# %%`. A line `# %% [markdown]`
starts a markdown cell.
"""
from __future__ import annotations

import json
import re
from pathlib import Path


CELL_RE = re.compile(r"^# %%(?:\s+\[(\w+)\])?\s*(.*)$")


def parse_cells(text: str):
    cells = []
    cur_lines: list[str] = []
    cur_type = "code"
    for line in text.splitlines():
        m = CELL_RE.match(line)
        if m:
            if cur_lines:
                cells.append((cur_type, "\n".join(cur_lines).rstrip()))
            cur_type = (m.group(1) or "code").lower()
            cur_lines = []
        else:
            cur_lines.append(line)
    if cur_lines:
        cells.append((cur_type, "\n".join(cur_lines).rstrip()))
    return cells


def to_nbformat(cells):
    out_cells = []
    for ctype, src in cells:
        if not src.strip():
            continue
        cell = {
            "cell_type": "markdown" if ctype == "markdown" else "code",
            "metadata": {},
            "source": src.splitlines(keepends=True),
        }
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        out_cells.append(cell)
    return {
        "cells": out_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main():
    here = Path(__file__).resolve().parent
    for py_file in sorted(here.glob("notebook_*.py")):
        cells = parse_cells(py_file.read_text())
        nb = to_nbformat(cells)
        out = py_file.with_suffix(".ipynb")
        out.write_text(json.dumps(nb, indent=1))
        print(f"Wrote {out.name} ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
