#!/usr/bin/env python
"""Rebuild every paper figure from the tracked data.

Runs each script in ``figures/`` in turn, in alphabetical order, and exits
non-zero if any of them fails. ``make_fig1_panels.py`` regenerates Figure 1's
panels only; the assembled ``fig1.pdf`` is not rebuilt.

    uv run python scripts/make_all_figures.py
"""

import subprocess
import sys
from pathlib import Path

FIGURES_DIR = Path(__file__).resolve().parents[1] / "figures"


def main() -> int:
    failed = []
    for script in sorted(FIGURES_DIR.glob("*.py")):
        print(f"--- {script.name}", flush=True)
        if subprocess.run([sys.executable, str(script)], cwd=FIGURES_DIR.parent).returncode:
            failed.append(script.name)
    if failed:
        print(f"Failed: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
