"""Regenerate the paper figures into figures/.

    python make_all_figures.py              # every figure
    python make_all_figures.py fig02b fig03 # only the named figures
"""

import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"


def main() -> None:
    scripts = sorted(SCRIPTS_DIR.glob("fig*.py"))
    wanted = sys.argv[1:]
    if wanted:
        scripts = [s for s in scripts if s.name.split("_")[0] in wanted]
        if not scripts:
            sys.exit(f"No figure script matches {wanted}; see scripts/fig*.py")
    # Each figure runs in its own process so one figure's style can't leak into another.
    for script in scripts:
        subprocess.run([sys.executable, str(script)], check=True)


if __name__ == "__main__":
    main()
