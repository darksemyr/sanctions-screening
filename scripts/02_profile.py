"""Step 2: profile the flat table, write reports/01-data-profile.md.

    python scripts/02_profile.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from screening.profile import run


def main() -> int:
    out = run()
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
