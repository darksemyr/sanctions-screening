"""Step 4 prep: names.txt -> data/interim/name_frequency.parquet

    python scripts/04_name_frequency.py
"""
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from screening.name_frequency import NAMES_TXT, FREQ_PARQUET, build


def main() -> int:
    if not NAMES_TXT.exists():
        print(f"missing: {NAMES_TXT}")
        print("curl -L -o data/raw/names.txt "
              "https://data.opensanctions.org/datasets/latest/default/names.txt")
        return 1
    t = time.time()
    df = build()
    print(f"names read   : {df.attrs['n_names']:,}")
    print(f"distinct toks: {len(df):,}")
    print(f"wrote        : {FREQ_PARQUET}")
    print(f"took         : {time.time()-t:.0f}s")
    print()
    print("ten commonest tokens:")
    print(df.head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
