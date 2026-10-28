"""Step 1: nested JSON Lines -> one flat parquet table.

    python scripts/01_parse.py
"""
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from screening import config as C
from screening.parse import parse


def main() -> int:
    if not C.RAW_PAIRS.exists():
        print(f"missing: {C.RAW_PAIRS}")
        print("download it into data/raw/ first, see README")
        return 1
    print(f"reading  {C.RAW_PAIRS}")
    start = time.time()
    stats = parse()
    print(f"rows written : {stats['rows_written']:,}")
    print(f"unparseable  : {stats['unparseable']:,}")
    print(f"wrote        : {C.PAIRS_FLAT}")
    print(f"took         : {time.time() - start:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
