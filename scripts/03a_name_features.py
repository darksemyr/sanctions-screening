"""Step 3a: name features (prefix n_) for every pair in PAIRS_INPUT.

    python scripts/03a_name_features.py            # all pairs, ~10 min
    python scripts/03a_name_features.py --sample 2000

Reads  config.PAIRS_INPUT and config.TARGETS_SIMPLE
Writes config.FEATURES_NAME (pair_id + 15 n_ columns)

When the cleaned data is ready, switch PAIRS_INPUT in config.py and re-run.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from screening import config as C
from screening import name_features as nf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0, help="only this many random pairs (quick check, not saved)")
    args = ap.parse_args()

    for path in (C.PAIRS_INPUT, C.TARGETS_SIMPLE):
        if not path.exists():
            print(f"missing: {path}")
            return 1

    start = time.time()
    pairs = pd.read_parquet(C.PAIRS_INPUT, columns=nf.INPUT_COLUMNS)
    if args.sample:
        pairs = pairs.sample(args.sample, random_state=C.RANDOM_SEED)
    print(f"input        : {C.PAIRS_INPUT.name} ({len(pairs):,} pairs)")

    tables = nf.build_frequency_tables()
    print(f"freq tables  : {time.time() - start:.0f}s")

    feats = nf.compute(pairs, tables)
    nf.check_contract(feats, len(pairs))
    print(f"features     : {time.time() - start:.0f}s")

    if args.sample:
        print(feats.describe(include="all").T.to_string())
        print("sample run: nothing written")
        return 0

    C.FEATURES_NAME.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(C.FEATURES_NAME, index=False)
    print(f"wrote        : {C.FEATURES_NAME} ({len(feats):,} rows)")
    print(f"took         : {time.time() - start:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
