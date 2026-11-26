"""Step 3b: record features (prefix r_) for every pair in PAIRS_INPUT.

    python scripts/03b_record_features.py            # all pairs, ~3 min
    python scripts/03b_record_features.py --sample 2000

Reads  config.PAIRS_INPUT and config.PAIRS_SPLIT (for the address median)
Writes config.FEATURES_RECORD (pair_id + 4 r_ columns), with the missing
values handled as set in config.R_MISSING_POLICY.

The median used for missing addresses comes from Vova's train + val pairs
only, never from his test pairs.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from screening import config as C
from screening import record_features as rf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0, help="only this many random pairs (quick check, not saved)")
    args = ap.parse_args()

    for path in (C.PAIRS_INPUT, C.PAIRS_SPLIT):
        if not path.exists():
            print(f"missing: {path}")
            return 1

    start = time.time()
    pairs = pd.read_parquet(C.PAIRS_INPUT, columns=rf.INPUT_COLUMNS)
    if args.sample:
        pairs = pairs.sample(args.sample, random_state=C.RANDOM_SEED)
    print(f"input        : {C.PAIRS_INPUT.name} ({len(pairs):,} pairs)")

    raw = rf.compute(pairs)
    print(f"features     : {time.time() - start:.0f}s")
    for c in C.R_CATEGORIES:
        print(f"  {c}: {raw[c].value_counts().to_dict()}")
    print(f"  r_address_sim missing: {raw['r_address_sim'].isna().mean():.1%}")

    fill = None
    if C.R_MISSING_POLICY == "different":
        split = pd.read_parquet(C.PAIRS_SPLIT, columns=["pair_id", "split"])
        dev_ids = split.loc[split["split"].isin(["train", "val"]), "pair_id"]
        fill = raw.loc[raw["pair_id"].isin(dev_ids), "r_address_sim"].median()
        print(f"missing policy: 'different'; missing address -> {fill:.4f} (median of train + val)")
    else:
        print(f"missing policy: '{C.R_MISSING_POLICY}'")
    feats = rf.apply_missing_policy(raw, C.R_MISSING_POLICY, fill)
    rf.check_contract(feats, len(pairs), C.R_MISSING_POLICY)

    if args.sample:
        print(feats.describe(include="all").T.to_string())
        print("sample run: nothing written")
        return 0

    C.FEATURES_RECORD.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(C.FEATURES_RECORD, index=False)
    print(f"wrote        : {C.FEATURES_RECORD} ({len(feats):,} rows)")
    print(f"took         : {time.time() - start:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
