"""Record features (prefix r_): birth date, country, ID number and address.

Moved from notebook 05 (sections 1, 5f); the logic is unchanged. Every feature
compares the two sides; no raw value is passed to the model.

    r_dob_status      exact / partial_match / swapped / day_month_typo /
                      same_year / close / different / missing
    r_country_status  same / different / missing   (country + nationality)
    r_id_status       same / different / missing   (idNumber field)
    r_address_sim     best token-set similarity of the addresses, 0..1, NaN if missing

compute() returns the raw comparison ("missing" kept). apply_missing_policy()
turns it into what the model sees, as set in config.R_MISSING_POLICY.
"""
import re

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from screening import config as C
from screening.lists import values
from screening.name_features import normalize

PROPS = ["birthDate", "country", "nationality", "idNumber", "address"]
INPUT_COLUMNS = ["pair_id"] + [f"{side}_{p}" for side in ("left", "right") for p in PROPS]

MAX_ADDRESSES = 10

# --- birth date ----------------------------------------------------------------

_DOB_RE = re.compile(r"^(\d{4})(?:-(\d{2}))?(?:-(\d{2}))?")
_WIDTH = (4, 2, 2)          # digits of year, month, day
DOB_ORDER = [c for c in C.R_CATEGORIES["r_dob_status"] if c != "missing"]   # best first


def parse_dob(s):
    """'1968-03-02' -> (1968, 3, 2); '1968' -> (1968,); unparseable -> None."""
    m = _DOB_RE.match(str(s).strip())
    return tuple(int(x) for x in m.groups() if x) if m else None


def one_edit(x, y, width):
    """True if two numbers differ by one changed digit or two neighbouring digits swapped."""
    a, b = f"{x:0{width}d}", f"{y:0{width}d}"
    diff = [i for i in range(width) if a[i] != b[i]]
    if len(diff) == 1:
        return True
    return (len(diff) == 2 and diff[1] == diff[0] + 1
            and a[diff[0]] == b[diff[1]] and a[diff[1]] == b[diff[0]])


def dob_relation(a, b):
    """Relation of two parsed birth dates (version 4 of notebook 05)."""
    if len(a) == 3 and a == b:
        return "exact"
    k = min(len(a), len(b))
    if a[:k] == b[:k]:
        return "partial_match"
    if k == 3 and a[0] == b[0] and a[1] == b[2] and a[2] == b[1]:
        return "swapped"
    diff = [i for i in range(k) if a[i] != b[i]]
    if len(diff) == 1 and diff[0] > 0 and one_edit(a[diff[0]], b[diff[0]], _WIDTH[diff[0]]):
        return "day_month_typo"       # same year, only month or only day has a typo
    if a[0] == b[0]:
        return "same_year"
    if abs(a[0] - b[0]) == 1:
        return "close"
    return "different"


def dob_status(row):
    """Best relation over all birth dates of both sides; 'missing' if a side has none."""
    left = [d for d in map(parse_dob, values(row, "left", ["birthDate"])) if d]
    right = [d for d in map(parse_dob, values(row, "right", ["birthDate"])) if d]
    if not left or not right:
        return "missing"
    rels = {dob_relation(a, b) for a in left for b in right}
    return next(x for x in DOB_ORDER if x in rels)


# --- country, ID number, address -------------------------------------------------

def set_status(row, props):
    """same if the two sides share a value (case-insensitive), different if not."""
    left = {str(v).strip().lower() for v in values(row, "left", props)} - {""}
    right = {str(v).strip().lower() for v in values(row, "right", props)} - {""}
    if not left or not right:
        return "missing"
    return "same" if left & right else "different"


_NON_ALNUM = re.compile(r"[^a-z0-9]")


def id_status(row):
    """ID numbers without punctuation, at least 4 characters; one contained in the
    other (at least 6 characters) also counts as the same."""
    def norm(side):
        return {x for x in (_NON_ALNUM.sub("", str(v).lower()) for v in values(row, side, ["idNumber"]))
                if len(x) >= 4}
    left, right = norm("left"), norm("right")
    if not left or not right:
        return "missing"
    if left & right or any((a in b or b in a) and min(len(a), len(b)) >= 6 for a in left for b in right):
        return "same"
    return "different"


def address_sim(row):
    """Best token-set similarity over up to 10 addresses per side, 0..1; NaN if a side has none."""
    left = [x for x in (normalize(v) for v in values(row, "left", ["address"])) if x][:MAX_ADDRESSES]
    right = [x for x in (normalize(v) for v in values(row, "right", ["address"])) if x][:MAX_ADDRESSES]
    if not left or not right:
        return np.nan
    return float(process.cdist(left, right, scorer=fuzz.token_set_ratio).max()) / 100


def pair_features(row):
    return {"pair_id": row.pair_id if hasattr(row, "pair_id") else row["pair_id"],
            "r_dob_status": dob_status(row),
            "r_country_status": set_status(row, ["country", "nationality"]),
            "r_id_status": id_status(row),
            "r_address_sim": address_sim(row)}


# --- whole table -------------------------------------------------------------------

def compute(pairs):
    """One row per pair: pair_id + the r_ columns, "missing" kept (raw comparison)."""
    out = pd.DataFrame([pair_features(r) for r in pairs.itertuples(index=False)],
                       columns=["pair_id"] + C.R_FEATURES)
    for c in C.R_NUMERIC:
        out[c] = out[c].astype(float)
    for c, cats in C.R_CATEGORIES.items():
        unknown = set(out[c].dropna()) - set(cats)
        if unknown:
            raise ValueError(f"{c}: values not in config.R_CATEGORIES: {sorted(unknown)}")
        out[c] = pd.Categorical(out[c], categories=cats)
    return out


def apply_missing_policy(feats, policy, address_fill=None):
    """What the model sees (config.R_MISSING_POLICY).

    "keep"      : unchanged.
    "different" : every "missing" category -> "different" (the category "missing"
                  is removed), missing r_address_sim -> address_fill (the median of
                  the training pairs, computed by the caller).
    """
    if policy == "keep":
        return feats
    if policy != "different":
        raise ValueError(f"unknown missing policy: {policy!r}")
    if address_fill is None or np.isnan(address_fill):
        raise ValueError("policy 'different' needs an address_fill value")
    out = feats.copy()
    for c, cats in C.R_CATEGORIES.items():
        kept = [x for x in cats if x != "missing"]
        out[c] = pd.Categorical(out[c].astype(str).replace("missing", "different"), categories=kept)
    out["r_address_sim"] = out["r_address_sim"].fillna(address_fill)
    return out


def check_contract(feats, n_pairs, policy):
    """Raise if the output breaks the contract in config.py."""
    problems = []
    if len(feats) != n_pairs:
        problems.append(f"{len(feats):,} rows, expected {n_pairs:,}")
    if feats["pair_id"].duplicated().any():
        problems.append("duplicate pair_id")
    if list(feats.columns) != ["pair_id"] + C.R_FEATURES:
        problems.append(f"unexpected columns: {list(feats.columns)}")
    for c in C.R_CATEGORIES:
        if feats[c].isna().any():
            problems.append(f"{c}: NaN or value outside the categories")
        if policy == "different" and (feats[c].astype(str) == "missing").any():
            problems.append(f"{c}: 'missing' left although the policy is 'different'")
    if policy == "different" and feats["r_address_sim"].isna().any():
        problems.append("r_address_sim: NaN left although the policy is 'different'")
    if problems:
        raise ValueError("record features break the contract: " + "; ".join(problems))
