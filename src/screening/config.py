"""Every path and every scope decision lives here, not in a notebook.

If a number or a filter is used in more than one place, it belongs in this
file. That is the difference between an analysis someone can re-run and an
analysis that only worked on your laptop that one afternoon.
"""

from pathlib import Path

# Project root, found relative to this file so it works from anywhere.
ROOT = Path(__file__).resolve().parents[2]

DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

# The source file, exactly as downloaded. Do not rename it: the date in the
# filename is the data version, and the report has to state which one we used.
RAW_PAIRS = DATA_RAW / "pairs-20251209.json.gz"

PAIRS_FLAT = DATA_INTERIM / "pairs_flat.parquet"
PAIRS_SCOPED = DATA_INTERIM / "pairs_scoped.parquet"
PAIRS_CLEAN = DATA_INTERIM / "pairs_clean.parquet"  # written by the cleaning step (Vova)

# Input of all feature scripts. Switch to PAIRS_CLEAN in the same pull request
# that adds the cleaning, so code and data on main always match.
PAIRS_INPUT = PAIRS_SCOPED

# --- Feature engineering (step 3), added 2026-11-11 ----------------------
# Every name and alias in the whole OpenSanctions database, one per line.
# Used to count how common a name is (name-frequency feature).
# Downloaded 2026-11-11 (latest default collection). The pairs are from
# 2025-12-09; older names exports need a data delivery token.
NAMES_TXT = DATA_RAW / "names-20261111.txt"

# Each person writes their own feature file, keyed by pair_id.
# Contract: pair_id + own columns only, same row count as PAIRS_SCOPED,
# column prefix n_ (names) or r_ (record), no NaN, categories from the
# lists defined in the owner's section below.
FEATURES_NAME = DATA_PROCESSED / "features_name.parquet"      # Person A
FEATURES_RECORD = DATA_PROCESSED / "features_record.parquet"  # Person B
FEATURES = DATA_PROCESSED / "features.parquet"                # merged by 03_features.py

# --- Scope decision, made 2026-10-28 -----------------------------------
# The file contains six kinds of record pair. Occupancy, Succession and
# Position are links between a person and a job, not names to be matched.
# We keep only the three schemas where two names are actually compared.
# A pair is kept only if BOTH sides are one of these (profile.in_scope_mask).
# 316,346 of 755,540 rows are dropped by this; 439,194 remain.
IN_SCOPE_SCHEMAS = ("Person", "Company", "Organization")

# Entity properties we pull out of the nested records. Anything not listed
# here is discarded at parse time, on purpose: a narrow table is easier to
# reason about than a wide one full of columns nobody uses.
LIST_PROPS = (
    "name",
    "alias",
    "firstName",
    "lastName",
    "middleName",
    "birthDate",
    "country",
    "nationality",
    "idNumber",
    "address",
    "topics",
)

RANDOM_SEED = 42

# --- Name features (Person A) -------------------------------------------
# Category lists and parameters for name features go here, and only here.


# --- Record features (Person B) -----------------------------------------
# Category lists and parameters for record features go here, and only here.
