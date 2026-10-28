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

# --- Scope decision, made 2026-10-28 -----------------------------------
# The file contains six kinds of record pair. Occupancy, Succession and
# Position are links between a person and a job, not names to be matched.
# We keep only the three schemas where two names are actually compared.
# Roughly 273k of 755k rows are dropped by this.
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
