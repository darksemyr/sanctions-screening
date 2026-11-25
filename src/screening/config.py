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
# Written by notebooks/02-vs-eda-cleaning-split.ipynb (Vova).
# In scope on both sides, self-pairs removed, duplicate pairs removed after
# normalising the order of the two sides. One row per pair_id.
PAIRS_CLEAN = DATA_PROCESSED / "pairs_clean.parquet"

# PAIRS_CLEAN plus two columns: split (train / val / test) and entity_group.
# Groups are built from positive pairs only, so a group is one entity as the
# analysts decided it. A whole group goes to one split; negative pairs whose
# two sides fall in different splits are dropped, so this file has fewer rows
# than PAIRS_CLEAN. Join features to it on pair_id.
PAIRS_SPLIT = DATA_PROCESSED / "pairs_split.parquet"

# Input of all feature scripts.
PAIRS_INPUT = PAIRS_CLEAN

# --- Feature engineering (step 3), added 2026-11-11 ----------------------
# Every name and alias in the whole OpenSanctions database, one per line.
# Used to count how common a name is (name-frequency feature).
# Downloaded 2026-11-11 (latest default collection). The pairs are from
# 2025-12-09; older names exports need a data delivery token.
NAMES_TXT = DATA_RAW / "names-20261111.txt"

# Each person writes their own feature file, keyed by pair_id.
# Contract: pair_id + own columns only, same row count as PAIRS_INPUT,
# column prefix n_ (names) or r_ (record), categories from the lists
# defined in the owner's section below. Categorical columns never NaN.
# Numeric columns may be NaN where a value is undefined (e.g. n_sim_core for
# persons); HistGradientBoosting, LightGBM and XGBoost handle NaN natively.
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
# Code: src/screening/name_features.py, runner: scripts/03a_name_features.py.
# Reasoning: docs/decisions.md and notebooks 03 + 04.

# OpenSanctions entity export (one row per entity, aliases ";"-separated).
# Used to count how many entities carry a name or a word (frequency bands).
TARGETS_SIMPLE = DATA_RAW / "targets.simple-20261111.csv"

# Cap on names + aliases per side; a few entities have hundreds.
MAX_NAMES = 30

# Frequency bands: (lowest count, highest count) -> band 0, 1, 2, ...
# 0 (not in targets.simple) is merged with 1-2 on purpose: both are almost
# always positive pairs (merge effect, notebook 06). -1 = nothing shared.
WORD_BANDS = [(0, 2), (3, 10), (11, 100), (101, 1000), (1001, 10**9)]
NAME_BANDS = [(0, 2), (3, 5), (6, 20), (21, 10**9)]
FREQ_GROUPS = {"person": ("Person",), "org": ("Company", "Organization", "LegalEntity")}

# The 15 name features (version 2, notebook 04, section 7) and their categories.
# "no_name" = at least one side has no name at all.
N_NUMERIC = ["n_sim_sort", "n_sim_set", "n_jaro_winkler", "n_exact_name",
             "n_token_coverage", "n_phonetic_match", "n_sim_core",
             "n_freq_word_band", "n_freq_name_band"]
N_CATEGORIES = {
    "n_initials": ["no_initials", "compatible", "conflict", "no_name"],
    "n_lastname_status": ["match", "similar", "conflict", "missing", "no_name"],
    "n_firstname_status": ["match", "similar", "conflict", "missing", "no_name"],
    "n_numbers": ["none", "one_side", "same", "different", "no_name"],
    "n_script": ["same", "different", "no_name"],
    "n_legal_form": ["none", "one_side", "same", "conflict", "n/a", "no_name"],
}
N_FEATURES = N_NUMERIC + list(N_CATEGORIES)


# --- Record features (Person B) -----------------------------------------
# Category lists and parameters for record features go here, and only here.
