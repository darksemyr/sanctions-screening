"""Turn names.txt into a frequency / IDF table.

`names.txt` is 3.16 million lines, one name per line, no structure. On its own
it is not something you can analyse. What we want from it is: how common is a
given name part across the whole sanctions database.

Output: data/interim/name_frequency.parquet, one row per distinct token.

    python scripts/04_name_frequency.py

Load it as a Series, which behaves as a dictionary for lookups and as a
Series for everything else:

    freq = load_frequency()          # Series: token -> idf
    df["idf"] = df["token"].map(freq)
    freq.sort_values().tail(50)      # the fifty commonest tokens
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter

import pandas as pd

from . import config as C

NAMES_TXT = C.DATA_RAW / "names.txt"
FREQ_PARQUET = C.DATA_INTERIM / "name_frequency.parquet"

# Keep letters, digits and inner apostrophes/hyphens; drop everything else.
_PUNCT = re.compile(r"[^\w'\-]+", re.UNICODE)


def normalise(text: str) -> str:
    """Casefold, strip accents, flatten punctuation to spaces.

    Accents only: this deliberately does NOT transliterate between alphabets,
    so Cyrillic stays Cyrillic. Transliteration is a separate decision.
    """
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return _PUNCT.sub(" ", text.casefold()).strip()


def tokenise(text: str) -> list[str]:
    return [t for t in normalise(text).split() if t]


def build(src=None, dest=None, min_count: int = 1) -> pd.DataFrame:
    src = src or NAMES_TXT
    dest = dest or FREQ_PARQUET

    counts: Counter[str] = Counter()
    n_names = 0
    with open(src, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            toks = tokenise(line)
            if not toks:
                continue
            n_names += 1
            # set(), not the list: a token appearing twice in one name is
            # still one name containing it. This is document frequency.
            counts.update(set(toks))

    items = [(tok, n) for tok, n in counts.items() if n >= min_count]
    df = pd.DataFrame(items, columns=["token", "n"])
    df["idf"] = df.n.map(lambda n: math.log(n_names / n))
    df = df.sort_values("n", ascending=False).reset_index(drop=True)
    df.attrs["n_names"] = n_names

    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest, index=False)
    return df


def load_frequency(column: str = "idf") -> pd.Series:
    """The table as a Series indexed by token. Use .map() against it."""
    df = pd.read_parquet(FREQ_PARQUET)
    return df.set_index("token")[column]


# ------------------------------------------------------------ feature helpers

def name_idf_features(name: str, idf: pd.Series, default: float | None = None) -> dict:
    """Rarity features for one name.

    A name is distinctive if it contains at least one rare token. Taking the
    maximum IDF captures that better than an average, which a few very common
    tokens would drag down.
    """
    toks = tokenise(name)
    if not toks:
        return {"n_tokens": 0, "max_idf": 0.0, "mean_idf": 0.0, "min_idf": 0.0}
    fallback = default if default is not None else float(idf.max())
    vals = [float(idf.get(t, fallback)) for t in toks]
    return {
        "n_tokens": len(toks),
        "max_idf": max(vals),
        "mean_idf": sum(vals) / len(vals),
        "min_idf": min(vals),
    }


def shared_token_idf(left: str, right: str, idf: pd.Series,
                     default: float | None = None) -> dict:
    """How distinctive is what the two names have in common?

    Two names sharing 'mohammed' tell you almost nothing. Two names sharing
    'torosov' tell you a great deal. A plain token-overlap count cannot see
    the difference; weighting the overlap by IDF can.
    """
    lt, rt = set(tokenise(left)), set(tokenise(right))
    if not lt or not rt:
        return {"shared_tokens": 0, "shared_idf_sum": 0.0, "shared_idf_max": 0.0,
                "jaccard": 0.0}
    fallback = default if default is not None else float(idf.max())
    shared = lt & rt
    vals = [float(idf.get(t, fallback)) for t in shared] or [0.0]
    return {
        "shared_tokens": len(shared),
        "shared_idf_sum": sum(vals),
        "shared_idf_max": max(vals),
        "jaccard": len(shared) / len(lt | rt),
    }
