"""Name features (prefix n_): 15 columns per pair, version 2 of notebook 04.

Explored and tested in notebooks 03 (frequency) and 04 (similarity, model).
The reasoning for each choice is in docs/decisions.md. This module is the
reference implementation; the notebooks keep their own copies for history.

Use:
    tables = build_frequency_tables()                 # ~1-2 min
    feats = compute(pairs, tables)                     # one row per pair_id

pairs needs: pair_id, left_schema and, for both sides, the list columns
name, alias, firstName, lastName (as in PAIRS_INPUT).
"""
import re
import unicodedata
from collections import Counter

import jellyfish
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler
from unidecode import unidecode

from screening import config as C
from screening.lists import values

INPUT_COLUMNS = ["pair_id", "left_schema", "right_schema"] + [
    f"{side}_{prop}" for side in ("left", "right") for prop in ("name", "alias", "firstName", "lastName")]

# --- normalisation ----------------------------------------------------------

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize(s):
    """Transliterate to ASCII, lowercase, non-alphanumerics -> single spaces."""
    if not s:
        return ""
    return " ".join(_NON_ALNUM.sub(" ", unidecode(s).lower()).split())


def name_key(s):
    """Word order does not matter: 'novak jan' == 'jan novak'."""
    return " ".join(sorted(s.split()))


def script_of(s):
    """Unicode script of the first letter (LATIN, CYRILLIC, ARABIC, ...)."""
    for ch in s or "":
        if ch.isalpha():
            n = unicodedata.name(ch, "")
            return n.split(" ")[0] if n else "OTHER"
    return "NONE"


# --- titles (persons only) --------------------------------------------------

TITLES = ["general de brigade", "general de division", "general de corps d armee", "brigadier general",
          "major general", "lieutenant general", "lieutenant colonel", "general", "brigadier", "colonel", "col",
          "major", "captain", "capt", "lieutenant", "lt", "admiral", "commander",
          "sheikh", "shaykh", "sheik", "maulavi", "mawlawi", "maulvi", "mullah", "mulla", "mufti", "haji", "hajji",
          "dr", "prof", "professor", "mr", "mrs", "ms", "miss", "sir", "his excellency", "eng", "engineer"]
_TITLE_RE = re.compile(r"\b(" + "|".join(re.escape(t) for t in sorted(TITLES, key=len, reverse=True)) + r")\b")


def strip_titles(name):
    """Remove titles from a normalised name; keep the name if nothing is left."""
    out = " ".join(_TITLE_RE.sub(" ", name).split())
    return out or name


# --- legal forms (companies only) -------------------------------------------

LEGAL_FORMS = {
    "llc":  ["llc", "l l c", "ooo", "limited liability company",
             "obshchestvo s ogranichennoi otvetstvennost iu", "obshchestvo s ogranichennoi otvetstvennostiu",
             "tovaristvo z obmezhenoiu vidpovidal nistiu", "tovarystvo z obmezhenoiu vidpovidalnistiu"],
    "ltd":  ["ltd", "limited", "pte ltd", "pty ltd",
             "ltd sti", "limited sirketi", "pvt ltd", "private limited", "sdn bhd", "bhd"],
    "jsc":  ["jsc", "ao", "oao", "zao", "cjsc", "ojsc", "aktsionernoe obshchestvo", "joint stock company",
             "otkrytoe aktsionernoe obshchestvo", "zakrytoe aktsionernoe obshchestvo",
             "aktsionerne tovaristvo", "aktsionerne tovarystvo",
             "privatne aktsionerne tovaristvo", "pryvatne aktsionerne tovarystvo"],
    "pjsc": ["pjsc", "pao", "publichnoe aktsionernoe obshchestvo", "public joint stock company"],
    "plc":  ["plc", "public limited company"],
    "inc":  ["inc", "incorporated"],
    "corp": ["corp", "corporation"],
    "gmbh": ["gmbh", "gesellschaft mit beschrankter haftung"],
    "kg":   ["gmbh co kg", "kg"],
    "ag":   ["ag", "aktiengesellschaft"],
    "sa":   ["sa", "s a", "sociedad anonima", "societe anonyme",
             "societate pe actiuni", "sa de cv", "s a de c v", "anonim sirketi", "a s"],
    "sarl": ["sarl", "s a r l"],
    "srl":  ["srl", "s r l", "societatea cu raspundere limitata", "societate cu raspundere limitata",
             "sociedad de responsabilidad limitada", "societa a responsabilita limitata"],
    "spa":  ["spa", "s p a"],
    "bv":   ["bv", "b v"],
    "nv":   ["nv", "n v"],
    "llp":  ["llp"],
    "fze":  ["fze", "fzco", "fzc", "fz llc"],
    "co":   ["co", "company"],
    "ou": ["ou"], "sro": ["s r o", "sro", "spol s r o"],
    "spzoo": ["sp z o o", "spolka z ograniczona odpowiedzialnoscia"],
    "kft": ["kft"], "doo": ["d o o", "doo"], "ltda": ["ltda"], "as": ["as", "asa"], "ab": ["ab"],
    "oy": ["oy", "oyj"], "aps": ["aps"],
}
_FORMS = sorted(((v, k) for k, vs in LEGAL_FORMS.items() for v in vs), key=lambda x: -len(x[0]))
_FORM_RE = re.compile(r"\b(" + "|".join(re.escape(v) for v, _ in _FORMS) + r")\b")
_FORM_OF = dict(_FORMS)


def split_legal(name):
    """(core name without legal forms, set of canonical legal forms found)."""
    found = {_FORM_OF[m.group(1)] for m in _FORM_RE.finditer(name)}
    return " ".join(_FORM_RE.sub(" ", name).split()), found


def legal_status(fl, fr):
    if not fl and not fr:
        return "none"
    if not fl or not fr:
        return "one_side"
    return "same" if fl & fr else "conflict"


# --- names of one side --------------------------------------------------------

def side_names(row, side, person):
    """Normalised, de-duplicated names + aliases (at most MAX_NAMES); titles stripped for persons."""
    out = []
    for n in values(row, side, ["name", "alias"]):
        k = normalize(n)
        if k and k not in out:
            out.append(k)
    out = out[:C.MAX_NAMES]
    return list(dict.fromkeys(strip_titles(n) for n in out)) if person else out


# --- word-level comparisons ----------------------------------------------------

def fuzzy_word_match(a, b):
    return a == b or (len(a) > 2 and len(b) > 2 and JaroWinkler.normalized_similarity(a, b) >= 0.9)


def _shorter_longer(tl, tr):
    return (tl, tr) if len(tl) <= len(tr) else (tr, tl)


def coverage(tl, tr):
    """Share of words (>1 letter) of the shorter name found in the longer one."""
    short, long_ = _shorter_longer(tl, tr)
    short = [t for t in short if len(t) > 1]
    if not short:
        return np.nan
    return sum(any(fuzzy_word_match(t, u) for u in long_) for t in short) / len(short)


def phonetic_share(tl, tr):
    """Share of words of the shorter name whose Metaphone code occurs in the longer one."""
    short, long_ = _shorter_longer(tl, tr)
    short = [t for t in short if len(t) > 1 and t.isalpha()]
    codes = {jellyfish.metaphone(u) for u in long_ if len(u) > 1 and u.isalpha()}
    if not short or not codes:
        return np.nan
    return sum(jellyfish.metaphone(t) in codes for t in short) / len(short)


def initials_status(tl, tr):
    il, ir = [t for t in tl if len(t) == 1], [t for t in tr if len(t) == 1]
    if not il and not ir:
        return "no_initials"
    ok = (all(any(u.startswith(t) for u in tr) for t in il)
          and all(any(u.startswith(t) for u in tl) for t in ir))
    return "compatible" if ok else "conflict"


def field_status(raw_l, raw_r):
    """Compare the firstName or lastName fields of both sides."""
    L = [x for x in (normalize(v) for v in raw_l) if x]
    R = [x for x in (normalize(v) for v in raw_r) if x]
    if not L or not R:
        return "missing"
    if set(L) & set(R):
        return "match"
    if max(JaroWinkler.normalized_similarity(a, b) for a in L for b in R) >= 0.9:
        return "similar"
    if any((len(a) == 1 or len(b) == 1) and a[0] == b[0] for a in L for b in R):
        return "similar"
    return "conflict"


def number_status(a, b):
    nl, nr = set(re.findall(r"\d+", a)), set(re.findall(r"\d+", b))
    if not nl and not nr:
        return "none"
    if not nl or not nr:
        return "one_side"
    return "same" if nl == nr else "different"


# --- frequency --------------------------------------------------------------------

def build_frequency_tables(path=None):
    """Count per entity how often each word and each full name occurs in targets.simple.

    Entities with the same name and the same birth date count once (notebook 03,
    section 1b). Returns {"word": {group: Counter}, "name": {group: Counter}}.
    """
    path = path or C.TARGETS_SIMPLE
    groups = C.FREQ_GROUPS
    tg = pd.read_csv(path, usecols=["id", "schema", "name", "aliases", "birth_date"],
                     dtype=str, keep_default_na=False)
    tg = tg[tg["schema"].isin(groups["person"] + groups["org"])]
    key = tg["name"].map(lambda n: name_key(normalize(n)))
    tg = tg.assign(identity=np.where(tg["birth_date"].str.strip() != "",
                                     key + " | dob " + tg["birth_date"], "id " + tg["id"]))
    tg = tg.drop_duplicates(subset=["schema", "identity"])

    tables = {"word": {}, "name": {}}
    for group, schemas in groups.items():
        words, names = Counter(), Counter()
        sub = tg[tg["schema"].isin(schemas)]
        for name, aliases in zip(sub["name"], sub["aliases"]):
            variants = [normalize(v) for v in [name] + (aliases.split(";") if aliases else [])]
            variants = [v for v in variants if v]
            words.update({w for v in variants for w in v.split() if len(w) > 1})
            names.update({name_key(v) for v in variants})
        tables["word"][group], tables["name"][group] = words, names
    return tables


def band(n, bands):
    return next(i for i, (lo, hi) in enumerate(bands) if lo <= n <= hi)


def frequency_features(L, R, person, tables):
    """Band of the rarest shared word and of the rarest shared full name (-1 = nothing shared)."""
    g = "person" if person else "org"
    sw = {w for n in L for w in n.split() if len(w) > 1} & {w for n in R for w in n.split() if len(w) > 1}
    sn = {name_key(n) for n in L} & {name_key(n) for n in R}
    return {
        "n_freq_word_band": band(min(tables["word"][g].get(w, 0) for w in sw), C.WORD_BANDS) if sw else -1,
        "n_freq_name_band": band(min(tables["name"][g].get(k, 0) for k in sn), C.NAME_BANDS) if sn else -1,
    }


# --- one pair ------------------------------------------------------------------------

def pair_features(r, tables):
    """All 15 name features of one pair (a row from itertuples, a Series or a dict)."""
    get = r.get if hasattr(r, "get") else (lambda k: getattr(r, k))
    person = get("left_schema") == "Person"
    L, R = side_names(r, "left", person), side_names(r, "right", person)
    f = {"pair_id": get("pair_id")}
    if not L or not R:
        f.update({c: np.nan for c in C.N_NUMERIC})
        f.update({c: "no_name" for c in C.N_CATEGORIES})
        f.update({"n_freq_word_band": -1, "n_freq_name_band": -1})
        return f

    m_sort = process.cdist(L, R, scorer=fuzz.token_sort_ratio)
    m_set = process.cdist(L, R, scorer=fuzz.token_set_ratio)
    m_jw = process.cdist(L, R, scorer=JaroWinkler.normalized_similarity)
    f["n_sim_sort"] = m_sort.max() / 100
    f["n_sim_set"] = m_set.max() / 100
    f["n_jaro_winkler"] = float(m_jw.max())
    f["n_exact_name"] = int(bool({name_key(x) for x in L} & {name_key(x) for x in R}))

    i, j = np.unravel_index(m_set.argmax(), m_set.shape)  # best pair of names
    tl, tr = L[i].split(), R[j].split()
    f["n_token_coverage"] = coverage(tl, tr)
    f["n_phonetic_match"] = phonetic_share(tl, tr)
    f["n_initials"] = initials_status(tl, tr)
    f["n_lastname_status"] = field_status(values(r, "left", ["lastName"]), values(r, "right", ["lastName"]))
    f["n_firstname_status"] = field_status(values(r, "left", ["firstName"]), values(r, "right", ["firstName"]))
    f["n_numbers"] = number_status(L[i], R[j])
    raw_l, raw_r = values(r, "left", ["name"]), values(r, "right", ["name"])
    f["n_script"] = ("same" if script_of(raw_l[0] if raw_l else "") == script_of(raw_r[0] if raw_r else "")
                     else "different")

    if person:
        f["n_sim_core"], f["n_legal_form"] = np.nan, "n/a"
    else:
        sl, sr = [split_legal(x) for x in L], [split_legal(x) for x in R]
        cl, cr = [c for c, _ in sl if c], [c for c, _ in sr if c]
        f["n_sim_core"] = (process.cdist(cl, cr, scorer=fuzz.token_sort_ratio).max() / 100
                           if cl and cr else np.nan)
        f["n_legal_form"] = legal_status(set().union(*(s for _, s in sl)), set().union(*(s for _, s in sr)))

    f.update(frequency_features(L, R, person, tables))
    return f


# --- whole table --------------------------------------------------------------------

def compute(pairs, tables):
    """One row per pair: pair_id + the 15 n_ columns, categories as pandas Categorical."""
    out = pd.DataFrame([pair_features(r, tables) for r in pairs.itertuples(index=False)])
    out = out[["pair_id"] + C.N_FEATURES]
    for c in C.N_NUMERIC:
        out[c] = out[c].astype(float)
    for c, cats in C.N_CATEGORIES.items():
        unknown = set(out[c].dropna()) - set(cats)
        if unknown:
            raise ValueError(f"{c}: values not in config.N_CATEGORIES: {sorted(unknown)}")
        out[c] = pd.Categorical(out[c], categories=cats)
    return out


def check_contract(feats, n_pairs):
    """Raise if the output breaks the contract in config.py."""
    problems = []
    if len(feats) != n_pairs:
        problems.append(f"{len(feats):,} rows, expected {n_pairs:,}")
    if feats["pair_id"].duplicated().any():
        problems.append("duplicate pair_id")
    if list(feats.columns) != ["pair_id"] + C.N_FEATURES:
        problems.append(f"unexpected columns: {list(feats.columns)}")
    for c in C.N_CATEGORIES:
        if feats[c].isna().any():
            problems.append(f"{c}: NaN or value outside {C.N_CATEGORIES[c]}")
    if problems:
        raise ValueError("name features break the contract: " + "; ".join(problems))
