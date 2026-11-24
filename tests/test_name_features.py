"""Tests for screening.name_features. Run: pytest -q

Small hand-made pairs; the frequency tables are built from a tiny CSV, so no
downloaded data is needed.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from screening import config as C
from screening import name_features as nf


# --- helpers -------------------------------------------------------------------

def test_normalize():
    assert nf.normalize("Jörn-Steinar  SELJELID") == "jorn steinar seljelid"
    assert nf.normalize("Владимир Путин") == "vladimir putin"
    assert nf.normalize(None) == ""


def test_name_key_ignores_word_order():
    assert nf.name_key("novak jan") == nf.name_key("jan novak")


def test_strip_titles_keeps_name_if_only_title():
    assert nf.strip_titles("general de brigade burhan qadour") == "burhan qadour"
    assert nf.strip_titles("dr he wei") == "he wei"
    assert nf.strip_titles("general") == "general"


def test_split_legal():
    core, forms = nf.split_legal(nf.normalize("NAI EUROPE ENERGY GMBH & CO. KG"))
    assert core == "nai europe energy"
    assert "kg" in forms
    assert nf.split_legal(nf.normalize("Kihonzi buzhaga OÜ")) == ("kihonzi buzhaga", {"ou"})


@pytest.mark.parametrize("fl, fr, expected", [
    (set(), set(), "none"), ({"ltd"}, set(), "one_side"),
    ({"ltd"}, {"ltd", "co"}, "same"), ({"llc"}, {"jsc"}, "conflict")])
def test_legal_status(fl, fr, expected):
    assert nf.legal_status(fl, fr) == expected


def test_initials_status():
    assert nf.initials_status(["jan", "novak"], ["novak", "jan"]) == "no_initials"
    assert nf.initials_status(["j", "novak"], ["jan", "novak"]) == "compatible"
    assert nf.initials_status(["p", "novak"], ["jan", "novak"]) == "conflict"


def test_field_status():
    assert nf.field_status([], ["Novak"]) == "missing"
    assert nf.field_status(["Novák"], ["NOVAK"]) == "match"
    assert nf.field_status(["Novakova"], ["Novakov"]) == "similar"
    assert nf.field_status(["Smith"], ["Novak"]) == "conflict"


def test_number_status():
    assert nf.number_status("abc", "abc") == "none"
    assert nf.number_status("dg tech 2", "dg tech") == "one_side"
    assert nf.number_status("unit 30", "unit 30") == "same"
    assert nf.number_status("unit 30", "unit 31") == "different"


def test_coverage_and_phonetic():
    assert nf.coverage(["jan", "novak"], ["jan", "petr", "novak"]) == 1.0
    assert np.isnan(nf.coverage(["j"], ["jan"]))
    assert nf.phonetic_share(["smith"], ["smyth"]) == 1.0


def test_band():
    assert nf.band(0, C.WORD_BANDS) == 0
    assert nf.band(2, C.WORD_BANDS) == 0
    assert nf.band(3, C.WORD_BANDS) == 1
    assert nf.band(5000, C.WORD_BANDS) == len(C.WORD_BANDS) - 1


# --- frequency tables + whole pairs ------------------------------------------------

@pytest.fixture
def tables(tmp_path):
    csv = tmp_path / "targets.csv"
    pd.DataFrame([
        # same name + same birth date -> counted once
        {"id": "a1", "schema": "Person", "name": "Jan Novak", "aliases": "", "birth_date": "1970-01-01"},
        {"id": "a2", "schema": "Person", "name": "Jan Novak", "aliases": "", "birth_date": "1970-01-01"},
        {"id": "a3", "schema": "Person", "name": "Jan Novak", "aliases": "Novak Jan", "birth_date": ""},
        {"id": "a4", "schema": "Person", "name": "Petr Novak", "aliases": "", "birth_date": ""},
        {"id": "c1", "schema": "Company", "name": "Acme Ltd", "aliases": "", "birth_date": ""},
        {"id": "x1", "schema": "Vessel", "name": "Jan Novak", "aliases": "", "birth_date": ""},
    ]).to_csv(csv, index=False)
    return nf.build_frequency_tables(csv)


def test_frequency_tables(tables):
    assert tables["name"]["person"]["jan novak"] == 2          # a1/a2 collapsed, a3 counts; vessel ignored
    assert tables["word"]["person"]["novak"] == 3
    assert tables["word"]["org"]["acme"] == 1


def _pair(left, right, schema="Person", **extra):
    row = {"pair_id": "p1", "left_schema": schema, "right_schema": schema,
           "left_name": [left] if left else [], "right_name": [right] if right else [],
           "left_alias": [], "right_alias": [], "left_firstName": [], "right_firstName": [],
           "left_lastName": [], "right_lastName": []}
    row.update(extra)
    return row


def test_person_pair(tables):
    f = nf.pair_features(_pair("Dr. Jan Novák", "NOVAK, Jan", left_lastName=["Novák"], right_lastName=["Novak"]),
                         tables)
    assert f["n_exact_name"] == 1 and f["n_sim_sort"] == 1.0
    assert f["n_lastname_status"] == "match" and f["n_firstname_status"] == "missing"
    assert f["n_legal_form"] == "n/a" and np.isnan(f["n_sim_core"])
    assert f["n_freq_name_band"] == 0 and f["n_freq_word_band"] == 0


def test_company_pair(tables):
    f = nf.pair_features(_pair("Acme Ltd", "ACME LLC", schema="Company"), tables)
    assert f["n_sim_core"] == 1.0 and f["n_legal_form"] == "conflict"


def test_missing_name_gives_no_name(tables):
    f = nf.pair_features(_pair("Jan Novak", None), tables)
    assert all(f[c] == "no_name" for c in C.N_CATEGORIES)
    assert f["n_freq_word_band"] == -1


def test_compute_meets_contract(tables):
    pairs = pd.DataFrame([_pair("Jan Novak", "Jan Novak"),
                          {**_pair("Acme Ltd", "Acme", schema="Company"), "pair_id": "p2"},
                          {**_pair(None, "Jan Novak"), "pair_id": "p3"}])
    feats = nf.compute(pairs, tables)
    nf.check_contract(feats, len(pairs))
    assert list(feats.columns) == ["pair_id"] + C.N_FEATURES
    assert len(C.N_FEATURES) == 15


def test_contract_catches_unknown_category(tables, monkeypatch):
    monkeypatch.setitem(C.N_CATEGORIES, "n_script", ["no_name"])  # "same" is no longer allowed
    with pytest.raises(ValueError):
        nf.compute(pd.DataFrame([_pair("Jan Novak", "Jan Novak")]), tables)


def test_contract_catches_wrong_row_count(tables):
    feats = nf.compute(pd.DataFrame([_pair("Jan Novak", "Jan Novak")]), tables)
    with pytest.raises(ValueError):
        nf.check_contract(feats, 2)
