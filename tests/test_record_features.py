"""Tests for screening.record_features. Run: pytest -q

Small hand-made pairs, no downloaded data needed.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from screening import config as C
from screening import record_features as rf


@pytest.mark.parametrize("a, b, expected", [
    ((1968, 3, 2), (1968, 3, 2), "exact"),
    ((1968,), (1968, 3, 2), "partial_match"),
    ((1968, 3, 2), (1968, 2, 3), "swapped"),
    ((1968, 3, 12), (1968, 3, 21), "day_month_typo"),   # day digits swapped
    ((1968, 3, 2), (1968, 8, 2), "day_month_typo"),     # one digit in the month
    ((1968, 3, 2), (1968, 3, 17), "same_year"),         # two digits differ: no typo
    ((1968, 3, 2), (1968, 11, 20), "same_year"),
    ((1968, 3, 2), (1969, 3, 2), "close"),
    ((1968,), (1986,), "different"),                    # year "typos" stay different
    ((1968, 3, 2), (1975, 3, 2), "different"),
])
def test_dob_relation(a, b, expected):
    assert rf.dob_relation(a, b) == expected


def test_parse_dob():
    assert rf.parse_dob("1968-03-02") == (1968, 3, 2)
    assert rf.parse_dob("1968") == (1968,)
    assert rf.parse_dob("unknown") is None


def _pair(pid="p1", **fields):
    row = {"pair_id": pid}
    for side in ("left", "right"):
        for p in rf.PROPS:
            row[f"{side}_{p}"] = []
    row.update(fields)
    return row


def test_best_relation_over_all_dates():
    r = _pair(left_birthDate=["1970-01-01", "1968-03-02"], right_birthDate=["1968-02-03"])
    assert rf.dob_status(r) == "swapped"
    assert rf.dob_status(_pair(left_birthDate=["1968"])) == "missing"


def test_country_uses_nationality_too():
    assert rf.set_status(_pair(left_country=["ru"], right_nationality=["RU"]), ["country", "nationality"]) == "same"
    assert rf.set_status(_pair(left_country=["ru"], right_country=["ua"]), ["country", "nationality"]) == "different"
    assert rf.set_status(_pair(left_country=["ru"]), ["country", "nationality"]) == "missing"


def test_id_status():
    assert rf.id_status(_pair(left_idNumber=["AB-1234"], right_idNumber=["ab 1234"])) == "same"
    assert rf.id_status(_pair(left_idNumber=["123456789"], right_idNumber=["XX123456789"])) == "same"
    assert rf.id_status(_pair(left_idNumber=["1234"], right_idNumber=["5678"])) == "different"
    assert rf.id_status(_pair(left_idNumber=["12"], right_idNumber=["12"])) == "missing"   # too short


def test_address_sim():
    assert rf.address_sim(_pair(left_address=["Main Street 1, Riga"], right_address=["main street 1 riga"])) == 1.0
    assert np.isnan(rf.address_sim(_pair(left_address=["Main Street 1"])))


def _table():
    return pd.DataFrame([
        _pair("p1", left_birthDate=["1968-03-02"], right_birthDate=["1968-03-02"],
              left_country=["ru"], right_country=["ru"], left_address=["a street 1"], right_address=["a street 1"]),
        _pair("p2", left_birthDate=["1968-03-02"], right_birthDate=["1990"]),
        _pair("p3"),
    ])


def test_compute_keeps_missing():
    f = rf.compute(_table())
    assert list(f.columns) == ["pair_id"] + C.R_FEATURES
    assert list(f["r_dob_status"].astype(str)) == ["exact", "different", "missing"]
    assert f["r_address_sim"].isna().sum() == 2
    rf.check_contract(f, 3, "keep")


def test_policy_different_removes_missing():
    f = rf.apply_missing_policy(rf.compute(_table()), "different", address_fill=0.8)
    assert "missing" not in set(f["r_country_status"].astype(str))
    assert list(f["r_id_status"].astype(str)) == ["different"] * 3
    assert list(f["r_address_sim"]) == [1.0, 0.8, 0.8]
    assert "missing" not in f["r_dob_status"].cat.categories
    rf.check_contract(f, 3, "different")


def test_contract_catches_missing_left():
    with pytest.raises(ValueError):
        rf.check_contract(rf.compute(_table()), 3, "different")
