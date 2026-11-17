"""Tests for screening.lists.values.

Both feature modules read every list property through values(), so a bug
here would silently change every feature. Run: pytest -q
"""
import sys
from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from screening.lists import values

ROW = {
    "left_name": ["Jan Novak"],
    "left_alias": ["J. Novak", ""],
    "left_birthDate": [],
    "right_name": np.array(["J\u00e1n Nov\u00e1k"], dtype=object),  # as read from parquet
    "right_alias": np.array([], dtype=object),
    "right_birthDate": None,
}


def test_combines_props_in_order():
    assert values(ROW, "left", ["name", "alias"]) == ["Jan Novak", "J. Novak"]


def test_drops_blank_strings():
    assert "" not in values(ROW, "left", ["alias"])


def test_empty_list_gives_empty_list():
    assert values(ROW, "left", ["birthDate"]) == []


def test_none_and_missing_column_give_empty_list():
    assert values(ROW, "right", ["birthDate"]) == []
    assert values(ROW, "right", ["country"]) == []


def test_numpy_arrays_become_plain_list():
    out = values(ROW, "right", ["name", "alias"])
    assert out == ["J\u00e1n Nov\u00e1k"]
    assert isinstance(out, list)
    assert not values(ROW, "right", ["alias"])  # safe to use in "if not ..."


def test_works_on_pandas_series_and_itertuples():
    df = pd.DataFrame([ROW])
    assert values(df.iloc[0], "left", ["name"]) == ["Jan Novak"]
    Row = namedtuple("Row", ROW.keys())
    assert values(Row(**ROW), "left", ["name"]) == ["Jan Novak"]


def test_single_prop_as_string():
    assert values(ROW, "left", "name") == ["Jan Novak"]


def test_rejects_unknown_side():
    with pytest.raises(ValueError):
        values(ROW, "middle", ["name"])
