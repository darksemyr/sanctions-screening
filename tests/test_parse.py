"""Tests for the parser.

Why test an analysis? Because `flatten_row` decides what every later number
means. If it silently drops a field, no error appears, the charts still
render, and the conclusion is wrong. Run: pytest -q
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from screening.parse import flatten_row

RECORD = {
    "judgement": "positive",
    "left": {
        "id": "x-1", "schema": "Person", "caption": "Jan Novak",
        "datasets": ["eu_fsf"],
        "properties": {"name": ["Jan Novak", "J. Novak"],
                       "birthDate": ["1970-01-01"], "country": ["cz"]},
    },
    "right": {
        "id": "y-2", "schema": "Person", "caption": "Ján Novák",
        "datasets": ["un_sc"],
        "properties": {"name": ["Ján Novák"]},
    },
}


def test_keeps_both_sides():
    row = flatten_row(RECORD, 0)
    assert row["left_id"] == "x-1"
    assert row["right_id"] == "y-2"
    assert row["judgement"] == "positive"


def test_keeps_every_alias():
    row = flatten_row(RECORD, 0)
    assert row["left_name"] == ["Jan Novak", "J. Novak"]


def test_missing_property_is_empty_not_none():
    """A missing field must be [], never None: downstream code calls len()."""
    row = flatten_row(RECORD, 0)
    assert row["right_birthDate"] == []
    assert row["right_country"] == []


def test_handles_entity_with_no_properties():
    row = flatten_row({"judgement": "negative", "left": {}, "right": {}}, 7)
    assert row["pair_id"] == 7
    assert row["left_name"] == []
