"""Turn the nested JSON Lines file into one flat table, one row per pair.

Why this is a module and not a notebook cell: every later step depends on
this output. If the parsing logic lives in a cell, nobody can re-run it, and
two people end up with slightly different tables without noticing.

Run it with:  python scripts/01_parse.py
"""

from __future__ import annotations

import gzip
import json
from typing import Any, Iterator

import pyarrow as pa
import pyarrow.parquet as pq

from . import config as C

BATCH = 50_000


def _props(entity: dict[str, Any]) -> dict[str, list[str]]:
    """Pull the properties we care about, always as a list of strings."""
    raw = entity.get("properties") or {}
    out = {}
    for key in C.LIST_PROPS:
        value = raw.get(key) or []
        out[key] = [str(v) for v in value]
    return out


def flatten_row(record: dict[str, Any], row_id: int) -> dict[str, Any]:
    """One JSON record becomes one flat dict."""
    left = record.get("left") or {}
    right = record.get("right") or {}
    lp = _props(left)
    rp = _props(right)

    row: dict[str, Any] = {
        "pair_id": row_id,
        "judgement": record.get("judgement"),
        "left_id": left.get("id"),
        "right_id": right.get("id"),
        "left_schema": left.get("schema"),
        "right_schema": right.get("schema"),
        "left_caption": left.get("caption"),
        "right_caption": right.get("caption"),
        "left_datasets": [str(d) for d in (left.get("datasets") or [])],
        "right_datasets": [str(d) for d in (right.get("datasets") or [])],
    }
    for key in C.LIST_PROPS:
        row[f"left_{key}"] = lp[key]
        row[f"right_{key}"] = rp[key]
    return row


def read_records(path) -> Iterator[dict[str, Any]]:
    """Stream the gzipped JSON Lines file. Never load it all into memory."""
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # Count these rather than crashing: a corrupt line in a 755k
                # row file is a data quality fact, not a reason to stop.
                yield {"_unparseable": True}


def _schema() -> pa.Schema:
    fields = [
        ("pair_id", pa.int64()),
        ("judgement", pa.string()),
        ("left_id", pa.string()),
        ("right_id", pa.string()),
        ("left_schema", pa.string()),
        ("right_schema", pa.string()),
        ("left_caption", pa.string()),
        ("right_caption", pa.string()),
        ("left_datasets", pa.list_(pa.string())),
        ("right_datasets", pa.list_(pa.string())),
    ]
    for key in C.LIST_PROPS:
        fields.append((f"left_{key}", pa.list_(pa.string())))
        fields.append((f"right_{key}", pa.list_(pa.string())))
    return pa.schema(fields)


def parse(src=None, dest=None) -> dict[str, int]:
    src = src or C.RAW_PAIRS
    dest = dest or C.PAIRS_FLAT
    dest.parent.mkdir(parents=True, exist_ok=True)

    schema = _schema()
    writer = pq.ParquetWriter(dest, schema, compression="zstd")
    buffer: list[dict[str, Any]] = []
    total = 0
    bad = 0

    def flush() -> None:
        if not buffer:
            return
        columns = {name: [row[name] for row in buffer] for name in schema.names}
        writer.write_table(pa.Table.from_pydict(columns, schema=schema))
        buffer.clear()

    try:
        for record in read_records(src):
            if record.get("_unparseable"):
                bad += 1
                continue
            buffer.append(flatten_row(record, total))
            total += 1
            if len(buffer) >= BATCH:
                flush()
        flush()
    finally:
        writer.close()

    return {"rows_written": total, "unparseable": bad}
