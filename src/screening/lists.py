"""Safe access to the list-valued properties of a flattened pair.

Every property in pairs_flat / pairs_scoped is a list, and it may be empty.
After a round trip through parquet the lists come back as NumPy arrays, where
"if not v" raises "truth value of an array is ambiguous" and v[0] fails on an
empty array. Use values() instead of touching these columns directly.
"""


def values(row, side, props):
    """Return all non-empty values of the given list properties on one side.

    row   : one pair - a dict, a pandas Series, or a row from itertuples()
    side  : "left" or "right"
    props : property names, e.g. ["name", "alias"]

    Returns a plain Python list (possibly empty). Values keep their stored
    order, props in the order given. Duplicates are kept; None and blank
    strings are dropped.
    """
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    if isinstance(props, str):
        props = [props]

    out = []
    for prop in props:
        col = f"{side}_{prop}"
        if hasattr(row, "get"):
            v = row.get(col)
        else:
            v = getattr(row, col, None)
        if v is None or isinstance(v, float):  # missing column or NaN
            continue
        if isinstance(v, str):  # a bare string: treat as one value
            v = [v]
        for x in v:
            if x is None:
                continue
            if isinstance(x, str) and not x.strip():
                continue
            out.append(x)
    return out
