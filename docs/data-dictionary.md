# Data dictionary

One row of `data/interim/pairs_flat.parquet` is **one judged pair of entity
records**. Written by `src/screening/parse.py`.

## Pair-level

| column | type | meaning |
| --- | --- | --- |
| `pair_id` | int64 | row number assigned at parse time. Our key, not theirs. |
| `judgement` | string | `positive` = same entity, `negative` = different. This is the label. |

## Per side (`left_` and `right_`)

| column | type | notes |
| --- | --- | --- |
| `*_id` | string | OpenSanctions entity id |
| `*_schema` | string | Person, Company, Organization, Occupancy, ... |
| `*_caption` | string | the display name. One string. |
| `*_datasets` | list | which source lists the record came from |
| `*_name` | list | **all** name spellings, not just the caption |
| `*_alias` | list | further alternative names |
| `*_firstName` / `*_middleName` / `*_lastName` | list | parsed name parts, often absent |
| `*_birthDate` | list | mixed precision: `1969` and `1969-06-16` both occur |
| `*_country` / `*_nationality` | list | two-letter codes |
| `*_idNumber` | list | passport, tax, registration numbers |
| `*_address` | list | free text |
| `*_topics` | list | risk tags, e.g. `sanction`, `role.pep` |

## Traps

Every property is a **list**, including when it holds one value. `len(x) == 0`
means absent. Never assume `x[0]` exists.

Birth dates have mixed precision. `1969` versus `1969-06-16` is a partial
match, not a mismatch, and treating it as a mismatch will quietly cost recall.

The caption is one of the names in `*_name`, but comparing captions only
throws away every other spelling. The median entity has 1 name, the 90th
percentile has 12, and the maximum is 293.

A pair can be `negative` with identical captions. 3,125 in-scope rows are like
this. They are different people who are spelled the same.

Fields are missing at very different rates, so a feature built on `idNumber`
exists for 9.1% of pairs while one built on `name` exists for 98.8%. Decide
how a model handles a missing feature before you build it, not after.
