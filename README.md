# Sanctions name screening: what safety costs

Group project, Introduction to Data Science.

Banks must check every customer against government sanctions lists. The check
runs on names, and names never match exactly, so the system scores similarity
instead. It is tuned loosely on purpose, because missing a real match is a
legal failure while a false alarm only costs an analyst's time. The result is
that most alerts are the wrong person and a human reads every one.

We are not asking "can we match names". That is close to solved. We are
asking:

> If you must almost never miss a true match, how many false alarms does that
> force on you, and what kinds of names cause them?

The deliverable is a cost curve and an error taxonomy, not an accuracy score.

## Get it running

```bash
git clone <this repo> && cd sanctions-screening
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# download the data (391 MB, not in git)
curl -L -o data/raw/pairs-20251209.json.gz \
  https://data.opensanctions.org/contrib/training/pairs-20251209.json.gz

make data      # ~95s: nested JSON -> flat parquet
make profile   # ~20s: writes reports/01-data-profile.md
make test      # 4 tests
```

If `make data` finishes and `make test` passes, your environment matches
everyone else's.

## Layout

```
src/screening/      importable code. Anything used twice lives here.
  config.py         every path and every scope decision, in one place
  parse.py          JSON Lines -> flat table
  profile.py        generates the data profile report
scripts/            thin runners, numbered in execution order
notebooks/          exploration only. Never the source of a result.
reports/            generated. Never hand-edited.
tests/              pytest
data/               gitignored. Reproducible from the URL above.
docs/               data dictionary and decisions
```

### Why not just a notebook

A notebook is for looking at things. It is a bad place to keep logic, because
cells run out of order, state hides between them, and two people end up with
different numbers from the same file without noticing.

The rule we follow: **if a result goes in the report, the code that produced
it lives in `src/` and runs from a script.** Notebooks explore, scripts
produce. When something in a notebook turns out to matter, move it into
`src/` and re-run the pipeline.

That is also why `reports/01-data-profile.md` says "do not edit by hand" at
the top. The report is an output of the code. If a number looks wrong, you fix
the code and regenerate, so the two can never drift apart.

## Data

OpenSanctions matcher training pairs: human analyst judgements on whether two
entity records are the same real-world entity.

- Source: https://data.opensanctions.org/contrib/training/pairs-20251209.json.gz
- Docs: https://www.opensanctions.org/docs/opensource/pairs/
- Licence: CC BY-NC 4.0. Fine for coursework. Cite OpenSanctions, do not
  republish a derivative dataset.
- Benchmark paper: https://arxiv.org/abs/2603.11051

See `docs/data-dictionary.md` for the fields and `reports/01-data-profile.md`
for what is actually in them.

## What we know so far

From the first profiling run, all reproducible with `make all`:

- 755,540 pairs, 76.9% positive. The pairs already survived a blocking step,
  so the familiar rare-event imbalance is **not** in this file. It is in the
  blocking that happens upstream.
- 442,054 rows are in scope (Person, Company, Organization). The other 313,486
  are Occupancy, Succession, Position and similar, which link a person to a
  job rather than comparing two names.
- Both sides carry a name in 98.8% of in-scope rows. A birth date on both
  sides: 48.9%. An ID number on both sides: only 9.1%.
- 6.2% of pairs are written in two different scripts.
- **3,125 negative pairs have byte-identical names.** Different people,
  same spelling. No string metric can ever separate those, which sets a hard
  ceiling on any name-only approach and is the single best argument for using
  birth date and country as features.

## Pipeline

| Step | Script | Output | Status |
| --- | --- | --- | --- |
| 1. Parse | `01_parse.py` | `data/interim/pairs_flat.parquet` | done |
| 2. Profile | `02_profile.py` | `reports/01-data-profile.md` | done |
| 3. Features | `03_features.py` | `data/processed/features.parquet` | next |
| 4. Model | `04_train.py` | `reports/02-model.md` | |
| 5. Threshold | `05_threshold.py` | `reports/03-cost-curve.md` | |
| 6. Errors | `06_errors.py` | `reports/04-error-taxonomy.md` | |

## Conventions

- Branch per step, pull request into `main`, one teammate reviews.
- Never commit anything from `data/`.
- Decisions that change what a number means go in `docs/decisions.md` with a
  date and a reason.
