# Decisions

Anything that changes what a number means goes here, with a date and a reason.

## 2026-10-28 — Scope: three schemas only

Keep `Person`, `Company`, `Organization`. Drop `Occupancy`, `Succession`,
`Position`, `LegalEntity`, `Vessel` and the rest.

Reason: those records link a person to a job or a company to its successor.
They are not two names being compared, which is the problem we are studying.
Keeping them would mean our metrics average over two different tasks.

Effect: 755,540 rows becomes 439,194. Implemented as
`config.IN_SCOPE_SCHEMAS`, applied by `profile.in_scope_mask()`.

Update 2026-11-06: the filter now checks **both** sides. Before, only
`left_schema` was checked, which kept 2,860 mixed pairs such as
Company / LegalEntity (442,054 rows). Both sides must be a name-bearing entity
for the pair to be the comparison we study.

## 2026-10-28 — We do not compete on F1

Published baselines on this exact benchmark: 91.3% F1 rule-based, 99.0%
GPT-4o. We cannot beat those and will not try.

Our contribution is the operating-point analysis: fix recall where a bank must
fix it, then measure and explain the false positives that result.

## 2026-10-28 — Our false-positive counts are not alert rates

The dataset is post-blocking, so 76.9% of pairs are positive. Production
systems see the opposite because blocking discards millions of obvious
non-matches first. The write-up must state this, or every number in it reads
as more alarming than it is.

## 2026-11-13 — Name frequency: source, counting unit, duplicates, bands

Explored in `notebooks/03-rs-name-frequency.ipynb`. Not yet in `src/`.

**Source: `targets.simple`, not `names.txt`.** Both from the OpenSanctions
default collection, downloaded 2026-11-11 (the pairs are from 2025-12-09; older
exports need a data delivery token).
`names.txt` (3,159,271 lines) has every name string exactly once, so a full
name always has frequency 1. It counts spellings, not people: five people
called Thomas Mueller count once, one person with five spellings counts five
times. It also mixes in positions ("mayor" 64,277, "councillor" 32,711) and
securities. `targets.simple` has one row per entity (1,248,755) with name,
aliases and `schema`, so we can count per entity and per type.
Cross-check: the two sources put 82.2% of words in the same frequency band and
99.2% in the same or a neighbouring band (Spearman 0.78 overall, 0.85 for words
with 5+ entities). The remaining differences are positions, securities and
spelling variants in `names.txt`, and aliases shared by many people that
`names.txt` lists once (e.g. "siwy", "gruby").

**Counting unit: once per entity.** For each entity we take name + aliases,
normalise (`unidecode`, lowercase, non-alphanumerics to spaces) and count
- words (one-letter words dropped), as a set per entity;
- full names, with the sorted words as key, so "Thomas Mueller" and
  "MUELLER, Thomas" are one name (one-letter words kept here).

Separate tables for persons (`Person`) and for companies/organisations
(`Company`, `Organization`, `LegalEntity`; LegalEntity is outside the pair
scope but carries company names). Vessels, securities, wallets are not counted.

**Duplicates: same name + same birth date = one person.** Some people appear as
many unmerged entities, e.g. "Morten Austestad" 109 times and "Jorn Steinar
Seljelid" 77 times with the same birth date (Norway State-Owned Enterprises
Leadership); most other cases are one person in two sources (French Mayors and
Wikidata PEPs). Collapsing removes 34,422 person entities (3.3%); only 0.1% of
collapsed groups disagree on country.
Rejected: "same name + same source, no birth date" (would remove another
23,514). It merged distinct people with common names, e.g. 21 x "Muhammad
Aslam" on the Pakistan NACTA list, 18 x "Jose Carlos da Silva" among Brazilian
PEPs, 24 x "First National Bank" in US OCC actions: exactly the names whose
frequency the feature must capture. Entities without a birth date are never
collapsed.

**Bands, with 1 and 2 in one band.** `targets.simple` is the state after
OpenSanctions merged duplicates, i.e. after the decisions that are our labels:
a name shared by a positive pair counts once (one merged entity), by a
negative pair at least twice. Merging 1 and 2 removes most of this leak; a
small part remains at the 2/3 boundary (one other carrier: positive counts 2,
negative 3). Planned check: train with and without the name-frequency features
and compare.

**Both features are kept: rarest shared word and rarest shared full name.**
On uncleaned pairs (persons), the share of negatives rises with the rarest
shared word from 0.9% (1-2 entities) to 46% (11-100) and 80% (>1000), and
with the rarest shared full name from 1.4% (1-2) to 37% (3-5) and 90% (6-20).
Within one word band the full name still separates (word band 11-100: 2.7%
negative if the full name is rare, 98% if 6-20), and vice versa, so each adds
information. To re-run after the cleaning.

## 2026-11-16 — Label leaks and data traps

Checked in `notebooks/06-rs-label-leakage-checks.ipynb` on the uncleaned pairs
(439,194). Each property was tested on its own as a predictor of the label
(AUC; 0.5 = chance). Anything that predicts the label without reading the
names describes the labelling process, not the entities, and must never become
a feature.

| # | Property | Result | Handling |
| --- | --- | --- | --- |
| 1 | same source list on both sides | AUC 0.86; 27.7% vs 95.0% positive | never a feature |
| 2 | both IDs still exist in `targets.simple` | AUC 0.83; 21.1% vs 87.6% positive | never a feature; no lookup of entity data by ID |
| 3 | form of the ID (`NK-`, Wikidata `Q`, source ID) | AUC 0.92; source-id/source-id 0.9% positive | never a feature |
| 4 | self-pairs; duplicate pairs | 38,217 self-pairs, all positive; 37,346 duplicate rows in 8,088 groups, 10 groups with conflicting labels | cleaning |
| 5 | number of aliases | AUC 0.63-0.67 | not a feature |
| 6 | pairs per entity; size of connected component | AUC 0.84 / 0.80 | not a feature |
| 7 | which fields are present on both sides | AUC 0.79 (persons 0.80, companies 0.72) | record features: compare models with and without the "missing" categories |
| 8 | name frequency counted after the merge | rarest shared full name carried by 1 entity: 99.1% positive, 2: 86.1%, not found: 98.8% | frequency bands put 0, 1 and 2 together |

Reason for 2 and 3: when OpenSanctions merges records, the merged entity gets a
new ID and the old IDs disappear, so whether and how an ID still exists tells
whether the pair was merged. IDs are used only to keep entities on one side of
the train/test split.

The project brief quoted 92% accuracy for the rule "same source list =>
different entity"; on the in-scope pairs it is 84.1%.

**Giant connected component.** 252,590 pairs (57.5%) are linked into one
component through heavily sanctioned organisations that appear on many lists
(e.g. Hayat Tahrir al-Sham, IRISL, Uralvagonzavod, Alfa-Bank). Inside it 39%
of pairs are positive, outside 94%. A split by connected component therefore
cannot be balanced (one fold gets 57% of the data). Instead we split by
entity: each entity goes to the test side with probability 0.3, a pair is used
only if both entities are on the same side (62% of pairs kept). Results are
reported separately for the giant component and the rest.

## 2026-11-16 — Name features: the final 15, and why all of them stay

**Decision.** The name side uses 15 features (version 2 of notebook 04:
titles stripped from person names, extended legal-form list). Code:
`src/screening/name_features.py`, run with `make features-name`.

| Feature | What it measures |
| --- | --- |
| n_sim_sort, n_sim_set, n_jaro_winkler | best string similarity over all name/alias combinations |
| n_exact_name | same full name, ignoring word order |
| n_token_coverage, n_phonetic_match | share of words of the shorter name found in the longer one (spelling / Metaphone) |
| n_initials, n_numbers, n_script | initials compatible, numbers in the names, same writing system |
| n_lastname_status, n_firstname_status | comparison of the firstName / lastName fields |
| n_sim_core, n_legal_form | companies: similarity without legal form, legal forms compared |
| n_freq_word_band, n_freq_name_band | how many entities carry the rarest shared word / full name |

**Evidence** (uncleaned pairs, entity split, 5 repetitions, gradient boosting):

- At 99% recall: 2.92% false alarms with all 15, 19.68% with name similarity alone.
- No overfitting: train vs test gap 0.2 percentage points at 99% recall.
- Dropping the three weakest (n_script, n_initials, n_numbers): 3.13% instead
  of 2.92%, worse in 5 of 5 splits. The choice was made on an inner validation
  split of the training data, not on the test set.
- Ridge / Lasso reach only 5.2% / 5.9% at 99% recall: the features work partly
  through interactions, so linear weights are not used to rank them.
- Redundant but kept: n_exact_name, n_token_coverage, n_sim_set,
  n_phonetic_match each add little on their own (the other similarities carry
  the same information). Dropping them as a group is untested.

**Feature contract relaxed.** Numeric name features may be NaN where a value
is undefined (n_sim_core for persons, coverage when a name has only initials).
Gradient boosting libraries handle NaN directly; imputing would invent a
similarity. Categorical columns are never NaN; "no_name" marks pairs where one
side has no name.

**Still to redo on the cleaned data** with LightGBM / XGBoost and a holdout
fixed in advance; all numbers above are from the uncleaned in-scope pairs.

## 2026-11-25 — Record features: the final 4, and how missing fields are handled

**Decision.** The record side uses 4 features. Code:
`src/screening/record_features.py`, run with `make features-record`.
Evidence: notebook 05 (cleaned pairs, Vova's train + val only, 5 splits by
person group, LightGBM).

| Feature | What it measures |
| --- | --- |
| r_dob_status | birth dates compared: exact / partial_match / swapped / day_month_typo / same_year / close / different |
| r_country_status | country and nationality: same / different |
| r_id_status | ID numbers (idNumber field, not the record ID): same / different |
| r_address_sim | best similarity of the addresses, 0..1 |

- **r_dob_status is version 4** (notebook 05, section 5f): version 1 plus
  day/month swapped and a one-digit typo in day or month. False alarms do not
  change (2.73% → 2.71% at 99% recall), but far fewer of these true matches
  are missed (4 instead of 12 over 5 splits). Year "typos" stay "different":
  they are mostly different people (2–4% positive).
- **Dropped:** `r_dob_year_gap` (adds nothing, section 5d) and topics
  (assigned per list = source-list leak).
- **Kept although small:** country (+0.09 pp when left out, without missing
  information; better in 5 of 5 splits).

**Missing fields.** Which fields are present predicts the label on its own
(AUC 0.86): it describes how the lists were built, not the entities. By
default the model cannot see it (`config.R_MISSING_POLICY = "different"`):
"missing" becomes "different", a missing address similarity becomes the
median of the train + val pairs.

| 99% recall, name + record | false alarms |
| --- | --- |
| names only | 3.86% |
| "missing" kept (optimistic) | 2.47% |
| **"different" stand-in (main result)** | **2.71%** |
| random stand-in (pessimistic) | 3.30% |

We report the range: comparing values cuts false alarms by 0.6–1.1 pp; the
rest up to 1.4 pp comes from the missingness pattern. Reweighting to remove
the pattern was considered and left out as too complex for what it adds.

## 2026-11-25 — Evaluation: Vova's test set stays closed, development splits by person group

**Decision.** Vova's split (`pairs_split.parquet`, notebook 02) is fixed:
train / val / test = 60 / 20 / 20 by entity group (groups from positive pairs
only, so a positive pair never crosses splits; negative pairs across splits
are dropped). All development work (feature choice, model comparison, error
analysis: notebooks 05, 07, 09) uses **train + val only**, with 5 repeated
inner splits **by person group** (30% of the groups to validation, salt per
repetition). The test pairs are used once, at the end, with the threshold
chosen on val.

**Why by person group and not by entity ID.** One real entity often has
several IDs (95% of IDs belong to an entity with at least two). Split by ID,
the same person sits on both sides, and the model looks better than it is:
+0.4 pp false alarms at 99% recall once the split respects the groups
(notebook 07, section 10, uncleaned pairs). The group split also drops fewer
pairs, because positive pairs always stay inside one group.

**Supersedes** the entity split of 2026-11-16 above.

## 2026-11-25 — Cleaned data: where the model stands before the final run

Notebook 05, cleaned pairs, train + val, 5 group splits, LightGBM default,
false alarms at 99% / 99.5% recall:

| Features | 99% | 99.5% |
| --- | --- | --- |
| names only (15) | 3.86% | 6.34% |
| names + record (4), missing as "different" | 2.71% | 3.86% |

Reproduced exactly from the feature files written by `make features`
(check run, 2026-11-25). Notebook 07 compares the models on these features;
the final number comes from Vova's test pairs.
