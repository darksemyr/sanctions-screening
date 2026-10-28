# Decisions

Anything that changes what a number means goes here, with a date and a reason.

## 2026-10-28 — Scope: three schemas only

Keep `Person`, `Company`, `Organization`. Drop `Occupancy`, `Succession`,
`Position`, `LegalEntity`, `Vessel` and the rest.

Reason: those records link a person to a job or a company to its successor.
They are not two names being compared, which is the problem we are studying.
Keeping them would mean our metrics average over two different tasks.

Effect: 755,540 rows becomes 442,054. Implemented as
`config.IN_SCOPE_SCHEMAS`.

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
