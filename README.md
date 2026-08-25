# Formula 1 Race Winner Prediction

Predicts win probability for every driver in a race, using only information
that would have been available before that race happened. Full experiment
log with every metric, config, and verdict: [REPORT.md](REPORT.md).

## The task, and why "95.7% accurate" was the wrong headline

A prior version of this project reported 95.7% accuracy on a driver-row
binary win target. That number is the majority-class rate: on a ~20-driver
grid, "predict every driver loses" is already right 19/20 times per driver
row, since only one driver per race actually wins. A classifier that always
predicts 0 scores 95.7% on that framing without having learned anything.

The metric that actually answers "can this system pick race winners" is
**Hit@1**: of the drivers in a race, does the model's top-ranked pick match
the actual winner? That's a per-*race* metric (one prediction per race, not
one per driver-row), and it's the one this project reports throughout.

## Data and the point-in-time database design

- **Kaggle** (`rohanrao/formula-1-world-championship-1950-2020`): seasons,
  races, circuits, drivers, constructors, results, qualifying — the bulk of
  history through 2024.
- **FastF1**: 2025+ results/qualifying (not in the Kaggle CSVs), plus
  race-weekend weather and live entry lists for predicting a race that
  hasn't happened yet.

Both land in Postgres (`docker compose up -d`), normalized, every row tagged
with its source and ingestion timestamp.

Every feature is a SQL query in `f1/features/queries.py` that takes a
required `as_of` date and computes trailing/cumulative stats with a window
frame ending at `1 PRECEDING` — the row being scored, and everything after
it, is structurally outside the window, not just conventionally excluded.
`tests/test_leakage.py` proves this for driver form, circuit history, and
weather specifically: a driver's first-ever race/circuit visit must show
NULL trailing stats, and a sentinel planted on a future race's weather must
not move a past `as_of`'s historical average.

## Evaluation protocol: rolling origin, then a locked holdout

A single train/test split gives a Bernoulli-trial-sized sample for Hit@1 —
Entry 4's first 24-race holdout had 95% CIs spanning [0.33, 0.71], too wide
to distinguish any of the systems tested. **Rolling-origin evaluation**
(Entry 9) fixed that: retrain on all seasons ≤ Y, test on season Y+1, for
Y = 2016…2024, pooling all 193 test races for the headline bootstrap CIs.

Iterating on the same pooled result for several stages (features, evaluation
design, model class) risks slowly fitting to it. Stage 4 split the
evaluation in two:

- **Development folds** — rolling-origin test seasons 2017–2023. Every
  decision from Stage 4 onward (which features to keep, model class,
  calibration) was made by looking only at these folds.
- **Locked holdout** — test seasons 2024–2025 (48 races). Evaluated exactly
  once (Entry 27), after every modeling decision was frozen on development
  folds, and never rerun or cherry-picked against.

## Results

**Baselines and models, pooled rolling-origin (193 races, Entries 9–10):**

| System | Hit@1 | 95% CI | Log loss | Spearman |
|---|---|---|---|---|
| pole_sitter | 0.5389 | — | 1.5936 | 0.7539 |
| xgb_v1 (classifier) | 0.5285 | [0.4611, 0.6010] | 0.1122 | 0.6507 |
| xgb_rank_pairwise | 0.5337 | — | 0.1446 | 0.7715 |
| xgb_rank_ndcg | 0.5337 | — | 0.1063 | 0.7678 |
| uniform (Entry 22) | — | — | 0.1986 | — |
| grid_logistic (Entry 22) | — | — | 0.1307 | — |

**Development-fold comparison, all five systems tested before the holdout
was locked (145 races, Entry 25):**

| System | Hit@1 | Hit@3 | MRR | NDCG@5 | Spearman | Log loss |
|---|---|---|---|---|---|---|
| pole_sitter | 0.5241 | 0.8759 | 0.6946 | 0.8560 | 0.7620 | 1.6436 |
| xgb_v1 (classifier) | 0.5379 | 0.8759 | 0.7096 | 0.8580 | 0.6887 | 0.1072 |
| xgb_rank_ndcg | 0.5586 | 0.8897 | 0.7261 | 0.8734 | 0.7833 | 0.1039 |
| conditional_logit | 0.5103 | 0.8966 | 0.7053 | 0.8637 | 0.7592 | 0.1113 |
| plackett_luce | 0.4897 | 0.8828 | 0.6928 | 0.8603 | 0.7939 | 0.1174 |

**Locked holdout, evaluated exactly once (48 races, 2024–2025, Entry 27) —
the model evaluated was `plackett_luce_calibrated_blend`, see "Methodology
notes" below for why:**

| Metric | Value | 95% CI | Δ vs pole-sitter | Verdict |
|---|---|---|---|---|
| hit_at_1 | 0.4792 | [0.3536, 0.6250] | -0.1042 | inconclusive |
| hit_at_3 | 0.9375 | [0.8542, 1.0000] | +0.0625 | inconclusive |
| mrr | 0.6960 | [0.6111, 0.7865] | -0.0460 | inconclusive |
| ndcg_at_5 | 0.8534 | [0.8160, 0.8918] | -0.0204 | inconclusive |
| spearman | 0.7217 | [0.6688, 0.7741] | -0.0080 | inconclusive |
| log_loss | 0.1167 | [0.1056, 0.1277] | -1.3255 | improved |
| brier_score | 0.0349 | [0.0315, 0.0384] | -0.0068 | inconclusive |

## Findings

**Noise floor (Entry 23).** A pure-noise control feature, permuted the same
way as every real feature, produces a log-loss-increase distribution of
mean=-0.00026, std=0.00225, 95% range [-0.00459, +0.00354] over 20 trials.
Re-read against that floor, only 4 of Entry 11's 18 original features
(quali_position, grid_position, constructor_wins_cum, avg_quali_last5) are
distinguishable from noise. Entry 28 adds a caveat to that re-read: 6 of the
other 14 features (circuit_win_rate, hist_wind_speed_avg, hist_track_temp_avg,
circuit_avg_finish, circuit_pass_rate_last5, hist_rain_rate) weren't actually
measured against real data — they were structurally null in the slice Entry
11 used (a dead `circuit_weather_history()` call and a circuit-id
cross-source resolution bug, see below), so "indistinguishable from noise"
should read "never measured" for those six.

**Nested feature selection (Entry 24, retracting Entry 20).** Entry 20's
original Stage 5 feature-selection pass reported hit_at_1=0.5793 after
choosing which of 9 candidate features to keep by looking at their score on
development folds, then reporting that same score on those same folds —
selection bias, not a measurement. Redone with nested selection (choose on
2017–2021, score fresh on held-out 2022–2023): the honest generalization
number is a flat Δhit@1 = +0.0000 [+0.0000, +0.0000]. Entry 28 found that 4
of the 9 candidates in that rerun were never actually given to the model due
to an ablation-script bug (see "Methodology notes"), so the honest reading
of Entry 24 is "5 of 9 candidates were tested and failed to clear the bar; 4
were never tested" rather than "9 of 9 failed." The feature set that
survives either way is the original batch-0 12 features, none of Stage 5's
additions.

**Upset breakdown (Entry 12).** Conditioning Hit@1 on whether the pole
sitter actually won: pole won (n=104) → Hit@1=0.8077 [0.7308, 0.8846]; pole
did not win (n=89) → Hit@1=0.2022 [0.1233, 0.2809]. The model beats a
1-in-19 random-guess floor on upset races (CI lower bound 0.1233 > 0.0526),
but the bulk of its apparent accuracy comes from races the pole sitter wins
outright — which grid position alone already predicts.

**Per-fold instability (Entry 9).** Rolling-origin per-season Hit@1 ranges
from 0.381 (2018, 2019) to 0.864 (2023) — a 48-point swing on comparable
20-24-race single-season samples, all from the same feature set and model
class. That range is evidence a two-season-or-shorter evaluation window on
this dataset is not stable enough to trust a single season's Hit@1 as
representative; it's part of why this project moved to pooled rolling-origin
evaluation (Entry 9) and a locked multi-season holdout (Entry 27) instead of
reporting any single season's number.

## Methodology notes

Three data-pipeline bugs and one measurement-methodology bug were found and
fixed or documented during this project, plus one retraction and one
selection error:

1. **`constructor_form` doubled feature rows** (Entry 7). A 2-car
   constructor's trailing-form query windowed over per-driver result rows
   instead of aggregating to one row per (constructor, race) first,
   doubling every feature row with an unspecified tie-break order between
   teammates. Affected Entries 1–4; fixed before Entry 5.
2. **Driver code is not a unique natural key** (Entry 13). Kaggle's `code`
   column collides across F1 history (Verstappen and Vergne are both
   "VER", plus 6 more collisions), corrupting ~25 rows in 2025's
   FastF1-sourced data before the resolver was rekeyed on full name.
   Development folds (2017–2023, 100% Kaggle-sourced) were confirmed
   unaffected by construction.
3. **Circuit-id cross-source resolution failure** (Entry 28, found while
   auditing Entry 11's exact-zero permutation importances). The FastF1
   ingest's circuit `IdResolver` is keyed on `(name, location)` but called
   with `(Location, Location)`, so it never matches an existing Kaggle
   circuit row and mints a new synthetic circuit_id for every 2025 race —
   confirmed directly: 24/24 2025 races have zero prior rows under their
   assigned circuit_id. Every circuit-keyed feature is therefore 100% null
   for all of 2025, which is exactly the slice Entry 11's permutation
   importance was computed on. A separate dead-code bug compounds this for
   weather features: `circuit_weather_history()` is defined but never
   called from `build_race_features()`, so `hist_track_temp_avg`,
   `hist_wind_speed_avg`, and `hist_rain_rate` are unconditionally null in
   every fold, not just 2025.
4. **Entry 20 retraction** (Entry 24). Stage 5's feature-selection pass
   selected and reported on the same development-fold data — selection
   bias. Retracted and redone with nested selection; nothing survived on
   the honest read.
5. **Entry 27 model-selection error** (Entry 29). The locked holdout was
   run on `plackett_luce_calibrated_blend`, which had the worst dev-fold
   Hit@1 of the five systems Entry 25 compared (0.4897, below both the
   pole-sitter baseline at 0.5241 and the plain classifier at 0.5379),
   while `xgb_rank_ndcg` led on Hit@1, Hit@3, MRR, NDCG@5, and log loss.
   No selection criterion was recorded before the choice was made — it
   should have been fixed, in writing, before Stage 8's calibration step
   touched any model. The locked holdout was **not** rerun on
   `xgb_rank_ndcg` to correct this: doing so would spend a second one-shot
   evaluation on a model chosen with the benefit of hindsight, which
   defeats the purpose of a locked holdout. Entry 27's 0.4792 Hit@1 is the
   honest generalization estimate for the model that was actually
   evaluated — not for the model development-fold evidence says should
   have been chosen.

## Repository layout

- `f1/db/` — SQLAlchemy models, session/engine.
- `f1/ingest/` — one-time loaders (Kaggle CSVs, FastF1 season backfill).
- `f1/features/queries.py` — point-in-time SQL feature queries.
- `f1/features/materialize.py` — builds and stores a feature table per split
  config, so the expensive query set isn't rerun on every experiment.
- `f1/eval/` — metrics (bootstrap CIs via `RaceLevelStats`), walk-forward
  prediction logging, rolling-origin fold runner.
- `f1/models/` — baselines, the XGBoost classifier and rankers, the PyTorch
  conditional logit / Plackett-Luce models, and the final calibrated blend.
- `f1/live/` — live single-race prediction from FastF1
  (`scripts/predict_race.py`), requires real qualifying-derived grid order.
- `scripts/` — one script per experiment/stage/fix; each appends its own
  REPORT.md entry.
- `alembic/` — schema migrations.
- `tests/` — referential integrity, leakage, and metrics-correctness tests.

## Running it

```
docker compose up -d
alembic upgrade head
python -m f1.ingest.kaggle_ingest
python -m f1.ingest.fastf1_results_ingest --year 2025   # seasons Kaggle doesn't have
pytest tests/
```

Then any `scripts/stageN_*.py` / `scripts/fixN_*.py` /
`scripts/precondition_diagnostics.py` to reproduce an experiment — each
appends its own REPORT.md entry. `scripts/predict_race.py --year Y --round R`
predicts a specific race live (requires qualifying to have happened). Do not
run `scripts/final_holdout.py --final` again for this project phase — the
holdout has been spent (see "Evaluation protocol" above).

## Status

This project phase is closed. The locked-holdout result (Entry 27) does not
decisively beat the pole-sitter baseline on Hit@1 (0.4792 vs an implied
~0.5834 for pole-sitter on the same 48 races, CI [0.35, 0.63] on the
difference, inconclusive), and — per the methodology note above — it
evaluated a model that development-fold evidence says was the wrong one to
lock in. The result that held up consistently across every evaluation
surface in this project is calibration: the model's probabilities are
meaningfully better than the pole-sitter's degenerate 1.0/0.0 predictions
(log loss improved, CI excluding zero, everywhere it was measured,
including the locked holdout).

Read plainly: this project does not produce a model that decisively beats
"predict the pole-position starting order" on Hit@1. It produces honestly
better-calibrated probabilities than a naive baseline, three confirmed
data-pipeline bugs (Entries 7, 13, 28), one retracted feature-selection
result (Entry 20/24), and one documented model-selection error (Entry 27/29)
— a full account of where the process went wrong, alongside where it held
up, rather than a single clean accuracy number.
