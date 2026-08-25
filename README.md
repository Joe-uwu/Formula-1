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

## Five pipeline defects, and the tell that caught each one

None of these crashed. Every one produced output that looked like a normal
result until someone noticed a number that was too clean to be real.

1. **`constructor_form` doubled feature rows** (Entry 7). A 2-car
   constructor's trailing-form query windowed over per-driver result rows
   instead of aggregating to one row per (constructor, race) first,
   giving every constructor two rows per race with an unspecified
   tie-break order between teammates. The tell: a feature-table row count
   exactly double what one row per driver per race implies. Affected
   Entries 1–4; fixed before Entry 5.
2. **Driver code is not a unique natural key** (Entry 13). Kaggle's `code`
   column collides across F1 history — Verstappen and Vergne are both
   "VER", plus six more collisions (ALB, HAR, DOO, MAG, MSC, BIA). Found
   while adding new features and noticing a Red Bull row that didn't fit;
   confirmed ~25 rows in 2025's FastF1-sourced data had silently landed on
   a driver retired since 2014. Fixed by keying the resolver on full name
   instead. Development folds (2017–2023, 100% Kaggle-sourced) were
   confirmed unaffected by construction.
3. **Weather queries defined but never called** (Entry 28). `f1/features/
   queries.py` defines `circuit_weather_history()`, but
   `build_race_features()` in `f1/features/materialize.py` never calls it.
   `hist_track_temp_avg`, `hist_wind_speed_avg`, and `hist_rain_rate` are
   therefore unconditionally null in every row, every fold, always. The
   tell: exact +0.00000 permutation importance with exact 0.00000 std —
   bootstrap resampling cannot produce a zero-width interval on real data.
4. **Circuit-id resolution failure across sources** (Entry 28). The FastF1
   ingest's circuit `IdResolver` is constructed with key `(name, location)`
   but called with `(event['Location'], event['Location'])`, which never
   matches an existing Kaggle circuit row (whose `name` and `location`
   differ) — so every FastF1-sourced (2025) race mints a brand-new
   synthetic circuit_id with zero prior history. Confirmed directly: 24/24
   2025 races have zero prior rows under their assigned circuit_id. Every
   circuit-keyed feature (`circuit_win_rate`, `circuit_avg_finish`,
   `circuit_pass_rate_last5`, `circuit_overtaking_difficulty`) is 100% null
   for all of 2025 as a result — the same "exact zero" tell as #3, on
   columns that carry real signal everywhere else.
5. **Ablation harness ignoring `keep_cols`** (Entry 28). `scripts/
   fix3_nested_reverse_ablation.py`'s `score()` computes `drop_cols`
   against `FEATURE_COLUMNS` and NaNs those out, intending that whatever
   remains reaches the model — but `WinModel.fit`/`predict_race` hardcode
   `df[FEATURE_COLUMNS]` regardless of `keep_cols`. Any candidate not
   already resident in `FEATURE_COLUMNS` was silently never handed to the
   model at all. The tell: a zero-width bootstrap CI on a log-loss delta
   for a feature that, checked independently, has thousands of distinct
   values — "with feature" and "without feature" were bit-identical runs.

The common thread: none of these five produced an error, a crash, or an
obviously wrong number. Each one produced a plausible-looking result that
was wrong specifically because it was *too* regular — an exact zero, a
zero-width confidence interval, a row count that was exactly double what it
should have been. A real measurement on noisy data doesn't come out that
clean. That instinct — treat suspicious tidiness as a bug report, not a good
result — is the most transferable finding in this project, more so than any
single accuracy number below.

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

Iterating on the same 193-race pooled result for several stages (features,
evaluation design, model class) risks slowly fitting to it. Stage 4 split
the evaluation in two:

- **Development folds** — rolling-origin test seasons 2017–2023. Every
  decision from Stage 4 onward (which features to keep, model class,
  calibration) was made by looking only at these folds.
- **Locked holdout** — test seasons 2024–2025 (48 races). Evaluated by
  `scripts/final_holdout.py`, which refuses to run without an explicit
  `--final` flag, exactly once (Entry 27), after every modeling decision
  was frozen on development folds, and never rerun or cherry-picked
  against.

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
the model evaluated was `plackett_luce_calibrated_blend`:**

| Metric | Value | 95% CI | Δ vs pole-sitter | Verdict |
|---|---|---|---|---|
| hit_at_1 | 0.4792 | [0.3536, 0.6250] | -0.1042 | inconclusive |
| hit_at_3 | 0.9375 | [0.8542, 1.0000] | +0.0625 | inconclusive |
| mrr | 0.6960 | [0.6111, 0.7865] | -0.0460 | inconclusive |
| ndcg_at_5 | 0.8534 | [0.8160, 0.8918] | -0.0204 | inconclusive |
| spearman | 0.7217 | [0.6688, 0.7741] | -0.0080 | inconclusive |
| log_loss | 0.1167 | [0.1056, 0.1277] | -1.3255 | improved |
| brier_score | 0.0349 | [0.0315, 0.0384] | -0.0068 | inconclusive |

**Caveat on this number (Entry 27, Entry 28):** `circuit_win_rate` is one of
this model's twelve features and was 100% null for all 24 of the 2025
holdout races, under defect #4 above. The model was trained and evaluated
under the same unpatched pipeline throughout, so 0.4792 is still a valid
one-shot read on the model that was actually run — but it's a read on a
model with a known data gap in one of its twelve features, not a clean
number.

## Findings

**Noise floor.** A pure-noise control feature, permuted the same way as
every real feature, produces a log-loss-increase distribution over 20
independent trials of mean=-0.00026, std=0.00225, 95% range
[-0.00459, +0.00354]. Re-read against that floor, only 4 of Entry 11's 18
original features (quali_position, grid_position, constructor_wins_cum,
avg_quali_last5) are distinguishable from noise. Six more of the fourteen
"within noise floor" verdicts (circuit_win_rate, hist_wind_speed_avg,
hist_track_temp_avg, circuit_avg_finish, circuit_pass_rate_last5,
hist_rain_rate) weren't actually measured against real data — they were
structurally null under defects #3 and #4 above, so "indistinguishable from
noise" should read "never measured" for those six.

**Nested feature selection, corrected.** Stage 5's original feature-selection
pass reported hit_at_1=0.5793 after choosing which of 9 candidate features
to keep by looking at their score on development folds, then reporting that
same score on those same folds — selection bias, not a measurement,
retracted and redone with nested selection (choose on 2017–2021, score
fresh on held-out 2022–2023). Of the 9 candidates, 5 were tested properly
and failed to clear a CI-excludes-zero bar; the other 4 never reached the
model at all, due to defect #5 above, and so were never actually tested.
Either way, the feature set that survives is the original batch-0 12
features — none of Stage 5's 9 additions.

**Upset breakdown.** Conditioning Hit@1 on whether the pole sitter actually
won: pole won (n=104) → Hit@1=0.8077 [0.7308, 0.8846]; pole did not win
(n=89) → Hit@1=0.2022 [0.1233, 0.2809]. Against the 1-in-19 (~0.0526)
random-guess floor for an upset race, the model clears it (CI lower bound
0.1233 > 0.0526) — but the bulk of its apparent accuracy comes from races
the pole sitter wins outright, which grid position alone already predicts.

**Per-fold instability.** Rolling-origin per-season Hit@1 ranges from 0.381
(2018, 2019) to 0.864 (2023) — a 48-point swing across comparable 20–24-race
single-season samples, all from the same feature set and model class. That
range is evidence a two-season-or-shorter evaluation window on this dataset
is not stable enough to trust a single season's Hit@1 as representative; it
is part of why this project moved to pooled rolling-origin evaluation and a
locked multi-season holdout instead of reporting any single season's number.

**The Entry 27 selection error.** The locked holdout evaluated
`plackett_luce_calibrated_blend`, which had the worst dev-fold Hit@1 of the
five systems Entry 25 compared (0.4897, below both the pole-sitter baseline
at 0.5241 and the plain classifier at 0.5379), while `xgb_rank_ndcg` led on
Hit@1, Hit@3, MRR, NDCG@5, and log loss. No selection criterion was recorded
before the choice was made — the holdout evaluated a model that
development-fold evidence argued against. It was not rerun on
`xgb_rank_ndcg` to correct this: doing so would spend the one 2024–2025
evaluation this project gets on a model chosen with the benefit of
hindsight, which destroys the property that makes a locked holdout worth
reporting in the first place. Entry 27's 0.4792 stands as the honest
generalization estimate for the model that was actually evaluated — not for
the model the development-fold evidence says should have been chosen.

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
difference, inconclusive), it evaluated a model that development-fold
evidence argued against (Entry 29), and it carries a known data gap in one
of its twelve features (`circuit_win_rate`, Entry 28). The result that held
up consistently across every evaluation surface in this project is
calibration: the model's probabilities are meaningfully better than the
pole-sitter's degenerate 1.0/0.0 predictions (log loss improved, CI
excluding zero, everywhere it was measured, including the locked holdout).

This project does not produce a model that decisively beats "predict the
pole-position starting order" on Hit@1. It produces honestly
better-calibrated probabilities than a naive baseline, five confirmed
pipeline defects (Entries 7, 13, 28), one retracted feature-selection result
(Entry 20/24), and one documented model-selection error (Entry 27/29) — a
full account of where the process went wrong, alongside where it held up,
rather than a single clean accuracy number.
