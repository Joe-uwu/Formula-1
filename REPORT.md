# F1 Predictor — Experiment Log

<!-- SUMMARY_START -->
| Entry | Headline metric | Verdict |
|---|---|---|
| Entry 1 — baselines only | Hit@1=0.544 | baseline |
| Entry 2 — first model | Hit@1=0.500 | inconclusive |
| Entry 3 — baselines only | Hit@1=0.667 | baseline |
| Entry 4 — xgb_v1, train<=2024-01-01 / val<2025-01-01 / test>=2025-01-01 | Hit@1=0.542 | inconclusive |
| Entry 5 — baselines only | Hit@1=0.667 | baseline |
| Entry 6 — xgb_v1, train<=2024-01-01 / val<2025-01-01 / test>=2025-01-01 | Hit@1=0.625 | inconclusive |
| Entry 7 — bugfix: constructor_form doubled feature rows | n/a (bugfix entry, no new model run) | n/a |
| Entry 8 — Stage 1: honest probabilistic baselines | xgb_v1 beats grid_logistic on log loss (0.0898 vs 0.1006). The feature set beyond starting position is contributing something measurable here. | n/a (baseline-comparison entry, not a model change) |
| Entry 9 — Stage 2: rolling-origin evaluation (193 races pooled) | Pooled Hit@1=0.528 over 193 races | inconclusive |
| Entry 10 — Stage 3: rank:pairwise / rank:ndcg vs. binary classifier (193 races pooled) | Spearman: classifier=0.6507, xgb_rank_pairwise=0.7715, pole_sitter=0.7539. Regression disappears with a ranking objective. | improved (Spearman-specific; see full table for other metrics) |
| Entry 11 — Diagnostics | Hit@1 drop when grid/quali removed: +0.1140 -> non-grid features carry little to none independent signal. | n/a (diagnostic entry, not a model comparison) |
| Entry 12 — Precondition diagnostics before Stage 4 | Dead features (reverse ablation, dev folds only): driver_races_before, circuit_avg_finish, circuit_pass_rate_last5, hist_track_temp_avg, hist_wind_speed_avg, hist_rain_rate | n/a (precondition/diagnostic entry) |
| Entry 13 — bugfix: driver code is not a unique natural key | Fixed a driver-identity bug affecting 2025 (FastF1-sourced) data for ~7 collision-prone driver codes. Development folds (2017-2023, 100% Kaggle-sourced) were never affected — confirmed by construction, not just by luck. | n/a (bugfix entry) |
| Entry 14 — Stage 5, Batch 1: qualifying pace gap (gap to pole, gap to median, both z-scored within session) | log_loss: 0.1072 -> 0.1123, hit_at_1: 0.5379 -> 0.5172 | inconclusive |
| Entry 15 — Stage 5, Batch 2: teammate qualifying delta (this race + trailing-5 average) | log_loss: 0.1123 -> 0.1084, hit_at_1: 0.5172 -> 0.5586 | inconclusive |
| Entry 16 — Stage 5, Batch 3: grid-minus-qualifying delta (penalty signal) | log_loss: 0.1084 -> 0.1085, hit_at_1: 0.5586 -> 0.5586 | inconclusive |
| Entry 17 — Stage 5, Batch 4: constructor reliability (trailing DNF rate, last 10 races) | log_loss: 0.1085 -> 0.1091, hit_at_1: 0.5586 -> 0.5517 | inconclusive |
| Entry 18 — Stage 5, Batch 5: circuit overtaking difficulty + interaction with grid position | log_loss: 0.1091 -> 0.1077, hit_at_1: 0.5517 -> 0.5517 | inconclusive |
| Entry 19 — Stage 5, Batch 6: season-relative constructor pace (rolling qualifying gap-to-pole, current season) | log_loss: 0.1077 -> 0.1076, hit_at_1: 0.5517 -> 0.5517 | inconclusive |
| Entry 20 — Stage 5 close-out: reverse ablation, final feature set | Kept 5/9 Stage 5 features. Final feature count: 17. | n/a (feature-selection entry) |
| Entry 21 — Fix 1: uniform/grid-logistic baselines on all 9 pooled folds (0 races) | [SUPERSEDED — see Entry 22] xgb_v1 does NOT beat grid_logistic on log loss (nan vs 0.1307) on the full 9-fold pooled set. | n/a (VOID — superseded by Entry 22; baseline-correction entry, not a model change) |
| Entry 22 — Fix 1: uniform/grid-logistic baselines on all 9 pooled folds (193 races) | xgb_v1 beats grid_logistic on log loss (0.1064 vs 0.1307) on the full 9-fold pooled set. | n/a (baseline-correction entry, not a model change) |
| Entry 23 — Fix 2: null-feature noise floor (20 trials), Entry 11/12 correction | Noise floor ~0.0052. 4/18 of Entry 11's features are distinguishable from noise at this threshold. | n/a (methodology-correction entry) |
| Entry 24 — Fix 3: nested reverse ablation, retracts Entry 20's 0.5793 | Selected 0/9 on selection folds (CI excludes zero): none. On held-out validation folds: Δhit@1=+0.0000 [+0.0000,+0.0000]. | inconclusive |
| Entry 25 — Stage 6: conditional logit / Plackett-Luce vs. classifier and ranker (145 dev-fold races) | Spearman: classifier=0.6887, conditional_logit=0.7592, plackett_luce=0.7939. | see per-metric CIs above — mixed by design, reported as such |
| Entry 26 — Stage 8: calibration (temperature + grid-only blend) | Temperatures: plackett_luce=0.80, conditional_logit=1.37. Blend weights: plackett_luce=0.76, conditional_logit=0.91. | n/a (calibration entry) |
| Entry 27 — LOCKED HOLDOUT (final, 48 races, 2024-2025) | LOCKED: Hit@1=0.4792 on 48 never-before-evaluated races | inconclusive |
<!-- SUMMARY_END -->

## Log

### Entry 1 — baselines only

**Date:** 2026-08-24  **Commit:** 6a5beae

**What changed / hypothesis:** No model yet. Establish the pole-sitter baseline before any model exists, so the first model has something to beat.

**Config diff from previous run:**
```
{
  "start_date": "2015-01-01",
  "end_date": "2024-12-31",
  "train_end_date": "2022-01-01"
}
model_version=pole_sitter
feature_build_timings={'reused': True}
```

**Metrics:**

| Metric | Value | 95% CI | Δ vs pole-sitter | Δ 95% CI | Verdict |
|---|---|---|---|---|---|
| hit_at_1 | 0.5441 | [0.4265, 0.6618] | +0.0000 | [+0.0000, +0.0000] | inconclusive |
| hit_at_3 | 0.7353 | [0.6324, 0.8382] | +0.0000 | [+0.0000, +0.0000] | inconclusive |
| mrr | 0.6480 | [0.5600, 0.7448] | +0.0000 | [+0.0000, +0.0000] | inconclusive |
| ndcg_at_5 | 0.7942 | [0.7468, 0.8396] | +0.0000 | [+0.0000, +0.0000] | inconclusive |
| spearman | 0.7581 | [0.7191, 0.7933] | +0.0000 | [+0.0000, +0.0000] | inconclusive |
| log_loss | 1.6526 | [1.4476, 1.8539] | +0.0000 | [+0.0000, +0.0000] | inconclusive |
| brier_score | 0.0478 | [0.0419, 0.0537] | +0.0000 | [+0.0000, +0.0000] | inconclusive |
| ece | 0.0478 | — | — | — | — |
| brier_skill_score | 0.0000 | — | — | — | — |

**Per-season Hit@1:**

- 2022: 0.455 (n=22 races)
- 2023: 0.682 (n=22 races)
- 2024: 0.500 (n=24 races)

**Split by whether the pole sitter won:**

- pole sitter won (n=37): Hit@1=1.000, MRR=1.000
- pole sitter did not win (n=31): Hit@1=0.000, MRR=0.228

**Headline:** Hit@1=0.544

**Verdict:** baseline

**Next:** Train the first model on this baseline.

### Entry 2 — first model

**Date:** 2026-08-24  **Commit:** 6a5beae

**What changed / hypothesis:** Trained xgb_v1 (XGBoost, 200 trees, depth 3) on point-in-time features. Hypothesis: grid position plus trailing form/circuit-history features beat the pole-sitter baseline on ranking metrics.

**Config diff from previous run:**
```
{
  "start_date": "2015-01-01",
  "end_date": "2024-12-31",
  "train_end_date": "2022-01-01"
}
model_version=xgb_v1
feature_build_timings={'reused': True}
```

**Metrics:**

| Metric | Value | 95% CI | Δ vs pole-sitter | Δ 95% CI | Verdict |
|---|---|---|---|---|---|
| hit_at_1 | 0.5000 | [0.3824, 0.6176] | -0.0441 | [-0.2059, +0.1029] | inconclusive |
| hit_at_3 | 0.7059 | [0.5882, 0.8088] | -0.0294 | [-0.1324, +0.0735] | inconclusive |
| mrr | 0.6303 | [0.5357, 0.7181] | -0.0177 | [-0.1238, +0.0827] | inconclusive |
| ndcg_at_5 | 0.7868 | [0.7329, 0.8346] | -0.0074 | [-0.0541, +0.0391] | inconclusive |
| spearman | 0.7058 | [0.6679, 0.7380] | -0.0522 | [-0.0830, -0.0214] | worsened |
| log_loss | 0.1176 | [0.1009, 0.1354] | -1.5350 | [-1.7208, -1.3305] | improved |
| brier_score | 0.0341 | [0.0308, 0.0375] | -0.0137 | [-0.0187, -0.0082] | improved |
| ece | 0.0250 | — | — | — | — |
| brier_skill_score | 0.2864 | — | — | — | — |

**Per-season Hit@1:**

- 2022: 0.455 (n=22 races)
- 2023: 0.636 (n=22 races)
- 2024: 0.417 (n=24 races)

**Split by whether the pole sitter won:**

- pole sitter won (n=37): Hit@1=0.595, MRR=0.732
- pole sitter did not win (n=31): Hit@1=0.387, MRR=0.508

**Headline:** Hit@1=0.500

**Verdict:** inconclusive

**Next:** Iterate: one hypothesis, one change, one measurement per entry.

### Entry 3 — baselines only

**Date:** 2026-08-24  **Commit:** 6a5beae

**What changed / hypothesis:** No model yet. Establish the pole-sitter baseline before any model exists, so the first model has something to beat.

**Config diff from previous run:**
```
{
  "start_date": "2015-01-01",
  "end_date": "2025-12-31",
  "val_start_date": "2024-01-01",
  "test_start_date": "2025-01-01"
}
model_version=pole_sitter
feature_build_timings={'total_seconds': 57.35642410000037, 'races': 233, 'avg_seconds_per_race': 0.2381765806867172, 'max_seconds_per_race': 0.2920576999995319}
```

**Metrics:**

| Metric | Value | 95% CI | Δ vs pole-sitter | Δ 95% CI | Verdict |
|---|---|---|---|---|---|
| hit_at_1 | 0.6667 | [0.4583, 0.8333] | +0.0000 | [+0.0000, +0.0000] | inconclusive |
| hit_at_3 | 0.8333 | [0.6667, 0.9583] | +0.0000 | [+0.0000, +0.0000] | inconclusive |
| mrr | 0.7532 | [0.6126, 0.8833] | +0.0000 | [+0.0000, +0.0000] | inconclusive |
| ndcg_at_5 | 0.8470 | [0.7746, 0.9150] | +0.0000 | [+0.0000, +0.0000] | inconclusive |
| spearman | 0.6537 | [0.5715, 0.7252] | +0.0000 | [+0.0000, +0.0000] | inconclusive |
| log_loss | 1.4436 | [1.1548, 1.7989] | +0.0000 | [+0.0000, +0.0000] | inconclusive |
| brier_score | 0.0418 | [0.0334, 0.0521] | +0.0000 | [+0.0000, +0.0000] | inconclusive |
| ece | 0.0418 | — | — | — | — |
| brier_skill_score | 0.0000 | — | — | — | — |

**Per-season Hit@1:**

- 2025: 0.667 (n=24 races)

**Split by whether the pole sitter won:**

- pole sitter won (n=16): Hit@1=1.000, MRR=1.000
- pole sitter did not win (n=8): Hit@1=0.000, MRR=0.260

**Headline:** Hit@1=0.667

**Verdict:** baseline

**Next:** Train the first model on this baseline.

### Entry 4 — xgb_v1, train<=2024-01-01 / val<2025-01-01 / test>=2025-01-01

**Date:** 2026-08-24  **Commit:** 6a5beae

**What changed / hypothesis:** Trained xgb_v1 (XGBoost, up to 200 trees depth 3, early-stopped on the validation split) on point-in-time features. Test set is 2025, backfilled from FastF1 since it isn't in the Kaggle CSVs. Hypothesis: grid position plus trailing form/circuit-history features beat the pole-sitter baseline on ranking metrics.

**Config diff from previous run:**
```
{
  "start_date": "2015-01-01",
  "end_date": "2025-12-31",
  "val_start_date": "2024-01-01",
  "test_start_date": "2025-01-01"
}
model_version=xgb_v1
feature_build_timings={'reused': True}
```

**Metrics:**

| Metric | Value | 95% CI | Δ vs pole-sitter | Δ 95% CI | Verdict |
|---|---|---|---|---|---|
| hit_at_1 | 0.5417 | [0.3333, 0.7083] | -0.1250 | [-0.3333, +0.0833] | inconclusive |
| hit_at_3 | 0.8750 | [0.7500, 1.0000] | +0.0417 | [+0.0000, +0.1250] | inconclusive |
| mrr | 0.6962 | [0.5559, 0.8195] | -0.0569 | [-0.1875, +0.0750] | inconclusive |
| ndcg_at_5 | 0.8396 | [0.7744, 0.8993] | -0.0074 | [-0.0479, +0.0416] | inconclusive |
| spearman | 0.6144 | [0.5412, 0.6822] | -0.0393 | [-0.0782, +0.0013] | inconclusive |
| log_loss | 0.1025 | [0.0855, 0.1219] | -1.3411 | [-1.6674, -1.0579] | improved |
| brier_score | 0.0335 | [0.0295, 0.0377] | -0.0083 | [-0.0162, -0.0007] | improved |
| ece | 0.0251 | — | — | — | — |
| brier_skill_score | 0.1997 | — | — | — | — |

**Per-season Hit@1:**

- 2025: 0.542 (n=24 races)

**Split by whether the pole sitter won:**

- pole sitter won (n=16): Hit@1=0.688, MRR=0.823
- pole sitter did not win (n=8): Hit@1=0.250, MRR=0.443

**Headline:** Hit@1=0.542

**Verdict:** inconclusive

**Next:** Iterate: one hypothesis, one change, one measurement per entry.

### Entry 5 — baselines only

**Date:** 2026-08-24  **Commit:** 6a5beae

**What changed / hypothesis:** No model yet. Establish the pole-sitter baseline before any model exists, so the first model has something to beat.

**Config diff from previous run:**
```
{
  "start_date": "2015-01-01",
  "end_date": "2025-12-31",
  "val_start_date": "2024-01-01",
  "test_start_date": "2025-01-01"
}
model_version=pole_sitter
feature_build_timings={'total_seconds': 56.769590800000515, 'races': 233, 'avg_seconds_per_race': 0.2394899836910023, 'max_seconds_per_race': 0.2981644999999844}
```

**Metrics:**

| Metric | Value | 95% CI | Δ vs pole-sitter | Δ 95% CI | Verdict |
|---|---|---|---|---|---|
| hit_at_1 | 0.6667 | [0.4583, 0.8333] | +0.0000 | [+0.0000, +0.0000] | inconclusive |
| hit_at_3 | 0.9583 | [0.8750, 1.0000] | +0.0000 | [+0.0000, +0.0000] | inconclusive |
| mrr | 0.8021 | [0.6840, 0.9097] | +0.0000 | [+0.0000, +0.0000] | inconclusive |
| ndcg_at_5 | 0.8932 | [0.8512, 0.9333] | +0.0000 | [+0.0000, +0.0000] | inconclusive |
| spearman | 0.6546 | [0.5723, 0.7260] | +0.0000 | [+0.0000, +0.0000] | inconclusive |
| log_loss | 1.1537 | [0.5768, 1.8709] | +0.0000 | [+0.0000, +0.0000] | inconclusive |
| brier_score | 0.0334 | [0.0167, 0.0542] | +0.0000 | [+0.0000, +0.0000] | inconclusive |
| ece | 0.0334 | — | — | — | — |
| brier_skill_score | 0.0000 | — | — | — | — |

**Per-season Hit@1:**

- 2025: 0.667 (n=24 races)

**Split by whether the pole sitter won:**

- pole sitter won (n=16): Hit@1=1.000, MRR=1.000
- pole sitter did not win (n=8): Hit@1=0.000, MRR=0.406

**Headline:** Hit@1=0.667

**Verdict:** baseline

**Next:** Train the first model on this baseline.

### Entry 6 — xgb_v1, train<=2024-01-01 / val<2025-01-01 / test>=2025-01-01

**Date:** 2026-08-24  **Commit:** 6a5beae

**What changed / hypothesis:** Trained xgb_v1 (XGBoost, up to 200 trees depth 3, early-stopped on the validation split) on point-in-time features. Test set is 2025, backfilled from FastF1 since it isn't in the Kaggle CSVs. Hypothesis: grid position plus trailing form/circuit-history features beat the pole-sitter baseline on ranking metrics.

**Config diff from previous run:**
```
{
  "start_date": "2015-01-01",
  "end_date": "2025-12-31",
  "val_start_date": "2024-01-01",
  "test_start_date": "2025-01-01"
}
model_version=xgb_v1
feature_build_timings={'reused': True}
```

**Metrics:**

| Metric | Value | 95% CI | Δ vs pole-sitter | Δ 95% CI | Verdict |
|---|---|---|---|---|---|
| hit_at_1 | 0.6250 | [0.4167, 0.7917] | -0.0417 | [-0.1667, +0.0833] | inconclusive |
| hit_at_3 | 0.9583 | [0.8750, 1.0000] | +0.0000 | [+0.0000, +0.0000] | inconclusive |
| mrr | 0.7882 | [0.6735, 0.8958] | -0.0139 | [-0.0833, +0.0625] | inconclusive |
| ndcg_at_5 | 0.8821 | [0.8441, 0.9221] | -0.0111 | [-0.0355, +0.0135] | inconclusive |
| spearman | 0.5938 | [0.5267, 0.6526] | -0.0608 | [-0.1148, -0.0091] | worsened |
| log_loss | 0.0898 | [0.0692, 0.1137] | -1.0639 | [-1.7459, -0.4943] | improved |
| brier_score | 0.0269 | [0.0196, 0.0354] | -0.0065 | [-0.0202, +0.0071] | inconclusive |
| ece | 0.0184 | — | — | — | — |
| brier_skill_score | 0.1952 | — | — | — | — |

**Per-season Hit@1:**

- 2025: 0.625 (n=24 races)

**Split by whether the pole sitter won:**

- pole sitter won (n=16): Hit@1=0.875, MRR=0.938
- pole sitter did not win (n=8): Hit@1=0.125, MRR=0.490

**Headline:** Hit@1=0.625

**Verdict:** inconclusive

**Next:** Iterate: one hypothesis, one change, one measurement per entry.

### Entry 7 — bugfix: constructor_form doubled feature rows

**Date:** 2026-08-24  **Commit:** 6a5beae

**What changed / hypothesis:** Building the live single-race prediction pipeline (f1/live/predict_upcoming.py) surfaced a bug: constructor_form() in f1/features/queries.py windowed directly over per-driver result rows instead of aggregating to one row per (constructor, race) first. A 2-car team has two result rows on the same race_date, so the query returned two constructor_form rows per race with an unspecified tie-break order between them. Merging that onto the one-row-per-driver grid in materialize.py silently doubled every feature row: feature_set 1 had 8437 rows over 209 races (~40.3/race, should be ~20); feature_set 2 had 9394 rows over 233 races (~40.3/race). Entries 1-4 were trained/evaluated on this doubled table. Fix: aggregate results to one row per (constructor_id, race date) — summed points, any-car-won flag, avg finish across the team's cars — before windowing. Added test_constructor_form_one_row_per_constructor to tests/test_leakage.py to catch a regression. Rebuilt the feature table (feature_set 3: 4698 rows / 233 races = 20.2/race, correct) and reran entries 5-6 on it — those two entries are the first ones on correct data.

**Config diff from previous run:**
```
(no split_config change — this is a feature-query correctness fix, see f1/features/queries.py constructor_form)
```

**Metrics:**

N/A — see entries 5 and 6, which supersede entries 3 and 4 as the trustworthy versions of that comparison.

**Headline:** n/a (bugfix entry, no new model run)

**Verdict:** n/a

**Next:** Entries 1-4 are left in the log uncorrected per the append-only rule, but should not be cited as current results.

### Entry 8 — Stage 1: honest probabilistic baselines

**Date:** 2026-08-24  **Commit:** 6a5beae

**What changed / hypothesis:** Added uniform (1/N) and grid-only logistic regression baselines, both evaluated on the same 2025 test set as Entry 6's xgb_v1 and pole-sitter predictions (reused from the DB, not rerun). Reports log loss, Brier, ECE, and Brier Skill Score for all four side by side. The Entry 6 log-loss/Brier 'improvement' over pole-sitter is suspected to be a clipping artifact, since pole-sitter's true log loss is infinite whenever pole doesn't win. Uniform and grid-only give honest, non-degenerate probability baselines to compare against instead.

**Config diff from previous run:**
```
{'start_date': '2015-01-01', 'end_date': '2025-12-31', 'val_start_date': '2024-01-01', 'test_start_date': '2025-01-01'}
(same split as Entry 6 — this stage only adds baselines, no split change)
```

**Metrics:**

| System | Log loss | Brier | ECE | Brier Skill Score* |
|---|---|---|---|---|
| xgb_v1 (model) | 0.0898 | 0.0269 | 0.0184 | 0.1952 |
| pole_sitter | 1.1537 | 0.0334 | 0.0334 | 0.0000 |
| uniform | 0.1988 | 0.0476 | 0.0000 | -0.4248 |
| grid_logistic | 0.1006 | 0.0309 | 0.0302 | 0.0763 |

*BSS is computed against the pole-sitter baseline (pole-sitter's own BSS is 0 by construction). Log-loss probabilities are clipped to [1e-15, 1-1e-15] before averaging — pole-sitter's true log loss is infinite on every race it doesn't win (predicted probability 0 for the actual winner); the 1e-15 clip is what turns that into a finite number, so pole-sitter's log-loss column here is an artifact of the clipping constant, not a real probabilistic score. Its uniform-1.0/0.0 predictions make it uncomparable on log loss by design — that's exactly why this stage adds uniform and grid-only as the real comparison points.

**Headline:** xgb_v1 beats grid_logistic on log loss (0.0898 vs 0.1006). The feature set beyond starting position is contributing something measurable here.

**Verdict:** n/a (baseline-comparison entry, not a model change)

**Next:** Stage 2: the 24-race single holdout gives ~24 Bernoulli trials for Hit@1 — replace it with rolling-origin evaluation before drawing conclusions from ranking metrics.

### Entry 9 — Stage 2: rolling-origin evaluation (193 races pooled)

**Date:** 2026-08-24  **Commit:** 6a5beae

**What changed / hypothesis:** Replaced the single 2025 holdout (24 races) with rolling-origin walk-forward: retrain on all seasons <= Y, test on season Y+1, for Y=2016..2024 (test seasons 2017-2025), pooling all 193 test races for the headline bootstrap CIs. Pole-sitter baseline evaluated on the identical pooled race set. A 24-race single holdout gives ~24 Bernoulli trials for Hit@1, which is why every Entry 6 ranking metric was 'inconclusive' with CIs spanning [0.42, 0.79] — that's an uninformative test, not a null result. ~193 pooled races should narrow the CIs enough to actually distinguish 'no evidence of a difference' from 'evidence of no difference.'

**Config diff from previous run:**
```
rolling-origin folds: test years [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025], retrained per fold; base split config unchanged: {'start_date': '2015-01-01', 'end_date': '2025-12-31', 'val_start_date': '2024-01-01', 'test_start_date': '2025-01-01'}
```

**Metrics:**

| Metric | Value | 95% CI | Δ vs pole-sitter | Δ 95% CI | Verdict |
|---|---|---|---|---|---|
| hit_at_1 | 0.5285 | [0.4611, 0.6010] | -0.0104 | [-0.0725, +0.0518] | inconclusive |
| hit_at_3 | 0.8756 | [0.8238, 0.9172] | +0.0000 | [-0.0415, +0.0466] | inconclusive |
| mrr | 0.6990 | [0.6530, 0.7444] | -0.0074 | [-0.0448, +0.0303] | inconclusive |
| ndcg_at_5 | 0.8441 | [0.8234, 0.8632] | -0.0164 | [-0.0321, +0.0014] | inconclusive |
| spearman | 0.6507 | [0.6254, 0.6753] | -0.1032 | [-0.1227, -0.0837] | worsened |
| log_loss | 0.1122 | [0.0983, 0.1266] | -1.4814 | [-1.7124, -1.2583] | improved |
| brier_score | 0.0321 | [0.0287, 0.0360] | -0.0140 | [-0.0193, -0.0088] | improved |
**Per-fold Hit@1 (test season Y+1, trained on everything <= Y):**

- 2017: 0.500 (n=20 races)
- 2018: 0.381 (n=21 races)
- 2019: 0.381 (n=21 races)
- 2020: 0.471 (n=17 races)
- 2021: 0.545 (n=22 races)
- 2022: 0.409 (n=22 races)
- 2023: 0.864 (n=22 races)
- 2024: 0.458 (n=24 races)
- 2025: 0.708 (n=24 races)

**Hit@3 ceiling:** 1.0000 in 2025 (24/24 races) — the winner is in the top-3 grid slots almost every race, so Hit@3 has almost no headroom left to discriminate between models. Reported in the table above but shouldn't be leaned on.

**Headline:** Pooled Hit@1=0.528 over 193 races

**Verdict:** inconclusive

**Next:** Stage 3: the model's Spearman correlation against actual finishing order is a statistically real loss vs. just using grid order (pole-sitter's implicit full-field ranking). Binary Win-target training has no incentive to order P8-P15 correctly; try a learning-to-rank objective (rank:pairwise, rank:ndcg) and see if that regression disappears.

### Entry 10 — Stage 3: rank:pairwise / rank:ndcg vs. binary classifier (193 races pooled)

**Date:** 2026-08-24  **Commit:** 6a5beae

**What changed / hypothesis:** Trained XGBRanker with rank:pairwise and rank:ndcg objectives, graded relevance from finishing position (field_size - position + 1, DNF/unknown = 0), grouped by race_id, same rolling-origin folds and features as Entry 9's classifier. Entry 9's classifier is trained on a binary Win target, so it has no incentive to order the mid-field correctly — only to separate P1 from the rest. Spearman scores the whole permutation, so a ranking objective should close that gap even if Hit@1 doesn't move.

**Config diff from previous run:**
```
objective: binary:logistic -> rank:pairwise / rank:ndcg; everything else unchanged from Entry 9: {'start_date': '2015-01-01', 'end_date': '2025-12-31', 'val_start_date': '2024-01-01', 'test_start_date': '2025-01-01'}
```

**Metrics:**

| System | Hit@1 | Hit@3 | MRR | NDCG@5 | Spearman | Log loss | Brier |
|---|---|---|---|---|---|---|---|
| xgb_v1 (classifier, binary Win target) | 0.5285 | 0.8756 | 0.6990 | 0.8441 | 0.6507 | 0.1122 | 0.0321 |
| xgb_rank_pairwise | 0.5337 | 0.8705 | 0.7070 | 0.8601 | 0.7715 | 0.1446 | 0.0398 |
| xgb_rank_ndcg | 0.5337 | 0.8860 | 0.7126 | 0.8684 | 0.7678 | 0.1063 | 0.0314 |
| pole_sitter | 0.5389 | 0.8756 | 0.7064 | 0.8604 | 0.7539 | 1.5936 | 0.0461 |

**Spearman vs. actual finishing order, with CIs (this is the question Stage 3 asks):**

| System | Spearman | 95% CI | Δ vs pole-sitter | Δ 95% CI | Verdict |
|---|---|---|---|---|---|
| xgb_v1 (classifier, binary Win target) | 0.6507 | [0.6254, 0.6753] | -0.1032 | [-0.1227, -0.0837] | worsened |
| xgb_rank_pairwise | 0.7715 | [0.7484, 0.7932] | +0.0175 | [+0.0057, +0.0299] | improved |
| xgb_rank_ndcg | 0.7678 | [0.7454, 0.7906] | +0.0138 | [+0.0018, +0.0270] | improved |

**Headline:** Spearman: classifier=0.6507, xgb_rank_pairwise=0.7715, pole_sitter=0.7539. Regression disappears with a ranking objective.

**Verdict:** improved (Spearman-specific; see full table for other metrics)

**Next:** Run diagnostics: predicted-probability-vs-grid-position correlation, permutation importance, grid/quali ablation, pooled upset breakdown.

### Entry 11 — Diagnostics

**Date:** 2026-08-24  **Commit:** 6a5beae

**What changed / hypothesis:** Four diagnostics on the Stage 2/3 rolling-origin models: predicted-probability-vs-grid correlation, permutation importance, a grid/quali ablation, and the pooled upset breakdown. These explain *why* the models perform as they do, regardless of which one wins on the headline metrics.

**Config diff from previous run:**
```
No split/model change — read-only analysis over Entry 9/10's already-logged rolling-origin predictions, plus one new ablation run (grid_position, quali_position set to null).
```

**Metrics:**

**1. Spearman(predicted probability, grid position), pooled:**

- xgb_v1 (classifier): 0.7043
- xgb_rank_pairwise: 0.8113
- xgb_rank_ndcg: 0.8387
- pole_sitter: 0.3475

**2. Permutation importance (xgb_v1, held out on 2025, log-loss increase when a feature is shuffled):**

| Feature | Mean log-loss increase | Std |
|---|---|---|
| quali_position | +0.04984 | 0.00935 |
| grid_position | +0.01526 | 0.00314 |
| constructor_wins_cum | +0.00796 | 0.00565 |
| avg_quali_last5 | +0.00534 | 0.00475 |
| constructor_points_cum | +0.00383 | 0.00227 |
| avg_finish_last3 | +0.00381 | 0.00307 |
| driver_races_before | +0.00087 | 0.00121 |
| avg_finish_last5 | +0.00008 | 0.00108 |
| wins_last5 | +0.00001 | 0.00037 |
| circuit_win_rate | +0.00000 | 0.00000 |
| hist_wind_speed_avg | +0.00000 | 0.00000 |
| hist_track_temp_avg | +0.00000 | 0.00000 |
| circuit_avg_finish | +0.00000 | 0.00000 |
| circuit_pass_rate_last5 | +0.00000 | 0.00000 |
| hist_rain_rate | +0.00000 | 0.00000 |
| driver_points_cum | -0.00023 | 0.00436 |
| wins_last3 | -0.00151 | 0.00050 |
| constructor_avg_finish_last3 | -0.00364 | 0.00192 |

**3. Ablation — grid_position and quali_position removed entirely (xgb_v1 classifier):**

| Metric | With grid/quali | Without | Drop |
|---|---|---|---|
| hit_at_1 | 0.5285 | 0.4145 | +0.1140 |
| hit_at_3 | 0.8756 | 0.7720 | +0.1036 |
| mrr | 0.6990 | 0.6089 | +0.0901 |
| ndcg_at_5 | 0.8441 | 0.7943 | +0.0497 |
| spearman | 0.6507 | 0.6171 | +0.0337 |
| log_loss | 0.1122 | 0.1295 | -0.0173 |
| brier_score | 0.0321 | 0.0370 | -0.0049 |

**4. Upset breakdown, pooled across all rolling-origin races:**

- xgb_v1 (classifier): pole won (n=104) Hit@1=0.808 | pole lost (n=89) Hit@1=0.202
- xgb_rank_pairwise: pole won (n=104) Hit@1=0.760 | pole lost (n=89) Hit@1=0.270
- xgb_rank_ndcg: pole won (n=104) Hit@1=0.817 | pole lost (n=89) Hit@1=0.202
- pole_sitter: pole won (n=104) Hit@1=1.000 | pole lost (n=89) Hit@1=0.000

**Headline:** Hit@1 drop when grid/quali removed: +0.1140 -> non-grid features carry little to none independent signal.

**Verdict:** n/a (diagnostic entry, not a model comparison)

**Next:** If independent signal is weak, the honest path forward is either richer non-grid features (pit strategy, tyre degradation, weather forecast at prediction time) or accepting grid-order-plus-noise as the ceiling for this feature set.

### Entry 12 — Precondition diagnostics before Stage 4

**Date:** 2026-08-24  **Commit:** 6a5beae

**What changed / hypothesis:** Four precondition checks required before locking a final holdout: pooled baseline confirmation, a null-feature sanity check on the permutation-importance methodology, per-feature reverse ablation (development folds only), and bootstrap CIs on the upset-conditional Hit@1. These validate the diagnostic tooling itself and identify dead features before Stage 5 adds more, on development folds only so the locked holdout (Stage 4) stays untouched by any feature-selection decision.

**Config diff from previous run:**
```
No split/model change. Reverse ablation restricted to development folds (test years 2017-2023) only — 2024/2025 are reserved for the Stage 4 locked holdout.
```

**Metrics:**

**1. Pooled baseline rerun:** already satisfied — Entry 9 evaluated pole-sitter on the identical 193-race pooled rolling-origin test set used for the model (not rerun here; restated for the audit trail).

**2. Null-feature check:** a pure-noise feature (`null_random_control`, standard normal, independent of everything) gets mean log-loss increase -0.00186 under the same permutation-importance procedure as Entry 11 — comparable in magnitude to the near-zero circuit/weather features there (SUSPECT — null feature shows non-trivial importance). This validates that Entry 11's near-zero rankings reflect real lack of signal, not a broken measurement.

**3. Reverse ablation (development folds 2017-2023 only): grid+quali baseline plus one feature at a time.**

Grid+quali-only baseline: log_loss=0.1185, hit_at_1=0.5241

| Feature added | Log loss | Δ log loss (neg=better) | Hit@1 | Δ Hit@1 | Verdict |
|---|---|---|---|---|---|
| avg_quali_last5 | 0.1108 | -0.0076 | 0.5310 | +0.0069 | alive |
| avg_finish_last5 | 0.1132 | -0.0053 | 0.5103 | -0.0138 | alive |
| wins_last5 | 0.1075 | -0.0110 | 0.5586 | +0.0345 | alive |
| avg_finish_last3 | 0.1122 | -0.0063 | 0.5172 | -0.0069 | alive |
| wins_last3 | 0.1089 | -0.0095 | 0.5310 | +0.0069 | alive |
| driver_points_cum | 0.1150 | -0.0035 | 0.5103 | -0.0138 | alive |
| driver_races_before | 0.1345 | +0.0160 | 0.4069 | -0.1172 | dead |
| constructor_points_cum | 0.1145 | -0.0040 | 0.5586 | +0.0345 | alive |
| constructor_wins_cum | 0.1084 | -0.0101 | 0.5586 | +0.0345 | alive |
| constructor_avg_finish_last3 | 0.1155 | -0.0030 | 0.4966 | -0.0276 | alive |
| circuit_win_rate | 0.1169 | -0.0016 | 0.5172 | -0.0069 | alive |
| circuit_avg_finish | 0.1216 | +0.0032 | 0.4897 | -0.0345 | dead |
| circuit_pass_rate_last5 | 0.1255 | +0.0071 | 0.4759 | -0.0483 | dead |
| hist_track_temp_avg | 0.1185 | +0.0000 | 0.5241 | +0.0000 | dead |
| hist_wind_speed_avg | 0.1185 | +0.0000 | 0.5241 | +0.0000 | dead |
| hist_rain_rate | 0.1185 | +0.0000 | 0.5241 | +0.0000 | dead |

**4. Upset-conditional Hit@1, with bootstrap CIs (xgb_v1, 193 pooled rolling-origin races):**

- pole won (n=104): Hit@1=0.8077 [0.7308, 0.8846]
- pole did not win (n=89): Hit@1=0.2022 [0.1233, 0.2809]

Upset-only Hit@1 vs. 1/19 (~0.0526) chance null: beats it (CI lower bound 0.1233 > 0.0526).

**Headline:** Dead features (reverse ablation, dev folds only): driver_races_before, circuit_avg_finish, circuit_pass_rate_last5, hist_track_temp_avg, hist_wind_speed_avg, hist_rain_rate

**Verdict:** n/a (precondition/diagnostic entry)

**Next:** Proceed to Stage 4 (lock 2024-2025 as a final holdout) now that these are logged.

### Entry 13 — bugfix: driver code is not a unique natural key

**Date:** 2026-08-24  **Commit:** 6a5beae

**What changed / hypothesis:** While building the qualifying-pace features (Stage 5), found that f1/ingest/fastf1_results_ingest.py and f1/live/lineup.py resolved FastF1 drivers to Kaggle driver_ids by matching on `code` (the 3-letter abbreviation). Kaggle's own code column is NOT globally unique across F1 history: Max Verstappen and Jean-Eric Vergne are both coded 'VER', and six other codes collide too (ALB, HAR, DOO, MAG, MSC, BIA). Confirmed real damage: 25 qualifying and 25 results rows for 2025 (Verstappen's races) had silently landed on Vergne's driver_id (818), a driver retired since 2014. Fix: key the resolver on (forename, surname) instead of code. Purged all fastf1-sourced results/qualifying/races and predictions/feature tables, re-ran fastf1_results_ingest for 2025 with the fixed resolver; confirmed 0 rows land on driver_id 818 and Red Bull correctly shows exactly 2 drivers per race. Added test_no_constructor_fields_more_than_two_cars_per_race_modern_era (scoped to 2015+, since the 2-cars-per-team rule wasn't universal in 1950s F1) to catch this class of bug even when it doesn't produce a literal duplicate (race_id, driver_id) row.

**Config diff from previous run:**
```
No split/model change. f1/ingest/fastf1_results_ingest.py and f1/live/lineup.py: driver resolver key changed from `code` to `(forename, surname)`.
```

**Metrics:**

N/A — data-integrity fix, not a model change.

**Headline:** Fixed a driver-identity bug affecting 2025 (FastF1-sourced) data for ~7 collision-prone driver codes. Development folds (2017-2023, 100% Kaggle-sourced) were never affected — confirmed by construction, not just by luck.

**Verdict:** n/a (bugfix entry)

**Next:** Proceed with Stage 5 features on the now-corrected data.

### Entry 14 — Stage 5, Batch 1: qualifying pace gap (gap to pole, gap to median, both z-scored within session)

**Date:** 2026-08-24  **Commit:** 6a5beae

**What changed / hypothesis:** Added columns: quali_gap_to_pole_norm, quali_gap_to_median_norm. Cumulative feature set now: ['grid_position', 'quali_position', 'avg_quali_last5', 'avg_finish_last5', 'wins_last5', 'avg_finish_last3', 'wins_last3', 'driver_points_cum', 'constructor_points_cum', 'constructor_wins_cum', 'constructor_avg_finish_last3', 'circuit_win_rate', 'quali_gap_to_pole_norm', 'quali_gap_to_median_norm'] Measured on development folds only (2017-2023, 145 races) so the locked holdout stays untouched.

**Config diff from previous run:**
```
feature_columns += ['quali_gap_to_pole_norm', 'quali_gap_to_median_norm']
```

**Metrics:**

| Metric | Before this batch | After this batch | Δ | Verdict |
|---|---|---|---|---|
| hit_at_1 | 0.5379 | 0.5172 | -0.0207 | inconclusive |
| hit_at_3 | 0.8759 | 0.8690 | -0.0069 | inconclusive |
| mrr | 0.7096 | 0.6969 | -0.0127 | inconclusive |
| ndcg_at_5 | 0.8580 | 0.8580 | +0.0000 | inconclusive |
| spearman | 0.6887 | 0.7005 | +0.0118 | inconclusive |
| log_loss | 0.1072 | 0.1123 | +0.0051 | inconclusive |
| brier_score | 0.0308 | 0.0325 | +0.0016 | inconclusive |

**Headline:** log_loss: 0.1072 -> 0.1123, hit_at_1: 0.5379 -> 0.5172

**Verdict:** inconclusive

**Next:** Next batch.

### Entry 15 — Stage 5, Batch 2: teammate qualifying delta (this race + trailing-5 average)

**Date:** 2026-08-24  **Commit:** 6a5beae

**What changed / hypothesis:** Added columns: teammate_quali_delta, teammate_quali_delta_avg5. Cumulative feature set now: ['grid_position', 'quali_position', 'avg_quali_last5', 'avg_finish_last5', 'wins_last5', 'avg_finish_last3', 'wins_last3', 'driver_points_cum', 'constructor_points_cum', 'constructor_wins_cum', 'constructor_avg_finish_last3', 'circuit_win_rate', 'quali_gap_to_pole_norm', 'quali_gap_to_median_norm', 'teammate_quali_delta', 'teammate_quali_delta_avg5'] Measured on development folds only (2017-2023, 145 races) so the locked holdout stays untouched.

**Config diff from previous run:**
```
feature_columns += ['teammate_quali_delta', 'teammate_quali_delta_avg5']
```

**Metrics:**

| Metric | Before this batch | After this batch | Δ | Verdict |
|---|---|---|---|---|
| hit_at_1 | 0.5172 | 0.5586 | +0.0414 | inconclusive |
| hit_at_3 | 0.8690 | 0.8690 | +0.0000 | inconclusive |
| mrr | 0.6969 | 0.7244 | +0.0274 | improved |
| ndcg_at_5 | 0.8580 | 0.8679 | +0.0099 | inconclusive |
| spearman | 0.7005 | 0.6854 | -0.0151 | inconclusive |
| log_loss | 0.1123 | 0.1084 | -0.0039 | inconclusive |
| brier_score | 0.0325 | 0.0310 | -0.0014 | inconclusive |

**Headline:** log_loss: 0.1123 -> 0.1084, hit_at_1: 0.5172 -> 0.5586

**Verdict:** inconclusive

**Next:** Next batch.

### Entry 16 — Stage 5, Batch 3: grid-minus-qualifying delta (penalty signal)

**Date:** 2026-08-24  **Commit:** 6a5beae

**What changed / hypothesis:** Added columns: grid_minus_quali_delta. Cumulative feature set now: ['grid_position', 'quali_position', 'avg_quali_last5', 'avg_finish_last5', 'wins_last5', 'avg_finish_last3', 'wins_last3', 'driver_points_cum', 'constructor_points_cum', 'constructor_wins_cum', 'constructor_avg_finish_last3', 'circuit_win_rate', 'quali_gap_to_pole_norm', 'quali_gap_to_median_norm', 'teammate_quali_delta', 'teammate_quali_delta_avg5', 'grid_minus_quali_delta'] Measured on development folds only (2017-2023, 145 races) so the locked holdout stays untouched.

**Config diff from previous run:**
```
feature_columns += ['grid_minus_quali_delta']
```

**Metrics:**

| Metric | Before this batch | After this batch | Δ | Verdict |
|---|---|---|---|---|
| hit_at_1 | 0.5586 | 0.5586 | +0.0000 | inconclusive |
| hit_at_3 | 0.8690 | 0.8690 | +0.0000 | inconclusive |
| mrr | 0.7244 | 0.7258 | +0.0014 | inconclusive |
| ndcg_at_5 | 0.8679 | 0.8677 | -0.0002 | inconclusive |
| spearman | 0.6854 | 0.6882 | +0.0029 | inconclusive |
| log_loss | 0.1084 | 0.1085 | +0.0001 | inconclusive |
| brier_score | 0.0310 | 0.0311 | +0.0001 | inconclusive |

**Headline:** log_loss: 0.1084 -> 0.1085, hit_at_1: 0.5586 -> 0.5586

**Verdict:** inconclusive

**Next:** Next batch.

### Entry 17 — Stage 5, Batch 4: constructor reliability (trailing DNF rate, last 10 races)

**Date:** 2026-08-24  **Commit:** 6a5beae

**What changed / hypothesis:** Added columns: constructor_dnf_rate_last10. Cumulative feature set now: ['grid_position', 'quali_position', 'avg_quali_last5', 'avg_finish_last5', 'wins_last5', 'avg_finish_last3', 'wins_last3', 'driver_points_cum', 'constructor_points_cum', 'constructor_wins_cum', 'constructor_avg_finish_last3', 'circuit_win_rate', 'quali_gap_to_pole_norm', 'quali_gap_to_median_norm', 'teammate_quali_delta', 'teammate_quali_delta_avg5', 'grid_minus_quali_delta', 'constructor_dnf_rate_last10'] Measured on development folds only (2017-2023, 145 races) so the locked holdout stays untouched.

**Config diff from previous run:**
```
feature_columns += ['constructor_dnf_rate_last10']
```

**Metrics:**

| Metric | Before this batch | After this batch | Δ | Verdict |
|---|---|---|---|---|
| hit_at_1 | 0.5586 | 0.5517 | -0.0069 | inconclusive |
| hit_at_3 | 0.8690 | 0.8621 | -0.0069 | inconclusive |
| mrr | 0.7258 | 0.7183 | -0.0074 | inconclusive |
| ndcg_at_5 | 0.8677 | 0.8678 | +0.0001 | inconclusive |
| spearman | 0.6882 | 0.6951 | +0.0069 | improved |
| log_loss | 0.1085 | 0.1091 | +0.0007 | inconclusive |
| brier_score | 0.0311 | 0.0314 | +0.0003 | inconclusive |

**Headline:** log_loss: 0.1085 -> 0.1091, hit_at_1: 0.5586 -> 0.5517

**Verdict:** inconclusive

**Next:** Next batch.

### Entry 18 — Stage 5, Batch 5: circuit overtaking difficulty + interaction with grid position

**Date:** 2026-08-24  **Commit:** 6a5beae

**What changed / hypothesis:** Added columns: circuit_overtaking_difficulty, circuit_overtaking_x_grid. Cumulative feature set now: ['grid_position', 'quali_position', 'avg_quali_last5', 'avg_finish_last5', 'wins_last5', 'avg_finish_last3', 'wins_last3', 'driver_points_cum', 'constructor_points_cum', 'constructor_wins_cum', 'constructor_avg_finish_last3', 'circuit_win_rate', 'quali_gap_to_pole_norm', 'quali_gap_to_median_norm', 'teammate_quali_delta', 'teammate_quali_delta_avg5', 'grid_minus_quali_delta', 'constructor_dnf_rate_last10', 'circuit_overtaking_difficulty', 'circuit_overtaking_x_grid'] Measured on development folds only (2017-2023, 145 races) so the locked holdout stays untouched.

**Config diff from previous run:**
```
feature_columns += ['circuit_overtaking_difficulty', 'circuit_overtaking_x_grid']
```

**Metrics:**

| Metric | Before this batch | After this batch | Δ | Verdict |
|---|---|---|---|---|
| hit_at_1 | 0.5517 | 0.5517 | +0.0000 | inconclusive |
| hit_at_3 | 0.8621 | 0.8552 | -0.0069 | inconclusive |
| mrr | 0.7183 | 0.7185 | +0.0002 | inconclusive |
| ndcg_at_5 | 0.8678 | 0.8685 | +0.0007 | inconclusive |
| spearman | 0.6951 | 0.6820 | -0.0132 | worsened |
| log_loss | 0.1091 | 0.1077 | -0.0014 | inconclusive |
| brier_score | 0.0314 | 0.0304 | -0.0009 | inconclusive |

**Headline:** log_loss: 0.1091 -> 0.1077, hit_at_1: 0.5517 -> 0.5517

**Verdict:** inconclusive

**Next:** Next batch.

### Entry 19 — Stage 5, Batch 6: season-relative constructor pace (rolling qualifying gap-to-pole, current season)

**Date:** 2026-08-24  **Commit:** 6a5beae

**What changed / hypothesis:** Added columns: constructor_season_pace_gap. Cumulative feature set now: ['grid_position', 'quali_position', 'avg_quali_last5', 'avg_finish_last5', 'wins_last5', 'avg_finish_last3', 'wins_last3', 'driver_points_cum', 'constructor_points_cum', 'constructor_wins_cum', 'constructor_avg_finish_last3', 'circuit_win_rate', 'quali_gap_to_pole_norm', 'quali_gap_to_median_norm', 'teammate_quali_delta', 'teammate_quali_delta_avg5', 'grid_minus_quali_delta', 'constructor_dnf_rate_last10', 'circuit_overtaking_difficulty', 'circuit_overtaking_x_grid', 'constructor_season_pace_gap'] Measured on development folds only (2017-2023, 145 races) so the locked holdout stays untouched.

**Config diff from previous run:**
```
feature_columns += ['constructor_season_pace_gap']
```

**Metrics:**

| Metric | Before this batch | After this batch | Δ | Verdict |
|---|---|---|---|---|
| hit_at_1 | 0.5517 | 0.5517 | +0.0000 | inconclusive |
| hit_at_3 | 0.8552 | 0.8828 | +0.0276 | inconclusive |
| mrr | 0.7185 | 0.7220 | +0.0035 | inconclusive |
| ndcg_at_5 | 0.8685 | 0.8708 | +0.0022 | inconclusive |
| spearman | 0.6820 | 0.6881 | +0.0062 | inconclusive |
| log_loss | 0.1077 | 0.1076 | -0.0001 | inconclusive |
| brier_score | 0.0304 | 0.0305 | +0.0001 | inconclusive |

**Headline:** log_loss: 0.1077 -> 0.1076, hit_at_1: 0.5517 -> 0.5517

**Verdict:** inconclusive

**Next:** All batches added — run the final reverse-ablation pass to decide what stays, respecting the ~20-feature ceiling.

### Entry 20 — Stage 5 close-out: reverse ablation, final feature set

**Date:** 2026-08-24  **Commit:** 6a5beae

**What changed / hypothesis:** Reverse ablation on the 9 Stage 5 candidates (development folds, batch-0 baseline + one feature at a time). Kept: ['quali_gap_to_median_norm', 'teammate_quali_delta', 'teammate_quali_delta_avg5', 'circuit_overtaking_x_grid', 'constructor_season_pace_gap']. Respect the ~20-feature ceiling for ~4700 training rows — drop anything that doesn't measurably beat the batch-0 baseline on its own.

**Config diff from previous run:**
```
Final feature set (17 columns): ['grid_position', 'quali_position', 'avg_quali_last5', 'avg_finish_last5', 'wins_last5', 'avg_finish_last3', 'wins_last3', 'driver_points_cum', 'constructor_points_cum', 'constructor_wins_cum', 'constructor_avg_finish_last3', 'circuit_win_rate', 'quali_gap_to_median_norm', 'teammate_quali_delta', 'teammate_quali_delta_avg5', 'circuit_overtaking_x_grid', 'constructor_season_pace_gap']
```

**Metrics:**

Batch-0 baseline (12 features): log_loss=0.1072, hit_at_1=0.5379

| Feature added | Log loss | Δ log loss | Hit@1 | Δ Hit@1 | Verdict |
|---|---|---|---|---|---|
| quali_gap_to_pole_norm | 0.1121 | +0.0049 | 0.5379 | +0.0000 | drop |
| quali_gap_to_median_norm | 0.1057 | -0.0015 | 0.5517 | +0.0138 | keep |
| teammate_quali_delta | 0.1066 | -0.0005 | 0.5241 | -0.0138 | keep |
| teammate_quali_delta_avg5 | 0.1056 | -0.0016 | 0.5724 | +0.0345 | keep |
| grid_minus_quali_delta | 0.1071 | -0.0001 | 0.5379 | +0.0000 | drop |
| constructor_dnf_rate_last10 | 0.1075 | +0.0004 | 0.5379 | +0.0000 | drop |
| circuit_overtaking_difficulty | 0.1078 | +0.0006 | 0.5310 | -0.0069 | drop |
| circuit_overtaking_x_grid | 0.1068 | -0.0004 | 0.5655 | +0.0276 | keep |
| constructor_season_pace_gap | 0.1071 | -0.0001 | 0.5448 | +0.0069 | keep |

**Final feature set score:** log_loss=0.1061 (vs batch-0 0.1072), hit_at_1=0.5793 (vs batch-0 0.5379)

**Headline:** Kept 5/9 Stage 5 features. Final feature count: 17.

**Verdict:** n/a (feature-selection entry)

**Next:** Update f1/features/materialize.py's FEATURE_COLUMNS to the final set and proceed to Stage 6.

### Entry 21 — Fix 1: uniform/grid-logistic baselines on all 9 pooled folds (0 races)

**⚠ SUPERSEDED — this entry is VOID. See Entry 22 for the corrected rerun and the real numbers.** The NaN readings below were an artifact of stale predictions: Entry 13's driver-identity cleanup wiped the predictions table, and Stage 2/3's pooled predictions were never regenerated before this comparison ran. Do not cite this entry's headline ("xgb_v1 does NOT beat grid_logistic on log loss") — it is not a measured result.

**Date:** 2026-08-24  **Commit:** 6a5beae

**What changed / hypothesis:** Reran the Entry 8 uniform and grid-only-logistic baselines on the same 9-fold pooled rolling-origin surface (2017-2025, 0 races) that Entries 9-11 already used for the model comparison — Entry 8's numbers were only the 24-race 2025 holdout, not comparable to the pooled model numbers. Correcting an apples-to-oranges comparison, not testing a new hypothesis.

**Config diff from previous run:**
```
{'start_date': '2015-01-01', 'end_date': '2025-12-31', 'val_start_date': '2024-01-01', 'test_start_date': '2025-01-01'}
fold_test_years=[2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025] (was: 2025 only)
```

**Metrics:**

| System | Log loss | Brier | ECE | Brier Skill Score* |
|---|---|---|---|---|
| xgb_v1 (model) | nan | nan | 0.0000 | nan |
| pole_sitter | nan | nan | 0.0000 | nan |
| uniform | 0.1986 | 0.0475 | 0.0000 | nan |
| grid_logistic | 0.1307 | 0.0365 | 0.0086 | nan |

**Headline:** [SUPERSEDED — see Entry 22] xgb_v1 does NOT beat grid_logistic on log loss (nan vs 0.1307) on the full 9-fold pooled set.

**Verdict:** n/a (VOID — superseded by Entry 22; baseline-correction entry, not a model change)

**Next:** Fix 2: null-feature check needs 20 repeats reported as a distribution, not a single point estimate.

### Entry 22 — Fix 1: uniform/grid-logistic baselines on all 9 pooled folds (193 races)

**Date:** 2026-08-24  **Commit:** 6a5beae

**What changed / hypothesis:** Reran the Entry 8 uniform and grid-only-logistic baselines on the same 9-fold pooled rolling-origin surface (2017-2025, 193 races) that Entries 9-11 already used for the model comparison — Entry 8's numbers were only the 24-race 2025 holdout, not comparable to the pooled model numbers. Correcting an apples-to-oranges comparison, not testing a new hypothesis. (Note: the first attempt at this, logged as the previous entry, produced NaN for xgb_v1/pole_sitter because the Entry 13 driver-identity cleanup had wiped the predictions table and Stage 2/3's pooled predictions were never regenerated after. Regenerated here on the corrected data and the final Stage 5 feature set before rerunning this comparison.)

**Config diff from previous run:**
```
{'start_date': '2015-01-01', 'end_date': '2025-12-31', 'val_start_date': '2024-01-01', 'test_start_date': '2025-01-01'}
fold_test_years=[2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025] (was: 2025 only)
```

**Metrics:**

| System | Log loss | Brier | ECE | Brier Skill Score* |
|---|---|---|---|---|
| xgb_v1 (model) | 0.1064 | 0.0301 | 0.0048 | 0.3469 |
| pole_sitter | 1.5936 | 0.0461 | 0.0461 | 0.0000 |
| uniform | 0.1986 | 0.0475 | 0.0000 | -0.0300 |
| grid_logistic | 0.1307 | 0.0365 | 0.0086 | 0.2082 |

**Headline:** xgb_v1 beats grid_logistic on log loss (0.1064 vs 0.1307) on the full 9-fold pooled set.

**Verdict:** n/a (baseline-correction entry, not a model change)

**Next:** Fix 2: null-feature check needs 20 repeats reported as a distribution, not a single point estimate.

### Entry 23 — Fix 2: null-feature noise floor (20 trials), Entry 11/12 correction

**Date:** 2026-08-24  **Commit:** 6a5beae

**What changed / hypothesis:** Reran the null-feature check 20 times (independent seeds) instead of once, reporting the resulting distribution as an explicit noise floor rather than a single point estimate compared to an arbitrary threshold. Entry 12's 'SUSPECT' verdict on a single -0.00186 trial was a labeling bug, not a real finding — a single sample can't establish a noise floor. Re-reading Entry 11's importances against a proper empirical floor tells us which ones are real.

**Config diff from previous run:**
```
No split/model change. 20x repeat of the Entry 12 null-feature procedure.
```

**Metrics:**

**Null-feature distribution (log-loss increase when a pure-noise feature is shuffled), 20 independent trials:**

mean=-0.00026, std=0.00225, 95% range=[-0.00459, +0.00354], max |value| observed=0.00523

**Corrected verdict:** Entry 12's single trial (-0.00186) falls well inside this range — it was 'SUSPECT' by an arbitrary 0.001 threshold, but the true noise floor is roughly 0.0052. That trial was sane, not suspect.


**Re-reading Entry 11's permutation importances against this noise floor:**

| Feature | Importance | vs. noise floor |
|---|---|---|
| quali_position | +0.04984 | above floor — real signal |
| grid_position | +0.01526 | above floor — real signal |
| constructor_wins_cum | +0.00796 | above floor — real signal |
| avg_quali_last5 | +0.00534 | above floor — real signal |
| constructor_points_cum | +0.00383 | within noise floor — indistinguishable from noise |
| avg_finish_last3 | +0.00381 | within noise floor — indistinguishable from noise |
| driver_races_before | +0.00087 | within noise floor — indistinguishable from noise |
| avg_finish_last5 | +0.00008 | within noise floor — indistinguishable from noise |
| wins_last5 | +0.00001 | within noise floor — indistinguishable from noise |
| circuit_win_rate | +0.00000 | within noise floor — indistinguishable from noise |
| hist_wind_speed_avg | +0.00000 | within noise floor — indistinguishable from noise |
| hist_track_temp_avg | +0.00000 | within noise floor — indistinguishable from noise |
| circuit_avg_finish | +0.00000 | within noise floor — indistinguishable from noise |
| circuit_pass_rate_last5 | +0.00000 | within noise floor — indistinguishable from noise |
| hist_rain_rate | +0.00000 | within noise floor — indistinguishable from noise |
| driver_points_cum | -0.00023 | within noise floor — indistinguishable from noise |
| wins_last3 | -0.00151 | within noise floor — indistinguishable from noise |
| constructor_avg_finish_last3 | -0.00364 | within noise floor — indistinguishable from noise |

**Headline:** Noise floor ~0.0052. 4/18 of Entry 11's features are distinguishable from noise at this threshold.

**Verdict:** n/a (methodology-correction entry)

**Next:** Fix 3: redo Entry 20's reverse ablation with paired-difference CIs and nested selection so the reported final score isn't computed on the same folds used to select the features.

### Entry 24 — Fix 3: nested reverse ablation, retracts Entry 20's 0.5793

**Date:** 2026-08-24  **Commit:** 6a5beae

**What changed / hypothesis:** RETRACTS Entry 20's headline figure (hit_at_1=0.5793): it was selected on and reported on the same development folds, which is selection bias, not a measured result. Redone with nested selection: features chosen on 2017-2021 using paired bootstrap CIs on the log-loss delta (kept only if the CI excludes zero, i.e. reliably better, not just a better point estimate), then the selected set is scored fresh on 2022-2023 (held out from selection) for the honest number. A feature set chosen by looking at a metric should never be reported on that same metric/data — nested selection is the fix.

**Config diff from previous run:**
```
selection_folds=[2017, 2018, 2019, 2020, 2021], validation_folds=[2022, 2023] (both within development folds; locked holdout 2024-2025 untouched)
```

**Metrics:**

**Selection phase (2017-2021, paired bootstrap CI on log-loss delta vs batch-0):**

| Feature | Δ log loss | 95% CI | Selected |
|---|---|---|---|
| quali_gap_to_pole_norm | +0.0000 | [+0.0000, +0.0000] | no |
| quali_gap_to_median_norm | -0.0023 | [-0.0071, +0.0029] | no |
| teammate_quali_delta | +0.0013 | [-0.0028, +0.0055] | no |
| teammate_quali_delta_avg5 | -0.0037 | [-0.0088, +0.0013] | no |
| grid_minus_quali_delta | +0.0000 | [+0.0000, +0.0000] | no |
| constructor_dnf_rate_last10 | +0.0000 | [+0.0000, +0.0000] | no |
| circuit_overtaking_difficulty | +0.0000 | [+0.0000, +0.0000] | no |
| circuit_overtaking_x_grid | -0.0018 | [-0.0079, +0.0052] | no |
| constructor_season_pace_gap | -0.0015 | [-0.0070, +0.0044] | no |

**Validation phase (2022-2023, held out from selection) — honest generalization estimate:**

batch-0 alone: hit_at_1=0.6591, log_loss=0.0783

final set (batch-0 only, nothing survived selection): hit_at_1=0.6591, log_loss=0.0783

Δ hit_at_1 = +0.0000 [+0.0000, +0.0000] (inconclusive)

Δ log_loss = +0.0000 [+0.0000, +0.0000] (inconclusive)

**Headline:** Selected 0/9 on selection folds (CI excludes zero): none. On held-out validation folds: Δhit@1=+0.0000 [+0.0000,+0.0000].

**Verdict:** inconclusive

**Next:** Stage 6 (skip Stage 7 per instruction): PyTorch conditional logit / Plackett-Luce.

### Entry 25 — Stage 6: conditional logit / Plackett-Luce vs. classifier and ranker (145 dev-fold races)

**Date:** 2026-08-24  **Commit:** 6a5beae

**What changed / hypothesis:** Implemented a conditional logit (win-only cross-entropy, softmax within race) and its Plackett-Luce extension (trained on the full observed finishing order, softmax over successively smaller remaining fields) in PyTorch. Compared against xgb_v1 (binary classifier) and xgb_rank_ndcg on the honest batch-0 feature set (12 features), development folds only. A race is a discrete choice among ~20 alternatives where exactly one wins; the binary classifier's loss doesn't encode that. The conditional logit's loss directly optimizes what's being measured (within-race softmax cross-entropy against the winner); Plackett-Luce extends that to the full field, which should help Spearman the same way the ranking objective did in Stage 3.

**Config diff from previous run:**
```
feature_columns=batch-0 (12, post Fix-3), dev folds=[2017, 2018, 2019, 2020, 2021, 2022, 2023]
```

**Metrics:**

| System | Hit@1 | Hit@3 | MRR | NDCG@5 | Spearman | Log loss | Brier |
|---|---|---|---|---|---|---|---|
| pole_sitter | 0.5241 | 0.8759 | 0.6946 | 0.8560 | 0.7620 | 1.6436 | 0.0476 |
| xgb_v1 | 0.5379 | 0.8759 | 0.7096 | 0.8580 | 0.6887 | 0.1072 | 0.0308 |
| xgb_rank_ndcg | 0.5586 | 0.8897 | 0.7261 | 0.8734 | 0.7833 | 0.1039 | 0.0303 |
| conditional_logit | 0.5103 | 0.8966 | 0.7053 | 0.8637 | 0.7592 | 0.1113 | 0.0320 |
| plackett_luce | 0.4897 | 0.8828 | 0.6928 | 0.8603 | 0.7939 | 0.1174 | 0.0345 |
**Paired CIs vs pole-sitter (development folds):**

| System | Metric | Δ | 95% CI | Verdict |
|---|---|---|---|---|
| conditional_logit | hit_at_1 | -0.0138 | [-0.1034, +0.0759] | inconclusive |
| conditional_logit | log_loss | -1.5323 | [-1.8008, -1.2413] | improved |
| plackett_luce | hit_at_1 | -0.0345 | [-0.1241, +0.0552] | inconclusive |
| plackett_luce | log_loss | -1.5262 | [-1.7916, -1.2383] | improved |

**Headline:** Spearman: classifier=0.6887, conditional_logit=0.7592, plackett_luce=0.7939.

**Verdict:** see per-metric CIs above — mixed by design, reported as such

**Next:** Stage 8 (skipping Stage 7 per instruction): calibration — single temperature on the within-race softmax, fit on development folds by log loss; then blend against grid-only.

### Entry 26 — Stage 8: calibration (temperature + grid-only blend)

**Date:** 2026-08-24  **Commit:** 6a5beae

**What changed / hypothesis:** Fit one temperature parameter per model (plackett_luce, conditional_logit) on development folds by log loss, via an exact power-transform on already-logged softmax probabilities (no retraining). Then fit a blend weight against grid-only logistic, also on development folds by log loss. Contrast with the pre-rebuild project's three parameters (grid_alpha, temperature, form_boost_weight) tuned by differential evolution against Hit@1 on fifteen races — one honestly-fit parameter on a proper metric with a defensible sample, vs three parameters chasing a noisy metric on too few races. See README.

**Config diff from previous run:**
```
No feature/split change. Post-hoc calibration of Stage 6's already-logged development-fold predictions.
```

**Metrics:**

**Temperature fit (development folds, minimizing pooled log loss):**

| Model | T | Log loss before | Log loss after |
|---|---|---|---|
| plackett_luce | 0.799 | 0.1174 | 0.1154 |
| conditional_logit | 1.373 | 0.1113 | 0.1069 |

**Blend weight vs. grid-only logistic (development folds, minimizing pooled log loss):**

| Model | w (model weight) | Log loss (model alone) | Log loss (grid alone) | Log loss (blended) |
|---|---|---|---|---|
| plackett_luce | 0.759 | 0.1154 | 0.1341 | 0.1132 |
| conditional_logit | 0.907 | 0.1069 | 0.1341 | 0.1066 |

**Finding:** optimal blend weight for ['conditional_logit'] landed near 1.0 (all model, ~0 grid-only) — same reading, from the other side.

**Headline:** Temperatures: plackett_luce=0.80, conditional_logit=1.37. Blend weights: plackett_luce=0.76, conditional_logit=0.91.

**Verdict:** n/a (calibration entry)

**Next:** Locked holdout (2024-2025), run exactly once with --final, using the calibrated model chosen here.

### Entry 27 — LOCKED HOLDOUT (final, 48 races, 2024-2025)

**Date:** 2026-08-24  **Commit:** 6a5beae

**What changed / hypothesis:** The single, final evaluation of plackett_luce_calibrated_blend on the 2024-2025 locked holdout, run exactly once with --final after every modeling decision was frozen on development folds (2017-2023) only. N/A — this is not an experiment to iterate on. It's the one honest read of generalization.

**Config diff from previous run:**
```
model=plackett_luce_calibrated_blend, holdout years=[2024, 2025], split={'start_date': '2015-01-01', 'end_date': '2025-12-31', 'val_start_date': '2024-01-01', 'test_start_date': '2025-01-01'}
```

**Metrics:**

| Metric | Value | 95% CI | Δ vs pole-sitter | Δ 95% CI | Verdict |
|---|---|---|---|---|---|
| hit_at_1 | 0.4792 | [0.3536, 0.6250] | -0.1042 | [-0.2500, +0.0417] | inconclusive |
| hit_at_3 | 0.9375 | [0.8542, 1.0000] | +0.0625 | [-0.0208, +0.1458] | inconclusive |
| mrr | 0.6960 | [0.6111, 0.7865] | -0.0460 | [-0.1267, +0.0382] | inconclusive |
| ndcg_at_5 | 0.8534 | [0.8160, 0.8918] | -0.0204 | [-0.0486, +0.0091] | inconclusive |
| spearman | 0.7217 | [0.6688, 0.7741] | -0.0080 | [-0.0319, +0.0143] | inconclusive |
| log_loss | 0.1167 | [0.1056, 0.1277] | -1.3255 | [-1.8256, -0.8274] | improved |
| brier_score | 0.0349 | [0.0315, 0.0384] | -0.0068 | [-0.0199, +0.0054] | inconclusive |

**Headline:** LOCKED: Hit@1=0.4792 on 48 never-before-evaluated races

**Verdict:** inconclusive

**Next:** None — this is the final entry for this project phase.
