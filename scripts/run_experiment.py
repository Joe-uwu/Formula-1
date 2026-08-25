"""Run one experiment end-to-end: materialize features (or reuse), walk
forward over the test races, compute metrics with bootstrap CIs, append a
REPORT.md entry. `--mode baseline` runs the pole-sitter baseline only (use
this for the very first log entry, before any model exists). `--mode model`
trains WinModel on the train split and walks forward on the test split.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from f1.config import SPLIT_CONFIG, config_hash
from f1.features.materialize import get_or_build, FEATURE_COLUMNS
from f1.eval import walk_forward, metrics as M
from f1.models.baseline import pole_sitter_predictions
from f1.models.train import WinModel
from f1 import report


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def metrics_table(preds: pd.DataFrame, baseline_preds: pd.DataFrame) -> str:
    # One real per-race groupby each, reused for every metric's point estimate
    # and every bootstrap draw below — not recomputed per metric per draw.
    stats = M.RaceLevelStats.from_predictions(preds)
    baseline_stats = M.RaceLevelStats.from_predictions(baseline_preds)

    rows = ["| Metric | Value | 95% CI | Δ vs pole-sitter | Δ 95% CI | Verdict |",
            "|---|---|---|---|---|---|"]
    for name in M.HEADLINE_METRIC_NAMES:
        point, lo, hi = M.bootstrap_ci(stats, name)
        _, dlo, dhi = M.paired_bootstrap_ci(stats, baseline_stats, name)
        v = M.verdict(dlo, dhi, metric=name)
        rows.append(f"| {name} | {point:.4f} | [{lo:.4f}, {hi:.4f}] | "
                     f"{point - baseline_stats.value(name):+.4f} | [{dlo:+.4f}, {dhi:+.4f}] | {v} |")
    ece, _ = M.expected_calibration_error(preds)
    rows.append(f"| ece | {ece:.4f} | — | — | — | — |")
    bss = M.brier_skill_score(preds, baseline_preds)
    rows.append(f"| brier_skill_score | {bss:.4f} | — | — | — | — |")
    return "\n".join(rows)


def breakdown_tables(preds: pd.DataFrame, feature_df: pd.DataFrame) -> str:
    merged = preds.merge(feature_df[["race_id", "as_of_date"]].drop_duplicates(), on="race_id")
    merged["season"] = pd.to_datetime(merged["as_of_date"]).dt.year
    lines = ["\n**Per-season Hit@1:**\n"]
    for season, g in merged.groupby("season"):
        lines.append(f"- {season}: {M.hit_at_k(g, 1):.3f} (n={g['race_id'].nunique()} races)")

    lines.append("\n**Split by whether the pole sitter won:**\n")
    if "quali_position" in feature_df:
        pole_won_races = set(
            feature_df.loc[(feature_df["quali_position"] == 1) & (feature_df["target_win"] == 1), "race_id"]
        )
        for label, race_set in [("pole sitter won", pole_won_races),
                                  ("pole sitter did not win", set(preds["race_id"]) - pole_won_races)]:
            subset = preds[preds["race_id"].isin(race_set)]
            if subset.empty:
                continue
            lines.append(f"- {label} (n={subset['race_id'].nunique()}): "
                          f"Hit@1={M.hit_at_k(subset, 1):.3f}, MRR={M.mean_reciprocal_rank(subset):.3f}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["baseline", "model"], required=True)
    args = ap.parse_args()

    feature_set_id, feature_df, timings = get_or_build(SPLIT_CONFIG)
    print(f"feature_set_id={feature_set_id} rows={len(feature_df)} timings={timings}")

    val_start = pd.Timestamp(SPLIT_CONFIG["val_start_date"])
    test_start = pd.Timestamp(SPLIT_CONFIG["test_start_date"])
    feature_df["as_of_date"] = pd.to_datetime(feature_df["as_of_date"])
    train_df = feature_df[feature_df["as_of_date"] < val_start]
    val_df = feature_df[(feature_df["as_of_date"] >= val_start) & (feature_df["as_of_date"] < test_start)]
    test_df = feature_df[feature_df["as_of_date"] >= test_start]

    import json
    chash = config_hash()

    walk_forward.run(test_df, pole_sitter_predictions, model_version="pole_sitter", config_hash=chash)
    # Metrics are computed from what's actually in the predictions table, not
    # the in-memory return value, so every reported number is DB-auditable.
    baseline_preds = walk_forward.fetch_predictions("pole_sitter", chash)

    n = report.next_entry_number()

    if args.mode == "baseline":
        model_preds = baseline_preds
        model_version = "pole_sitter"
        title = f"Entry {n} — baselines only"
        what_changed = "No model yet."
        hypothesis = "Establish the pole-sitter baseline before any model exists, so the first model has something to beat."
    else:
        model = WinModel().fit(train_df, val_df)
        model.save(chash)
        walk_forward.run(test_df, model.predict_race, model_version=model.version, config_hash=chash)
        model_preds = walk_forward.fetch_predictions(model.version, chash)
        model_version = model.version
        title = f"Entry {n} — {model.version}, train<={SPLIT_CONFIG['val_start_date']} / val<{SPLIT_CONFIG['test_start_date']} / test>={SPLIT_CONFIG['test_start_date']}"
        what_changed = (f"Trained {model.version} (XGBoost, up to 200 trees depth 3, early-stopped on the "
                          f"validation split) on point-in-time features. Test set is 2025, backfilled from FastF1 "
                          f"since it isn't in the Kaggle CSVs.")
        hypothesis = "Hypothesis: grid position plus trailing form/circuit-history features beat the pole-sitter baseline on ranking metrics."

    table = metrics_table(model_preds, baseline_preds)
    breakdown = breakdown_tables(model_preds, feature_df)
    hit1 = M.hit_at_k(model_preds, 1)

    report.append_entry(
        title=title,
        commit_sha=git_sha(),
        what_changed=what_changed,
        hypothesis=hypothesis,
        config_diff=json.dumps(SPLIT_CONFIG, indent=2) + f"\nmodel_version={model_version}\nfeature_build_timings={timings}",
        metrics_table_md=table + "\n" + breakdown,
        headline=f"Hit@1={hit1:.3f}",
        verdict=("baseline" if args.mode == "baseline" else
                 M.verdict(*M.paired_bootstrap_ci(
                     M.RaceLevelStats.from_predictions(model_preds),
                     M.RaceLevelStats.from_predictions(baseline_preds), "hit_at_1")[1:])),
        next_steps=("Train the first model on this baseline." if args.mode == "baseline"
                     else "Iterate: one hypothesis, one change, one measurement per entry."),
    )
    print("REPORT.md updated.")


if __name__ == "__main__":
    main()
