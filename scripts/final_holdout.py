"""The locked-holdout evaluation. Test seasons 2024-2025 — untouched by every
decision made from Stage 4 onward (feature selection, hyperparameters, model
class, calibration all happen on DEV_TEST_YEARS = 2017-2023 only).

Refuses to run without --final. Run this exactly ONCE, after every modeling
decision is frozen (i.e. after Stage 8), and log the single resulting entry.
Running it more than once and picking the better result defeats the entire
point — the holdout stops being a measurement of generalization the moment
it's used to choose between candidates. See README.md.
"""
import argparse
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from f1.config import SPLIT_CONFIG, config_hash
from f1.features.materialize import get_or_build
from f1.eval import walk_forward, metrics as M
from f1.eval.rolling_origin import run_rolling_origin, LOCKED_HOLDOUT_YEARS
from f1.models.baseline import pole_sitter_predictions
from f1 import report

HOLDOUT_CONFIG_HASH = config_hash() + "_locked_holdout"


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true",
                     help="Required. Confirms every modeling decision (features, hyperparameters, "
                          "model class, calibration) is frozen and this is the single, final run.")
    ap.add_argument("--model-factory", default=None,
                     help="dotted path to a zero-arg callable returning a fresh, unfit model, "
                          "e.g. f1.models.train:WinModel. Required with --final.")
    ap.add_argument("--model-version", default=None, help="model_version tag for the predictions table.")
    args = ap.parse_args()

    if not args.final:
        print("Refusing to run: pass --final only once every decision (features, hyperparameters, "
              "model class, calibration) has been frozen on the development folds (2017-2023).")
        print("This holdout (2024-2025) gets evaluated exactly once, ever.")
        sys.exit(1)

    if not args.model_factory or not args.model_version:
        print("--final requires --model-factory and --model-version (no default model — that would be "
              "a silent decision about what 'final' means).")
        sys.exit(1)

    module_path, attr = args.model_factory.split(":")
    import importlib
    factory = getattr(importlib.import_module(module_path), attr)

    feature_set_id, feature_df, _ = get_or_build(SPLIT_CONFIG)
    feature_df["as_of_date"] = pd.to_datetime(feature_df["as_of_date"])

    model_preds = run_rolling_origin(feature_df, factory, model_version=args.model_version,
                                       config_hash=HOLDOUT_CONFIG_HASH, test_years=LOCKED_HOLDOUT_YEARS)

    pooled_start = pd.Timestamp(f"{min(LOCKED_HOLDOUT_YEARS)}-01-01")
    pooled_end = pd.Timestamp(f"{max(LOCKED_HOLDOUT_YEARS) + 1}-01-01")
    pooled_test_df = feature_df[(feature_df["as_of_date"] >= pooled_start) & (feature_df["as_of_date"] < pooled_end)]
    pole_preds = walk_forward.run(pooled_test_df, pole_sitter_predictions,
                                    model_version="pole_sitter", config_hash=HOLDOUT_CONFIG_HASH)

    n_races = model_preds["race_id"].nunique()
    stats = M.RaceLevelStats.from_predictions(model_preds)
    baseline_stats = M.RaceLevelStats.from_predictions(pole_preds)

    rows = ["| Metric | Value | 95% CI | Δ vs pole-sitter | Δ 95% CI | Verdict |",
            "|---|---|---|---|---|---|"]
    for name in M.HEADLINE_METRIC_NAMES:
        point, lo, hi = M.bootstrap_ci(stats, name)
        _, dlo, dhi = M.paired_bootstrap_ci(stats, baseline_stats, name)
        v = M.verdict(dlo, dhi, metric=name)
        rows.append(f"| {name} | {point:.4f} | [{lo:.4f}, {hi:.4f}] | "
                     f"{point - baseline_stats.value(name):+.4f} | [{dlo:+.4f}, {dhi:+.4f}] | {v} |")
    table = "\n".join(rows)

    report.append_entry(
        title=f"Entry {report.next_entry_number()} — LOCKED HOLDOUT (final, {n_races} races, 2024-2025)",
        commit_sha=git_sha(),
        what_changed=(
            f"The single, final evaluation of {args.model_version} on the 2024-2025 locked holdout, "
            f"run exactly once with --final after every modeling decision was frozen on development folds "
            f"(2017-2023) only."
        ),
        hypothesis="N/A — this is not an experiment to iterate on. It's the one honest read of generalization.",
        config_diff=f"model={args.model_version}, holdout years={list(LOCKED_HOLDOUT_YEARS)}, split={SPLIT_CONFIG}",
        metrics_table_md=table,
        headline=f"LOCKED: Hit@1={stats.value('hit_at_1'):.4f} on {n_races} never-before-evaluated races",
        verdict=M.verdict(*M.paired_bootstrap_ci(stats, baseline_stats, "hit_at_1")[1:], metric="hit_at_1"),
        next_steps="None — this is the final entry for this project phase.",
    )
    print("REPORT.md updated. This holdout has now been spent — do not run this script again for this project phase.")
    print(table)


if __name__ == "__main__":
    main()
