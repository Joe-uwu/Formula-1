"""Fix 1: Entry 8's uniform and grid-only-logistic baselines were only
evaluated on the 24-race 2025 holdout. Entries 9-11 pooled all 9
rolling-origin folds (2017-2025) for the model/pole-sitter comparison —
this reruns the two Entry-8 baselines on that same 9-fold pooled surface so
they're apples-to-apples with the model numbers, not narrower.

Not a new touch of the pristine locked holdout: Entries 9-11 already used
all 9 years (2017-2025, including 2024-2025) before Stage 4 existed. This
just brings Entry 8's baselines up to the same evaluation surface.
"""
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from f1.config import SPLIT_CONFIG, config_hash
from f1.features.materialize import get_or_build
from f1.eval import walk_forward, metrics as M
from f1.eval.rolling_origin import run_rolling_origin, FOLD_TEST_YEARS
from f1.models.baseline import pole_sitter_predictions, uniform_predictions
from f1.models.grid_logistic import GridLogisticModel
from f1 import report

ROLLING_CONFIG_HASH = config_hash() + "_rolling"


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def main():
    feature_set_id, feature_df, _ = get_or_build(SPLIT_CONFIG)
    feature_df["as_of_date"] = pd.to_datetime(feature_df["as_of_date"])

    uniform_preds = run_rolling_origin(feature_df, lambda: _UniformModel(), model_version="uniform",
                                         config_hash=ROLLING_CONFIG_HASH, test_years=FOLD_TEST_YEARS)
    grid_logistic_preds = run_rolling_origin(feature_df, lambda: GridLogisticModel(), model_version="grid_logistic",
                                               config_hash=ROLLING_CONFIG_HASH, test_years=FOLD_TEST_YEARS)

    # already logged under this config_hash by Stage 2/Stage 3 scripts
    model_preds = walk_forward.fetch_predictions("xgb_v1", ROLLING_CONFIG_HASH)
    pole_preds = walk_forward.fetch_predictions("pole_sitter", ROLLING_CONFIG_HASH)

    n_races = model_preds["race_id"].nunique()
    systems = [("xgb_v1 (model)", model_preds), ("pole_sitter", pole_preds),
                ("uniform", uniform_preds), ("grid_logistic", grid_logistic_preds)]

    rows = ["| System | Log loss | Brier | ECE | Brier Skill Score* |", "|---|---|---|---|---|"]
    values = {}
    for name, preds in systems:
        stats = M.RaceLevelStats.from_predictions(preds)
        ll, brier = stats.value("log_loss"), stats.value("brier_score")
        ece, _ = M.expected_calibration_error(preds)
        bss = M.brier_skill_score(preds, pole_preds)
        values[name] = ll
        rows.append(f"| {name} | {ll:.4f} | {brier:.4f} | {ece:.4f} | {bss:.4f} |")
    table = "\n".join(rows)

    model_vs_grid = "beats" if values["xgb_v1 (model)"] < values["grid_logistic"] else "does NOT beat"

    report.append_entry(
        title=f"Entry {report.next_entry_number()} — Fix 1: uniform/grid-logistic baselines on all 9 pooled folds ({n_races} races)",
        commit_sha=git_sha(),
        what_changed=(
            f"Reran the Entry 8 uniform and grid-only-logistic baselines on the same 9-fold pooled rolling-origin "
            f"surface (2017-2025, {n_races} races) that Entries 9-11 already used for the model comparison — "
            f"Entry 8's numbers were only the 24-race 2025 holdout, not comparable to the pooled model numbers."
        ),
        hypothesis=(
            "Correcting an apples-to-oranges comparison, not testing a new hypothesis. (Note: the first attempt "
            "at this, logged as the previous entry, produced NaN for xgb_v1/pole_sitter because the Entry 13 "
            "driver-identity cleanup had wiped the predictions table and Stage 2/3's pooled predictions were "
            "never regenerated after. Regenerated here on the corrected data and the final Stage 5 feature set "
            "before rerunning this comparison.)"
        ),
        config_diff=f"{SPLIT_CONFIG}\nfold_test_years={list(FOLD_TEST_YEARS)} (was: 2025 only)",
        metrics_table_md=table,
        headline=f"xgb_v1 {model_vs_grid} grid_logistic on log loss ({values['xgb_v1 (model)']:.4f} vs {values['grid_logistic']:.4f}) on the full 9-fold pooled set.",
        verdict="n/a (baseline-correction entry, not a model change)",
        next_steps="Fix 2: null-feature check needs 20 repeats reported as a distribution, not a single point estimate.",
    )
    print("REPORT.md updated.")
    print(table)


class _UniformModel:
    def fit(self, train_df, val_df=None):
        return self

    def predict_race(self, race_df):
        return uniform_predictions(race_df)


if __name__ == "__main__":
    main()
