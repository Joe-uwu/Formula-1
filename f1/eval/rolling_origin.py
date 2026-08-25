"""Rolling-origin walk-forward: train on all seasons <= Y, test on season
Y+1, for Y in a given range, retraining fresh each fold. Pools predictions
across folds under one (model_version, config_hash) so headline metrics are
computed on ~200 races instead of one season's ~24 — race_id is globally
unique, so pooling under one key is exactly "concatenate the folds."
"""
import sys
from pathlib import Path
from typing import Callable

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from f1.eval import walk_forward

FOLD_TEST_YEARS = range(2017, 2026)  # train <= Y, test = Y+1, for Y in 2016..2024 — superseded below

# Stage 4: locked holdout. Every decision from here on — feature selection,
# hyperparameters, model class, calibration — is made by looking ONLY at
# DEV_TEST_YEARS. LOCKED_HOLDOUT_YEARS is evaluated exactly once, at the very
# end, by scripts/final_holdout.py (which refuses to run without --final).
# See README.md for why this matters more than any single accuracy number.
DEV_TEST_YEARS = range(2017, 2024)       # test seasons 2017-2023
LOCKED_HOLDOUT_YEARS = range(2024, 2026)  # test seasons 2024-2025 — do not evaluate until Stage 8 is done


def run_rolling_origin(feature_df: pd.DataFrame, model_factory: Callable[[], object],
                         model_version: str, config_hash: str,
                         test_years=FOLD_TEST_YEARS) -> pd.DataFrame:
    """model_factory() -> a fresh, unfit model with .fit(train_df) and
    .predict_race(race_df). Returns the pooled predictions across all folds,
    each also written to walk_forward under (model_version, config_hash)."""
    feature_df = feature_df.copy()
    feature_df["as_of_date"] = pd.to_datetime(feature_df["as_of_date"])

    all_preds = []
    for test_year in test_years:
        train_end = pd.Timestamp(f"{test_year}-01-01")
        test_end = pd.Timestamp(f"{test_year + 1}-01-01")
        train_df = feature_df[feature_df["as_of_date"] < train_end]
        test_df = feature_df[(feature_df["as_of_date"] >= train_end) & (feature_df["as_of_date"] < test_end)]
        if test_df.empty or train_df.empty:
            continue

        model = model_factory().fit(train_df)
        preds = walk_forward.run(test_df, model.predict_race,
                                   model_version=model_version, config_hash=config_hash)
        preds["fold_test_year"] = test_year
        all_preds.append(preds)

    return pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
