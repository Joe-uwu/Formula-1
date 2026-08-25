"""First honest model: gradient-boosted binary win classifier trained on the
materialized point-in-time feature table. Deliberately simple — no tuning
toward a target number, per the brief. Iterate from here with one hypothesis
per REPORT.md entry.
"""
from pathlib import Path

import joblib
import pandas as pd
from xgboost import XGBClassifier

from f1.features.materialize import FEATURE_COLUMNS

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


class WinModel:
    version = "xgb_v1"

    def __init__(self):
        self.clf = XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            eval_metric="logloss", early_stopping_rounds=20, random_state=0,
        )

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None):
        X = train_df[FEATURE_COLUMNS].astype(float)
        y = train_df["target_win"]
        if val_df is not None and not val_df.empty:
            X_val = val_df[FEATURE_COLUMNS].astype(float)
            y_val = val_df["target_win"]
            self.clf.fit(X, y, eval_set=[(X_val, y_val)], verbose=False)
        else:
            self.clf.set_params(early_stopping_rounds=None)
            self.clf.fit(X, y)
        return self

    def predict_race(self, race_df: pd.DataFrame) -> pd.DataFrame:
        """race_df: one row per driver for one race with FEATURE_COLUMNS + actual_position."""
        df = race_df.copy()
        X = df[FEATURE_COLUMNS].astype(float)
        raw = self.clf.predict_proba(X)[:, 1]
        # normalize within-race so probabilities sum to 1 across the field
        total = raw.sum()
        df["predicted_probability"] = raw / total if total > 0 else 1.0 / len(df)
        df["predicted_rank"] = pd.Series(-df["predicted_probability"].values, index=df.index) \
            .rank(method="first").astype(int)
        return df

    def save(self, config_hash: str) -> Path:
        MODELS_DIR.mkdir(exist_ok=True)
        path = MODELS_DIR / f"{self.version}_{config_hash}.joblib"
        joblib.dump(self.clf, path)
        return path

    @classmethod
    def load(cls, config_hash: str) -> "WinModel":
        path = MODELS_DIR / f"{cls.version}_{config_hash}.joblib"
        model = cls()
        model.clf = joblib.load(path)
        return model
