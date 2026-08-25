"""Learning-to-rank models: XGBoost ranker trained on graded relevance
derived from finishing position, grouped by race — instead of the binary
Win target WinModel uses. A binary target only has to separate P1 from
everyone else; it has no incentive to order P8 through P15 correctly, yet
Spearman (and any full-field ranking metric) scores exactly that ordering.
"""
import numpy as np
import pandas as pd
from xgboost import XGBRanker

from f1.features.materialize import FEATURE_COLUMNS


def _relevance(df: pd.DataFrame) -> pd.Series:
    """P1 gets the highest label (field size), last place gets 1, DNF/unknown gets 0."""
    field_size = df.groupby("race_id")["driver_id"].transform("size")
    return (field_size - df["target_position"] + 1).fillna(0).clip(lower=0)


class RankerModel:
    def __init__(self, objective: str = "rank:pairwise"):
        self.objective = objective
        self.version = f"xgb_{objective.replace(':', '_')}"
        self.ranker = XGBRanker(
            objective=objective, n_estimators=200, max_depth=3, learning_rate=0.05, random_state=0,
        )

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None):
        train_df = train_df.sort_values("race_id")
        X = train_df[FEATURE_COLUMNS].astype(float)
        y = _relevance(train_df)
        self.ranker.fit(X, y, qid=train_df["race_id"].to_numpy())
        return self

    def predict_race(self, race_df: pd.DataFrame) -> pd.DataFrame:
        df = race_df.copy()
        X = df[FEATURE_COLUMNS].astype(float)
        raw = self.ranker.predict(X)
        # Ranker scores aren't probabilities. Softmax within the race so log
        # loss / Brier / BSS stay comparable against the classifier and
        # baselines, which all report a proper per-race distribution.
        exp = np.exp(raw - raw.max())
        probs = exp / exp.sum()
        df["predicted_probability"] = probs
        df["predicted_rank"] = pd.Series(-probs, index=df.index).rank(method="first").astype(int)
        return df
