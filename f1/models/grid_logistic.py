"""Grid-only baseline: a single-feature logistic regression on grid position.
This is the real bar to beat — it's what "the answer is just the starting
order" looks like as a calibrated probability model, as opposed to the
pole-sitter baseline's degenerate 1.0/0.0 predictions.
"""
import pandas as pd
from sklearn.linear_model import LogisticRegression


class GridLogisticModel:
    version = "grid_logistic"

    def __init__(self):
        self.clf = LogisticRegression()
        self._fallback_grid = 10.0

    def fit(self, train_df: pd.DataFrame):
        grid = train_df["grid_position"].astype(float)
        self._fallback_grid = float(grid.median())
        X = grid.fillna(self._fallback_grid).to_frame()
        y = train_df["target_win"]
        self.clf.fit(X, y)
        return self

    def predict_race(self, race_df: pd.DataFrame) -> pd.DataFrame:
        df = race_df.copy()
        X = df["grid_position"].astype(float).fillna(self._fallback_grid).to_frame()
        raw = self.clf.predict_proba(X)[:, 1]
        total = raw.sum()
        df["predicted_probability"] = raw / total if total > 0 else 1.0 / len(df)
        df["predicted_rank"] = pd.Series(-df["predicted_probability"].values, index=df.index) \
            .rank(method="first").astype(int)
        return df
