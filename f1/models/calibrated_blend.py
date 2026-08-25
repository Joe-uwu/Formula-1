"""The final frozen model for the locked holdout: Plackett-Luce (Stage 6's
best full-field Spearman, and the only model giving both win probability
and full-field ranking from one set of driver strengths), temperature-
calibrated and blended with grid-only logistic — both calibration
parameters frozen from Stage 8's development-fold fit (T=0.799, w=0.759),
NOT refit here. Only the underlying Plackett-Luce and grid-logistic models
retrain per rolling-origin fold, same as everywhere else in this project.
"""
import pandas as pd

from f1.models.grid_logistic import GridLogisticModel
from f1.models.conditional_logit import PlackettLuceModel

TEMPERATURE = 0.799  # Stage 8, fit on development folds by log loss
BLEND_WEIGHT = 0.759  # Stage 8, fit on development folds by log loss (model weight; grid gets 1 - this)


class CalibratedBlendedPlackettLuce:
    version = "plackett_luce_calibrated_blend"

    def __init__(self):
        self.pl = PlackettLuceModel()
        self.grid = GridLogisticModel()

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None):
        self.pl.fit(train_df)
        self.grid.fit(train_df)
        return self

    def predict_race(self, race_df: pd.DataFrame) -> pd.DataFrame:
        pl_out = self.pl.predict_race(race_df)
        grid_out = self.grid.predict_race(race_df)

        p = pl_out["predicted_probability"].clip(lower=1e-12) ** (1.0 / TEMPERATURE)
        p = p / p.sum()

        blended = BLEND_WEIGHT * p.to_numpy() + (1 - BLEND_WEIGHT) * grid_out["predicted_probability"].to_numpy()
        df = race_df.copy()
        df["predicted_probability"] = blended
        df["predicted_rank"] = pd.Series(-blended, index=df.index).rank(method="first").astype(int)
        return df
