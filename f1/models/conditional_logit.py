"""Conditional logit / Plackett-Luce: a scoring function over driver-race
features, softmaxed within race, trained by cross-entropy against the
observed outcome. This directly encodes "exactly one driver wins" in the
loss, which an independent per-driver binary classifier does not.

Win-only model: score = linear(features), softmax within race_id,
cross-entropy against the winner. This is the conditional logit.

Full-field model (Plackett-Luce): the likelihood of the observed finishing
order factorizes as a product of softmaxes over successively smaller
remaining fields — P(order) = prod_i [ exp(s_i) / sum_{j in remaining} exp(s_j) ].
One set of driver strengths gives both win probability (first softmax term)
and a full-field ranking (repeatedly renormalizing over what's left), which
is exactly the pair of things the classifier and the ranker were previously
getting from two separate models.
"""
import numpy as np
import pandas as pd
import torch
from torch import nn

from f1.features.materialize import FEATURE_COLUMNS


class _ScoreNet(nn.Module):
    """Deliberately small: a linear layer is the conditional logit itself;
    one hidden layer gives it a little capacity without risking overfit on
    ~4700 rows. Either way the output is a single per-driver scalar score."""

    def __init__(self, n_features: int, hidden: int = 0):
        super().__init__()
        if hidden > 0:
            self.net = nn.Sequential(nn.Linear(n_features, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        else:
            self.net = nn.Linear(n_features, 1)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class ConditionalLogitModel:
    """Win-only conditional logit: cross-entropy on the winner, softmax
    within race. Equivalent to Plackett-Luce truncated to the first pick."""
    version = "conditional_logit"

    def __init__(self, hidden: int = 8, epochs: int = 200, lr: float = 0.01, weight_decay: float = 1e-3, seed: int = 0):
        torch.manual_seed(seed)
        self.hidden, self.epochs, self.lr, self.weight_decay = hidden, epochs, lr, weight_decay
        self.mean_ = None
        self.std_ = None
        self.net = None

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None):
        X_raw = train_df[FEATURE_COLUMNS].astype(float).fillna(0.0).to_numpy()
        self.mean_ = X_raw.mean(axis=0, keepdims=True)
        self.std_ = X_raw.std(axis=0, keepdims=True)
        self.std_[self.std_ == 0] = 1.0
        X = self._standardize(X_raw)

        self.net = _ScoreNet(len(FEATURE_COLUMNS), hidden=self.hidden)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        groups = train_df.groupby("race_id", sort=False)
        race_indices = [g.index.to_numpy() for _, g in groups]
        winner_local_idx = []
        for idx in race_indices:
            win_flags = train_df.loc[idx, "target_win"].to_numpy()
            winner_local_idx.append(int(np.argmax(win_flags)) if win_flags.any() else None)

        X_t = torch.tensor(X, dtype=torch.float32)
        pos_of = {row_idx: i for i, row_idx in enumerate(train_df.index)}

        self.net.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            scores = self.net(X_t)
            loss = torch.tensor(0.0)
            n_races_with_winner = 0
            for idx, winner_i in zip(race_indices, winner_local_idx):
                if winner_i is None:
                    continue
                local_pos = [pos_of[i] for i in idx]
                race_scores = scores[local_pos]
                log_probs = torch.log_softmax(race_scores, dim=0)
                loss = loss - log_probs[winner_i]
                n_races_with_winner += 1
            loss = loss / max(n_races_with_winner, 1)
            loss.backward()
            opt.step()
        return self

    def predict_race(self, race_df: pd.DataFrame) -> pd.DataFrame:
        df = race_df.copy()
        X_raw = df[FEATURE_COLUMNS].astype(float).fillna(0.0).to_numpy()
        X = self._standardize(X_raw)
        self.net.eval()
        with torch.no_grad():
            scores = self.net(torch.tensor(X, dtype=torch.float32)).numpy()
        probs = np.exp(scores - scores.max())
        probs = probs / probs.sum()
        df["predicted_probability"] = probs
        df["predicted_rank"] = pd.Series(-probs, index=df.index).rank(method="first").astype(int)
        return df


class PlackettLuceModel(ConditionalLogitModel):
    """Full-field extension: trains on the whole observed finishing order,
    not just the winner. Likelihood factorizes as a product of softmaxes
    over successively smaller remaining fields (each step removes whoever
    finished next-best and renormalizes over who's left)."""
    version = "plackett_luce"

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None):
        X_raw = train_df[FEATURE_COLUMNS].astype(float).fillna(0.0).to_numpy()
        self.mean_ = X_raw.mean(axis=0, keepdims=True)
        self.std_ = X_raw.std(axis=0, keepdims=True)
        self.std_[self.std_ == 0] = 1.0
        X = self._standardize(X_raw)

        self.net = _ScoreNet(len(FEATURE_COLUMNS), hidden=self.hidden)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        groups = train_df.groupby("race_id", sort=False)
        # Plackett-Luce order: best-finishing first. Unclassified (DNF/no
        # position) drivers are dropped from the order — we don't know where
        # in the sequence they'd fall, and forcing them last is a modeling
        # choice this dataset doesn't need: enough classified finishers
        # remain per race to define the ordering that matters (who's ahead
        # of whom among those who actually finished).
        race_order_local_idx = []
        race_row_idx = []
        for race_id, g in groups:
            classified = g.dropna(subset=["target_position"]).sort_values("target_position")
            if len(classified) < 2:
                continue
            race_row_idx.append(g.index.to_numpy())
            pos_map = {row: i for i, row in enumerate(g.index)}
            race_order_local_idx.append([pos_map[row] for row in classified.index])

        X_t = torch.tensor(X, dtype=torch.float32)
        pos_of = {row_idx: i for i, row_idx in enumerate(train_df.index)}

        self.net.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            scores = self.net(X_t)
            loss = torch.tensor(0.0)
            n = 0
            for idx, order in zip(race_row_idx, race_order_local_idx):
                local_pos = [pos_of[i] for i in idx]
                race_scores = scores[local_pos]
                remaining = list(range(len(order)))
                for step, winner_local in enumerate(order[:-1]):  # last-place needs no softmax (denominator of 1)
                    winner_pos_in_remaining = remaining.index(winner_local)
                    subset_scores = race_scores[remaining]
                    log_probs = torch.log_softmax(subset_scores, dim=0)
                    loss = loss - log_probs[winner_pos_in_remaining]
                    remaining.remove(winner_local)
                n += 1
            loss = loss / max(n, 1)
            loss.backward()
            opt.step()
        return self
