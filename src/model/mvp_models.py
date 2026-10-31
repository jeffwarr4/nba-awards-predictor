"""
MVP model specs, fitting, scoring and the leave-one-season-out backtest.

A spec says which rows train the model, which features it sees, its target, the
model family and any post-hoc blend. The shipped model and every experiment in
docs/nba-2026-evaluation-and-plan.md are specs in SPECS, so the backtest table and
the shipped artifact come from the same code.

Evaluation is leave-one-season-out over every season in the history: fit on all
other seasons, score every player in the held-out season's pool, rank. Seasons are
never split randomly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

SEASON_GAMES = 82

# Totals that grow with games played. Scaled to an 82-game season by team games, in
# training (lockout/bubble seasons) and live (pace projection). Rates are not scaled.
PACE_COLS = ["g", "gs", "mp_total", "ows", "dws", "ws", "vorp"]

BOX = ["g", "mp", "pts", "trb", "ast", "stl", "blk", "fg_pct", "fg3_pct", "ft_pct"]   # the shipped 10


@dataclass
class Spec:
    name: str
    features: list[str]
    model: str = "ridge"                 # ridge | logit_share | gbr_top5 | logit_top5
    train_rows: str = "pool"             # pool (every scored player) | voters (vote-getters only)
    blend_team: bool = False             # shipped post-hoc 0.7 model + 0.2 win% + 0.1 conf rank
    alpha: float = 10.0
    min_games_2024: int | None = None    # 65-game rule era: drop players under this many games
    notes: str = ""
    params: dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────
# Rows
# ─────────────────────────────────────────────────────────────────
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Pace-normalize totals and fill shooting % blanks (0 attempts) with 0."""
    df = df.copy()
    factor = SEASON_GAMES / df["team_gp"].replace(0, np.nan)
    for c in PACE_COLS:
        df[c] = df[c] * factor
    df["g"] = df["g"].clip(upper=SEASON_GAMES)
    df["g_share"] = df["g"] / SEASON_GAMES
    for c in ("fg_pct", "fg3_pct", "ft_pct", "efg_pct", "ts_pct"):
        df[c] = df[c].fillna(0.0)
    df["neg_conf_rank"] = -df["conf_rank"]
    return df


def pool_mask(df: pd.DataFrame) -> pd.Series:
    """Who is on the board: the live candidate filter (≥5 games, ≥15 min/game)."""
    return (df["g"] >= 5) & (df["mp"] >= 15)


# ─────────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────────
def _standardize(X: np.ndarray):
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1.0
    return mu, sd


def _ridge_closed_form(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """As shipped in train_mvp_ridge_np.py (the intercept column is penalized too)."""
    Xi = np.c_[np.ones(len(X)), X]
    return np.linalg.solve(Xi.T @ Xi + alpha * np.eye(Xi.shape[1]), Xi.T @ y)


class Model:
    """Fit on a training frame, score any frame. Imputes feature NaNs with training medians."""

    def __init__(self, spec: Spec):
        self.spec = spec

    def _X(self, df: pd.DataFrame) -> np.ndarray:
        return df[self.spec.features].astype(float).fillna(self.medians).to_numpy()

    def fit(self, train: pd.DataFrame) -> "Model":
        s = self.spec
        if s.train_rows == "voters":
            train = train[train["vote_share"] > 0]
        self.medians = train[s.features].astype(float).median()
        X = self._X(train)
        self.mu, self.sd = _standardize(X)
        Z = (X - self.mu) / self.sd
        if s.model == "ridge":
            self.w = _ridge_closed_form(Z, train["vote_share"].to_numpy(float), s.alpha)
        elif s.model == "logit_share":
            from sklearn.linear_model import LogisticRegression
            y = (train["vote_share"] > 0).astype(int).to_numpy()
            wt = np.where(y == 1, 1 + 20 * train["vote_share"].to_numpy(float), 1.0)
            self.clf = LogisticRegression(C=s.params.get("C", 0.5), max_iter=2000, class_weight="balanced")
            self.clf.fit(Z, y, sample_weight=wt)
        elif s.model == "logit_top5":
            from sklearn.linear_model import LogisticRegression
            self.clf = LogisticRegression(C=s.params.get("C", 0.5), max_iter=2000, class_weight="balanced")
            self.clf.fit(Z, train["is_top5"].to_numpy(int))
        elif s.model == "gbr_top5":
            from sklearn.ensemble import GradientBoostingRegressor
            from sklearn.linear_model import LogisticRegression
            self.reg = GradientBoostingRegressor(random_state=42, n_estimators=300, learning_rate=0.05, max_depth=3)
            self.reg.fit(Z, train["vote_share"].to_numpy(float))
            self.clf = LogisticRegression(max_iter=2000, class_weight="balanced")
            self.clf.fit(Z, train["is_top5"].to_numpy(int))
        else:
            raise ValueError(s.model)
        return self

    def raw(self, df: pd.DataFrame) -> np.ndarray:
        Z = (self._X(df) - self.mu) / self.sd
        m = self.spec.model
        if m == "ridge":
            return np.c_[np.ones(len(Z)), Z] @ self.w
        if m in ("logit_share", "logit_top5"):
            return self.clf.decision_function(Z)
        if m == "gbr_top5":   # NHL's blend: 0.5 × min-max vote share + 0.4 × top-5 prob + 0.1 × softmax win prob
            vs = self.reg.predict(Z)
            p5 = self.clf.predict_proba(Z)[:, 1]
            win = np.zeros_like(vs)
            top = np.argsort(-vs)[:25]
            e = np.exp(vs[top] - vs[top].max())
            win[top] = e / e.sum()
            return 0.5 * minmax(vs) + 0.4 * p5 + 0.1 * win
        raise ValueError(m)

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """Ranking score for one season's pool."""
        raw = self.raw(df)
        if not self.spec.blend_team:
            return raw
        return (0.70 * minmax(raw) + 0.20 * minmax(df["team_win_pct"].fillna(0.5).to_numpy(float))
                + 0.10 * minmax(-df["conf_rank"].fillna(8).to_numpy(float)))


def minmax(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    span = np.nanmax(x) - np.nanmin(x)
    return (x - np.nanmin(x)) / span if span > 0 else np.zeros_like(x)


# ─────────────────────────────────────────────────────────────────
# Backtest
# ─────────────────────────────────────────────────────────────────
def season_pool(df: pd.DataFrame, spec: Spec) -> pd.DataFrame:
    pool = df[pool_mask(df)]
    if spec.min_games_2024 is not None:
        pool = pool[(pool["season_end_year"] < 2024) | (pool["g_raw"] >= spec.min_games_2024)]
    return pool


def rank_season(model: Model, season_df: pd.DataFrame, spec: Spec) -> pd.DataFrame:
    """Score and rank one season's pool. Vote-getters outside the pool are appended
    unranked (model_rank = pool size + 1), so a filter can't hide a miss."""
    pool = season_pool(season_df, spec).copy()
    pool["score"] = model.score(pool)
    pool = pool.sort_values("score", ascending=False).reset_index(drop=True)
    pool["model_rank"] = np.arange(1, len(pool) + 1)
    missing = season_df[(season_df["vote_share"] > 0) & ~season_df["player_id"].isin(pool["player_id"])].copy()
    missing["model_rank"] = len(pool) + 1
    return pd.concat([pool, missing], ignore_index=True)


def season_metrics(ranked: pd.DataFrame) -> dict:
    real = ranked[ranked["vote_share"] > 0].sort_values("vote_share", ascending=False)
    winner, top5 = real.iloc[0], real.head(5)
    return {
        "season": int(ranked["season_end_year"].iloc[0]),
        "winner": winner["player_name"],
        "winner_rank": int(winner["model_rank"]),
        "top5_in_top5": int((top5["model_rank"] <= 5).sum()),
        "top5_in_top10": int((top5["model_rank"] <= 10).sum()),
        "model_top3": ", ".join(ranked.sort_values("model_rank").head(3)["player_name"]),
    }


def backtest(hist: pd.DataFrame, spec: Spec, seasons=None, verbose: bool = False) -> pd.DataFrame:
    """Leave-one-season-out. Training rows: every other season's pool (or its vote-getters)."""
    rows = []
    seasons = seasons or sorted(hist["season_end_year"].unique())
    for season in seasons:
        train = hist[(hist["season_end_year"] != season)]
        train = train[pool_mask(train) | (train["vote_share"] > 0)]
        model = Model(spec).fit(train)
        ranked = rank_season(model, hist[hist["season_end_year"] == season], spec)
        rows.append(season_metrics(ranked))
        if verbose:
            print(f"  {season}: winner {rows[-1]['winner']} ranked #{rows[-1]['winner_rank']}")
    return pd.DataFrame(rows)


def summarize(bt: pd.DataFrame) -> dict:
    return {
        "seasons": len(bt),
        "winner_rank1_pct": round(float((bt["winner_rank"] == 1).mean()), 3),
        "winner_top3_pct": round(float((bt["winner_rank"] <= 3).mean()), 3),
        "winner_top5_pct": round(float((bt["winner_rank"] <= 5).mean()), 3),
        "winner_rank_mean": round(float(bt["winner_rank"].mean()), 2),
        "top5_in_top5_mean": round(float(bt["top5_in_top5"].mean()), 2),
    }


def load_history(path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")
    df["g_raw"] = df["g"]
    return add_features(df)
