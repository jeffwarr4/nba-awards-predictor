from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import json

REPO_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = REPO_ROOT / "data" / "processed" / "training" / "nba_mvp_training_dataset.csv"
MODEL_DIR = REPO_ROOT / "models"
MODEL_PATH = MODEL_DIR / "nba_mvp_ridge_np.json"

FEATURES = [
    "games_played",
    "minutes_played",
    "pts",
    "trb",
    "ast",
    "stl",
    "blk",
    "fg_pct",
    "fg3_pct",
    "ft_pct",
]
TARGET = "vote_share"

def standardize_fit(X: np.ndarray):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    return mu, sd

def standardize_apply(X: np.ndarray, mu: np.ndarray, sd: np.ndarray):
    return (X - mu) / sd

def ridge_fit_closed_form(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """
    Closed form ridge:
      w = (X^T X + alpha I)^-1 X^T y
    Note: X should already include intercept column if desired.
    """
    n_features = X.shape[1]
    A = X.T @ X + alpha * np.eye(n_features)
    b = X.T @ y
    w = np.linalg.solve(A, b)
    return w

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def group_kfold_seasons(seasons: np.ndarray, n_splits: int = 5):
    """
    Simple grouped CV splitter by season_end_year without sklearn.
    """
    uniq = np.array(sorted(np.unique(seasons)))
    # deterministic split: chunk seasons
    folds = np.array_split(uniq, n_splits)
    for i in range(n_splits):
        test_seasons = set(folds[i].tolist())
        test_mask = np.array([s in test_seasons for s in seasons])
        train_idx = np.where(~test_mask)[0]
        test_idx = np.where(test_mask)[0]
        yield train_idx, test_idx

def main(alpha: float = 10.0, n_splits: int = 5) -> None:
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Missing training dataset: {TRAIN_PATH}")

    df = pd.read_csv(TRAIN_PATH).copy()

    needed = FEATURES + [TARGET, "season_end_year"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Training dataset missing columns: {missing}")

    # numeric + drop NA
    for c in FEATURES + [TARGET]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["season_end_year"] = pd.to_numeric(df["season_end_year"], errors="coerce")
    df = df.dropna(subset=FEATURES + [TARGET, "season_end_year"]).copy()

    X_raw = df[FEATURES].to_numpy(dtype=float)
    y = df[TARGET].to_numpy(dtype=float)
    seasons = df["season_end_year"].to_numpy(dtype=int)

    # Fit standardization on full data (for final model), but CV will refit per fold.
    cv_maes = []
    fold = 1
    for tr_idx, te_idx in group_kfold_seasons(seasons, n_splits=n_splits):
        X_tr_raw, X_te_raw = X_raw[tr_idx], X_raw[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        mu, sd = standardize_fit(X_tr_raw)
        X_tr = standardize_apply(X_tr_raw, mu, sd)
        X_te = standardize_apply(X_te_raw, mu, sd)

        # add intercept
        X_tr_i = np.c_[np.ones(len(X_tr)), X_tr]
        X_te_i = np.c_[np.ones(len(X_te)), X_te]

        w = ridge_fit_closed_form(X_tr_i, y_tr, alpha=alpha)
        preds = X_te_i @ w

        fold_mae = mae(y_te, preds)
        cv_maes.append(fold_mae)
        print(f"[CV] Fold {fold} MAE={fold_mae:.4f} (test seasons={sorted(set(seasons[te_idx].tolist()))[:3]}...)")
        fold += 1

    print(f"[CV] Mean MAE={float(np.mean(cv_maes)):.4f}  Std={float(np.std(cv_maes)):.4f}")

    # Train final model on all data
    mu, sd = standardize_fit(X_raw)
    X = standardize_apply(X_raw, mu, sd)
    X_i = np.c_[np.ones(len(X)), X]
    w = ridge_fit_closed_form(X_i, y, alpha=alpha)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_type": "ridge_closed_form",
        "alpha": alpha,
        "target": TARGET,
        "features": FEATURES,
        "intercept_and_weights": w.tolist(),  # w[0] is intercept
        "x_mean": mu.tolist(),
        "x_std": sd.tolist(),
        "train_rows": int(len(df)),
        "train_seasons": int(df["season_end_year"].nunique()),
    }
    MODEL_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[SAVED MODEL] {MODEL_PATH}")
    print(f"[TRAIN ROWS] {payload['train_rows']}  [SEASONS] {payload['train_seasons']}")

if __name__ == "__main__":
    main(alpha=10.0, n_splits=5)
