from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import os
import json
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

CURRENT_PATH = REPO_ROOT / "data" / "processed" / "current" / "nba_mvp_current_candidates.csv"
STANDINGS_PATH = REPO_ROOT / "data" / "processed" / "current" / "nba_team_standings.csv"
OUT_DIR = REPO_ROOT / "data" / "processed" / "outputs"

# Allow workflow/env to override the prior path (default keeps existing behavior of looking in outputs)
PRIOR_TOP10_PATH = Path(
    os.getenv("MVP_PRIOR_TOP10_PATH", str(OUT_DIR / "web_prior_top10.csv"))
)

# If 1, missing prior is a hard error. If 0, movers are skipped (rank_delta stays NaN, mover_label stays NEW)
FAIL_ON_MISSING_PRIOR = os.getenv("FAIL_ON_MISSING_PRIOR", "1") == "1"

# Optional: write a local snapshot file (helpful for local dev). In GH Actions you typically want 0.
WRITE_LOCAL_PRIOR_SNAPSHOT = os.getenv("WRITE_LOCAL_PRIOR_SNAPSHOT", "0") == "1"

MODEL_PATH = REPO_ROOT / "models" / "nba_mvp_ridge_np.json"

TEAM_ABBR_TO_NAME = {
    "ATL": "Hawks", "BOS": "Celtics", "BKN": "Nets", "CHA": "Hornets", "CHI": "Bulls", "CLE": "Cavaliers",
    "DAL": "Mavericks", "DEN": "Nuggets", "DET": "Pistons", "GSW": "Warriors", "HOU": "Rockets", "IND": "Pacers",
    "LAC": "Clippers", "LAL": "Lakers", "MEM": "Grizzlies", "MIA": "Heat", "MIL": "Bucks", "MIN": "Timberwolves",
    "NOP": "Pelicans", "NYK": "Knicks", "OKC": "Thunder", "ORL": "Magic", "PHI": "76ers", "PHX": "Suns",
    "POR": "Trail Blazers", "SAC": "Kings", "SAS": "Spurs", "TOR": "Raptors", "UTA": "Jazz", "WAS": "Wizards",
}


def zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mu = s.mean(skipna=True)
    sd = s.std(skipna=True)
    if sd == 0 or np.isnan(sd):
        return s * 0
    return (s - mu) / sd


def minmax01(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    lo, hi = s.min(skipna=True), s.max(skipna=True)
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return s * 0
    return (s - lo) / (hi - lo)


def ridge_predict(df: pd.DataFrame, payload: dict) -> np.ndarray:
    features = payload["features"]
    mu = np.array(payload["x_mean"], dtype=float)
    sd = np.array(payload["x_std"], dtype=float)
    w = np.array(payload["intercept_and_weights"], dtype=float)

    X_raw = df[features].to_numpy(dtype=float)
    X = (X_raw - mu) / sd
    X_i = np.c_[np.ones(len(X)), X]
    return X_i @ w


def main() -> None:
    if not CURRENT_PATH.exists():
        raise FileNotFoundError(f"Missing current candidates file: {CURRENT_PATH}")
    if not STANDINGS_PATH.exists():
        raise FileNotFoundError(f"Missing team standings file: {STANDINGS_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}. Run train_mvp_ridge_np first.")

    df = pd.read_csv(CURRENT_PATH)
    st = pd.read_csv(STANDINGS_PATH)

    payload = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    features = payload["features"]

    # Join standings
    df["team_name_lookup"] = df["team"].map(TEAM_ABBR_TO_NAME)
    st["team_name_lookup"] = st["team_name"].astype(str)

    merged = df.merge(
        st[["team_name_lookup", "win_pct", "conf_rank", "conference"]],
        on="team_name_lookup",
        how="left"
    ).rename(columns={
        "win_pct": "team_win_pct",
        "conf_rank": "team_conf_rank",
        "conference": "team_conference",
    })

    merged["team_win_pct"] = merged["team_win_pct"].fillna(0.5)
    merged["team_conf_rank"] = merged["team_conf_rank"].fillna(8)

    # Ensure numeric features
    for c in features:
        merged[c] = pd.to_numeric(merged[c], errors="coerce")
    merged = merged.dropna(subset=features).copy()

    merged["ridge_pred_vote_share"] = ridge_predict(merged, payload)

    # Hybrid score: ridge prediction + team context
    z_win = zscore(merged["team_win_pct"])
    z_seed = zscore(-pd.to_numeric(merged["team_conf_rank"], errors="coerce"))

    merged["mvp_score"] = (
        0.70 * minmax01(merged["ridge_pred_vote_share"]) +
        0.20 * minmax01(z_win) +
        0.10 * minmax01(z_seed)
    )

    merged = merged.sort_values("mvp_score", ascending=False).reset_index(drop=True)
    merged["mvp_rank"] = merged.index + 1

    def tier(r: int) -> str:
        if r <= 3:
            return "Top 3"
        if r <= 5:
            return "Top 5"
        if r <= 10:
            return "6-10"
        return "Outside"

    merged["mvp_tier"] = merged["mvp_rank"].apply(tier)

    # Top 10 + movers
    web_current_top10 = merged.head(10).copy()
    web_current_top10["rank"] = web_current_top10["mvp_rank"].astype(int)
    web_current_top10["rank_delta"] = np.nan
    web_current_top10["mover_label"] = "NEW"

    # Movers (requires prior)
    if not PRIOR_TOP10_PATH.exists():
        msg = f"[WARN] Prior Top10 not found at {PRIOR_TOP10_PATH}. Movers cannot be computed."
        if FAIL_ON_MISSING_PRIOR:
            raise FileNotFoundError(msg + " (Set FAIL_ON_MISSING_PRIOR=0 to skip movers.)")
        print(msg + " Proceeding without movers.")
    else:
        prior_df = pd.read_csv(PRIOR_TOP10_PATH)

        required = {"player_key", "rank"}
        missing = required - set(prior_df.columns)
        if missing:
            raise ValueError(f"[ERROR] Prior file {PRIOR_TOP10_PATH} missing columns: {sorted(missing)}")

        prior_map = (
            prior_df[["player_key", "rank"]]
            .dropna()
            .set_index("player_key")["rank"]
            .to_dict()
        )

        def mover(row):
            prev_rank = prior_map.get(row["player_key"])
            if prev_rank is None:
                return np.nan, "NEW"
            delta = int(prev_rank) - int(row["rank"])
            if delta > 0:
                return delta, f"↑{delta}"
            if delta < 0:
                return delta, f"↓{abs(delta)}"
            return 0, "—"

        movers = web_current_top10.apply(lambda r: pd.Series(mover(r)), axis=1)
        web_current_top10[["rank_delta", "mover_label"]] = movers

        # Helpful warning if prior == current (often indicates prior source issue)
        try:
            cur_cmp = web_current_top10[["player_key", "rank"]].astype(str).reset_index(drop=True)
            pri_cmp = prior_df[["player_key", "rank"]].astype(str).reset_index(drop=True)
            if cur_cmp.equals(pri_cmp):
                print("[WARN] Prior Top10 appears identical to current Top10 (movement will be zero). Verify prior source.")
        except Exception as e:
            print(f"[WARN] Could not compare prior vs current: {e}")

    # Meta
    season = str(merged["season"].iloc[0])
    season_end_year = int(merged["season_end_year"].iloc[0])
    last_updated_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    web_meta = pd.DataFrame([{
        "season": season,
        "season_end_year": season_end_year,
        "week_label_current": "",
        "week_label_prior": "",
        "last_updated_utc": last_updated_utc,
        "model_version": "ridge_np_v1_hybrid_team",
        "notes": "Ridge (numpy) predicts vote_share; blended with team win% + conf rank."
    }])

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    out_all = OUT_DIR / "model_all_candidates.csv"
    merged.to_csv(out_all, index=False, encoding="utf-8-sig")

    out_top10 = OUT_DIR / "web_current_top10.csv"
    cols_top10 = [
        "season", "season_end_year",
        "rank", "player_key", "player_name", "team",
        "mvp_score", "ridge_pred_vote_share",
        "team_win_pct", "team_conf_rank", "team_conference",
        "pts", "trb", "ast", "stl", "blk", "games_played",
        "rank_delta", "mover_label",
        "last_updated_utc"
    ]

    # Ensure last_updated_utc exists in the top10 frame
    web_current_top10["last_updated_utc"] = last_updated_utc

    web_current_top10[cols_top10].to_csv(out_top10, index=False, encoding="utf-8-sig")

    out_meta = OUT_DIR / "web_meta.csv"
    web_meta.to_csv(out_meta, index=False, encoding="utf-8-sig")

    # Optional local snapshot (for dev). In GitHub Actions, prior should come from Pages/seed, not from this run.
    if WRITE_LOCAL_PRIOR_SNAPSHOT:
        snap_path = OUT_DIR / "web_prior_top10_snapshot.csv"
        web_current_top10[["player_key", "rank"]].to_csv(snap_path, index=False, encoding="utf-8-sig")
        print(f"[SAVED] {snap_path}")

    print(f"[SAVED] {out_all}")
    print(f"[SAVED] {out_top10}")
    print(f"[SAVED] {out_meta}")
    print(f"[TOP10 #1] {web_current_top10['player_name'].iloc[0]}")


if __name__ == "__main__":
    main()
