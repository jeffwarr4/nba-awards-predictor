from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

CURRENT_PATH = REPO_ROOT / "data" / "processed" / "current" / "nba_mvp_current_candidates.csv"
OUT_DIR = REPO_ROOT / "data" / "processed" / "outputs"

def zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mu = s.mean(skipna=True)
    sd = s.std(skipna=True)
    if sd == 0 or np.isnan(sd):
        return s * 0
    return (s - mu) / sd

def compute_mvp_score_v1(df: pd.DataFrame) -> pd.Series:
    # Basic stat impact
    z_pts = zscore(df["pts"])
    z_trb = zscore(df["trb"])
    z_ast = zscore(df["ast"])
    z_stl = zscore(df["stl"])
    z_blk = zscore(df["blk"])

    # Efficiency (bounded 0-1); z-score them
    z_fg  = zscore(df["fg_pct"])
    z_fg3 = zscore(df["fg3_pct"])
    z_ft  = zscore(df["ft_pct"])

    # Availability (penalize low GP/MP)
    z_gp = zscore(df["games_played"])
    z_mp = zscore(df["minutes_played"])

    # Weights (v1): PTS drives most, then playmaking/rebounding, then defense, then availability, then efficiency
    score = (
        1.25 * z_pts +
        0.55 * z_ast +
        0.45 * z_trb +
        0.25 * z_stl +
        0.20 * z_blk +
        0.30 * z_gp +
        0.20 * z_mp +
        0.15 * z_fg +
        0.10 * z_fg3 +
        0.10 * z_ft
    )
    return score

def main() -> None:
    if not CURRENT_PATH.exists():
        raise FileNotFoundError(f"Missing current candidates file: {CURRENT_PATH}")

    df = pd.read_csv(CURRENT_PATH)

    # Basic validation
    needed = ["season", "season_end_year", "player_name", "player_key", "team",
              "games_played", "minutes_played", "pts", "trb", "ast", "stl", "blk",
              "fg_pct", "fg3_pct", "ft_pct", "last_updated_utc"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Current candidates missing columns: {missing}")

    df = df.copy()
    df["mvp_score"] = compute_mvp_score_v1(df)

    # Rank all candidates
    df = df.sort_values("mvp_score", ascending=False).reset_index(drop=True)
    df["mvp_rank"] = df.index + 1

    # Tier labels
    def tier(r: int) -> str:
        if r <= 3:
            return "Top 3"
        if r <= 5:
            return "Top 5"
        if r <= 10:
            return "6-10"
        return "Outside"

    df["mvp_tier"] = df["mvp_rank"].apply(tier)

    # Web outputs
    web_current_top10 = df.head(10).copy()
    web_current_top10 = web_current_top10.assign(
        rank=web_current_top10["mvp_rank"].astype(int),
        rank_delta=np.nan,
        mover_label="NEW"  # Step 8 initializes; Step 8.3 will compute deltas if prior exists
    )

    # Meta
    season = str(df["season"].iloc[0])
    season_end_year = int(df["season_end_year"].iloc[0])
    last_updated_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    web_meta = pd.DataFrame([{
        "season": season,
        "season_end_year": season_end_year,
        "week_label_current": "",   # we’ll fill this in Step 10 (weekly labeling)
        "week_label_prior": "",
        "last_updated_utc": last_updated_utc,
        "model_version": "v1_baseline_zscore",
        "notes": "Baseline MVP score from z-scored box stats + availability + shooting splits"
    }])

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save full ranked candidate list (maps to your `model_all_candidates`)
    out_all = OUT_DIR / "model_all_candidates.csv"
    df.to_csv(out_all, index=False)

    # Save current top 10 (maps to your `web_current_top10`)
    out_top10 = OUT_DIR / "web_current_top10.csv"
    cols_top10 = [
        "season", "season_end_year",
        "rank", "player_key", "player_name", "team",
        "mvp_score", "pts", "trb", "ast", "games_played",
        "rank_delta", "mover_label",
        "last_updated_utc"
    ]
    web_current_top10[cols_top10].to_csv(out_top10, index=False)

    # Save meta
    out_meta = OUT_DIR / "web_meta.csv"
    web_meta.to_csv(out_meta, index=False)

    print(f"[SAVED] {out_all}")
    print(f"[SAVED] {out_top10}")
    print(f"[SAVED] {out_meta}")
    print(f"[TOP10 #1] {web_current_top10['player_name'].iloc[0]}")

if __name__ == "__main__":
    main()
