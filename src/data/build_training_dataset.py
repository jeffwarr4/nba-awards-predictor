from __future__ import annotations

from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

VOTING_NORM_PATH = REPO_ROOT / "data" / "processed" / "voting" / "nba_mvp_voting_history_normalized.csv"
OUT_DIR = REPO_ROOT / "data" / "processed" / "training"
OUT_PATH = OUT_DIR / "nba_mvp_training_dataset.csv"

# Features we can rely on from the BR MVP voting table (already in your seed)
FEATURE_COLS = [
    "age",
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
    "ws",
    "ws_per_48",
]

LABEL_COLS = [
    "mvp_rank",
    "vote_share",
    "points_won",
    "first_place_votes",
]

ID_COLS = [
    "season",
    "season_end_year",
    "player_name",
    "player_key",
    "br_player_id",
    "team",
]

def main() -> None:
    if not VOTING_NORM_PATH.exists():
        raise FileNotFoundError(f"Missing normalized voting file: {VOTING_NORM_PATH}")

    df = pd.read_csv(VOTING_NORM_PATH)

    # Ensure required columns exist
    required = set(ID_COLS + FEATURE_COLS + LABEL_COLS)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Training build missing columns: {missing}")

    # Basic cleaning: remove rows missing key label/rank
    df = df.dropna(subset=["mvp_rank"]).copy()

    # Create a binary label for "Top 5 finish" (useful later if we want MLB-style probability too)
    df["is_top5"] = (df["mvp_rank"] <= 5).astype(int)

    # Also create "is_top10" (for ranking evaluation)
    df["is_top10"] = (df["mvp_rank"] <= 10).astype(int)

    # Keep a clean modeling frame
    out_cols = ID_COLS + FEATURE_COLS + LABEL_COLS + ["is_top5", "is_top10"]
    df_out = df[out_cols].copy()

    # Sort for readability
    df_out = df_out.sort_values(["season_end_year", "mvp_rank", "player_name"]).reset_index(drop=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUT_PATH, index=False)

    print(f"[SAVED] {OUT_PATH}")
    print(f"[ROWS] {len(df_out)}")
    print(f"[SEASONS] {df_out['season'].nunique()}  ({df_out['season'].min()} -> {df_out['season'].max()})")
    print(f"[TOP5 ROWS] {df_out['is_top5'].sum()}")

if __name__ == "__main__":
    main()
