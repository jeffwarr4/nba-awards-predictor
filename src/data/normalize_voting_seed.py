from __future__ import annotations

import re
import unicodedata
from pathlib import Path
import pandas as pd

# Repo root = .../nba-awards-predictor
REPO_ROOT = Path(__file__).resolve().parents[2]

RAW_PATH = REPO_ROOT / "data" / "raw" / "voting" / "nba_mvp_voting_history_seed.csv"
OUT_DIR = REPO_ROOT / "data" / "processed" / "voting"
OUT_PATH = OUT_DIR / "nba_mvp_voting_history_normalized.csv"

# Expected headers from your Basketball-Reference CSV exports
EXPECTED_COLUMNS = [
    "Rank", "Player", "Age", "Tm", "First", "Pts Won", "Pts Max", "Share",
    "G", "MP", "PTS", "TRB", "AST", "STL", "BLK",
    "FG%", "3P%", "FT%", "WS", "WS/48",
    "PlayerID", "season"
]


def normalize_name_to_key(name: str) -> str:
    """
    Create a stable, human-readable player_key from Player name.
    - ascii fold (Jokić -> Jokic)
    - lowercase
    - remove suffixes (jr/sr/ii/iii/iv/v)
    - remove punctuation
    - spaces -> underscores
    """
    if pd.isna(name):
        return ""

    s = str(name).strip()

    # Fold accents to ascii
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")

    s = s.lower()

    # Remove common suffixes as standalone tokens
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", s)

    # Remove punctuation (keep spaces)
    s = re.sub(r"[^a-z\s]", "", s)

    # Collapse whitespace -> underscore
    s = re.sub(r"\s+", "_", s.strip())
    return s


def parse_rank(rank_val) -> tuple[int, bool]:
    """
    Basketball-Reference sometimes shows ties like '12T'.
    We normalize:
      - mvp_rank: int (12)
      - is_tied_rank: bool (True)
    """
    s = str(rank_val).strip()
    is_tied = s.endswith("T")
    s = s[:-1] if is_tied else s
    # Some exports can include blanks or weird values; let it fail loudly
    return int(s), is_tied


def season_end_year_from_season_str(season_str: str) -> int:
    """
    Your season column looks like '1980-81'.
    End year = start + 1 => 1981.
    """
    s = str(season_str).strip()
    # Expect first 4 chars are start year
    start_year = int(s[:4])
    return start_year + 1


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Seed file not found: {RAW_PATH}")

    df = pd.read_csv(RAW_PATH, encoding="cp1252", encoding_errors="replace")


    # ---- Basic validation ----
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "Seed file is missing expected columns:\n"
            f"{missing}\n\n"
            f"Found columns:\n{list(df.columns)}"
        )

    # Keep raw rank for traceability
    df["rank_raw"] = df["Rank"].astype(str)

    # Normalize rank
    df[["mvp_rank", "is_tied_rank"]] = df["Rank"].apply(lambda x: pd.Series(parse_rank(x)))

    # Canonical player fields
    df["player_name"] = df["Player"].astype(str).str.strip()
    df["player_key"] = df["player_name"].apply(normalize_name_to_key)

    # Keep season as-is (e.g., '1980-81'), derive season_end_year
    df["season"] = df["season"].astype(str).str.strip()
    df["season_end_year"] = df["season"].apply(season_end_year_from_season_str)

    # Rename stats columns to snake_case
    df = df.rename(
        columns={
            "Tm": "team",
            "Age": "age",
            "First": "first_place_votes",
            "Pts Won": "points_won",
            "Pts Max": "points_max",
            "Share": "vote_share",
            "G": "games_played",
            "MP": "minutes_played",
            "PTS": "pts",
            "TRB": "trb",
            "AST": "ast",
            "STL": "stl",
            "BLK": "blk",
            "FG%": "fg_pct",
            "3P%": "fg3_pct",
            "FT%": "ft_pct",
            "WS": "ws",
            "WS/48": "ws_per_48",
            "PlayerID": "br_player_id",
        }
    )

    # Optional: ensure numeric types where appropriate (won't crash if blanks exist)
    num_cols = [
        "age", "first_place_votes", "points_won", "points_max", "vote_share",
        "games_played", "minutes_played", "pts", "trb", "ast", "stl", "blk",
        "fg_pct", "fg3_pct", "ft_pct", "ws", "ws_per_48"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Helpful duplicate check: same season + same br_player_id should be unique
    # (If BR player id missing for some reason, we fallback to player_key)
    dup_key = ["season", "br_player_id"]
    if df["br_player_id"].isna().any():
        dup_key = ["season", "player_key"]

    dups = df.duplicated(subset=dup_key, keep=False)
    if dups.any():
        dup_rows = df.loc[dups, ["season", "player_name", "br_player_id", "mvp_rank"]].sort_values(["season", "mvp_rank"])
        raise ValueError(
            "Duplicate rows detected for the same player in the same season.\n"
            f"{dup_rows.to_string(index=False)}"
        )

    # Output column order (stable contract)
    out_cols = [
        "season",
        "season_end_year",
        "player_name",
        "player_key",
        "br_player_id",
        "team",
        "age",
        "rank_raw",
        "mvp_rank",
        "is_tied_rank",
        "first_place_votes",
        "points_won",
        "points_max",
        "vote_share",
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

    df_out = (
        df[out_cols]
        .sort_values(["season_end_year", "mvp_rank", "player_name"])
        .reset_index(drop=True)
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUT_PATH, index=False)

    print(f"[SAVED] {OUT_PATH}")
    print(f"[ROWS] {len(df_out)}")
    print(f"[SEASONS] {df_out['season'].nunique()}  ({df_out['season'].min()} -> {df_out['season'].max()})")


if __name__ == "__main__":
    main()
