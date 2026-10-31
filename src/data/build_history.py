"""
Every player-season TRAIN_START..TRAIN_END from Basketball-Reference (per-game,
advanced, team record) joined to the MVP vote → config.HISTORY_CSV.

The backtest has to score every player, not just vote-getters: ranking only the
players who got votes hands the model the hardest part of the problem.

~3 throttled fetches per season (cached in data/raw/bbref_cache/), ~8 minutes cold.

Run:
    python -m src.data.build_history
"""

from __future__ import annotations

import sys

import pandas as pd

from config import HISTORY_CSV, TRAIN_END, TRAIN_START
from src.nba_common import season_players

VOTES_CSV = HISTORY_CSV.parent.parent / "voting" / "nba_mvp_voting_history_normalized.csv"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    seasons = []
    for y in range(TRAIN_START, TRAIN_END + 1):
        df = season_players(y)
        seasons.append(df)
        print(f"  {y}: {len(df)} players, team record missing for {df['team_gp'].isna().sum()}")
    hist = pd.concat(seasons, ignore_index=True)

    votes = pd.read_csv(VOTES_CSV)[["season_end_year", "br_player_id", "mvp_rank", "vote_share",
                                    "points_won", "first_place_votes"]]
    votes = votes.rename(columns={"br_player_id": "player_id"})
    hist = hist.merge(votes, on=["season_end_year", "player_id"], how="left", indicator=True)
    unmatched = votes.merge(hist[["season_end_year", "player_id"]], how="left", indicator="m")
    unmatched = unmatched[unmatched["m"] == "left_only"]
    if len(unmatched):
        raise SystemExit(f"ERROR: {len(unmatched)} vote-getters not found in player tables:\n{unmatched.to_string()}")
    hist = hist.drop(columns="_merge")
    hist["vote_share"] = hist["vote_share"].fillna(0.0)
    hist["got_votes"] = (hist["vote_share"] > 0).astype(int)
    hist["is_top5"] = (hist["mvp_rank"] <= 5).fillna(False).astype(int)

    HISTORY_CSV.parent.mkdir(parents=True, exist_ok=True)
    hist.to_csv(HISTORY_CSV, index=False, encoding="utf-8")
    print(f"[SAVED] {HISTORY_CSV} ({len(hist):,} player-seasons, {hist['got_votes'].sum()} vote-getters)")


if __name__ == "__main__":
    main()
