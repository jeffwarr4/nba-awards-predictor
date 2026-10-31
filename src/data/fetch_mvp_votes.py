"""
Pull NBA MVP vote tables from Basketball-Reference into the voting seed
(data/raw/voting/nba_mvp_voting_history_seed.csv), in the seed's column format.

The seed was first saved as cp1252, which turned 15 accented names into '?'.
It is now written as UTF-8. When re-pulling seasons already in the seed, every
numeric column is compared with the old rows and any difference is printed, so
a re-pull can't silently change history.

Run:
    python -m src.data.fetch_mvp_votes                  # TRAIN_START..TRAIN_END, rewrite the seed
    python -m src.data.fetch_mvp_votes --seasons 2026   # just these seasons, merged into the seed
Then: python -m src.data.normalize_voting_seed && python -m src.data.build_training_dataset
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from config import TRAIN_END, TRAIN_START, VOTING_SEED_CSV
from src.data.bbref import fetch_html, parse_table

# seed column ← data-stat
SEED_COLUMNS = {
    "Rank": "rank", "Player": "player", "Age": "age", "Tm": "team_id", "First": "votes_first",
    "Pts Won": "points_won", "Pts Max": "points_max", "Share": "award_share", "G": "g",
    "MP": "mp_per_g", "PTS": "pts_per_g", "TRB": "trb_per_g", "AST": "ast_per_g",
    "STL": "stl_per_g", "BLK": "blk_per_g", "FG%": "fg_pct", "3P%": "fg3_pct", "FT%": "ft_pct",
    "WS": "ws", "WS/48": "ws_per_48",
}
NUMERIC = [c for c in SEED_COLUMNS if c not in ("Rank", "Player", "Tm")]


def season_label(end_year: int) -> str:
    return f"{end_year - 1}-{str(end_year)[-2:]}"


def fetch_season(end_year: int) -> pd.DataFrame:
    rows = parse_table(fetch_html(f"/awards/awards_{end_year}.html"), "mvp")
    if not rows:
        raise SystemExit(f"ERROR: no MVP table on awards_{end_year}.html")
    df = pd.DataFrame([{col: r.get(stat, "") for col, stat in SEED_COLUMNS.items()} | {"PlayerID": r.get("player_id", "")}
                       for r in rows])
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["season"] = season_label(end_year)
    return df


def compare(old: pd.DataFrame, new: pd.DataFrame) -> int:
    """Numeric differences between old and re-pulled rows, joined on season + PlayerID."""
    m = old.merge(new, on=["season", "PlayerID"], how="outer", suffixes=("_old", "_new"), indicator=True)
    diffs = 0
    for _, r in m[m["_merge"] != "both"].iterrows():
        print(f"  [DIFF] {r['season']} {r['PlayerID']}: only in {'old seed' if r['_merge'] == 'left_only' else 'new pull'}")
        diffs += 1
    both = m[m["_merge"] == "both"]
    for c in NUMERIC:
        a = pd.to_numeric(both[f"{c}_old"], errors="coerce")
        b = pd.to_numeric(both[f"{c}_new"], errors="coerce")
        bad = ~((a - b).abs().le(1e-6) | (a.isna() & b.isna()))
        for _, r in both[bad].iterrows():
            print(f"  [DIFF] {r['season']} {r['Player_new']} {c}: {r[f'{c}_old']} -> {r[f'{c}_new']}")
            diffs += 1
    renamed = both[both["Player_old"].astype(str) != both["Player_new"].astype(str)]
    for _, r in renamed.iterrows():
        print(f"  [NAME] {r['season']} {r['Player_old']!r} -> {r['Player_new']!r}")
    return diffs


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser()
    p.add_argument("--seasons", type=int, nargs="*", help="end years (default TRAIN_START..TRAIN_END)")
    seasons = p.parse_args().seasons or list(range(TRAIN_START, TRAIN_END + 1))

    new = pd.concat([fetch_season(y) for y in seasons], ignore_index=True)
    print(f"[FETCHED] {len(new)} vote-getters, {len(seasons)} season(s)")

    old = pd.DataFrame()
    if VOTING_SEED_CSV.exists():
        raw = VOTING_SEED_CSV.read_bytes()
        try:
            text_enc = "utf-8"; raw.decode("utf-8")
        except UnicodeDecodeError:
            text_enc = "cp1252"
        old = pd.read_csv(VOTING_SEED_CSV, encoding=text_enc)
        overlap = old[old["season"].isin(new["season"])]
        if len(overlap):
            diffs = compare(overlap, new[new["season"].isin(overlap["season"])])
            print(f"[CHECK] {diffs} numeric/row difference(s) vs the existing seed")
        old = old[~old["season"].isin(new["season"])]

    seed = pd.concat([old, new], ignore_index=True)
    seed["_end"] = seed["season"].str[:4].astype(int) + 1
    seed["_rank"] = seed["Rank"].astype(str).str.rstrip("T").astype(int)
    seed = seed.sort_values(["_end", "_rank", "Player"]).drop(columns=["_end", "_rank"])
    VOTING_SEED_CSV.parent.mkdir(parents=True, exist_ok=True)
    seed[list(SEED_COLUMNS) + ["PlayerID", "season"]].to_csv(VOTING_SEED_CSV, index=False, encoding="utf-8")
    print(f"[SAVED] {VOTING_SEED_CSV} ({len(seed)} rows, {seed['season'].nunique()} seasons, UTF-8)")


if __name__ == "__main__":
    main()
