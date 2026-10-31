"""
Score the live NBA season and write the weekly MVP board.

Pipeline
1. Every player's season-to-date per-game stats + team record from Basketball-Reference
   (src/nba_common.season_players, the same table builder training uses; live pages, no cache).
2. Week = 7-day blocks from opening night; stats_through = last date with completed games
   (Basketball-Reference schedule pages).
3. Eligibility: ≥ MIN_TEAM_GAME_SHARE of his team's games, ≥ MIN_MINUTES per game, and still
   able to reach ELIGIBLE_GAMES (65-game rule; Basketball-Reference's G undercounts, so 62).
4. Games played projected to an 82-game pace; every other feature is a per-game average.
5. Score with models/nba_mvp_ridge_np.json + the team blend (0.70 ridge, 0.20 win%, 0.10 conf rank),
   min-max'd over the eligible pool. docs/nba-2026-evaluation-and-plan.md has the backtest.

Outputs
  predictions/{season}/...           see src/publish/board.py
  data/processed/outputs/            legacy Google Sheet feed (unchanged columns): model_all_candidates.csv,
                                     web_current_top10.csv, web_meta.csv → src/publish/build_canva_flat_nba.py,
                                     publish_pages.py → GitHub Pages

Exits 0 without writing when there are no completed games yet, or none since the last board
(before opening night, after the regular season). In Actions it sets the step output skipped=true.

Run:
    python -m src.predict_current
    python -m src.predict_current --outdir <scratch> --no-publish   # board CSVs only, elsewhere
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from config import (
    ELIGIBLE_GAMES, LEGACY_OUT_DIR, MIN_MINUTES, MIN_TEAM_GAME_SHARE,
    MODEL_PATH, PREDICTIONS, SEASON_END_YEAR, SEASON_GAMES, TOP_N,
)
from src.data.bbref import fetch_html, parse_table
from src.data.normalize_voting_seed import normalize_name_to_key
from src.model.mvp_models import minmax
from src.nba_common import season_players
from src.publish.board import publish_board, season_week

MONTHS = ["october", "november", "december", "january", "february", "march", "april"]

# Legacy feed (Google Sheet IMPORTDATA of GitHub Pages). Column order is a contract.
LEGACY_TOP10_COLS = [
    "season", "season_end_year", "rank", "player_key", "player_name", "team",
    "mvp_score", "ridge_pred_vote_share", "team_win_pct", "team_conf_rank", "team_conference",
    "pts", "trb", "ast", "stl", "blk", "games_played", "rank_delta", "mover_label", "last_updated_utc",
]
LEGACY_ALL_COLS = [
    "season", "season_end_year", "player_name", "team", "games_played", "minutes_played", "pts", "trb", "ast",
    "stl", "blk", "fg_pct", "fg3_pct", "ft_pct", "player_key", "last_updated_utc", "team_name_lookup",
    "team_win_pct", "team_conf_rank", "team_conference", "ridge_pred_vote_share", "mvp_score", "mvp_rank", "mvp_tier",
]
TEAM_NAMES = {
    "ATL": "Hawks", "BOS": "Celtics", "BKN": "Nets", "CHA": "Hornets", "CHI": "Bulls", "CLE": "Cavaliers",
    "DAL": "Mavericks", "DEN": "Nuggets", "DET": "Pistons", "GSW": "Warriors", "HOU": "Rockets", "IND": "Pacers",
    "LAC": "Clippers", "LAL": "Lakers", "MEM": "Grizzlies", "MIA": "Heat", "MIL": "Bucks", "MIN": "Timberwolves",
    "NOP": "Pelicans", "NYK": "Knicks", "OKC": "Thunder", "ORL": "Magic", "PHI": "76ers", "PHX": "Suns",
    "POR": "Trail Blazers", "SAC": "Kings", "SAS": "Spurs", "TOR": "Raptors", "UTA": "Jazz", "WAS": "Wizards",
}


# ------------------------
# Schedule
# ------------------------
def season_dates(season: int) -> tuple[date | None, date | None]:
    """(opening night, last date with a completed game), from the monthly schedule pages."""
    first = through = None
    for month in MONTHS:
        try:
            rows = parse_table(fetch_html(f"/leagues/NBA_{season}_games-{month}.html", cache=False), "schedule")
        except requests.HTTPError:
            break   # month page not published yet
        days = [(datetime.strptime(r["date_game"], "%a, %b %d, %Y").date(), bool(r.get("home_pts")))
                for r in rows if r.get("date_game")]
        if days and first is None:
            first = min(d for d, _ in days)
        done = [d for d, played in days if played]
        if not done:
            break
        through = max(done)
    return first, through


# ------------------------
# Scoring
# ------------------------
def eligible(df: pd.DataFrame) -> pd.Series:
    share = df["g"] / df["team_gp"].replace(0, np.nan)
    can_reach = df["g"] + (SEASON_GAMES - df["team_gp"]) >= ELIGIBLE_GAMES
    return (share >= MIN_TEAM_GAME_SHARE) & (df["mp"] >= MIN_MINUTES) & can_reach


def score(players: pd.DataFrame) -> pd.DataFrame:
    """Eligible players, ranked: ridge_pred_vote_share (pace-projected games), mvp_score, mvp_rank."""
    payload = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    df = players[eligible(players)].copy()
    if df.empty:
        return df
    live = pd.DataFrame({
        "games_played": (df["g"] * SEASON_GAMES / df["team_gp"]).clip(upper=SEASON_GAMES),   # 82-game pace
        "minutes_played": df["mp"],
        **{c: df[c] for c in ("pts", "trb", "ast", "stl", "blk")},
        **{c: df[c].fillna(0.0) for c in ("fg_pct", "fg3_pct", "ft_pct")},
    })
    X = (live[payload["features"]].to_numpy(float) - np.array(payload["x_mean"])) / np.array(payload["x_std"])
    df["ridge_pred_vote_share"] = np.c_[np.ones(len(X)), X] @ np.array(payload["intercept_and_weights"])
    df["mvp_score"] = (0.70 * minmax(df["ridge_pred_vote_share"])
                       + 0.20 * minmax(df["team_win_pct"].fillna(0.5))
                       + 0.10 * minmax(-df["conf_rank"].fillna(8)))
    df = df.sort_values("mvp_score", ascending=False).reset_index(drop=True)
    df["mvp_rank"] = np.arange(1, len(df) + 1)
    return df


# ------------------------
# Legacy feed
# ------------------------
def write_legacy(scored: pd.DataFrame, board: pd.DataFrame, season: int, week: int) -> None:
    """model_all_candidates.csv, web_current_top10.csv, web_meta.csv with the old columns and
    old-style player_key (first_last, hyphens dropped). mover_label follows the board's movement."""
    LEGACY_OUT_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    label = f"{season - 1}-{str(season)[-2:]}"
    d = scored.copy()
    d["season"], d["season_end_year"] = label, season
    d["games_played"], d["minutes_played"] = d["g"].astype(int), d["mp"]
    d["player_key"] = d["player_name"].map(normalize_name_to_key)
    d["last_updated_utc"] = now
    d["team_name_lookup"] = d["team"].map(TEAM_NAMES)
    d["team_conf_rank"], d["team_conference"] = d["conf_rank"], d["conference"]
    d["mvp_tier"] = pd.cut(d["mvp_rank"], [0, 3, 5, 10, np.inf], labels=["Top 3", "Top 5", "6-10", "Outside"])
    d[LEGACY_ALL_COLS].to_csv(LEGACY_OUT_DIR / "model_all_candidates.csv", index=False, encoding="utf-8-sig")

    top = d.head(TOP_N).copy()
    top["rank"] = top["mvp_rank"]
    moves = board.head(TOP_N)
    top["rank_delta"] = pd.to_numeric(moves["rank_delta"], errors="coerce").to_numpy()
    top["mover_label"] = moves["movement"].replace({"–": "—"}).to_numpy()   # the Sheet expects an em dash
    top[LEGACY_TOP10_COLS].to_csv(LEGACY_OUT_DIR / "web_current_top10.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame([{
        "season": label, "season_end_year": season,
        "week_label_current": f"Week {week}", "week_label_prior": f"Week {week - 1}" if week > 1 else "",
        "last_updated_utc": now, "model_version": "ridge_np_v2_hybrid_team",
        "notes": "Ridge (numpy) predicts vote_share; blended with team win% + conf rank. Basketball-Reference stats.",
    }]).to_csv(LEGACY_OUT_DIR / "web_meta.csv", index=False, encoding="utf-8-sig")
    print(f"Legacy feed -> {LEGACY_OUT_DIR}")


# ------------------------
# Main
# ------------------------
def _skip(reason: str) -> None:
    print(f"[SKIP] {reason}")
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write("skipped=true\n")
    raise SystemExit(0)


def run(season: int = SEASON_END_YEAR, outdir: Path | None = None, publish: bool = True, force: bool = False) -> None:
    first, through = season_dates(season)
    if first is None or through is None:
        _skip(f"no completed {season - 1}-{str(season)[-2:]} games yet")
    latest = Path(outdir or PREDICTIONS / str(season)) / "board_mvp_latest.csv"
    if not force and latest.exists():
        prev = pd.read_csv(latest)
        if len(prev) and int(prev["season"].iloc[0]) == season and str(prev["stats_through"].iloc[0]) == through.isoformat():
            _skip(f"no new games since the last board (stats through {through})")

    players = season_players(season, cache=False)
    if players["team_gp"].isna().all():
        raise SystemExit("ERROR: no team records parsed from the standings page")
    scored = score(players)
    print(f"{len(players)} players, {len(scored)} eligible; week {season_week(first, through)}, stats through {through}")

    board = publish_board(scored, season, first, through, outdir=outdir, publish=publish)
    if outdir is None:
        write_legacy(scored, board, season, season_week(first, through))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")   # ↑/↓ arrows on a cp1252 console
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--no-publish", action="store_true", help="skip candidates, chips, graphics and caption")
    parser.add_argument("--force", action="store_true", help="rebuild even if no games were played since the last board")
    args = parser.parse_args()
    run(SEASON_END_YEAR, args.outdir, publish=not args.no_publish, force=args.force)


if __name__ == "__main__":
    main()
