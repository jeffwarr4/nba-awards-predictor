"""
Shared Basketball-Reference cleaning + feature engineering for the MVP pipeline.

src/data/build_history.py (training rows) and live scoring both build their player
tables through season_players(), so a feature can't be computed one way in training
and another way live.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.bbref import fetch_html, parse_table

# Basketball-Reference code → the NBA's own code, where they differ (current franchises).
# Applied to the display/key `team` only; joins use the raw code (team_bbref).
NBA_TEAM_CODES = {"BRK": "BKN", "CHO": "CHA", "PHO": "PHX"}

MULTI_TEAM = ("TOT",)   # plus "2TM", "3TM", ...

PER_GAME = {
    "games": "g", "games_started": "gs", "mp_per_g": "mp", "pts_per_g": "pts", "trb_per_g": "trb",
    "ast_per_g": "ast", "stl_per_g": "stl", "blk_per_g": "blk", "tov_per_g": "tov",
    "fg_pct": "fg_pct", "fg3_pct": "fg3_pct", "ft_pct": "ft_pct", "efg_pct": "efg_pct",
    "fga_per_g": "fga", "fta_per_g": "fta",
}
ADVANCED = {
    "mp": "mp_total", "per": "per", "ts_pct": "ts_pct", "usg_pct": "usg_pct",
    "ows": "ows", "dws": "dws", "ws": "ws", "ws_per_48": "ws_per_48",
    "obpm": "obpm", "dbpm": "dbpm", "bpm": "bpm", "vorp": "vorp",
}


def _is_multi(team: str) -> bool:
    return team in MULTI_TEAM or (len(team) == 3 and team[0].isdigit() and team.endswith("TM"))


def collapse_multi_team(rows: list[dict]) -> list[dict]:
    """One row per player. A traded player keeps his season-total row (2TM/TOT),
    labelled with the team of his last per-team row (Basketball-Reference lists them
    in the order played), so his team record and headshot key are his current club."""
    by_player: dict[str, list[dict]] = {}
    for r in rows:
        if r.get("player_id"):
            by_player.setdefault(r["player_id"], []).append(r)
    out = []
    for pid, rs in by_player.items():
        total = [r for r in rs if _is_multi(r.get("team_name_abbr", ""))]
        if total:
            singles = [r for r in rs if not _is_multi(r.get("team_name_abbr", ""))]
            row = dict(total[0])
            row["team_name_abbr"] = singles[-1]["team_name_abbr"] if singles else ""
            row["n_teams"] = len(singles)
        else:
            row = dict(rs[0])
            row["n_teams"] = 1
        out.append(row)
    return out


def _numeric(df: pd.DataFrame, cols) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].replace("", np.nan), errors="coerce")
    return df


def team_standings(end_year: int, cache: bool = True) -> pd.DataFrame:
    """One row per team: team_bbref, W, L, team_gp, team_win_pct, srs, conference, conf_rank
    (rank by win% within the conference; ties share the better rank)."""
    html = fetch_html(f"/leagues/NBA_{end_year}_standings.html", cache=cache)
    parts = []
    for conf in ("E", "W"):
        t = pd.DataFrame(parse_table(html, f"divs_standings_{conf}"))
        if t.empty:
            continue
        t = t[t.get("team_name_href", pd.Series(dtype=str)).notna() & t["team_name_href"].ne("")]
        t["conference"] = "East" if conf == "E" else "West"
        parts.append(t)
    if not parts:
        return pd.DataFrame(columns=["team_bbref", "W", "L", "team_gp", "team_win_pct", "srs", "conference", "conf_rank"])
    st = pd.concat(parts, ignore_index=True).rename(
        columns={"team_name_href": "team_bbref", "wins": "W", "losses": "L", "srs": "srs"})
    st = _numeric(st, ["W", "L", "srs"])
    st["team_gp"] = st["W"] + st["L"]
    st["team_win_pct"] = st["W"] / st["team_gp"].replace(0, np.nan)
    st["conf_rank"] = st.groupby("conference")["team_win_pct"].rank(ascending=False, method="min")
    return st[["team_bbref", "W", "L", "team_gp", "team_win_pct", "srs", "conference", "conf_rank"]]


def season_players(end_year: int, cache: bool = True) -> pd.DataFrame:
    """Every player in a season: per-game + advanced + his (last) team's record.
    Columns: player_id, player_name, age, pos, team_bbref, team, n_teams, PER_GAME/ADVANCED
    renames, W, L, team_gp, team_win_pct, srs, conference, conf_rank, season_end_year."""
    pg_rows = collapse_multi_team(parse_table(fetch_html(f"/leagues/NBA_{end_year}_per_game.html", cache=cache), "per_game_stats"))
    adv_rows = collapse_multi_team(parse_table(fetch_html(f"/leagues/NBA_{end_year}_advanced.html", cache=cache), "advanced"))
    if not pg_rows:
        raise SystemExit(f"ERROR: no per-game table for {end_year}")

    pg = pd.DataFrame(pg_rows).rename(columns={"name_display": "player_name", "team_name_abbr": "team_bbref", **PER_GAME})
    pg = pg[["player_id", "player_name", "age", "pos", "team_bbref", "n_teams", *PER_GAME.values()]]
    adv = pd.DataFrame(adv_rows).rename(columns=ADVANCED)
    adv = adv[["player_id", *[c for c in ADVANCED.values() if c in adv.columns]]]
    df = pg.merge(adv, on="player_id", how="left")
    df = _numeric(df, ["age", *PER_GAME.values(), *ADVANCED.values()])

    df = df.merge(team_standings(end_year, cache=cache), on="team_bbref", how="left")
    df["team"] = df["team_bbref"].map(lambda t: NBA_TEAM_CODES.get(t, t))
    df["pos"] = df["pos"].astype(str).str.split("-").str[0]   # 'SG-PG' → 'SG'
    df["season_end_year"] = end_year
    return df
