"""
Weekly MVP board: the scored current-season frame from src/predict_current.py
→ predictions/{season}/ CSVs, movement, headshot candidates, then chips, graphics, caption.

Port of nhl-awards-predictor/publish/board.py.

Outputs (predictions/{season}/)
  top10_mvp_latest.csv    ← top TOP_N, BOARD_COLS (ends with chips, why_blurb)
  board_mvp_latest.csv    ← BOARD_DEPTH deep, same columns; the website will read it from GitHub
  {stem}_{ts}.csv         ← archives (ARCHIVES_TO_KEEP most recent of each)
  .prev_board_mvp.csv     ← last week's 30-deep board; drives rank_delta / movement
  chip_data/, graphics/   ← see generate_chips.py, render_graphics.py, generate_captions.py
predictions/nba_candidates.csv ← everyone on the board, for stinger-assets' headshot sync
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from config import ARCHIVES_TO_KEEP, BOARD_DEPTH, BUBBLE_RANKS, CANDIDATES_CSV, PREDICTIONS, TOP_N

# Column order of top10_mvp_latest.csv / board_mvp_latest.csv (a contract: the website
# will read board_mvp_latest.csv by these names). chips (top 3, pipe-separated) then why_blurb end it.
BOARD_COLS = [
    "player_key", "rank", "rank_delta", "movement", "player_name", "team", "position",
    "model_score",
    "games", "team_games", "minutes", "pts", "reb", "ast", "stl", "blk", "fg_pct", "fg3_pct", "ft_pct",
    "team_record", "team_win_pct", "conf_rank", "conference",
    "season", "week", "stats_through", "updated_utc",
    "chips", "why_blurb",
]


# ------------------------
# Keys
# ------------------------
def headshot_key(name: str, team: str) -> str:
    """firstname_lastname_team slug for stinger-assets headshots.
    Must match safe_filename(f"{name} {team}") in stinger-assets/sync_espn_headshots.py:
    accents stripped, apostrophes and periods dropped, hyphens split.
    e.g. 'Shai Gilgeous-Alexander', 'OKC' → 'shai_gilgeous_alexander_okc'; 'Nikola Jokić', 'DEN' → 'nikola_jokic_den'
    """
    n = unicodedata.normalize("NFKD", f"{name} {team}")
    n = "".join(ch for ch in n if not unicodedata.combining(ch)).lower()
    n = n.replace("'", "").replace(".", "").replace("-", " ")
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", n)).strip("_")


def season_week(first_game: date, through: date) -> int:
    """Week N of the season, counted in 7-day blocks from opening night."""
    return (through - first_game).days // 7 + 1


# ------------------------
# Board
# ------------------------
def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def build_board(scored: pd.DataFrame, season: int, week: int, through: date) -> pd.DataFrame:
    """BOARD_DEPTH rows from the ranked frame, display columns only (actual stats, not pace)."""
    df = scored.sort_values("mvp_rank").head(BOARD_DEPTH).reset_index(drop=True)
    w, l = (_num(df[c]).fillna(0).astype(int) for c in ("W", "L"))
    board = pd.DataFrame({
        "rank": df["mvp_rank"].astype(int),
        "player_name": df["player_name"],
        "team": df["team"],
        "position": df["pos"],
        "model_score": _num(df["mvp_score"]).round(6),
        "games": _num(df["g"]).astype(int),
        "team_games": _num(df["team_gp"]).astype(int),
        "minutes": _num(df["mp"]).round(1),
        "pts": _num(df["pts"]).round(1),
        "reb": _num(df["trb"]).round(1),
        "ast": _num(df["ast"]).round(1),
        "stl": _num(df["stl"]).round(1),
        "blk": _num(df["blk"]).round(1),
        "fg_pct": _num(df["fg_pct"]).round(3),
        "fg3_pct": _num(df["fg3_pct"]).round(3),
        "ft_pct": _num(df["ft_pct"]).round(3),
        "team_record": [f"{a}-{b}" for a, b in zip(w, l)],
        "team_win_pct": _num(df["team_win_pct"]).round(3),
        "conf_rank": _num(df["conf_rank"]).astype("Int64"),
        "conference": df["conference"],
    })
    keys = [headshot_key(n, t) for n, t in zip(board["player_name"], board["team"])]
    # Two players with one name on one team: the lower-ranked one gets a position suffix
    # so movement and chips stay per-player.
    seen: set[str] = set()
    for i, (k, pos) in enumerate(zip(keys, board["position"])):
        if k in seen:
            keys[i] = f"{k}_{str(pos).lower()}"
        seen.add(keys[i])
    board["player_key"] = keys
    board["season"] = season
    board["week"] = week
    board["stats_through"] = through.isoformat()
    board["updated_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    board["chips"] = ""
    board["why_blurb"] = ""
    return board


def add_movement(board: pd.DataFrame, prev_path: Path) -> pd.DataFrame:
    """rank_delta: +up / -down vs last week ("" if NEW); movement: ↑3 / ↓2 / – / NEW.
    No snapshot yet (the first 2026-27 board): everyone is NEW."""
    board = board.copy()
    prev = {}
    if prev_path.exists():
        p = pd.read_csv(prev_path)
        prev = dict(zip(p["player_key"], p["rank"].astype(int)))
    deltas, moves = [], []
    for key, rank in zip(board["player_key"], board["rank"]):
        if key not in prev:
            deltas.append(""); moves.append("NEW")
        else:
            d = prev[key] - int(rank)
            deltas.append(d)
            moves.append(f"↑{d}" if d > 0 else (f"↓{abs(d)}" if d < 0 else "–"))
    board["rank_delta"] = deltas
    board["movement"] = moves
    return board


def roll_prev_snapshot(latest_path: Path, prev_path: Path, season: int, week: int) -> bool:
    """The movement baseline is the last board from an EARLIER week of this season, so a
    same-week rerun (manual dispatch, a fix) keeps last week's arrows. True if rolled."""
    if not latest_path.exists():
        return False
    latest = pd.read_csv(latest_path)
    if len(latest) and int(latest["season"].iloc[0]) == season and int(latest["week"].iloc[0]) < week:
        latest[["player_key", "rank", "week"]].to_csv(prev_path, index=False)
        return True
    return False


def guard_same_week(board: pd.DataFrame, latest_path: Path, season: int, week: int) -> None:
    """Never replace this week's published board with a shorter one."""
    if not latest_path.exists():
        return
    latest = pd.read_csv(latest_path)
    if (len(latest) and int(latest["season"].iloc[0]) == season and int(latest["week"].iloc[0]) == week
            and len(board) < len(latest)):
        raise SystemExit(f"ERROR: new week-{week} board has {len(board)} rows; the published one has "
                         f"{len(latest)}. Board not written.")


def write_candidates(board: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"sport": "NBA", "player_name": board["player_name"],
                  "team": board["team"], "player_key": board["player_key"]}) \
      .drop_duplicates("player_key").to_csv(path, index=False)
    print(f"Headshot candidates -> {path} ({len(board)} players)")


def write_boards(board: pd.DataFrame, outdir: Path) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    board = board[BOARD_COLS]
    for df, stem in ((board.head(TOP_N), "top10_mvp"), (board, "board_mvp")):
        df.to_csv(outdir / f"{stem}_latest.csv", index=False)
        df.to_csv(outdir / f"{stem}_{ts}.csv", index=False)
        for old in sorted(outdir.glob(f"{stem}_2*.csv"))[:-ARCHIVES_TO_KEEP]:
            old.unlink()


# ------------------------
# Main
# ------------------------
def publish_board(scored: pd.DataFrame, season: int, first_game: date, through: date,
                  outdir: Path | None = None, publish: bool = True) -> pd.DataFrame:
    """Board CSVs + movement + candidates; then chips, graphics and caption when publish."""
    default_out = PREDICTIONS / str(season)
    outdir = Path(outdir or default_out)
    outdir.mkdir(parents=True, exist_ok=True)
    # A dry run into another folder must not overwrite the live candidates feed.
    candidates = CANDIDATES_CSV if outdir.resolve() == default_out.resolve() else outdir / CANDIDATES_CSV.name

    week = season_week(first_game, through)
    board = build_board(scored, season, week, through)
    if len(board) < BUBBLE_RANKS[1]:
        # Fail loudly — never replace the published board with one too short for the bubble slide.
        raise SystemExit(f"ERROR: only {len(board)} eligible players for {season} week {week}; board not written.")

    latest, prev = outdir / "board_mvp_latest.csv", outdir / ".prev_board_mvp.csv"
    guard_same_week(board, latest, season, week)
    roll_prev_snapshot(latest, prev, season, week)
    board = add_movement(board, prev)
    write_boards(board, outdir)

    print(f"\n=== {season - 1}–{str(season)[-2:]} NBA MVP — week {week}, stats through {through} ===")
    show = ["rank", "movement", "player_name", "team", "position", "games", "team_games",
            "pts", "reb", "ast", "team_record"]
    print(board.head(TOP_N)[show].to_string(index=False))
    print(f"Saved -> {latest} ({len(board)} deep)")

    if publish:
        write_candidates(board, candidates)
        board = publish_outputs(board, outdir, season, week, through)
    return board


def publish_outputs(board: pd.DataFrame, outdir: Path, season: int, week: int, through: date) -> pd.DataFrame:
    """Chips + bubble Whys (needs ANTHROPIC_API_KEY) → board CSVs get chips + why_blurb →
    board and bubble graphics → carousel caption (needs key). Returns the board with chips."""
    from src.publish.generate_captions import generate_caption_file
    from src.publish.generate_chips import generate_chips, load_chips
    from src.publish.render_graphics import render_board, render_bubble_graphic

    gfx = outdir / "graphics"
    chip_df = generate_chips(board, outdir)
    if chip_df.empty:
        # No key or the API failed: keep this week's earlier chips rather than blanking them.
        chip_df = load_chips(outdir, week)
    if not chip_df.empty:
        whys = dict(zip(chip_df["player_key"], chip_df["Why"]))
        chips = dict(zip(chip_df["player_key"], chip_df["Chip"].fillna("")))
        board = board.assign(chips=board["player_key"].map(chips).fillna(""),
                             why_blurb=board["player_key"].map(whys).fillna(""))
        for df, name in ((board.head(TOP_N), "top10_mvp_latest.csv"), (board, "board_mvp_latest.csv")):
            df[BOARD_COLS].to_csv(outdir / name, index=False)

    render_board(board.head(TOP_N), chip_df, gfx, season, week, through)
    # The bubble slide is secondary: a failure there must not cost the main board's caption.
    try:
        render_bubble_graphic(board, chip_df, gfx, season, week, through)
    except Exception as e:
        print(f"  [WARN] Bubble graphic failed: {e}")
    generate_caption_file(board, chip_df, gfx, season, week)
    return board
