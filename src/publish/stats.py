"""Board stat columns, display labels and formatting shared by chips, captions and the graphics."""

from __future__ import annotations

import pandas as pd

# Columns of board_mvp_latest.csv that describe performance; games is context, not a standout.
STAT_COLS = ["pts", "reb", "ast", "stl", "blk", "fg_pct", "fg3_pct", "ft_pct", "team_win_pct"]

STAT_LABELS = {
    "pts": "PPG", "reb": "RPG", "ast": "APG", "stl": "SPG", "blk": "BPG",
    "fg_pct": "FG%", "fg3_pct": "3P%", "ft_pct": "FT%", "team_win_pct": "Team W%",
    "games": "GP", "minutes": "MPG",
}

# Guards are compared with guards, forwards with forwards, centers with centers.
POSITION_GROUP = {"PG": "G", "SG": "G", "G": "G", "SF": "F", "PF": "F", "F": "F", "C": "C"}
ROLE = {"G": "guard", "F": "forward", "C": "center"}


def position_group(pos) -> str:
    return POSITION_GROUP.get(str(pos), "F")


def format_stat(col: str, value) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if col in ("team_win_pct", "fg_pct", "fg3_pct", "ft_pct"):
        return f"{v:.3f}".lstrip("0") if v < 1 else "1.000"
    return f"{v:.1f}"


def stat_line(player: pd.Series, cols: list[str] | None = None) -> str:
    """'PPG: 31.2, RPG: 5.1, ...' — only stats present for this player."""
    return ", ".join(f"{STAT_LABELS.get(c, c)}: {format_stat(c, player[c])}"
                     for c in (cols or STAT_COLS) if c in player and pd.notna(player[c]))
