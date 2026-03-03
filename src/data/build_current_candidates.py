from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import re
import unicodedata
import pandas as pd
import time
import random

from requests import Session
from requests.exceptions import ReadTimeout, ConnectionError, Timeout

from nba_api.stats.endpoints import leaguedashplayerstats
from nba_api.stats.library.parameters import SeasonTypeAllStar
from nba_api.stats.library.http import NBAStatsHTTP

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "data" / "processed" / "current"


def normalize_name_to_key(name: str) -> str:
    if pd.isna(name):
        return ""
    s = str(name).strip()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", s)
    s = re.sub(r"[^a-z\s]", "", s)
    s = re.sub(r"\s+", "_", s.strip())
    return s


def season_label_from_end_year(end_year: int) -> str:
    return f"{end_year-1}-{str(end_year)[-2:]}"


def make_session() -> Session:
    """
    Create a browser-like session for stats.nba.com.
    This is the #1 fix for hanging/timeouts.
    """
    s = Session()
    s.headers.update({
        # Browser-ish UA; stats.nba.com is picky
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nba.com/",
        "Origin": "https://www.nba.com",
        "Connection": "keep-alive",
    })
    return s


def main(season_end_year: int = 2026) -> None:
    """
    Pull current-season per-game boxscore + shooting % from nba_api.
    season_end_year=2026 corresponds to season '2025-26'.
    """
    season = season_label_from_end_year(season_end_year)

    # Attach our hardened session to nba_api's HTTP layer
    # nba_api uses NBAStatsHTTP internally; we can override its session
    NBAStatsHTTP._session = make_session()

    max_tries = 6
    base_sleep = 4

    # Use shorter timeouts per attempt; retries are better than one giant hang.
    # nba_api passes this "timeout" to requests. If a tuple is supported in your version,
    # you can switch to (connect, read). If not, it will use this as total seconds.
    timeout_seconds = 45

    last_err = None
    df = None

    for attempt in range(1, max_tries + 1):
        try:
            endpoint = leaguedashplayerstats.LeagueDashPlayerStats(
                season=season,
                season_type_all_star=SeasonTypeAllStar.regular,
                per_mode_detailed="PerGame",
                timeout=timeout_seconds,
            )
            df = endpoint.get_data_frames()[0].copy()
            last_err = None
            break
        except (ReadTimeout, ConnectionError, Timeout, TimeoutError) as e:
            last_err = e
            # exponential backoff + jitter
            sleep_s = base_sleep * (2 ** (attempt - 1)) + random.uniform(0, 2)
            print(f"[WARN] stats.nba.com request failed (attempt {attempt}/{max_tries}): {e}")
            print(f"[WARN] sleeping {sleep_s:.1f}s then retrying...")
            time.sleep(sleep_s)

    if last_err is not None or df is None:
        raise RuntimeError(f"Failed to fetch LeagueDashPlayerStats after {max_tries} attempts") from last_err

    needed = {
        "PLAYER_NAME", "TEAM_ABBREVIATION", "GP", "MIN",
        "PTS", "REB", "AST", "STL", "BLK",
        "FG_PCT", "FG3_PCT", "FT_PCT"
    }
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"nba_api response missing expected columns: {missing}")

    out = pd.DataFrame({
        "season": season,
        "season_end_year": season_end_year,
        "player_name": df["PLAYER_NAME"].astype(str),
        "team": df["TEAM_ABBREVIATION"].astype(str),
        "games_played": pd.to_numeric(df["GP"], errors="coerce"),
        "minutes_played": pd.to_numeric(df["MIN"], errors="coerce"),
        "pts": pd.to_numeric(df["PTS"], errors="coerce"),
        "trb": pd.to_numeric(df["REB"], errors="coerce"),
        "ast": pd.to_numeric(df["AST"], errors="coerce"),
        "stl": pd.to_numeric(df["STL"], errors="coerce"),
        "blk": pd.to_numeric(df["BLK"], errors="coerce"),
        "fg_pct": pd.to_numeric(df["FG_PCT"], errors="coerce"),
        "fg3_pct": pd.to_numeric(df["FG3_PCT"], errors="coerce"),
        "ft_pct": pd.to_numeric(df["FT_PCT"], errors="coerce"),
    })

    out["player_key"] = out["player_name"].apply(normalize_name_to_key)

    out["last_updated_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    out = out[(out["games_played"] >= 5) & (out["minutes_played"] >= 15)].copy()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "nba_mvp_current_candidates.csv"
    out.to_csv(out_path, index=False)

    print(f"[SAVED] {out_path}")
    print(f"[ROWS] {len(out)}")
    print(f"[SEASON] {season}")


if __name__ == "__main__":
    main(season_end_year=2026)