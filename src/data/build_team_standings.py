from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import time
import random
import pandas as pd

from requests import Session
from requests.exceptions import ReadTimeout, ConnectionError, Timeout

from nba_api.stats.endpoints import leaguestandings
from nba_api.stats.library.http import NBAStatsHTTP

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "data" / "processed" / "current"


def season_label_from_end_year(end_year: int) -> str:
    return f"{end_year-1}-{str(end_year)[-2:]}"


def make_session() -> Session:
    s = Session()
    s.headers.update({
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
    season = season_label_from_end_year(season_end_year)

    # Force nba_api to use our hardened session
    NBAStatsHTTP._session = make_session()

    max_tries = 6
    base_sleep = 3
    timeout_seconds = 45

    last_err = None
    df = None

    for attempt in range(1, max_tries + 1):
        try:
            endpoint = leaguestandings.LeagueStandings(season=season, timeout=timeout_seconds)
            df = endpoint.get_data_frames()[0].copy()
            last_err = None
            break
        except (ReadTimeout, ConnectionError, Timeout, TimeoutError) as e:
            last_err = e
            sleep_s = base_sleep * (2 ** (attempt - 1)) + random.uniform(0, 2)
            print(f"[WARN] stats.nba.com request failed (attempt {attempt}/{max_tries}): {e}")
            print(f"[WARN] sleeping {sleep_s:.1f}s then retrying...")
            time.sleep(sleep_s)

    if last_err is not None or df is None:
        raise RuntimeError(f"Failed to fetch LeagueStandings after {max_tries} attempts") from last_err

    needed = {"TeamID", "TeamCity", "TeamName", "WINS", "LOSSES", "WinPCT", "Conference", "PlayoffRank"}
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Standings response missing columns: {missing}")

    out = pd.DataFrame({
        "season": season,
        "season_end_year": season_end_year,
        "team_id": df["TeamID"],
        "team_city": df["TeamCity"].astype(str),
        "team_name": df["TeamName"].astype(str),
        "wins": pd.to_numeric(df["WINS"], errors="coerce"),
        "losses": pd.to_numeric(df["LOSSES"], errors="coerce"),
        "win_pct": pd.to_numeric(df["WinPCT"], errors="coerce"),
        "conference": df["Conference"].astype(str),
        "conf_rank": pd.to_numeric(df["PlayoffRank"], errors="coerce"),
        "last_updated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "nba_team_standings.csv"
    out.to_csv(out_path, index=False)

    print(f"[SAVED] {out_path}")
    print(f"[ROWS] {len(out)}")
    print(f"[SEASON] {season}")


if __name__ == "__main__":
    main(season_end_year=2026)