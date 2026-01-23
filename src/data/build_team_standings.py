from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import pandas as pd

from nba_api.stats.endpoints import leaguestandings

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "data" / "processed" / "current"

def season_label_from_end_year(end_year: int) -> str:
    return f"{end_year-1}-{str(end_year)[-2:]}"

def main(season_end_year: int = 2026) -> None:
    season = season_label_from_end_year(season_end_year)

    # LeagueStandings returns current standings for the season
    endpoint = leaguestandings.LeagueStandings(season=season, timeout=60)
    df = endpoint.get_data_frames()[0].copy()

    # Expected columns include:
    # TeamCity, TeamName, TeamID, WINS, LOSSES, WinPCT, Conference, PlayoffRank, etc.
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
