from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "data" / "processed" / "current"

def main() -> None:
    # Stub: we will replace with nba_api pull in Step 7
    cols = [
        "season",
        "season_end_year",
        "player_name",
        "player_key",
        "team",
        "age",
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
        "last_updated_utc",
    ]
    df = pd.DataFrame(columns=cols)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    df["last_updated_utc"] = ts

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "nba_mvp_current_candidates.csv"
    df.to_csv(out_path, index=False)
    print(f"[SAVED] {out_path} (stub; will be populated in Step 7)")

if __name__ == "__main__":
    main()
