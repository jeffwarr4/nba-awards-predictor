from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

CURRENT_PATH = REPO_ROOT / "data" / "processed" / "current" / "nba_mvp_current_candidates.csv"
STANDINGS_PATH = REPO_ROOT / "data" / "processed" / "current" / "nba_team_standings.csv"
OUT_DIR = REPO_ROOT / "data" / "processed" / "outputs"

PRIOR_TOP10_PATH = OUT_DIR / "web_prior_top10.csv"

# Map nba_api TEAM_ABBREVIATION to standings team_name/city is annoying without TeamID.
# We'll join using a small abbreviation lookup via a static map that we can expand.
# For now, we’ll derive a join key based on common team abbreviations -> full name.
TEAM_ABBR_TO_NAME = {
    "ATL": "Hawks",
    "BOS": "Celtics",
    "BKN": "Nets",
    "CHA": "Hornets",
    "CHI": "Bulls",
    "CLE": "Cavaliers",
    "DAL": "Mavericks",
    "DEN": "Nuggets",
    "DET": "Pistons",
    "GSW": "Warriors",
    "HOU": "Rockets",
    "IND": "Pacers",
    "LAC": "Clippers",
    "LAL": "Lakers",
    "MEM": "Grizzlies",
    "MIA": "Heat",
    "MIL": "Bucks",
    "MIN": "Timberwolves",
    "NOP": "Pelicans",
    "NYK": "Knicks",
    "OKC": "Thunder",
    "ORL": "Magic",
    "PHI": "76ers",
    "PHX": "Suns",
    "POR": "Trail Blazers",
    "SAC": "Kings",
    "SAS": "Spurs",
    "TOR": "Raptors",
    "UTA": "Jazz",
    "WAS": "Wizards",
}

def zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mu = s.mean(skipna=True)
    sd = s.std(skipna=True)
    if sd == 0 or np.isnan(sd):
        return s * 0
    return (s - mu) / sd

def compute_mvp_score_v2(df: pd.DataFrame) -> pd.Series:
    # Player impact (same core as v1)
    z_pts = zscore(df["pts"])
    z_trb = zscore(df["trb"])
    z_ast = zscore(df["ast"])
    z_stl = zscore(df["stl"])
    z_blk = zscore(df["blk"])

    # Efficiency
    z_fg  = zscore(df["fg_pct"])
    z_fg3 = zscore(df["fg3_pct"])
    z_ft  = zscore(df["ft_pct"])

    # Availability
    z_gp = zscore(df["games_played"])
    z_mp = zscore(df["minutes_played"])

    # Team success (new)
    z_win = zscore(df["team_win_pct"])

    # Conf rank: lower is better; invert then z-score
    inv_rank = -pd.to_numeric(df["team_conf_rank"], errors="coerce")
    z_seed = zscore(inv_rank)

    # Weights tuned for NBA narrative:
    # player impact still dominates, but team success matters a lot
    score = (
        1.15 * z_pts +
        0.55 * z_ast +
        0.45 * z_trb +
        0.25 * z_stl +
        0.20 * z_blk +
        0.25 * z_gp +
        0.15 * z_mp +
        0.12 * z_fg +
        0.08 * z_fg3 +
        0.08 * z_ft +
        0.55 * z_win +
        0.25 * z_seed
    )
    return score

def main() -> None:
    if not CURRENT_PATH.exists():
        raise FileNotFoundError(f"Missing current candidates file: {CURRENT_PATH}")
    if not STANDINGS_PATH.exists():
        raise FileNotFoundError(f"Missing team standings file: {STANDINGS_PATH}")

    df = pd.read_csv(CURRENT_PATH)
    st = pd.read_csv(STANDINGS_PATH)

    # Build join helper: team abbreviation -> team_name
    df["team_name_lookup"] = df["team"].map(TEAM_ABBR_TO_NAME)

    # Standings join key (team nickname)
    st["team_name_lookup"] = st["team_name"].astype(str)

    merged = df.merge(
        st[["team_name_lookup", "win_pct", "conf_rank", "conference"]],
        on="team_name_lookup",
        how="left"
    )

    # Rename for clarity in output
    merged = merged.rename(columns={
        "win_pct": "team_win_pct",
        "conf_rank": "team_conf_rank",
        "conference": "team_conference",
    })

    # If a join failed, keep it visible (we can patch mapping if needed)
    if merged["team_win_pct"].isna().any():
        missing_teams = sorted(merged.loc[merged["team_win_pct"].isna(), "team"].unique().tolist())
        print(f"[WARN] Missing team standings join for: {missing_teams}")
        # Fill neutral values to avoid crashing score
        merged["team_win_pct"] = merged["team_win_pct"].fillna(0.5)
        merged["team_conf_rank"] = merged["team_conf_rank"].fillna(8)

    merged["mvp_score"] = compute_mvp_score_v2(merged)

    merged = merged.sort_values("mvp_score", ascending=False).reset_index(drop=True)
    merged["mvp_rank"] = merged.index + 1

    def tier(r: int) -> str:
        if r <= 3:
            return "Top 3"
        if r <= 5:
            return "Top 5"
        if r <= 10:
            return "6-10"
        return "Outside"

    merged["mvp_tier"] = merged["mvp_rank"].apply(tier)

    web_current_top10 = merged.head(10).copy()
    web_current_top10["rank"] = web_current_top10["mvp_rank"].astype(int)
    web_current_top10["rank_delta"] = np.nan
    web_current_top10["mover_label"] = "NEW"

    # Movers if prior exists
    if PRIOR_TOP10_PATH.exists():
        prior_df = pd.read_csv(PRIOR_TOP10_PATH)
        prior_map = (
            prior_df[["player_key", "rank"]]
            .dropna()
            .set_index("player_key")["rank"]
            .to_dict()
        )

        def mover(row):
            prev_rank = prior_map.get(row["player_key"])
            if prev_rank is None:
                return np.nan, "NEW"
            delta = int(prev_rank) - int(row["rank"])
            if delta > 0:
                return delta, f"↑{delta}"
            if delta < 0:
                return delta, f"↓{abs(delta)}"
            return 0, "—"

        movers = web_current_top10.apply(lambda r: pd.Series(mover(r)), axis=1)
        web_current_top10[["rank_delta", "mover_label"]] = movers

    # Meta
    season = str(merged["season"].iloc[0])
    season_end_year = int(merged["season_end_year"].iloc[0])
    last_updated_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    web_meta = pd.DataFrame([{
        "season": season,
        "season_end_year": season_end_year,
        "week_label_current": "",
        "week_label_prior": "",
        "last_updated_utc": last_updated_utc,
        "model_version": "v2_team_success",
        "notes": "Baseline score with team win% + conf rank added to box stats + availability + shooting."
    }])

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    out_all = OUT_DIR / "model_all_candidates.csv"
    merged.to_csv(out_all, index=False)

    out_top10 = OUT_DIR / "web_current_top10.csv"
    cols_top10 = [
        "season", "season_end_year",
        "rank", "player_key", "player_name", "team",
        "mvp_score",
        "team_win_pct", "team_conf_rank", "team_conference",
        "pts", "trb", "ast", "games_played",
        "rank_delta", "mover_label",
        "last_updated_utc"
    ]
    web_current_top10[cols_top10].to_csv(out_top10, index=False)

    out_meta = OUT_DIR / "web_meta.csv"
    web_meta.to_csv(out_meta, index=False)

    # Snapshot for next run
    web_current_top10[["player_key", "rank"]].to_csv(PRIOR_TOP10_PATH, index=False)

    print(f"[SAVED] {out_all}")
    print(f"[SAVED] {out_top10}")
    print(f"[SAVED] {PRIOR_TOP10_PATH}")
    print(f"[SAVED] {out_meta}")
    print(f"[TOP10 #1] {web_current_top10['player_name'].iloc[0]}")

if __name__ == "__main__":
    main()
