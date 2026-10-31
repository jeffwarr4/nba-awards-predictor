# config.py
# ---------------------------------------------------------------
# Single source of truth for nba-awards-predictor: seasons and paths.
#
# Seasons are Basketball-Reference *end* years: 2027 = the 2026–27 season.
# Bump SEASON_END_YEAR and TRAIN_END each season; nothing else hardcodes a year.
# ---------------------------------------------------------------

from pathlib import Path

# ── Seasons ───────────────────────────────────────────────────────
SEASON_END_YEAR = 2027   # season being predicted RIGHT NOW
TRAIN_START     = 1981   # first season of MVP vote history (1980-81)
TRAIN_END       = 2026   # last season with a complete MVP vote

# ── Paths ─────────────────────────────────────────────────────────
ROOT_DIR       = Path(__file__).resolve().parent
DATA_DIR       = ROOT_DIR / "data"
MODELS_DIR     = ROOT_DIR / "models"

VOTING_SEED_CSV = DATA_DIR / "raw" / "voting" / "nba_mvp_voting_history_seed.csv"
HISTORY_CSV     = DATA_DIR / "processed" / "training" / "nba_player_seasons.csv"   # every player-season + votes
