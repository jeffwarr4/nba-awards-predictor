# config.py
# ---------------------------------------------------------------
# Single source of truth for nba-awards-predictor: seasons and paths.
#
# Seasons are Basketball-Reference *end* years: 2027 = the 2026–27 season.
# Bump SEASON_END_YEAR and TRAIN_END each season; nothing else hardcodes a year.
# ---------------------------------------------------------------

import os
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
MODEL_PATH      = MODELS_DIR / "nba_mvp_ridge_np.json"

# ── In-season scoring ─────────────────────────────────────────────
SEASON_GAMES        = 82
MIN_TEAM_GAME_SHARE = 0.60   # played ≥60% of his team's games (works from week 1)
MIN_MINUTES         = 15     # minutes per game
ELIGIBLE_GAMES      = 62     # must still be able to reach this many games (65-game rule; BBRef's G undercounts)

# ── Weekly board + publishing (src/publish/) ──────────────────────
PREDICTIONS      = ROOT_DIR / "predictions"
CURRENT_PRED_DIR = PREDICTIONS / str(SEASON_END_YEAR)
CANDIDATES_CSV   = PREDICTIONS / "nba_candidates.csv"   # pulled by stinger-assets' headshot sync
LEGACY_OUT_DIR   = DATA_DIR / "processed" / "outputs"     # Google Sheet feed, published to Pages from docs/nba/mvp/
FONTS_DIR        = ROOT_DIR / "assets" / "fonts"

# 1080×1350. The bubble background, overlay and slot geometry come from src/publish/make_bubble_bg.py.
BOARD_BACKGROUND  = ROOT_DIR / "NBA_background.png"
BUBBLE_BACKGROUND = ROOT_DIR / "NBA_Bubble_bg.png"
BUBBLE_OVERLAY    = ROOT_DIR / "NBA_Bubble_overlay.png"
BUBBLE_SLOTS      = ROOT_DIR / "assets" / "bubble_slots.json"

# Sibling repo with headshots/logos. CI checks it out and sets the env var.
STINGER_ASSETS_DIR = Path(os.environ.get("STINGER_ASSETS_DIR", str(ROOT_DIR.parent / "stinger-assets")))

TOP_N            = 10
BOARD_DEPTH      = 30        # saved board depth; everyone on it is a headshot candidate
BUBBLE_RANKS     = (11, 15)  # inclusive; the "On the Bubble" slide
ARCHIVES_TO_KEEP = 4
