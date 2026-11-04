# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NBA Awards Predictor: ranks NBA MVP candidates weekly from Basketball-Reference stats and publishes a 30-deep board,
AI-written chips/Whys, two social graphics and a caption, plus the legacy Google Sheet feed on GitHub Pages. It was
brought in line with the sibling `nhl-awards-predictor` (closest reference) and `nfl-awards-predictor` in Oct 2026;
the evaluation, backtests and decisions are in `docs/nba-2026-evaluation-and-plan.md`.

## Environment

- Venv lives outside the repo: `C:\DevVenvs2\nba-awards\Scripts\python.exe` (Python 3.12). `pip install -r requirements.txt`.
- Run everything as a module from the repo root (`python -m src.predict_current`); imports rely on it.
- `ANTHROPIC_API_KEY` enables chips, Whys and the caption; without it they are skipped with a warning.
  `STINGER_ASSETS_DIR` overrides the default `../stinger-assets` for headshots/logos.
- No tests or linters.

## Commands

```powershell
# Weekly board (what the workflow runs)
python -m src.predict_current
python -m src.predict_current --outdir <scratch> --no-publish   # board CSVs only, elsewhere; no legacy feed
python -m src.predict_current --force                           # rebuild even with no new games

# Re-run one publish step from the latest board
python -m src.publish.render_graphics
python -m src.publish.generate_chips
python -m src.publish.generate_captions
python -m src.publish.make_bubble_bg [--preview <png>]          # regenerate bubble background/overlay/slots

# History, backtest, retraining
python -m src.data.fetch_mvp_votes [--seasons 2027]   # Basketball-Reference vote tables → seed (UTF-8), diffed vs the old seed
python -m src.data.normalize_voting_seed; python -m src.data.build_training_dataset
python -m src.data.build_history                      # every player-season + votes → data/processed/training/nba_player_seasons.csv
python -m src.model.train_mvp                         # leave-one-season-out backtest + fit → models/nba_mvp_ridge_np.json, backtest_mvp.*
python -m src.model.train_mvp --outdir <scratch>      # same, without touching models/
python -m src.model.experiments [E0 E8 ...]           # every spec in SPECS → comparison table, models/experiments/*.csv
```

Each season: bump `SEASON_END_YEAR` and `TRAIN_END` in `config.py`, `fetch_mvp_votes --seasons {TRAIN_END}`, rebuild
history, retrain, commit the model and data, and re-enable the weekly workflow if GitHub disabled it (see Automation).

## Architecture

**`config.py` is the single source of truth**: seasons (Basketball-Reference *end* years, 2027 = 2026-27), paths,
in-season knobs, board depth. Nothing else hardcodes a year.

**Data: Basketball-Reference only** (`src/data/bbref.py`: 3.5 s throttle, 429 backoff, page cache in
`data/raw/bbref_cache/`, gitignored, ~190 MB; live scoring passes `cache=False`). It answers from GitHub runners
(`probe_sources.yml`, Oct 2026); stats.nba.com times out there. The old `nba_api` scripts (`build_current_candidates.py`,
`build_team_standings.py`, `build_rankings_ridge_np_v1.py`) are a local fallback only; nothing scheduled calls them.

**`src/nba_common.py: season_players()` builds every season's player table** (per-game + advanced + team record and
conference rank) for both training and live scoring. A traded player's `2TM`/`TOT` row is kept under his last team.
Basketball-Reference `BRK/CHO/PHO` become the NBA's `BKN/CHA/PHX` in `team`; joins use the raw `team_bbref`.

**Model** (`models/nba_mvp_ridge_np.json`, from `src/model/train_mvp.py`, spec `E0` in `src/model/experiments.py`):
closed-form ridge (alpha 10) on vote share, trained on vote-getters, 10 per-game features; then
`mvp_score = 0.70 minmax(ridge) + 0.20 minmax(team win%) + 0.10 minmax(-conf rank)` over the eligible pool. Backtest
(leave-one-season-out, 1981–2026, every player scored): winner #1 72%, top 3 93%, mean winner rank 1.70. NHL's
GBR + classifier with advanced stats (E8) was narrowly better on full seasons but unverifiable in-season; deferred.
Backtests score every player with ≥5 G and ≥15 min, never just vote-getters, and seasons are never split randomly.

**Live scoring** (`src/predict_current.py`), in order:
1. Week = 7-day blocks from opening night; `stats_through` = last date with a completed game (schedule pages).
   No completed games, or none since the last board: exit 0 and set the Actions output `skipped=true`.
2. Eligibility: ≥60% of team games, ≥15 min/game, and still able to reach 62 games (the 65-game rule; Basketball-
   Reference's G undercounts: three 2025-26 top-5 finishers had 64, so never filter at 65).
3. Games played projected to an 82-game pace; every other feature is a per-game average. Displayed stats are actual.
4. Board (`src/publish/board.py`) → legacy feed (`write_legacy`).

## Weekly board (`src/publish/`, port of nhl-awards-predictor)

`predictions/{season}/`:
- `top10_mvp_latest.csv` and `board_mvp_latest.csv` (30 deep), same columns (`BOARD_COLS`), 4 rolling timestamped
  archives each. `chips` (top 3, pipe-separated) and `why_blurb` (top 3 + bubble) are blank without a key.
- **Movement**: `.prev_board_mvp.csv` (hidden, 30 rows) is last week's board, rolled only when `week` advances within
  the season, so a same-week rerun keeps last week's arrows. No snapshot (first 2026-27 board): everyone is NEW.
- **Guards**: nothing is written if fewer than 15 players are eligible, or if a same-week rerun would write a shorter
  board than the published one. If chips fail or the key is missing on a same-week rerun, this week's saved
  `chip_data/chip_data_latest.csv` is reused.
- `player_key = first_last_team`, lowercase, accents stripped, apostrophes/periods dropped, hyphens split; equals
  `safe_filename(f"{name} {team}")` in stinger-assets (`shai_gilgeous_alexander_okc`). A traded player's key changes
  with his team. Two same-name players on one team: the lower-ranked gets a `_{pos}` suffix.
- `predictions/nba_candidates.csv` (`sport, player_name, team, player_key`): the 30-deep board, for headshot sync.
- **Chips** (`generate_chips.py`, `claude-haiku-5-5`, effort `low`): Chip + Why for the top 3 (no "leads"/"ranks
  first" claims; the model sees one player's stats); Why ≤75 chars (enforced) for ranks 11–15, compared with the
  lowest-ranked top-10 player in the same role: guard (PG/SG), forward (SF/PF), center.
- **API client** (`generate_chips.make_client`, also used by captions) strips whitespace from the key. A secret pasted
  with a trailing newline (Oct 2026) made every call fail as `Connection error.`: an invalid auth header, not a
  network problem. Failures log only the underlying error type (`describe()`), never its message, which can echo the
  key into the public Actions log.
- **Graphics** (`render_graphics.py`): main board on `NBA_background.png` (light hardwood, so text sits on dark brown
  plates); "On the Bubble" on `NBA_Bubble_bg.png` + `NBA_Bubble_overlay.png` + `assets/bubble_slots.json`, generated by
  `make_bubble_bg.py` from the board background. A missing top-3 headshot shows a large team logo instead. Fonts are
  OFL copies in `assets/fonts/`. `graphics/caption_mvp.txt` covers the two-slide carousel; its team hashtags are in
  `generate_captions.py: TEAM_HASHTAGS` (written from memory; not yet checked against the teams' current tags).
  Player hashtags transliterate accents (`name_tag`: `#LukaDoncic`); stripping non-ASCII gave `#LukaDoni`.

## Output contract (website)

When the site switches over it reads `predictions/{season}/board_mvp_latest.csv` from raw.githubusercontent.com, by
column name. Renaming any of these breaks it:

`player_key, rank, rank_delta, movement, player_name, team, position, model_score, games, team_games, minutes, pts,
reb, ast, stl, blk, fg_pct, fg3_pct, ft_pct, team_record, team_win_pct, conf_rank, conference, season, week,
stats_through, updated_utc, chips, why_blurb`

`rank_delta`: positive = up, blank = NEW. `movement`: `↑2` / `↓2` / `–` / `NEW`. `team_record` is `W-L`.
Percentages are fractions (`.512`). `team` is an NBA code (`OKC`, `GSW`, `NYK`).

**Legacy feed, until the site switches.** The /nba page reads a Google Sheet that IMPORTDATAs
`https://jeffwarr4.github.io/nba-awards-predictor/nba/mvp/`: `web_current_top10.csv`, `web_prior_top10.csv`,
`web_meta.csv`, `model_all_candidates.csv`, `canva_top5_flat.csv`, `canva_6_10_flat.csv`. `predict_current.write_legacy`
keeps their exact columns (`LEGACY_TOP10_COLS`, `LEGACY_ALL_COLS`) and the old `first_last` key with hyphens dropped
(`shai_gilgeousalexander`); `mover_label` uses an em dash for no change. `web_prior_top10.csv` is the previously
published `web_current_top10.csv`, downloaded by the workflow.

## stinger-assets contract

Headshots: `nba/headshots/{player_key}.png`. Logos: `nba/logos/{team}.png`, lowercase NBA codes; all 30 exist
(`gsw, nyk, sas, nop, uta, was`; the ESPN short duplicates `gs, ny, sa, no, utah, wsh` are unused). The 19 older
`first_last.png` headshots stay for the legacy Sheet. Since stinger-assets PR #3 (Oct 2026) its
`scripts/merge_candidates.py` pulls `predictions/nba_candidates.csv`, and `SPORT_TEAM_CODE_ALIASES["NBA"]` maps
`GSW→GS, NOP→NO, NYK→NY, SAS→SA, UTA→UTAH, WAS→WSH` for ESPN roster matching. Its "Sync Headshots" workflow runs daily
at 11:37 UTC (or manually), and the weekly workflow fetches any missing ranks 1–15 headshot in-run (`sync_espn_headshots.py fetch --strict`,
re-rendering if it downloaded one; the run fails before committing if one can't be found). The runner is case-sensitive; Windows isn't.

## Automation

`.github/workflows/nba_mvp_weekly_pages.yml`: started Saturdays 15:17 UTC by the stinger-scheduler Cloudflare Worker (`stinger-website/scheduler`); October–April; its own cron (Saturdays 17:17 UTC) is a late backup. Plus manual dispatch (`force` input). Checks out stinger-assets as a sibling, runs `predict_current` with the
`ANTHROPIC_API_KEY` secret, fails on an empty board, builds the Canva flats, copies the legacy feed to `docs/nba/mvp/`,
uploads artifacts, commits `predictions/` and `docs/nba/mvp/*.csv` (plus `git add -f` of `.prev_board_mvp.csv`) with
`[skip ci]`, and deploys Pages. Skipped runs (no new games) do none of that. The model is not retrained in CI.

GitHub disables a scheduled workflow after 60 days without repo activity, which the May–October off-season can
trigger (it was disabled in Oct 2026). Before opening night: `gh workflow enable nba_mvp_weekly_pages.yml`.
The secret must be named exactly `ANTHROPIC_API_KEY` (it was first created as `ANTRHOPIC_API_KEY`, and the
chips silently skipped).

`probe_sources.yml` (manual, commits nothing) prints which data sources answer from a runner.

## Git

This repo lives in OneDrive, which has deleted tracked files before. Stage files by name (never `git add -A`) and
stop if `git status` shows deletions you didn't make. Commit only when asked. Push as `jeffwarr4`, never the
`outdoor-escapes` account; `git -c credential.helper= -c 'credential.helper=!gh auth git-credential' push` uses
gh's active `jeffwarr4` login. `.vscode/`, `requirements.freeze.txt` and the untracked files under
`data/processed/outputs/` are local leftovers; leave them alone.
