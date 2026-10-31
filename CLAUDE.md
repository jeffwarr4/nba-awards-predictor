# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NBA Awards Predictor: ranks NBA MVP candidates weekly and publishes them for the Stinger website and social posts.
It is being brought in line with the sibling `nhl-awards-predictor` (closest reference) and `nfl-awards-predictor`
(Oct 2026 rebuild in progress). This file describes the pipeline **as it exists before that rebuild**. Update it
as the rebuild lands.

## Environment

- Venv lives outside the repo: `C:\DevVenvs2\nba-awards\Scripts\python.exe` (Python 3.12). It has pandas, numpy,
  nba_api, bs4, lxml and scikit-learn, but not `anthropic` or `pillow` yet.
- `requirements.txt` is **UTF-16 encoded** (a PowerShell `pip freeze >` artifact). pip copes; most other tools don't.
- Run scripts as modules from the repo root: `python -m src.model.build_rankings_ridge_np_v1`.

## History, backtest and retraining (Oct 2026)

`config.py` (repo root) holds the seasons (`SEASON_END_YEAR`, `TRAIN_START`, `TRAIN_END`; Basketball-Reference
*end* years, 2027 = 2026-27) and data paths. Results and the decision are in `docs/nba-2026-evaluation-and-plan.md`.

```powershell
python -m src.data.fetch_mvp_votes                 # Basketball-Reference vote tables → seed (UTF-8); diffs vs the old seed
python -m src.data.fetch_mvp_votes --seasons 2027  # add one season
python -m src.data.normalize_voting_seed; python -m src.data.build_training_dataset
python -m src.data.build_history                   # every player-season + votes → data/processed/training/nba_player_seasons.csv
python -m src.model.train_mvp                      # leave-one-season-out backtest + fit → models/nba_mvp_ridge_np.json, backtest_mvp.*
python -m src.model.train_mvp --outdir <scratch>   # same, without touching models/
python -m src.model.experiments [E0 E8 ...]        # every spec in SPECS → comparison table, models/experiments/*.csv
```

- `src/data/bbref.py`: throttled fetch (3.5 s apart, 429 backoff) with a page cache in `data/raw/bbref_cache/`
  (gitignored, ~190 MB; pass `cache=False` for live pages) and a `data-stat` table parser.
- `src/nba_common.py: season_players()` builds one season's player table (per-game + advanced + team record, traded
  players collapsed to their season total under their last team). Training and live scoring must both use it.
- `src/model/mvp_models.py`: specs, models, `backtest()`. Backtests score **every** player with ≥5 G and ≥15 min,
  not just vote-getters; vote-getters filtered out of a pool count as unranked misses. Season totals are scaled to 82
  team games. A hard G ≥ 65 filter is wrong for 2024+: Basketball-Reference's G undercounts (three 2025-26 top-5
  finishers had 64); use 62.
- The seed, normalized votes and training CSVs are tracked in git (they weren't before Oct 2026).

## Pipeline (current)

```
data/raw/voting/nba_mvp_voting_history_seed.csv      (Basketball-Reference MVP vote tables 1980-81..2024-25; gitignored)
  → src/data/normalize_voting_seed.py   → data/processed/voting/nba_mvp_voting_history_normalized.csv   (gitignored)
  → src/data/build_training_dataset.py  → data/processed/training/nba_mvp_training_dataset.csv          (gitignored)
  → src/model/train_mvp_ridge_np.py     → models/nba_mvp_ridge_np.json   (closed-form ridge, alpha 10, 10 box-score features)

Local, by hand (stats.nba.com blocks GitHub runners):
  src/data/build_current_candidates.py  → data/processed/current/nba_mvp_current_candidates.csv  (nba_api per-game)
  src/data/build_team_standings.py      → data/processed/current/nba_team_standings.csv          (nba_api standings)
  → commit + push

Weekly workflow (.github/workflows/nba_mvp_weekly_pages.yml):
  src/model/build_rankings_ridge_np_v1.py → data/processed/outputs/{model_all_candidates,web_current_top10,web_meta}.csv
  src/publish/build_canva_flat_nba.py     → canva_top5_flat.csv, canva_6_10_flat.csv
  src/publish/publish_pages.py            → docs/nba/mvp/ → deployed to GitHub Pages as a workflow artifact
```

`build_rankings_v1.py` and `build_rankings_v2_team.py` are older ranking scripts nothing calls.

## Known issues (found Oct 2026)

- **Training covers vote-getters only** (714 rows, 9–30 per season); live scoring runs on every player (~380).
- **No team strength in the model.** `mvp_score = 0.70 × minmax(ridge) + 0.20 × minmax(z win%) + 0.10 × minmax(z −conf rank)`
  is blended on after the ridge.
- The ridge's own CV is 5-fold MAE on vote-getters, not a ranking metric; there is no backtest of the blended board.
- The closed-form ridge also penalizes the intercept.
- The raw seed was saved as cp1252, so 15 names lost their accents to `?` (`Nikola Joki?` → key `nikola_joki`).
- `player_key` is `first_last` with hyphens dropped (`shai_gilgeousalexander`). NFL/NHL use `first_last_team`.
- The committed `docs/nba/mvp/` copies are stale (2026-01-24). Pages serves the workflow's artifact, so the live
  site files are from the last run (2026-04-15).
- Movement compares against the last published Pages board, so a same-week rerun shows `—` for everyone (as the
  2026-04-15 run did).

## Output contract (website, legacy)

The website's /nba page reads a Google Sheet that IMPORTDATAs the Pages files under
`https://jeffwarr4.github.io/nba-awards-predictor/nba/mvp/`: `web_current_top10.csv`, `web_prior_top10.csv`,
`web_meta.csv`, `canva_top5_flat.csv`, `canva_6_10_flat.csv` (and `model_all_candidates.csv`). Keep their columns
unchanged until the site switches to `board_mvp_latest.csv`.

## stinger-assets

Headshots: `../stinger-assets/nba/headshots/` (19 files, `first_last.png`, no team). Logos: `nba/logos/`, lowercase,
with both ESPN short and 3-letter duplicates (`gs`/`gsw`, `ny`/`nyk`, `sa`/`sas`, `no`/`nop`, `utah`/`uta`, `was`/`wsh`).
Its `scripts/merge_candidates.py` has no NBA source yet. Don't change stinger-assets without asking.

## Git

This repo lives in OneDrive, which has deleted tracked files before. Stage files by name (never `git add -A`) and
stop if `git status` shows deletions you didn't make. Commit only when asked. Push as `jeffwarr4`, never the
`outdoor-escapes` account; `git -c credential.helper= -c 'credential.helper=!gh auth git-credential' push` uses
gh's active `jeffwarr4` login.
