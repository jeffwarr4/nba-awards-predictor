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

**Weekly and final boards (`scripts/snapshot_board.py`, the same file in all four awards repos).** The
workflow's "Keep this week's board" step, just before the commit, copies `board_mvp_latest.csv` to `board_mvp_wkNN.csv` (NN = `week`), which is never
pruned (`ARCHIVES_TO_KEEP` only prunes the timestamped run files). A same-week rerun overwrites that copy. After the
season's last regular-season board, run `python scripts/snapshot_board.py --final` once and commit: it writes
`board_mvp_final.csv`, which the website's season archive pages read. Don't edit or delete the weekly or final files; they're the
season history. Weeks 2–4 of 2026–27 were backfilled from the workflow's commits (the step started at Week 5).

## Weekly reel (`src/publish/render_reel.py`, port of the NHL and NFL reels)

One story per week as a silent 1080×1920 MP4 (20 s + 1 s hold) plus a narration script Jeff voices over in
Instagram (he adds the Captions sticker there; nothing is burned in). The reel posts first, the afternoon the board completes; the static board posts the next day (order reversed 2026-11-24).
`python -m src.publish.render_reel [--outdir DIR] [--frames-only] [--no-script] [--story FILE]`; tests:
`python -m pytest tests/`.

Reworked Nov 2026 for retention. The first reels lost viewers in the hook (NHL: ~6 of 27 s watched), so: four
scenes, `REEL_SCENES` hook 0–3 / proof 3–13 / context 13–16 / ask 16–20, 40–50 words (`REEL_WORDS`) at `REEL_WPS` (3) words a second. Frame 0 already shows the hook text,
headshot, name and last week's rank (no fade-ins), and the rank flips to this week's by 0.8 s. The reel ends on a
comment question, not a spoken URL (the URL is small on screen and ends the Instagram caption).
- `reel_story.pick_story` (pure) takes the first match: new No. 1 (or "back on top" if he held it on an earlier
  `board_mvp_wkNN.csv`), biggest climber into the top 5, 11+ into the top 10, biggest drop by a former top-5
  player, else a profile of No. 1. Players are matched by `player_key`, as in `.prev_board_mvp.csv`, so a player
  traded since last week reads as NEW. Hooks are short and lead with the number. For a climb, `season_lows`
  (his worst rank on an earlier weekly board) makes the hook his path when it's bigger than last week's move and
  at least 2 weeks back: "No. 30 → No. 3 in 3 weeks". The reel fonts have no "→" glyph, so `hook_line_img` draws it.
- `reel_script.py` writes four timed beats in Jeff's first-person voice on `REEL_MODEL` (Opus 5.5; chips stay on
  Haiku), plus the on-screen `question`, the Instagram caption and 5 hashtags. v3 (Nov 2026) makes it flow: the hook
  names him in one natural sentence ("Giannis jumped from 30 to 3 in three weeks."), the whole script uses ONE time window, the proof gives ONE
  reason after a lead-in that rotates by week (`PROOF_OPENERS`: "So what changed?", "Here's why.", "The reason?", "What happened?", offset per sport by `OPENER_OFFSET` so the three sports differ; the model may skip it when the hook makes the reason obvious, and `problems()` flags any other lead-in) in connected sentences (at most two numbers), the context is the gap to the
  milestone, and the ask is a yes/no or two-way question about it. `story_frame()` hands the script:
  - the window: the `comparison_row` week to now ("the last three weeks (Week 2 to Week 5), his last ten games"),
    or "this week" with no earlier board; the hook's ranks are the window's
  - the driver: `player` if a `REEL_FLAT_STATS` stat (pts/reb/ast, absolute per game) moved by at least its amount
    with his rank, else `team` if the team's win rate in the window is `REEL_TEAM_SWING` above/below its rate before
    (3+ games), else `board` if he played no games in the window and players ahead of him dropped, else `unclear`;
    the script may only claim a cause the driver states
  - `crossed()`: the players ahead of him at the window's start who are behind him now (or, for a drop, the
    reverse), from the window-start board
  - the milestone (`milestone()`): hold No. 1; catch No. 1 from No. 2-3; the top 3 from 4-5; the top 5 from 6-10;
    the top 10 from 11+
  `problems()` checks `REEL_WORDS` (40-50), the hook (≤ `REEL_HOOK_MAX_WORDS`, names him, no "our model", and reads like a fan saying it: "Jack Eichel jumped from 15 to 5 in one week." with a verb and bare numbers; `ROBOT_HOOK` flags a colon or "No." before a rank, and the team name before his name is left out as a tongue twister), the
  proof (≤ `REEL_PROOF_MAX_WORDS`), his full name or `REEL_NICKNAMES` short name somewhere (accents folded), no
  then/since/now without a time anchor in the same sentence (`unanchored_time`: "this week", "Week 8", "three weeks
  ago", "his last three games"...), a context that doesn't repeat the hook's ranks or name the No. 1 (unless it's a
  No. 1 story or he's No. 2-3), the last beat ending with "?", no stock phrases or "my model", no probability
  language, and 5 hashtags. The caption's site line is added in code (`caption_with_cta`): whatever site line the model wrote is cut off and "Full board at stingercollectibles.com." follows as its own sentence. Percentages: a shooting, completion, K% or save stat is fine ("60%
  from the field"; `STAT_PCT`); a win % never is, nor any other percent (`probability_language`). The prompt carries
  two style examples (Jeff's, edited to pass the checks). One rewrite request; then `finish()` swaps in
  `fallback_question()` about the milestone ("Can SGA crack the top 5?"). The result, `reel_meta.json` and the
  script file keep the frame, `first_draft_problems` and `rewritten`. `render_reel --script-only` writes just the
  script and meta (for reviewing scripts on replays); a `--story` file without its own cards now gets the
  then-vs-now comparison too. Tests: `tests/test_reel_script.py`.
  The player hashtag uses `generate_captions.name_tag` (accents transliterated). Needs anthropic 2.x
  (`fallbacks`); an old local SDK fails the script call.
- **Explain, don't recite.** The script and the Proof cards compare the player then vs now, so the reel says
  why the model moved him. `comparison_row` picks his row on an earlier weekly board: the path hook's week, else
  the earliest within `REEL_LOOKBACK_WEEKS` (3). The script gets his rank, team record (with the record since
  then), conference rank and per-game stats then and now (`then_now_lines`), plus `MODEL_NOTE`: what the model
  weighs, in plain words. (`models/nba_mvp_ridge_np.json` blends 0.70 ridge on per-game stats, with points,
  assists and rebounds weighted most, + 0.20 team win rate + 0.10 conference rank. Update `MODEL_NOTE` if the
  model changes.) Example, 2026–27 Week 5: "Giannis's scoring barely moved, 27.6 a game then, 28.2 now. Miami
  did. They were 1-4, now 9-6. That's my model reacting to winning."
- The background is the board's hardwood tinted dark brown (`PLATE`), and the header band is a gold-bordered
  badge (small on the hook, larger on the ask). With a comparison, the Proof scene shows "WEEK 2 → WEEK 5" over
  then → now cards (`then_now_cards`): team record, conference rank, PPG, and the board stat that moved the model
  most (weight × change / spread, read from `MODEL_PATH`). Without one (first board, or he's new to the board), it
  shows the board card's four stats (`player_stats`, win % left out) in box-score order, the record line and the
  `why_blurb`.
- Outputs in `predictions/{season}/reel/`: `reel_script.txt` and `reel_meta.json` are committed; `reel_latest.mp4`
  is gitignored and the workflow uploads it to the repo's `reel-latest` release as `nba_reel_{season}_wk{NN}.mp4`.
  stinger-website's social-drafts job emails it with the script. Both workflow steps are `continue-on-error`, after
  the headshot fetch and skipped on a no-games week. `predict_current.py` doesn't render the reel; run it on its own
  locally.
- One-off reels (a correction, a feature, or a hand-written week): a story file names the player and hook and can
  set labels, cards and a script (four beats for `REEL_SCENES`, `question`, caption, hashtags); it writes
  `reel_<name>.*` beside the weekly files. Upload the MP4 by hand.
- `reel_meta.json` keeps the keys stinger-website reads (`through_week`, `release_asset`, `release_tag`, `player`,
  `hook`, `story_type`) and adds `question` and `compare_week`; `captions` is now always empty.

## Output contract (website)

When the site switches over it reads `predictions/{season}/board_mvp_latest.csv` from raw.githubusercontent.com, by
column name. Renaming any of these breaks it:

`player_key, rank, rank_delta, movement, player_name, team, position, model_score, games, team_games, minutes, pts,
reb, ast, stl, blk, fg_pct, fg3_pct, ft_pct, team_record, team_win_pct, conf_rank, conference, season, week,
stats_through, updated_utc, chips, why_blurb`

`rank_delta`: positive = up, blank = NEW. `movement`: `↑2` / `↓2` / `–` / `NEW`. `team_record` is `W-L`.
Percentages are fractions (`.512`). `team` is an NBA code (`OKC`, `GSW`, `NYK`).

**Legacy feed** (the old Squarespace site was replaced by the Cloudflare site on 2026-11-23, which doesn't read it; keep it until Squarespace is cancelled (early Dec 2026, the rollback window), then it can be retired). The old Squarespace /nba page read a Google Sheet that IMPORTDATAs
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
