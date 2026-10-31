# NBA MVP Model — 2025-26 Evaluation & 2026-27 Build Plan
_Written 2026-10-30 (2026-27 season, first week)_

## How 2025-26 went
Final model board (2026-04-15 run, live on Pages): Jokić #1, Gilgeous-Alexander #2, Wembanyama #3, Dončić #4, Cunningham #5, Brown #6.
Actual vote: Gilgeous-Alexander 939 pts (83 first-place), Jokić 634 (10), Wembanyama 569 (5), Dončić 250, Cunningham 117, Brown 89; Mitchell and Leonard 1 pt each.
The model had the actual top 6 in order except #1/#2 swapped: winner ranked 2nd, top 3 3/3, top 5 5/5. Mitchell was #8, Leonard #23.
Misses were false positives, not vote-getters: Duren (#7), Murray (#9) and Holmgren (#10) got no votes. The 0.20 team-win% blend lifts role players on top teams.

In-season (shipped model replayed on the 9 weekly cached inputs in git history, 2026-02-02 → 04-14):
- SGA #2 every week; Jokić #1 every week. The ridge alone had SGA 3rd (0.39 vs Jokić 0.62, Dončić 0.46); the team blend moved him to 2nd.
- The actual top 5 was inside the model's top 5 every week.
- The same pattern in 2024-25 (backtest, final stats): SGA (winner) #2 behind Jokić. Two straight seasons the model's one miss was SGA vs Jokić.

## Backtest
Leave-one-season-out, 1980-81 through 2025-26 (46 seasons). Every player with ≥5 games and ≥15 min/game is scored (median 284 per season), not just vote-getters. Season totals (games, WS, VORP) are scaled to 82 team games, so 1999, 2012 and the 2020–21 seasons compare with full ones.

| model | winner #1 | top 3 | top 5 | mean winner rank | actual top 5 in model top 5 |
|---|---|---|---|---|---|
| **Shipped (E0): ridge on vote-getters + team blend** | **72%** | **93%** | **96%** | **1.70** | **3.65** |
| 2011+ only (16 seasons) | 69% | 100% | 100% | 1.38 | 3.62 |

The shipped model had never been backtested as a board; its training script reported 5-fold MAE on vote-getters only. It is better than that suggested.

In-season (date-cut stats via nba_api at Dec 1, Jan 1, Feb 1, Mar 1, Apr 1 of 2021-22 through 2025-26; 25 checkpoints; models trained without the season being replayed):

| model | mean winner rank | winner top 3 | winner top 10 | actual top 5 in top 10 |
|---|---|---|---|---|
| **Shipped (E0)** | **2.28** | **88%** | 100% | 4.4 |
| Ridge with team win% inside (E2) | 2.44 | 84% | 100% | 4.5 |
| NFL target, box + win% (E10) | 2.64 | 80% | 100% | 4.6 |
| NHL blend, box + win% (E9) | 3.20 | 64% | 100% | 4.6 |

Worst checkpoints for the shipped model: Embiid 2022-23 #5 on Dec 1, #4 on Jan 1; Jokić 2021-22 #4 on Dec 1. Every eventual winner was top 5 by Jan 1 in all five seasons.

## Problems found
1. **Training data had a duplicated season.** The hand-exported vote seed had 1983-84's vote table filed as 1984-85 (Jordan's and Olajuwon's rookie season missing, Bird credited with 0.858 instead of 0.978), Caldwell Jones under the wrong season, and two truncated player IDs (`thomais0`, `leonaka0`).
2. **Accents lost.** The seed was saved as cp1252: 15 rows had `?` in names (`Nikola Joki?` → key `nikola_joki`).
3. **Training data not in git.** Seed, normalized and training CSVs were all gitignored; the model couldn't be rebuilt from the repo.
4. **No board-level validation.** No backtest existed; the team blend weights (0.70 / 0.20 / 0.10) were never tested.
5. **Movement broke on reruns.** The prior was "whatever is on Pages", so the 04-14 → 04-15 rerun showed `—` for everyone.
6. Committed `docs/nba/mvp/` copies are from 2026-01-24; Pages serves the workflow's artifact, so the site was current but the repo wasn't.
7. Local-only data refresh (stats.nba.com blocks runners), a 10-day freshness gate, no position column, `player_key` without team, no chips/graphics automation, `requirements.txt` saved as UTF-16.

## Experiments
Same backtest for all. "Pool" = trained on every scored player; "voters" = vote-getters only (as shipped).

| spec | change | winner #1 | top 3 | top 5 | mean rank | top 5 in top 5 | 2011+ #1 / mean |
|---|---|---|---|---|---|---|---|
| E0 | **shipped** | 72% | 93% | 96% | 1.70 | 3.65 | 69% / 1.38 |
| E0b | shipped ridge, no team blend | 46% | 76% | 89% | 2.83 | 3.35 | 50% / 2.19 |
| E1 | E0 trained on every player | 50% | 87% | 96% | 2.33 | 3.72 | 44% / 2.06 |
| E2 | team win% inside the ridge (voters) | 70% | 93% | 96% | 1.74 | 3.76 | 75% / 1.31 |
| E3 | team win% inside (pool) | 46% | 80% | 89% | 2.96 | 3.33 | 44% / 2.19 |
| E4 | E3 + WS, WS/48 | 52% | 91% | 93% | 2.28 | 3.50 | 56% / 1.50 |
| E5 | E3 + WS, WS/48, BPM, VORP, USG% | 50% | 87% | 93% | 2.57 | 3.41 | 50% / 1.50 |
| E6 | E5 features, NFL target (any votes, share-weighted logit) | 70% | 96% | 96% | 1.85 | 3.83 | 69% / 1.31 |
| E7 | E5 features, top-5 classifier | 70% | 96% | 96% | 1.80 | 3.76 | 69% / 1.31 |
| E8 | E5 features, NHL blend (GBR vote share + top-5 classifier) | 76% | 96% | 96% | 1.67 | 3.80 | 75% / 1.25 |
| E9 | E8 without advanced stats | 70% | 98% | 98% | 1.57 | 3.70 | 63% / 1.44 |
| E10 | E6 without advanced stats | 67% | 96% | 96% | 1.80 | 3.70 | 69% / 1.38 |
| E11 | E2 + advanced stats | 70% | 93% | 96% | 1.87 | 3.78 | 75% / 1.31 |
| E12 | E8 + team SRS, conference rank | 74% | 96% | 96% | 1.74 | 3.83 | 75% / 1.31 |
| E13 | E8 + 2024+ eligibility (G ≥ 62) | 76% | 96% | 96% | 1.67 | 3.80 | 75% / 1.25 |
| E14 | E0 + 2024+ eligibility (G ≥ 62) | 72% | 93% | 96% | 1.70 | 3.65 | 69% / 1.38 |

What the experiments say:
- **Team strength matters most**, and how it enters doesn't: blended after (E0) or inside the ridge (E2) score the same. Without it the winner is #1 only 46% of the time (E0b).
- **Selection bias: adding non-vote-getters hurts a ridge** (E1 50%, E3 46%). A linear fit on a target that is 0 for 97% of rows flattens the top. Classifiers (E6–E8) handle every-player training fine, but don't beat E0 by much.
- **Advanced stats add nothing reliable.** WS/BPM/VORP help the classifiers a little (E8 vs E9 on winner #1, not on mean rank) and don't help the ridge (E11 vs E2).
- **E8 is the best full-season model, narrowly.** Season by season vs E0: winner ranked better in 7 seasons, worse in 5; top-5 overlap better in 12, worse in 7. It gets SGA #1 in 2024-25 and 2025-26, where E0 had him #2. But it can't be replayed in-season (Basketball-Reference has no date-cut advanced stats), and its replayable version (E9) is the **worst** model in-season (mean winner rank 3.20 vs 2.28), with the eventual winner #5–9 in December and January of the Jokić and Embiid seasons.
- **65-game rule.** Basketball-Reference's G undercounts eligibility: Dončić, Wembanyama and Cunningham got votes in 2025-26 with 64 G. A hard G ≥ 65 filter would have removed 3 of the actual top 5. G ≥ 62 removes no 2024–26 contender (E13, E14 unchanged).
- **In-season eligibility.** Tested on the replay: "on pace for 62 games" dropped an eventual winner (Embiid, 2022-12-01, after early missed games); "can still reach 62" (G + team games left ≥ 62) and NHL's "≥ 60% of team games" dropped nobody and left ranks unchanged.
- **Pace projection** (games to an 82-game pace) made no difference in the replay: every other feature is already a per-game average. Kept for scale consistency with training, not for accuracy.
- The 2005 and 2006 Steve Nash MVPs are the stats-only ceiling: every spec has him #8–#13 (E9's #1 in 2005 is a fluke, not reproduced in 2006).

Rejected: training a ridge on every player (E1, E3–E5), advanced stats in the ridge (E4, E5, E11), team SRS / conference rank as extra features (E12), and the NFL target and top-5 classifier (E6, E7, E10). Each was even with or worse than E0.

## Decision (approved 2026-10-30)
**Keep the shipped model (E0), retrained through 2025-26 on the corrected history.** Shipped 2026-10-30. Same 10 features, same post-hoc team blend, same JSON format, so `build_rankings_ridge_np_v1.py` reads it unchanged. Weights barely move (largest change: blocks 0.034 → 0.042, steals 0.026 → 0.021). Candidate artifact: `models/nba_mvp_ridge_np.json` from `python -m src.model.train_mvp`, which now prints the leave-one-season-out table on every retrain and saves `models/backtest_mvp.{csv,json}`.

E8 (NHL blend + advanced stats) is the one challenger worth watching. Running it as an unpublished weekly shadow score was considered and deferred; revisit after the 2026-27 vote with `python -m src.model.experiments E0 E8`.

Weekly scoring changes:
- Eligibility: ≥5 games, ≥15 min/game (as before) **and** can still reach 62 games (G + team games remaining ≥ 62).
- Games played projected to an 82-game pace before scoring; displayed stats stay actual.

## 2026-27 build plan (port of nhl-awards-predictor patterns)
1. **Data from Basketball-Reference, unattended.** One source for training and live (`src/nba_common.py: season_players`): per-game + standings weekly (2–3 fetches), positions included. `probe_sources.yml` (manual) confirms it answers from a runner before the freshness gate and local `nba_api` refresh are retired; `nba_api` scripts stay as a local fallback.
2. `config.py` single source of truth (`SEASON_END_YEAR = 2027`, `TRAIN_END = 2026`); `requirements.txt` re-saved as UTF-8 with `anthropic` and `pillow`; workflow on Python 3.12.
3. **Board** in `predictions/2027/`: `top10_mvp_latest.csv` and 30-deep `board_mvp_latest.csv` (same columns, ending `chips, why_blurb`), 4 timestamped archives each, hidden `.prev_board_mvp.csv` rolled only when the week advances. First week shows NEW for everyone. Never written if fewer than 15 players are eligible or if a same-week rerun would be shorter; chips reused from this week's saved file if the API fails.
4. **Columns** (the contract, listed in CLAUDE.md): `player_key, rank, rank_delta, movement, player_name, team, position, model_score, games, team_games, minutes, pts, reb, ast, stl, blk, fg_pct, fg3_pct, ft_pct, team_record, team_win_pct, conf_rank, conference, season, week, stats_through, updated_utc, chips, why_blurb`.
5. **Keys and assets.** `player_key = first_last_team` (NHL convention, `safe_filename(f"{name} {team}")`), e.g. `shai_gilgeous_alexander_okc`. Teams use NBA codes (Basketball-Reference `BRK/CHO/PHO` → `BKN/CHA/PHX`), so every board team maps to an existing lowercase logo (`gsw, nyk, sas, nop, uta, was`; the ESPN short duplicates `gs, ny, sa, no, utah, wsh` go unused) with no alias map. `predictions/nba_candidates.csv` lists the 30-deep board. stinger-assets (separate proposal): NBA candidates URL in `merge_candidates.py`; NBA section in `SPORT_TEAM_CODE_ALIASES` (`GSW→GS, NOP→NO, NYK→NY, SAS→SA, UTA→UTAH`; `WAS→WSH` already shared). The 19 `first_last.png` headshots stay; the legacy feed keeps the old keys.
6. **Chips** (`claude-haiku-5-5`): Chip + Why for the top 3; ≤75-char Why for ranks 11–15 vs the lowest-ranked top-10 player in the same role (G = PG/SG, F = SF/PF, C).
7. **Graphics**: main board on `NBA_background.png` (light wood: text on dark plates, as on NHL's ice); "On the Bubble" slide from a ported `make_bubble_bg.py`; `caption_mvp.txt` for the carousel.
8. **Legacy feed**: `web_current_top10.csv`, `web_prior_top10.csv`, `web_meta.csv`, `model_all_candidates.csv`, `canva_top5_flat.csv`, `canva_6_10_flat.csv` keep their columns and keep deploying to Pages until the site switches to `board_mvp_latest.csv`.
9. **Workflow**: Tuesdays 17:43 UTC, Oct–Apr, skipping before opening night and after the regular season; stinger-assets sibling checkout; `ANTHROPIC_API_KEY` to the chips/caption step; commits `predictions/` with `git add -f` of the snapshot.
