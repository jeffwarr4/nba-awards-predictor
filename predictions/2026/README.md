# 2025-26 NBA season: final board

`board_mvp_final.csv` is the model's final MVP top 10 for 2025-26. That season's workflow published to GitHub Pages
without committing its outputs, and the April runs' artifacts and logs have expired, so this file was rebuilt on
2026-11-16 by re-running that season's model (`src.model.build_rankings_ridge_np_v1`, commit 99995e4) on the cached
inputs committed on 2026-04-14, two days after the regular season ended. Columns are renamed to the 2027 board's names
(`games` = games_played, `reb` = trb, `model_score` = mvp_score) so the site's Past Results page can read it. Frozen.
