"""
Retrain the MVP model: leave-one-season-out backtest over every season in
config.HISTORY_CSV, then fit the shipped spec on all of them.

Writes (to --outdir, default models/):
  nba_mvp_ridge_np.json   the model, in the format src/model/build_rankings_ridge_np_v1.py reads
  backtest_mvp.csv        per-season winner rank and top-5 overlap
  backtest_mvp.json       summary (all seasons and 2011+), spec, trained seasons

Run:
    python -m src.model.train_mvp                       # backtest + fit + save to models/
    python -m src.model.train_mvp --outdir <scratch>    # try it without touching models/
    python -m src.model.train_mvp --no-backtest
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from config import HISTORY_CSV, MODELS_DIR
from src.model.experiments import SPECS
from src.model.mvp_models import Model, backtest, load_history, pool_mask, summarize

SHIPPED = "E0"   # docs/nba-2026-evaluation-and-plan.md

# History column → the live candidates column the weekly scorer reads
LIVE_NAMES = {"g_raw": "games_played", "mp": "minutes_played"}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", type=Path, default=MODELS_DIR)
    p.add_argument("--no-backtest", action="store_true")
    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    spec = next(s for s in SPECS if s.name == SHIPPED)
    hist = load_history(HISTORY_CSV)
    seasons = sorted(int(s) for s in hist["season_end_year"].unique())
    print(f"[INFO] {len(hist):,} player-seasons, {seasons[0]}–{seasons[-1]}; spec {spec.name}: {spec.notes}")

    summary = None
    if not args.no_backtest:
        print("[INFO] Leave-one-season-out backtest...")
        bt = backtest(hist, spec)
        bt.to_csv(args.outdir / "backtest_mvp.csv", index=False)
        summary = {"all": summarize(bt), "2011+": summarize(bt[bt["season"] >= 2011])}
        print("\n" + bt.to_string(index=False))
        print(f"\n[METRIC] all seasons: {summary['all']}")
        print(f"[METRIC] 2011+:       {summary['2011+']}")

    train = hist[pool_mask(hist) | (hist["vote_share"] > 0)]
    model = Model(spec).fit(train)
    if spec.model != "ridge":
        raise SystemExit(f"Saving is only wired for ridge specs; {spec.name} is {spec.model}")
    voters = train[train["vote_share"] > 0]
    payload = {
        "model_type": "ridge_closed_form",
        "alpha": spec.alpha,
        "target": "vote_share",
        "features": [LIVE_NAMES.get(f, f) for f in spec.features],
        "intercept_and_weights": model.w.tolist(),
        "x_mean": model.mu.tolist(),
        "x_std": model.sd.tolist(),
        "train_rows": int(len(voters) if spec.train_rows == "voters" else len(train)),
        "train_seasons": len(seasons),
        "trained_seasons": [seasons[0], seasons[-1]],
        "post_blend": "0.70 minmax(ridge) + 0.20 minmax(team win%) + 0.10 minmax(-conf rank)" if spec.blend_team else None,
    }
    (args.outdir / "nba_mvp_ridge_np.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (args.outdir / "backtest_mvp.json").write_text(json.dumps(
        {"spec": spec.name, "notes": spec.notes, "features": spec.features, "trained_seasons": seasons,
         "evaluation": "leave-one-season-out, every player with >=5 games and >=15 min/game scored",
         "backtest_summary": summary}, indent=2), encoding="utf-8")
    print(f"[SAVED] {args.outdir / 'nba_mvp_ridge_np.json'}, backtest_mvp.csv, backtest_mvp.json")


if __name__ == "__main__":
    main()
