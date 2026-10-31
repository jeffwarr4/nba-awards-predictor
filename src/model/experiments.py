"""
Run every spec in SPECS through the leave-one-season-out backtest and print one
comparison table (all seasons, and 2011+ separately). Per-season results go to
models/experiments/{name}.csv. This produced the tables in
docs/nba-2026-evaluation-and-plan.md.

Run:
    python -m src.model.experiments              # all specs
    python -m src.model.experiments E0 E5        # just these
"""

from __future__ import annotations

import sys

import pandas as pd

from config import HISTORY_CSV, MODELS_DIR
from src.model.mvp_models import BOX, Spec, backtest, load_history, summarize

BOX_RAW_G = ["g_raw" if c == "g" else c for c in BOX]   # the shipped model saw raw games played
WIN = ["team_win_pct"]
WS = ["ws", "ws_per_48"]
ADV = ["ws", "ws_per_48", "bpm", "vorp", "usg_pct"]

SPECS = [
    Spec("E0", BOX_RAW_G, train_rows="voters", blend_team=True, notes="shipped: ridge on vote-getters + team blend"),
    Spec("E0b", BOX_RAW_G, train_rows="voters", notes="shipped ridge, no team blend"),
    Spec("E1", BOX, train_rows="pool", blend_team=True, notes="E0 trained on every player"),
    Spec("E2", BOX + WIN, train_rows="voters", notes="team win% inside, vote-getters"),
    Spec("E3", BOX + WIN, train_rows="pool", notes="team win% inside, every player"),
    Spec("E4", BOX + WIN + WS, train_rows="pool", notes="+ WS, WS/48"),
    Spec("E5", BOX + WIN + ADV, train_rows="pool", notes="+ WS, WS/48, BPM, VORP, USG%"),
    Spec("E6", BOX + WIN + ADV, model="logit_share", notes="NFL target: any votes, share-weighted"),
    Spec("E7", BOX + WIN + ADV, model="logit_top5", notes="top-5 finish classifier"),
    Spec("E8", BOX + WIN + ADV, model="gbr_top5", notes="NHL: GBR vote share + top-5 classifier"),
    # Follow-ups
    Spec("E9", BOX + WIN, model="gbr_top5", notes="E8 without advanced stats"),
    Spec("E10", BOX + WIN, model="logit_share", notes="E6 without advanced stats"),
    Spec("E11", BOX + WIN + ADV, train_rows="voters", notes="E2 + advanced stats"),
    Spec("E12", BOX + WIN + ADV + ["srs", "neg_conf_rank"], model="gbr_top5", notes="E8 + team SRS, conf rank"),
    Spec("E13", BOX + WIN + ADV, model="gbr_top5", min_games_2024=62, notes="E8 + 2024+ eligibility (G >= 62)"),
    Spec("E14", BOX_RAW_G, train_rows="voters", blend_team=True, min_games_2024=62, notes="E0 + 2024+ eligibility (G >= 62)"),
]


def table_row(spec: Spec, bt: pd.DataFrame) -> dict:
    a, r = summarize(bt), summarize(bt[bt["season"] >= 2011])
    return {"spec": spec.name, "winner#1": a["winner_rank1_pct"], "top3": a["winner_top3_pct"],
            "top5": a["winner_top5_pct"], "mean_rank": a["winner_rank_mean"], "top5_in_top5": a["top5_in_top5_mean"],
            "2011+ #1": r["winner_rank1_pct"], "2011+ top3": r["winner_top3_pct"],
            "2011+ mean": r["winner_rank_mean"], "2011+ t5in5": r["top5_in_top5_mean"], "notes": spec.notes}


def main(names: list[str]) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    hist = load_history(HISTORY_CSV)
    out = MODELS_DIR / "experiments"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in [s for s in SPECS if not names or s.name in names]:
        bt = backtest(hist, spec)
        bt.to_csv(out / f"{spec.name}.csv", index=False)
        rows.append(table_row(spec, bt))
        print(f"{spec.name}: {rows[-1]}", flush=True)
    print("\n" + pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:])
