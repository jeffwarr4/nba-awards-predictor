from __future__ import annotations

from pathlib import Path
import shutil

REPO_ROOT = Path(__file__).resolve().parents[2]

OUT_DIR = REPO_ROOT / "data" / "processed" / "outputs"
PAGES_DIR = REPO_ROOT / "docs" / "nba" / "mvp"

FILES = [
    "model_all_candidates.csv",
    "web_current_top10.csv",
    "web_prior_top10.csv",
    "web_meta.csv",
    "canva_top5_flat.csv",
    "canva_6_10_flat.csv"
]

def main() -> None:
    if not OUT_DIR.exists():
        raise FileNotFoundError(f"Missing outputs directory: {OUT_DIR}")

    PAGES_DIR.mkdir(parents=True, exist_ok=True)

    copied = 0
    for fname in FILES:
        src = OUT_DIR / fname
        if not src.exists():
            raise FileNotFoundError(f"Missing output file: {src}")

        dst = PAGES_DIR / fname
        shutil.copyfile(src, dst)
        copied += 1

    print(f"[PUBLISHED] {copied} files to {PAGES_DIR}")

if __name__ == "__main__":
    main()
