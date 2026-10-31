"""
Basketball-Reference fetch + table parsing, shared by the vote fetcher, the history
builder and live scoring.

- fetch_html: throttled (Sports-Reference allows ~20 requests/minute), retries 429/5xx
  with backoff, and caches pages under data/raw/bbref_cache/ (completed seasons never
  change; pass cache=False for live pages).
- parse_table: rows of a table by id as dicts keyed by each cell's data-stat, plus
  player_id from data-append-csv. Basketball-Reference hides some tables inside HTML
  comments, so comment markers are stripped first.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / "data" / "raw" / "bbref_cache"
BASE = "https://www.basketball-reference.com"

MIN_INTERVAL = 3.5   # seconds between requests
_last_request = 0.0


def fetch_html(path: str, cache: bool = True, retries: int = 5) -> str:
    """GET BASE + path ('/leagues/NBA_2026_per_game.html')."""
    global _last_request
    cache_path = CACHE_DIR / path.strip("/").replace("/", "__")
    if cache and cache_path.exists():
        return cache_path.read_text(encoding="utf-8")

    for attempt in range(retries + 1):
        wait = MIN_INTERVAL - (time.time() - _last_request)
        if wait > 0:
            time.sleep(wait)
        _last_request = time.time()
        r = requests.get(BASE + path, timeout=30)
        if r.status_code == 200:
            r.encoding = "utf-8"
            if cache:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(r.text, encoding="utf-8")
            return r.text
        if (r.status_code == 429 or r.status_code >= 500) and attempt < retries:
            backoff = int(r.headers.get("Retry-After", 0) or 0) or 30 * (attempt + 1)
            print(f"  [WARN] {r.status_code} on {path}; retrying in {backoff}s")
            time.sleep(backoff)
            continue
        r.raise_for_status()
    raise RuntimeError(f"Failed to fetch {path}")


def parse_table(html: str, table_id: str) -> list[dict]:
    """Body rows of <table id=table_id>. Header/spacer rows (class thead) are skipped.
    Each row: {data-stat: text}, plus 'player_id' (and 'team_href' code) when present."""
    html = html.replace("<!--", "").replace("-->", "")
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", id=table_id)
    if table is None:
        return []
    rows = []
    for tr in table.find("tbody").find_all("tr"):
        if "thead" in (tr.get("class") or []):
            continue
        row = {}
        for cell in tr.find_all(["th", "td"]):
            stat = cell.get("data-stat")
            if not stat:
                continue
            row[stat] = cell.get_text(strip=True)
            if stat in ("player", "name_display") and cell.get("data-append-csv"):
                row["player_id"] = cell["data-append-csv"]
            a = cell.find("a", href=re.compile(r"^/teams/"))
            if a is not None:
                row[f"{stat}_href"] = a["href"].split("/")[2]
        if row:
            rows.append(row)
    return rows
