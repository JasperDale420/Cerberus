from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Iterable, List

import httpx
import pandas as pd


@dataclass(frozen=True)
class UniverseSource:
    name: str
    url: str
    symbol_column_candidates: List[str]


SOURCES = [
    UniverseSource(
        name="sp500",
        url="https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        symbol_column_candidates=["Symbol", "Ticker symbol", "Ticker"],
    ),
    UniverseSource(
        name="nasdaq100",
        url="https://en.wikipedia.org/wiki/Nasdaq-100",
        symbol_column_candidates=["Ticker", "Ticker symbol", "Symbol"],
    ),
]


def _normalize_symbol(sym: str) -> str:
    s = str(sym).strip().upper()
    # Keep '.' tickers (e.g., BRK.B) as-is; broker adapters can map if needed.
    return s


def _extract_symbols_from_tables(
    tables: Iterable[pd.DataFrame], column_candidates: List[str]
) -> List[str]:
    for t in tables:
        cols = [str(c) for c in t.columns]
        for c in column_candidates:
            if c in cols:
                syms = [_normalize_symbol(x) for x in t[c].tolist()]
                out = [
                    s
                    for s in syms
                    if s
                    and s.isascii()
                    and s.replace(".", "").replace("-", "").isalnum()
                ]
                if out:
                    return out
    raise RuntimeError(f"Could not find a symbol column in tables: {column_candidates}")


def _write_list(path: Path, name: str, url: str, symbols: List[str]) -> None:
    now = datetime.now(timezone.utc).isoformat()
    symbols_sorted = sorted(set(symbols))
    lines = [
        f"# {name} universe snapshot\n",
        f"# source: {url}\n",
        f"# generated_at_utc: {now}\n",
        "\n",
    ] + [f"{s}\n" for s in symbols_sorted]
    path.write_text("".join(lines))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "config"
    out_dir.mkdir(parents=True, exist_ok=True)

    for src in SOURCES:
        resp = httpx.get(
            src.url,
            headers={"User-Agent": "CerberusBot/1.0 (repo-to-PRD audit)"},
            timeout=30,
        )
        resp.raise_for_status()
        tables = pd.read_html(StringIO(resp.text))
        symbols = _extract_symbols_from_tables(tables, src.symbol_column_candidates)
        out_path = out_dir / f"universe_{src.name}.txt"
        _write_list(out_path, src.name, src.url, symbols)
        print(f"Wrote {len(set(symbols))} symbols to {out_path}")


if __name__ == "__main__":
    main()
