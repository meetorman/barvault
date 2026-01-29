from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from market_data.normalize.symbols import normalize_symbol
from market_data.types import ArchiveConfig


@dataclass(frozen=True, slots=True)
class ArchiveFileDetail:
    symbol: str
    year: int
    month: int
    path: str


def partition_path(cfg: ArchiveConfig, *, symbol: str, year: int, month: int) -> str:
    sym = normalize_symbol(symbol)
    mm = f"{month:02d}"
    return f"{cfg.dataset_uri()}/symbol={sym}/year={year}/month={mm}/data.parquet"


def partition_from_month_start(ts: pd.Timestamp) -> tuple[int, int]:
    ts = pd.Timestamp(ts)
    ts = ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")
    return int(ts.year), int(ts.month)

