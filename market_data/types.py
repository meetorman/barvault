from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Literal, Optional, Sequence, Union

import pandas as pd

Timeframe = Literal["1m"]
Dataset = Literal["bars"]

BarsFrame = pd.DataFrame


def parse_ts(x: Union[str, datetime, pd.Timestamp]) -> pd.Timestamp:
    ts = x if isinstance(x, pd.Timestamp) else pd.Timestamp(x)
    if ts.tz is None:
        # Treat naive timestamps as UTC for determinism.
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


@dataclass(frozen=True, slots=True)
class Bar:
    symbol: str
    ts: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass(frozen=True, slots=True)
class ArchiveConfig:
    """
    Stateless archive configuration.

    Layout:
      {root}/{dataset}/{timeframe}/symbol={SYMBOL}/year=YYYY/month=MM/data.parquet

    Boundary semantics for reads are standardized in ParquetReader:
      start is inclusive, end is exclusive: [start, end)
    """

    root: str
    dataset: Dataset = "bars"
    timeframe: Timeframe = "1m"
    source: Optional[str] = None

    @staticmethod
    def local(root: str, *, dataset: Dataset = "bars", timeframe: Timeframe = "1m") -> "ArchiveConfig":
        return ArchiveConfig(root=root, dataset=dataset, timeframe=timeframe)

    @staticmethod
    def s3(bucket: str, prefix: str = "", *, dataset: Dataset = "bars", timeframe: Timeframe = "1m") -> "ArchiveConfig":
        prefix = prefix.lstrip("/")
        base = f"s3://{bucket}"
        root = f"{base}/{prefix}" if prefix else base
        return ArchiveConfig(root=root, dataset=dataset, timeframe=timeframe)

    def base_uri(self) -> str:
        return self.root.rstrip("/")

    def dataset_uri(self) -> str:
        return f"{self.base_uri()}/{self.dataset}/{self.timeframe}"


def ensure_str_seq(symbols: Union[str, Sequence[str], None]) -> list[str]:
    if symbols is None:
        return []
    if isinstance(symbols, str):
        return [symbols]
    return list(symbols)


def iter_month_starts(start: pd.Timestamp, end: pd.Timestamp) -> Iterable[pd.Timestamp]:
    """
    Generate month-start timestamps for partitions intersecting [start, end).
    """
    start = parse_ts(start).normalize().replace(day=1)
    end = parse_ts(end).normalize().replace(day=1)
    cur = start
    while cur <= end:
        yield cur
        cur = (cur + pd.offsets.MonthBegin(1)).normalize()

