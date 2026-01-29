from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import pandas as pd

from market_data.normalize.bars import normalize_1m_bars_frame
from market_data.normalize.calendars import Calendar


@dataclass(frozen=True, slots=True)
class GapReport:
    start: pd.Timestamp
    end: pd.Timestamp
    missing_by_symbol: dict[str, pd.DatetimeIndex]

    @property
    def total_missing(self) -> int:
        return int(sum(len(v) for v in self.missing_by_symbol.values()))


def find_gaps(
    bars_1m: pd.DataFrame,
    *,
    calendar: Calendar,
    start: pd.Timestamp,
    end: pd.Timestamp,
    symbols: Sequence[str] | None = None,
) -> GapReport:
    """
    Compute missing expected 1-minute timestamps per symbol for [start, end).
    """
    start_utc = pd.Timestamp(start, tz="UTC") if pd.Timestamp(start).tz is None else pd.Timestamp(start).tz_convert("UTC")
    end_utc = pd.Timestamp(end, tz="UTC") if pd.Timestamp(end).tz is None else pd.Timestamp(end).tz_convert("UTC")

    df = normalize_1m_bars_frame(bars_1m)
    if symbols is not None:
        want = set(symbols)
        df = df[df["symbol"].isin(want)]

    expected = calendar.expected_minutes(start_utc, end_utc)
    missing_by_symbol: dict[str, pd.DatetimeIndex] = {}

    for sym, g in df.groupby("symbol", sort=True):
        observed = pd.DatetimeIndex(g["ts"], tz="UTC")
        missing = expected.difference(observed)
        missing_by_symbol[sym] = missing

    # If caller passed symbols and some are absent, still return them as fully missing.
    if symbols is not None:
        for sym in sorted(set(symbols)):
            if sym not in missing_by_symbol:
                missing_by_symbol[sym] = expected

    return GapReport(start=start_utc, end=end_utc, missing_by_symbol=missing_by_symbol)

