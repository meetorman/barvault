from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import pandas as pd

from market_data.providers.base import BaseProvider
from market_data.normalize.symbols import normalize_symbol
from market_data.types import Bar, iter_month_starts, parse_ts


@dataclass(frozen=True, slots=True)
class PolygonConfig:
    api_key: Optional[str] = None
    base_url: str = "https://api.polygon.io"


class PolygonProvider(BaseProvider):
    """
    Polygon provider using `polygon-api-client`.

    Notes:
    - No env var lookups (caller passes `PolygonConfig(api_key=...)`).
    - No network calls at import-time.
    """

    def __init__(self, config: PolygonConfig):
        if not config.api_key:
            raise ValueError("PolygonConfig.api_key is required")
        self._config = config

        # Lazy-ish: constructing the client is cheap; calls happen only in fetch methods.
        from polygon import RESTClient

        self._client = RESTClient(api_key=config.api_key, base=config.base_url)

    def fetch_1m_bars(self, symbol: str, *, start: pd.Timestamp, end: pd.Timestamp) -> Iterable[Bar]:
        sym = normalize_symbol(symbol)
        start_ts = parse_ts(start)
        end_ts = parse_ts(end)
        if end_ts <= start_ts:
            return []

        out: list[Bar] = []
        # Fetch in month-sized chunks to avoid large-range truncation and reduce requests.
        for month_start in iter_month_starts(start_ts, end_ts):
            if month_start >= end_ts:
                break
            month_end = (month_start + pd.offsets.MonthBegin(1)).normalize()
            if month_end > end_ts:
                month_end = end_ts

            # Polygon aggregates are day-based; request full days for the month chunk
            # then filter to [start,end).
            aggs = self._client.get_aggs(
                ticker=sym,
                multiplier=1,
                timespan="minute",
                from_=month_start.strftime("%Y-%m-%d"),
                to=month_end.strftime("%Y-%m-%d"),
                limit=50000,
            )

            for bar in aggs or []:
                # Polygon timestamps are epoch milliseconds (UTC).
                ts = pd.Timestamp(bar.timestamp, unit="ms", tz="UTC")
                if ts < start_ts or ts >= end_ts:
                    continue
                out.append(
                    Bar(
                        symbol=sym,
                        ts=ts,
                        open=float(bar.open),
                        high=float(bar.high),
                        low=float(bar.low),
                        close=float(bar.close),
                        volume=int(bar.volume),
                    )
                )

        # Deterministic order.
        out.sort(key=lambda b: (b.symbol, b.ts))
        return out

