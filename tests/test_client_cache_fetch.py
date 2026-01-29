from __future__ import annotations

import pandas as pd
import pytest

from market_data import ArchiveConfig, MarketDataClient, NYSECalendar, ParquetWriter
from market_data.providers.base import BaseProvider
from market_data.types import Bar


class FakeProvider(BaseProvider):
    def __init__(self, make_bars):
        self.calls: list[tuple[str, pd.Timestamp, pd.Timestamp]] = []
        self._make_bars = make_bars

    def fetch_1m_bars(self, symbol: str, *, start: pd.Timestamp, end: pd.Timestamp):
        self.calls.append((symbol, start, end))
        return list(self._make_bars(symbol, start, end))


def _make_linear_bars(symbol: str, start: pd.Timestamp, end: pd.Timestamp):
    # Deterministic synthetic bars for every minute in [start,end).
    mins = pd.date_range(start, end, freq="1min", inclusive="left", tz="UTC")
    for i, ts in enumerate(mins):
        base = float(i + 1)
        yield Bar(
            symbol=symbol,
            ts=ts,
            open=base,
            high=base + 0.2,
            low=base - 0.1,
            close=base + 0.1,
            volume=100 + i,
        )


def test_client_cache_miss_fetches_and_archives(tmp_path):
    cfg = ArchiveConfig.local(root=str(tmp_path))
    provider = FakeProvider(_make_linear_bars)
    client = MarketDataClient(cfg, provider=provider, calendar=NYSECalendar())

    out = client.get_bars(
        ["AAPL"],
        start="2025-01-02T14:30:00Z",
        end="2025-01-02T14:33:00Z",
        timeframe="1min",
    )

    assert len(provider.calls) == 1
    assert out[["symbol", "ts"]].values.tolist() == [
        ["AAPL", pd.Timestamp("2025-01-02T14:30:00Z")],
        ["AAPL", pd.Timestamp("2025-01-02T14:31:00Z")],
        ["AAPL", pd.Timestamp("2025-01-02T14:32:00Z")],
    ]


def test_client_cache_hit_does_not_call_provider(tmp_path):
    cfg = ArchiveConfig.local(root=str(tmp_path))
    writer = ParquetWriter(cfg)

    # Pre-seed the archive with complete data.
    bars = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL", "AAPL"],
            "ts": [
                pd.Timestamp("2025-01-02T14:30:00Z"),
                pd.Timestamp("2025-01-02T14:31:00Z"),
                pd.Timestamp("2025-01-02T14:32:00Z"),
            ],
            "open": [10.0, 11.0, 12.0],
            "high": [10.2, 11.2, 12.2],
            "low": [9.9, 10.9, 11.9],
            "close": [10.1, 11.1, 12.1],
            "volume": [1000, 1001, 1002],
        }
    )
    writer.write_1m(bars)

    provider = FakeProvider(_make_linear_bars)
    client = MarketDataClient(cfg, provider=provider, calendar=NYSECalendar())

    out = client.get_bars(
        ["AAPL"],
        start="2025-01-02T14:30:00Z",
        end="2025-01-02T14:33:00Z",
        timeframe="1min",
    )

    assert provider.calls == []
    assert out[["symbol", "ts"]].values.tolist() == [
        ["AAPL", pd.Timestamp("2025-01-02T14:30:00Z")],
        ["AAPL", pd.Timestamp("2025-01-02T14:31:00Z")],
        ["AAPL", pd.Timestamp("2025-01-02T14:32:00Z")],
    ]


def test_client_partial_gap_fetches_and_merges(tmp_path):
    cfg = ArchiveConfig.local(root=str(tmp_path))
    writer = ParquetWriter(cfg)

    # Archive is missing 14:31.
    existing = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL"],
            "ts": [
                pd.Timestamp("2025-01-02T14:30:00Z"),
                pd.Timestamp("2025-01-02T14:32:00Z"),
            ],
            "open": [999.0, 999.0],
            "high": [999.2, 999.2],
            "low": [998.9, 998.9],
            "close": [999.1, 999.1],
            "volume": [9990, 9992],
        }
    )
    writer.write_1m(existing)

    provider = FakeProvider(_make_linear_bars)
    client = MarketDataClient(cfg, provider=provider, calendar=NYSECalendar())

    out = client.get_bars(
        ["AAPL"],
        start="2025-01-02T14:30:00Z",
        end="2025-01-02T14:33:00Z",
        timeframe="1min",
    )

    # Provider called due to missing expected minute.
    assert len(provider.calls) == 1

    # Merge semantics keep provider values for duplicated minutes (writer drop_duplicates keep='last').
    assert out["open"].tolist() == [1.0, 2.0, 3.0]


def test_client_returns_resampled_timeframe(tmp_path):
    cfg = ArchiveConfig.local(root=str(tmp_path))
    provider = FakeProvider(_make_linear_bars)
    client = MarketDataClient(cfg, provider=provider, calendar=NYSECalendar())

    out = client.get_bars(
        ["AAPL"],
        start="2025-01-02T14:30:00Z",
        end="2025-01-02T14:40:00Z",
        timeframe="5min",
    )

    # 10 minutes -> two 5-minute bars labeled at the window start.
    assert out[["symbol", "ts"]].values.tolist() == [
        ["AAPL", pd.Timestamp("2025-01-02T14:30:00Z")],
        ["AAPL", pd.Timestamp("2025-01-02T14:35:00Z")],
    ]
    assert out["volume"].tolist() == [sum(100 + i for i in range(5)), sum(100 + i for i in range(5, 10))]


def test_client_rejects_ambiguous_minutes_timeframe(tmp_path):
    cfg = ArchiveConfig.local(root=str(tmp_path))
    provider = FakeProvider(_make_linear_bars)
    client = MarketDataClient(cfg, provider=provider, calendar=NYSECalendar())

    with pytest.raises(ValueError):
        client.get_bars(
            ["AAPL"],
            start="2025-01-02T14:30:00Z",
            end="2025-01-02T14:33:00Z",
            timeframe="1m",  # in pandas this means months
        )


def test_client_records_does_not_require_pandas_inputs(tmp_path):
    from datetime import datetime, timezone

    cfg = ArchiveConfig.local(root=str(tmp_path))
    provider = FakeProvider(_make_linear_bars)
    client = MarketDataClient(cfg, provider=provider, calendar=NYSECalendar())

    recs = client.get_bars_records(
        ["AAPL"],
        start=datetime(2025, 1, 2, 14, 30, tzinfo=timezone.utc),
        end=datetime(2025, 1, 2, 14, 33, tzinfo=timezone.utc),
        timeframe="1min",
    )

    assert [r["symbol"] for r in recs] == ["AAPL", "AAPL", "AAPL"]

