from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Sequence

import pandas as pd

from market_data.archive.reader import ParquetReader
from market_data.archive.writer import ParquetWriter
from market_data.normalize.bars import REQUIRED_COLUMNS, normalize_1m_bars_frame, resample_from_1m
from market_data.normalize.calendars import Calendar, NYSECalendar
from market_data.providers.base import BaseProvider
from market_data.providers.polygon import PolygonConfig, PolygonProvider
from market_data.quality.gaps import find_gaps
from market_data.types import ArchiveConfig, Bar, parse_ts


@dataclass(frozen=True, slots=True)
class MarketDataClientConfig:
    """
    High-level behavior flags for MarketDataClient.
    """

    # If True, re-read the archive after writing fetched data. This guarantees
    # the returned frame reflects canonical archive normalization/deduping.
    reread_after_write: bool = True


class MarketDataClient:
    """
    Cache-first market data client.

    - Golden source in the archive is always **1-minute** bars.
    - If requested data is missing in the archive for [start,end), we fetch from the provider,
      write 1m to the archive (merge+dedupe), and then return a complete result.
    - If the caller requests a higher timeframe, we resample from the 1m golden source.
    """

    def __init__(
        self,
        archive_cfg: ArchiveConfig,
        *,
        polygon: PolygonConfig | None = None,
        provider: BaseProvider | None = None,
        calendar: Calendar | None = None,
        config: MarketDataClientConfig = MarketDataClientConfig(),
    ):
        self._archive_cfg = archive_cfg
        self._calendar = calendar or NYSECalendar()
        self._config = config

        if provider is None:
            if polygon is None:
                raise ValueError("Either provider=... or polygon=PolygonConfig(...) must be provided")
            provider = PolygonProvider(polygon)

        self._provider = provider
        self._reader = ParquetReader(archive_cfg, calendar=self._calendar)
        self._writer = ParquetWriter(archive_cfg)

    def get_bars(
        self,
        symbols: Sequence[str],
        *,
        start: str | datetime,
        end: str | datetime,
        timeframe: str = "1min",
        columns: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """
        Return bars for `symbols` in [start,end).

        - Always fetches/archives 1-minute bars.
        - If `timeframe` is not 1 minute, returns a derived timeframe via resampling.

        Timeframe must be a pandas-compatible offset string (examples: "1min", "5min", "15min", "1h", "1D").
        Note: In pandas, the unit "m" means **months**, not minutes. Use "min" for minutes.
        """

        tf = _validate_timeframe(timeframe)
        want_1m = _is_one_minute(tf)

        start_ts = parse_ts(start)
        end_ts = parse_ts(end)

        df_1m = self._get_1m(symbols, start=start_ts, end=end_ts, columns=None if not want_1m else columns)
        if want_1m:
            return df_1m

        out = resample_from_1m(df_1m, tf, label="left", closed="left")
        if columns is not None:
            keep = [c for c in columns if c in out.columns]
            out = out.loc[:, keep]
        return out.reset_index(drop=True)

    def get_bars_records(
        self,
        symbols: Sequence[str],
        *,
        start: str | datetime,
        end: str | datetime,
        timeframe: str = "1min",
    ) -> list[dict]:
        """
        Like `get_bars`, but returns plain Python records (no pandas usage required).

        Record schema:
        - symbol: str
        - ts: datetime (timezone-aware UTC)
        - open/high/low/close: float
        - volume: int
        """

        df = self.get_bars(symbols, start=start, end=end, timeframe=timeframe, columns=None)
        if df.empty:
            return []

        # Convert timestamps to timezone-aware Python datetimes.
        out: list[dict] = []
        # Avoid pandas FutureWarning around `.dt.to_pydatetime()` by converting per-element.
        ts_py = [pd.Timestamp(t).to_pydatetime() for t in pd.to_datetime(df["ts"], utc=True)]
        for i, row in enumerate(df.itertuples(index=False)):
            # row fields: symbol, ts, open, high, low, close, volume (and maybe others if timeframe != 1min)
            d = row._asdict()
            d["ts"] = ts_py[i]
            out.append(d)
        return out

    def _get_1m(
        self,
        symbols: Sequence[str],
        *,
        start: pd.Timestamp,
        end: pd.Timestamp,
        columns: Sequence[str] | None,
    ) -> pd.DataFrame:
        cached = self._reader.read(symbols, start=start, end=end, columns=columns)

        # If caller asked for a subset of columns, we still need the full schema to compute gaps/fetch.
        cached_full = cached if columns is None else self._reader.read(symbols, start=start, end=end, columns=None)
        report = find_gaps(cached_full, calendar=self._calendar, start=start, end=end, symbols=symbols)
        if report.total_missing == 0:
            return cached

        fetched_frames: list[pd.DataFrame] = []
        for sym in symbols:
            missing = report.missing_by_symbol.get(sym)
            if missing is None or len(missing) == 0:
                continue

            bars = self._provider.fetch_1m_bars(sym, start=start, end=end)
            fetched_frames.append(_bars_to_frame(bars))

        if fetched_frames:
            fetched = pd.concat(fetched_frames, ignore_index=True)
            # Writer normalizes + merges/dedupes.
            self._writer.write_1m(fetched, mode="merge_overwrite")

        if self._config.reread_after_write:
            return self._reader.read(symbols, start=start, end=end, columns=columns)

        # Return in-memory merge as a fallback.
        merged = pd.concat([cached_full] + fetched_frames, ignore_index=True) if fetched_frames else cached_full
        merged = normalize_1m_bars_frame(merged)
        merged = merged[(merged["ts"] >= start) & (merged["ts"] < end)]
        if columns is not None:
            keep = [c for c in columns if c in merged.columns]
            merged = merged.loc[:, keep]
        return merged.reset_index(drop=True)


def _validate_timeframe(timeframe: str) -> str:
    tf = str(timeframe).strip()
    # In pandas offset aliases, `m` means months. We only want minutes as `min`.
    if tf.endswith("m") and not tf.endswith("min"):
        raise ValueError(f"Invalid timeframe {timeframe!r}. Use 'min' for minutes (e.g. '5min'); 'm' means months in pandas.")
    try:
        pd.tseries.frequencies.to_offset(tf)
    except Exception as e:  # pragma: no cover
        raise ValueError(f"Invalid timeframe {timeframe!r}. Provide a pandas offset string like '5min', '1h', or '1D'.") from e
    return tf


def _is_one_minute(timeframe: str) -> bool:
    return pd.tseries.frequencies.to_offset(timeframe) == pd.tseries.frequencies.to_offset("1min")


def _bars_to_frame(bars: Iterable[Bar]) -> pd.DataFrame:
    rows = [
        {
            "symbol": b.symbol,
            "ts": b.ts,
            "open": b.open,
            "high": b.high,
            "low": b.low,
            "close": b.close,
            "volume": b.volume,
        }
        for b in bars
    ]
    if not rows:
        # empty but with required columns
        df = pd.DataFrame({c: pd.Series(dtype="object") for c in REQUIRED_COLUMNS})
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        return df
    return pd.DataFrame(rows)

