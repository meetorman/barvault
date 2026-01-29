from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import pandas as pd
import pyarrow.parquet as pq
from fsspec.core import url_to_fs

from market_data.archive.metadata import ArchiveFileDetail, partition_path
from market_data.normalize.bars import REQUIRED_COLUMNS, normalize_1m_bars_frame
from market_data.normalize.calendars import Calendar
from market_data.normalize.symbols import normalize_symbol
from market_data.types import ArchiveConfig, iter_month_starts, parse_ts


@dataclass(frozen=True, slots=True)
class ArchiveBrowser:
    cfg: ArchiveConfig
    _fs: object
    _dataset_path: str

    def details(
        self,
        *,
        symbols: Sequence[str] | None = None,
        month: str | None = None,
    ) -> list[ArchiveFileDetail]:
        """
        List archive partitions as typed records.

        - symbols: optional filter (case-insensitive)
        - month: optional "YYYY-MM" filter
        """
        want_symbols = None
        if symbols is not None:
            want_symbols = {normalize_symbol(s) for s in symbols}

        want_year = want_month = None
        if month is not None:
            dt = pd.Timestamp(f"{month}-01", tz="UTC")
            want_year, want_month = int(dt.year), int(dt.month)

        glob_pat = f"{self._dataset_path}/symbol=*/year=*/month=*/data.parquet"
        paths = sorted(self._fs.glob(glob_pat))
        out: list[ArchiveFileDetail] = []
        for p in paths:
            # p is rel-path for the filesystem
            parts = p.split("/")
            sym = None
            year = None
            month_s = None
            for part in parts:
                if part.startswith("symbol="):
                    sym = part.split("=", 1)[1]
                elif part.startswith("year="):
                    year = int(part.split("=", 1)[1])
                elif part.startswith("month="):
                    month_s = part.split("=", 1)[1]
            if sym is None or year is None or month_s is None:
                continue

            sym_n = normalize_symbol(sym)
            mm = int(month_s)
            if want_symbols is not None and sym_n not in want_symbols:
                continue
            if want_year is not None and (year != want_year or mm != want_month):
                continue

            out.append(
                ArchiveFileDetail(
                    symbol=sym_n,
                    year=year,
                    month=mm,
                    path=self._relpath_to_uri(p),
                )
            )

        out.sort(key=lambda d: (d.symbol, d.year, d.month, d.path))
        return out

    def _relpath_to_uri(self, rel_path: str) -> str:
        proto = getattr(self._fs, "protocol", "file")
        if isinstance(proto, (list, tuple)):
            proto = proto[0] if proto else "file"
        if proto in ("file", None):
            # local filesystem: rel_path is typically absolute already
            return rel_path
        return f"{proto}://{rel_path}"


class ParquetReader:
    """
    Deterministic archive reader for 1-minute bars.

    Semantics:
    - `start` inclusive, `end` exclusive: [start, end)
    - output is stably sorted by (symbol, ts)
    """

    def __init__(self, cfg: ArchiveConfig, *, fs=None, calendar: Optional[Calendar] = None):
        if cfg.timeframe != "1m":
            raise ValueError("ArchiveConfig.timeframe must be '1m' for ParquetReader")
        self._cfg = cfg
        base_fs, base_path = url_to_fs(cfg.base_uri())
        self._fs = base_fs if fs is None else fs
        self._base_path = base_path
        self._dataset_path = f"{self._base_path}/{cfg.dataset}/{cfg.timeframe}".rstrip("/")
        self.calendar = calendar
        self.archive = ArchiveBrowser(cfg=cfg, _fs=self._fs, _dataset_path=self._dataset_path)

    def read(
        self,
        symbols: Sequence[str],
        *,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        columns: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        syms = sorted({normalize_symbol(s) for s in symbols})
        start_ts = parse_ts(start)
        end_ts = parse_ts(end)
        if end_ts <= start_ts:
            return self._empty_frame(columns)

        files = self._files_for_range(syms, start_ts, end_ts)
        if not files:
            return self._empty_frame(columns)

        frames: list[pd.DataFrame] = []
        for rel_path in files:
            # Read via the fsspec filesystem to avoid pyarrow attempting to interpret
            # an fsspec FS as a native pyarrow filesystem (can cause schema merge issues).
            with self._fs.open(rel_path, "rb") as f:
                table = pq.read_table(f)
            frames.append(table.to_pandas())

        df = pd.concat(frames, ignore_index=True) if frames else self._empty_frame(None)
        df = normalize_1m_bars_frame(df)

        # Filter
        df = df[df["symbol"].isin(syms)]
        df = df[(df["ts"] >= start_ts) & (df["ts"] < end_ts)]
        df = df.sort_values(["symbol", "ts"], kind="mergesort").reset_index(drop=True)

        if columns is not None:
            cols = list(columns)
            # Always allow caller to request any subset, but enforce deterministic core ordering when present.
            keep = [c for c in cols if c in df.columns]
            df = df.loc[:, keep]

        return df.reset_index(drop=True)

    def _files_for_range(self, symbols: Sequence[str], start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
        rel_paths: list[str] = []
        for sym in symbols:
            for month_start in iter_month_starts(start, end):
                y = int(month_start.year)
                m = int(month_start.month)
                part_uri = partition_path(self._cfg, symbol=sym, year=y, month=m)
                rel = self._uri_to_relpath(part_uri)
                if self._fs.exists(rel):
                    rel_paths.append(rel)
        # Deterministic file ordering.
        rel_paths.sort()
        return rel_paths

    def _uri_to_relpath(self, uri: str) -> str:
        fs2, rel = url_to_fs(uri)
        if fs2.protocol != self._fs.protocol:
            raise ValueError(f"URI protocol mismatch: reader fs={self._fs.protocol} uri={uri}")
        return rel

    def _empty_frame(self, columns: Sequence[str] | None) -> pd.DataFrame:
        base_cols = list(REQUIRED_COLUMNS)
        df = pd.DataFrame({c: pd.Series(dtype="object") for c in base_cols})
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        if columns is not None:
            cols = [c for c in columns if c in df.columns]
            return df.loc[:, cols]
        return df

