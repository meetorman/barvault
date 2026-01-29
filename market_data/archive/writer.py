from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from fsspec.core import url_to_fs

from market_data.archive.metadata import partition_path
from market_data.normalize.bars import normalize_1m_bars_frame
from market_data.normalize.symbols import normalize_symbol
from market_data.types import ArchiveConfig


@dataclass(frozen=True, slots=True)
class WriteResult:
    written_partitions: list[str]


class ParquetWriter:
    """
    Writer for the golden source: 1-minute bars only.

    Behavior:
    - partitions by (symbol, year, month)
    - writes `data.parquet` per partition
    - default mode merges with existing file and de-dupes by (symbol, ts), then overwrites
    """

    def __init__(self, cfg: ArchiveConfig):
        if cfg.timeframe != "1m":
            raise ValueError("ArchiveConfig.timeframe must be '1m' for ParquetWriter")
        self._cfg = cfg
        self._fs, self._base_path = url_to_fs(cfg.base_uri())
        self._dataset_path = f"{self._base_path}/{cfg.dataset}/{cfg.timeframe}".rstrip("/")

    def write_1m(self, bars_1m: pd.DataFrame, *, mode: Literal["merge_overwrite", "overwrite"] = "merge_overwrite") -> WriteResult:
        df = normalize_1m_bars_frame(bars_1m)

        written: list[str] = []
        df["_year"] = df["ts"].dt.year.astype("int32")
        df["_month"] = df["ts"].dt.month.astype("int32")

        for (sym, year, month), g in df.groupby(["symbol", "_year", "_month"], sort=True):
            sym = normalize_symbol(sym)
            part_uri = partition_path(self._cfg, symbol=sym, year=int(year), month=int(month))
            rel_path = self._uri_to_relpath(part_uri)

            out_df = g.drop(columns=["_year", "_month"]).copy()
            if mode == "merge_overwrite" and self._fs.exists(rel_path):
                existing = self._read_partition(rel_path)
                out_df = pd.concat([existing, out_df], ignore_index=True)
                out_df = normalize_1m_bars_frame(out_df)
                out_df = out_df.drop_duplicates(subset=["symbol", "ts"], keep="last")
                out_df = out_df.sort_values(["symbol", "ts"], kind="mergesort").reset_index(drop=True)
            else:
                out_df = out_df.sort_values(["symbol", "ts"], kind="mergesort").reset_index(drop=True)

            parent = rel_path.rsplit("/", 1)[0]
            self._fs.makedirs(parent, exist_ok=True)

            table = pa.Table.from_pandas(out_df, preserve_index=False)
            with self._fs.open(rel_path, "wb") as f:
                pq.write_table(table, f)

            written.append(part_uri)

        return WriteResult(written_partitions=written)

    def _uri_to_relpath(self, uri: str) -> str:
        fs2, rel = url_to_fs(uri)
        if fs2.protocol != self._fs.protocol:
            raise ValueError(f"URI protocol mismatch: writer fs={self._fs.protocol} uri={uri}")
        return rel

    def _read_partition(self, rel_path: str) -> pd.DataFrame:
        # Read via the fsspec filesystem to avoid pyarrow attempting to interpret
        # an fsspec FS as a native pyarrow filesystem (can cause schema merge issues).
        with self._fs.open(rel_path, "rb") as f:
            table = pq.read_table(f)
        return table.to_pandas()

