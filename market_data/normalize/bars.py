from __future__ import annotations

from typing import Iterable, Literal, Optional

import pandas as pd

from market_data.normalize.symbols import normalize_symbol


REQUIRED_COLUMNS = ("symbol", "ts", "open", "high", "low", "close", "volume")


def normalize_1m_bars_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize a bars DataFrame into a deterministic schema.

    Expected columns:
      symbol, ts, open, high, low, close, volume

    - ts is tz-aware UTC
    - symbol is normalized to uppercase
    - rows are sorted by (symbol, ts)
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"bars missing required columns: {missing}")

    out = df.loc[:, list(REQUIRED_COLUMNS)].copy()
    out["symbol"] = out["symbol"].astype("string").map(normalize_symbol)
    out["ts"] = pd.to_datetime(out["ts"], utc=True)

    for c in ("open", "high", "low", "close"):
        out[c] = pd.to_numeric(out[c], errors="raise").astype("float64")

    # Allow int-like or float-like volume; coerce to int64.
    out["volume"] = pd.to_numeric(out["volume"], errors="raise").astype("int64")

    out = out.sort_values(["symbol", "ts"], kind="mergesort").reset_index(drop=True)
    return out


def resample_from_1m(
    df_1m: pd.DataFrame,
    timeframe: str,
    *,
    label: Literal["left", "right"] = "left",
    closed: Literal["left", "right"] = "left",
) -> pd.DataFrame:
    """
    Derive higher timeframes from 1m bars without writing them to the archive.
    """
    df = normalize_1m_bars_frame(df_1m)
    df = df.set_index("ts")

    pieces: list[pd.DataFrame] = []
    for sym, g in df.groupby("symbol", sort=True):
        agg = (
            g.resample(timeframe, label=label, closed=closed)
            .agg(
                open=("open", "first"),
                high=("high", "max"),
                low=("low", "min"),
                close=("close", "last"),
                volume=("volume", "sum"),
            )
            .dropna(subset=["open", "high", "low", "close"])
        )
        agg.insert(0, "symbol", sym)
        pieces.append(agg.reset_index())

    out = pd.concat(pieces, ignore_index=True) if pieces else df.reset_index().iloc[0:0]
    out = out.sort_values(["symbol", "ts"], kind="mergesort").reset_index(drop=True)
    return out

