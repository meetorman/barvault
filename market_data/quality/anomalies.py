from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from market_data.normalize.bars import normalize_1m_bars_frame


@dataclass(frozen=True, slots=True)
class AnomalyReport:
    negative_volume: pd.DataFrame
    duplicate_rows: pd.DataFrame
    non_monotonic: pd.DataFrame
    invalid_ohlc: pd.DataFrame

    @property
    def is_clean(self) -> bool:
        return (
            self.negative_volume.empty
            and self.duplicate_rows.empty
            and self.non_monotonic.empty
            and self.invalid_ohlc.empty
        )


def detect_anomalies(bars_1m: pd.DataFrame) -> AnomalyReport:
    """
    Basic anomaly checks as pure functions returning structured results.
    """
    df = normalize_1m_bars_frame(bars_1m)

    negative_volume = df[df["volume"] < 0]

    dup_mask = df.duplicated(subset=["symbol", "ts"], keep=False)
    duplicate_rows = df[dup_mask].sort_values(["symbol", "ts"], kind="mergesort")

    # Non-monotonic timestamps within a symbol
    non_mono_pieces: list[pd.DataFrame] = []
    for sym, g in df.groupby("symbol", sort=True):
        ts = g["ts"]
        if not ts.is_monotonic_increasing:
            non_mono_pieces.append(g)
    non_monotonic = pd.concat(non_mono_pieces, ignore_index=True) if non_mono_pieces else df.iloc[0:0]

    invalid_ohlc = df[
        (df["high"] < df["low"])
        | (df["open"] < df["low"])
        | (df["open"] > df["high"])
        | (df["close"] < df["low"])
        | (df["close"] > df["high"])
    ]

    return AnomalyReport(
        negative_volume=negative_volume.reset_index(drop=True),
        duplicate_rows=duplicate_rows.reset_index(drop=True),
        non_monotonic=non_monotonic.reset_index(drop=True),
        invalid_ohlc=invalid_ohlc.reset_index(drop=True),
    )

