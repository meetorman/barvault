from __future__ import annotations

import pandas as pd

from market_data import ArchiveConfig, ParquetReader, ParquetWriter


def test_reader_deterministic_sort_and_boundaries(tmp_path):
    cfg = ArchiveConfig.local(root=str(tmp_path))
    writer = ParquetWriter(cfg)

    bars = pd.DataFrame(
        {
            "symbol": ["MSFT", "AAPL", "AAPL", "MSFT"],
            "ts": [
                pd.Timestamp("2025-01-02T14:31:00Z"),
                pd.Timestamp("2025-01-02T14:30:00Z"),
                pd.Timestamp("2025-01-02T14:31:00Z"),
                pd.Timestamp("2025-01-02T14:30:00Z"),
            ],
            "open": [10.0, 100.0, 101.0, 11.0],
            "high": [10.5, 100.5, 101.5, 11.5],
            "low": [9.5, 99.5, 100.5, 10.5],
            "close": [10.1, 100.1, 101.1, 11.1],
            "volume": [200, 1000, 1100, 220],
        }
    )
    writer.write_1m(bars)

    reader = ParquetReader(cfg)

    out1 = reader.read(["MSFT", "AAPL"], start="2025-01-02T14:30:00Z", end="2025-01-02T14:32:00Z")
    out2 = reader.read(["AAPL", "MSFT"], start="2025-01-02T14:30:00Z", end="2025-01-02T14:32:00Z")

    # Same inputs, deterministic outputs (ordering + values).
    pd.testing.assert_frame_equal(out1, out2)

    # Sorted by (symbol, ts)
    assert out1[["symbol", "ts"]].values.tolist() == [
        ["AAPL", pd.Timestamp("2025-01-02T14:30:00Z")],
        ["AAPL", pd.Timestamp("2025-01-02T14:31:00Z")],
        ["MSFT", pd.Timestamp("2025-01-02T14:30:00Z")],
        ["MSFT", pd.Timestamp("2025-01-02T14:31:00Z")],
    ]

    # End is exclusive: [start, end)
    out3 = reader.read(["AAPL"], start="2025-01-02T14:30:00Z", end="2025-01-02T14:31:00Z")
    assert out3[["symbol", "ts"]].values.tolist() == [
        ["AAPL", pd.Timestamp("2025-01-02T14:30:00Z")],
    ]

