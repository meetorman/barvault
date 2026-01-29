from __future__ import annotations

import pandas as pd

from market_data import NYSECalendar, find_gaps


def test_gap_detection_one_missing_minute():
    bars = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL"],
            "ts": [pd.Timestamp("2025-01-02T14:30:00Z"), pd.Timestamp("2025-01-02T14:32:00Z")],
            "open": [100.0, 100.2],
            "high": [100.1, 100.3],
            "low": [99.9, 100.1],
            "close": [100.0, 100.25],
            "volume": [1000, 1200],
        }
    )

    cal = NYSECalendar()
    report = find_gaps(
        bars,
        calendar=cal,
        start=pd.Timestamp("2025-01-02T14:30:00Z"),
        end=pd.Timestamp("2025-01-02T14:33:00Z"),
        symbols=["AAPL"],
    )

    missing = report.missing_by_symbol["AAPL"]
    assert list(missing) == [pd.Timestamp("2025-01-02T14:31:00Z")]

