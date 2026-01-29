# Barvault

Cache-first market bar archive: fetch from Polygon (or other providers), store 1-minute bars in S3 or local, return any timeframe. Callers can use plain Python (`datetime`, `list[dict]`) — no pandas required.

- deterministic archive reads (stable ordering + explicit boundary semantics)
- a single golden source stored in the archive: **1-minute bars**
- stateless configuration (no module globals, no env lookups at import-time)

## Install (Poetry)

```bash
poetry install
```

## Quick usage

```python
from __future__ import annotations

from datetime import datetime, timezone

from market_data import ArchiveConfig, MarketDataClient, PolygonConfig

cfg = ArchiveConfig.s3(bucket="public-market-data-truth-trader", prefix="md-archive")
client = MarketDataClient(cfg, polygon=PolygonConfig(api_key="YOUR_POLYGON_KEY"))

records = client.get_bars_records(
    ["AAPL"],
    start=datetime(2025, 1, 2, 14, 30, tzinfo=timezone.utc),
    end=datetime(2025, 1, 2, 15, 30, tzinfo=timezone.utc),
    timeframe="5min",
)
print(records[:2])
```

## Fetch + cache (Polygon) + return any timeframe

The archive’s golden source is always **1-minute** bars. `MarketDataClient` will:

- read what’s already in the archive for `[start, end)`
- detect missing expected minutes (using the provided calendar)
- fetch missing data from Polygon
- write 1m bars back to the archive (merge + de-dupe)
- return **1m** or any derived timeframe (resampled from 1m)

```python
from __future__ import annotations

from market_data import ArchiveConfig, MarketDataClient, PolygonConfig

# Local archive:
# cfg = ArchiveConfig.local(root="/tmp/md-archive")
#
# S3 archive:
cfg = ArchiveConfig.s3(bucket="public-market-data-truth-trader", prefix="md-archive")

client = MarketDataClient(
    cfg,
    polygon=PolygonConfig(api_key="YOUR_POLYGON_KEY"),
)

# Always returns data for [start,end) (start inclusive, end exclusive).
bars_1m = client.get_bars(
    ["AAPL", "MSFT"],
    start="2025-01-02T14:30:00Z",
    end="2025-01-02T15:30:00Z",
    timeframe="1min",
)

# Derived timeframe (resampled from 1m):
bars_5m = client.get_bars(
    ["AAPL"],
    start="2025-01-02T14:30:00Z",
    end="2025-01-02T15:30:00Z",
    timeframe="5min",
)

# If you don't want to work with pandas DataFrames, use plain Python records:
records_5m = client.get_bars_records(
    ["AAPL"],
    start="2025-01-02T14:30:00Z",
    end="2025-01-02T15:30:00Z",
    timeframe="5min",
)
```

Timeframe notes:

- Use pandas-compatible offset strings like `"5min"`, `"1h"`, `"1D"`.
- Do **not** use `"1m"` to mean 1 minute — in pandas `"m"` means **months**.

## Design constraints

- `import market_data` must not start servers, read env vars, or require credentials
- all configuration is passed via constructors/args (`ArchiveConfig`, provider constructors, etc.)
- only **1m** is accepted by the writer; higher timeframes are derived from 1m (never written)

