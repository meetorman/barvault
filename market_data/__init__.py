"""
market_data

Pure-Python market data engine primitives:
- stateless configuration via ArchiveConfig
- deterministic parquet archive reads/writes for 1-minute bars
- basic calendar + quality checks as pure functions
"""

from market_data.archive.reader import ParquetReader
from market_data.archive.writer import ParquetWriter
from market_data.client import MarketDataClient, MarketDataClientConfig
from market_data.normalize.calendars import Calendar, NYSECalendar
from market_data.providers.polygon import PolygonConfig, PolygonProvider
from market_data.quality.anomalies import AnomalyReport, detect_anomalies
from market_data.quality.gaps import GapReport, find_gaps
from market_data.types import ArchiveConfig, Bar, BarsFrame, Timeframe

__all__ = [
    "AnomalyReport",
    "ArchiveConfig",
    "Bar",
    "BarsFrame",
    "Calendar",
    "GapReport",
    "MarketDataClient",
    "MarketDataClientConfig",
    "NYSECalendar",
    "ParquetReader",
    "ParquetWriter",
    "PolygonConfig",
    "PolygonProvider",
    "Timeframe",
    "detect_anomalies",
    "find_gaps",
]

