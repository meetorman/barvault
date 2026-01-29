from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import pandas as pd

from market_data.providers.base import BaseProvider
from market_data.types import Bar


@dataclass(frozen=True, slots=True)
class TradingViewConfig:
    username: Optional[str] = None
    password: Optional[str] = None


class TradingViewProvider(BaseProvider):
    """
    Stub adapter (no authentication or env lookups at import-time).
    """

    def __init__(self, config: TradingViewConfig):
        self._config = config

    def fetch_1m_bars(self, symbol: str, *, start: pd.Timestamp, end: pd.Timestamp) -> Iterable[Bar]:
        raise NotImplementedError("TradingViewProvider is a stub; implement network calls in your integration layer.")

