from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

import pandas as pd

from market_data.types import Bar


class BaseProvider(ABC):
    """
    Providers are pure adapters.

    - No env var lookups inside the package (caller passes credentials/config explicitly).
    - No module-level global state.
    """

    @abstractmethod
    def fetch_1m_bars(self, symbol: str, *, start: pd.Timestamp, end: pd.Timestamp) -> Iterable[Bar]:
        """
        Fetch 1-minute bars for [start, end).
        """

