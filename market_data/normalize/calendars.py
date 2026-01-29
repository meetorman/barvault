from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class Calendar(Protocol):
    """
    Calendar protocol for generating expected 1-minute timestamps.
    """

    def expected_minutes(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
        """
        Return expected minute timestamps for the interval [start, end) in UTC.
        """


@dataclass(frozen=True, slots=True)
class NYSECalendar:
    """
    Minimal NYSE-ish calendar:
    - Weekdays only (Mon-Fri)
    - Regular session 09:30-16:00 America/New_York
    - Does NOT include market holidays / early closes (caller can supply a richer calendar).
    """

    tz: str = "America/New_York"
    open_time: str = "09:30"
    close_time: str = "16:00"

    def expected_minutes(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
        start_utc = pd.Timestamp(start, tz="UTC") if pd.Timestamp(start).tz is None else pd.Timestamp(start).tz_convert("UTC")
        end_utc = pd.Timestamp(end, tz="UTC") if pd.Timestamp(end).tz is None else pd.Timestamp(end).tz_convert("UTC")
        if end_utc <= start_utc:
            return pd.DatetimeIndex([], tz="UTC")

        # Work in local session TZ, then convert to UTC.
        start_local = start_utc.tz_convert(self.tz)
        end_local = end_utc.tz_convert(self.tz)

        # Build business days spanning the interval in local time.
        days = pd.date_range(
            start=start_local.normalize(),
            end=end_local.normalize(),
            freq="B",
            tz=self.tz,
        )

        pieces: list[pd.DatetimeIndex] = []
        for day in days:
            session_start = pd.Timestamp(f"{day.date()} {self.open_time}", tz=self.tz)
            session_end = pd.Timestamp(f"{day.date()} {self.close_time}", tz=self.tz)

            # Generate minute bars for [open, close) in local TZ.
            mins = pd.date_range(session_start, session_end, freq="1min", inclusive="left", tz=self.tz)
            pieces.append(mins)

        all_minutes_local = pieces[0].append(pieces[1:]) if pieces else pd.DatetimeIndex([], tz=self.tz)
        all_minutes_utc = all_minutes_local.tz_convert("UTC")

        # Clip to [start, end) in UTC for deterministic semantics.
        clipped = all_minutes_utc[(all_minutes_utc >= start_utc) & (all_minutes_utc < end_utc)]
        return clipped

