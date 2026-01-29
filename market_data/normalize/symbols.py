from __future__ import annotations


def normalize_symbol(symbol: str) -> str:
    """
    Deterministic symbol normalization.

    This package intentionally does not enforce exchange-specific mapping rules.
    """
    s = (symbol or "").strip().upper()
    if not s:
        raise ValueError("symbol must be non-empty")
    return s

