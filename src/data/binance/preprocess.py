# preprocess_external.py
from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
from typing import List, Callable, Dict

from data.binance.constants import FEATURES, EXTRA_RAW_FEATURES, INTERVAL_TO_PANDAS

def set_index_and_sort(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    df = df.copy()

    open_cols = ["open_time_utc", "openTimeUTC"]
    base_open = next((c for c in open_cols if c in df.columns), None)

    if base_open is None:
        close_cols = ["close_time_utc", "closeTimeUTC"]
        base_close = next((c for c in close_cols if c in df.columns), None)
        if base_close is None:
            raise KeyError("Input dataframe must contain one of open_time_utc/openTimeUTC or close_time_utc/closeTimeUTC.")
        df[base_close] = pd.to_datetime(df[base_close], utc=True, errors="coerce")
        if interval not in INTERVAL_TO_PANDAS:
            raise KeyError(f"Interval '{interval}' not found in INTERVAL_TO_PANDAS mapping.")
        freq = INTERVAL_TO_PANDAS[interval]
        df["bar_end_utc"] = (df[base_close] + pd.Timedelta(milliseconds=1)).dt.floor(freq)
    else:
        df[base_open] = pd.to_datetime(df[base_open], utc=True, errors="coerce")
        if interval not in INTERVAL_TO_PANDAS:
            raise KeyError(f"Interval '{interval}' not found in INTERVAL_TO_PANDAS mapping.")
        offset = to_offset(INTERVAL_TO_PANDAS[interval])
        df["bar_end_utc"] = df[base_open] + offset

    df = df.dropna(subset=["bar_end_utc"])

    for c in FEATURES + [c for c in EXTRA_RAW_FEATURES if c in df.columns]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = (df.sort_values("bar_end_utc")
            .drop_duplicates("bar_end_utc", keep="last")
            .set_index("bar_end_utc"))

    if df.index.tz is None:
        raise ValueError("Index must be timezone-aware (UTC).")
    if df.empty:
        raise ValueError("No valid rows after datetime parsing & deduping.")
    return df


def sanity_check(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """
    - Prices(open/high/low/close) <= 0 or non-finite → NaN
    - Volume: < 0 or non-finite → NaN (keep 0 as 0)
    - OHLC logical consistency:
        * low <= high
        * low <= open, close
        * high >= open, close
      If violated → set all of (open, high, low, close) = NaN for that row.
    - taker_buy_base_asset_volume (if present):
        * non-finite or < 0 → NaN
        * if volume <= 0 → NaN
        * if TB > volume → NaN
    """
    df = df.copy()

    for c in FEATURES:
        if c not in df.columns:
            df[c] = np.nan
    tb_col = EXTRA_RAW_FEATURES[0] if EXTRA_RAW_FEATURES else None
    if tb_col and tb_col not in df.columns:
        df[tb_col] = np.nan

    for c in FEATURES + ([tb_col] if tb_col else []):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 1) Prices must be positive
    for c in ["open", "high", "low", "close"]:
        s = df[c]
        df[c] = s.where(s.notna() & np.isfinite(s) & (s > 0), np.nan)

    # 2) Volume: keep zeros; only negative/non-finite → NaN
    v = df["volume"]
    df["volume"] = v.where(v.notna() & np.isfinite(v) & (v >= 0), np.nan)

    # 3) OHLC logical sanity
    O, H, L, C = df["open"], df["high"], df["low"], df["close"]
    finite = O.notna() & H.notna() & L.notna() & C.notna()
    bad = pd.Series(False, index=df.index)
    bad |= finite & (L > H)
    bad |= finite & ((L > O) | (L > C))
    bad |= finite & ((H < O) | (H < C))
    if bad.any():
        df.loc[bad, ["open", "high", "low", "close"]] = np.nan

    # 4) Taker-buy sanity (optional)
    if tb_col:
        TB = df[tb_col]
        TB = TB.where(TB.notna() & np.isfinite(TB) & (TB >= 0), np.nan)
        V  = df["volume"]
        TB = TB.where((V > 0), np.nan)      # invalid if volume <= 0
        TB = TB.where(~(TB > V), np.nan)    # TB cannot exceed V
        df[tb_col] = TB

    return df


def ensure_regular_grid(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """
    Reindex to the expected fixed time grid for the interval.
    Missing timestamps are filled with NaN rows (no forward-fill here).
    """
    if interval not in INTERVAL_TO_PANDAS:
        raise KeyError(f"Interval '{interval}' not found in INTERVAL_TO_PANDAS mapping.")
    freq = INTERVAL_TO_PANDAS[interval]
    if df.index.tz is None:
        raise ValueError("Index must be timezone-aware (UTC). Did you run set_index_and_sort?")
    full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq, tz="UTC")
    return df.reindex(full_index)


def drop_outside_features(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Keep only raw columns needed for feature engineering."""
    keep = [c for c in FEATURES if c in df.columns]
    for c in EXTRA_RAW_FEATURES:
        if c in df.columns and c not in keep:
            keep.append(c)
    return df[keep].copy()


PreprocessFn = Callable[[pd.DataFrame, str], pd.DataFrame]
_REGISTRY: Dict[str, PreprocessFn] = {
    "set_index_and_sort": set_index_and_sort,
    "sanity_check": sanity_check,
    "ensure_regular_grid": ensure_regular_grid,
    "drop_outside_features": drop_outside_features,
}

class PreprocessPipeline:
    def __init__(self, steps: List[str]):
        self.steps = steps

    def __call__(self, df: pd.DataFrame, interval: str) -> pd.DataFrame:
        for name in self.steps:
            if name not in _REGISTRY:
                raise KeyError(f"Unknown preprocess step: {name}. Available: {list(_REGISTRY.keys())}")
            df = _REGISTRY[name](df, interval)
        return df
