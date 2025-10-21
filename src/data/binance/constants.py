# vision_constants.py
from __future__ import annotations

from typing import Dict

# Raw columns expected from Binance (case-insensitive by meaning, but use these exact names in your dataframes)
FEATURES = ["open", "high", "low", "close", "volume"]  # raw OHLCV
EXTRA_RAW_FEATURES = ["taker_buy_base_asset_volume"]   # optional but recommended

# Final engineered feature names per (symbol, interval) in this order
DERIVED_FEATURES = ["r_oc_prev", "r_co", "r_ho", "r_lo", "v_log1p", "taker_imbalance"]
TI_IDX_IN_DERIVED = DERIVED_FEATURES.index("taker_imbalance")

def mask_name(f: str) -> str:
    return f"{f}__mask"

def elapsed_name(f: str) -> str:
    return f"{f}__elapsed"

# Binance interval -> pandas frequency mapping
INTERVAL_TO_PANDAS: Dict[str, str] = {
    "1m": "1min", "3m": "3min", "5m": "5min", "15m": "15min", "30m": "30min",
    "1h": "1h", "2h": "2h", "4h": "4h", "6h": "6h", "8h": "8h", "12h": "12h",
    "1d": "1D", "3d": "3D", "1w": "7D", "1M": "MS",  # monthly bars aligned to month start
}

def interval_to_minutes(interval: str) -> int:
    """Return interval length in minutes (used for sorting and elapsed-time construction)."""
    if interval.endswith("m"):
        return int(interval[:-1])
    if interval.endswith("h"):
        return int(interval[:-1]) * 60
    if interval.endswith("d"):
        return int(interval[:-1]) * 60 * 24
    if interval.endswith("w"):
        return int(interval[:-1]) * 60 * 24 * 7
    if interval.endswith("M"):
        # Treat month as ~30 days; exact alignment handled by pandas freq ("MS").
        return 30 * 24 * 60
    raise ValueError(f"Unknown interval: {interval}")

__all__ = [
    "FEATURES",
    "EXTRA_RAW_FEATURES",
    "DERIVED_FEATURES",
    "TI_IDX_IN_DERIVED",
    "mask_name",
    "elapsed_name",
    "INTERVAL_TO_PANDAS",
    "interval_to_minutes",
]
