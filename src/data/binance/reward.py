from __future__ import annotations
from typing import Dict, Iterable, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from pandas.tseries.frequencies import to_offset
from data.binance.constants import INTERVAL_TO_PANDAS

def _interval_to_offset(interval: str) -> pd.DateOffset:
    """
    Convert a Binance-like interval string into a pandas DateOffset.
    Priority:
      1) Use INTERVAL_TO_PANDAS mapping (e.g., '1m'->'1min', '1d'->'1D', '1M'->'MS').
      2) Fallback to pandas' to_offset() if unmapped.
      3) Raise ValueError if nothing works.
    """
    key = str(interval).strip()
    if key in INTERVAL_TO_PANDAS:
        return to_offset(INTERVAL_TO_PANDAS[key])

    key_l = key.lower()
    key_u = key.upper()
    if key_l in INTERVAL_TO_PANDAS:
        return to_offset(INTERVAL_TO_PANDAS[key_l])
    if key_u in INTERVAL_TO_PANDAS:
        return to_offset(INTERVAL_TO_PANDAS[key_u])

    try:
        return to_offset(key)
    except Exception as e:
        raise ValueError(
            f"Unknown interval '{interval}'. Add it to INTERVAL_TO_PANDAS in constants.py."
        ) from e


def _to_utc_ts_list(anchor_times: Iterable) -> List[pd.Timestamp]:
    """
    Normalize anchor times into a list of tz-aware (UTC) pandas Timestamps.
    Non-parsable entries become NaT.
    """
    ts = pd.to_datetime(list(anchor_times), utc=True, errors="coerce")
    if isinstance(ts, pd.Timestamp):
        ts = pd.DatetimeIndex([ts])
    return [pd.Timestamp(t) if pd.notna(t) else pd.NaT for t in ts]


def extract_series_for_symbols(
    data: Dict[str, Dict[str, pd.DataFrame]],
    base_interval: str,
    anchor_times: Iterable,        # length B
    window_size: int,              # L
    symbols: List[str],            # length S (order preserved)
    columns: List[str],            # length C (any numeric-ish columns)
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      y_tp1: (B, S, C)  - exact values at t + base_interval, per symbol/column.
      y_t:   (B, S, C)  - exact values at t, per symbol/column.
      W:     (B, S, L, C) - right-aligned rolling window up to and including t,
                            left-padded with NaN to length L.

    Rules & assumptions:
      - Each DataFrame index must be a UTC tz-aware DatetimeIndex. If tz-naive,
        it will be localized to UTC; if another tz, it will be converted to UTC.
      - Only exact timestamp matches populate y_t / y_tp1; otherwise they remain NaN.
      - The window consists of the last L rows whose timestamps are <= t.
        If fewer than L rows exist, the window is left-padded with NaN.
      - Any requested column missing in a DataFrame is filled with NaN.
      - All numeric data handled as float32 consistently (inputs coerced).
    """
    # Enforce float32 consistently
    dtype_np = np.float32
    dtype_torch = torch.float32

    anchors = _to_utc_ts_list(anchor_times)
    B, S, C = len(anchors), len(symbols), len(columns)
    off = _interval_to_offset(base_interval)

    # Pre-allocate result buffers as float32 and NaN-filled where appropriate
    y_t  = np.full((B, S, C), np.nan, dtype=dtype_np)
    y_p1 = np.full((B, S, C), np.nan, dtype=dtype_np)
    W    = np.full((B, S, window_size, C), np.nan, dtype=dtype_np)

    # Early exit on empty shapes
    dev = device if device is not None else torch.device("cpu")
    if B == 0 or S == 0 or C == 0:
        return (
            torch.tensor(y_p1, dtype=dtype_torch, device=dev),
            torch.tensor(y_t,  dtype=dtype_torch, device=dev),
            torch.tensor(W,    dtype=dtype_torch, device=dev),
        )

    # Vectorize anchor handling once (reuse for every symbol)
    anchors_idx = pd.DatetimeIndex(anchors)         # tz-aware UTC
    mask_nat = anchors_idx.isna()
    if mask_nat.any():
        bad_pos = np.nonzero(mask_nat.to_numpy())[0].tolist()
        raise ValueError(
            f"anchor_times contain NaT at positions {bad_pos}. "
            "All anchors must be valid, tz-aware timestamps parsable by pandas."
        )
    anchors_next_idx = anchors_idx + off            # vectorized t + offset; NaT stays NaT

    # Iterate symbols
    for s_idx, sym in enumerate(symbols):
        df = data.get(sym, {}).get(base_interval, None)
        if df is None or df.empty:
            continue

        dfx = df.copy()

        # Normalize index tz to UTC
        if dfx.index.tz is None:
            dfx.index = dfx.index.tz_localize("UTC")
        else:
            dfx.index = dfx.index.tz_convert("UTC")

        # Ensure requested columns exist and are numeric; coerce to float32
        for c in columns:
            if c not in dfx.columns:
                dfx[c] = np.nan
            else:
                # coerce to numeric, then cast to float32
                dfx[c] = pd.to_numeric(dfx[c], errors="coerce").astype(dtype_np)
        dfx = dfx[columns]  # preserve column order

        idx = dfx.index
        if not idx.is_monotonic_increasing:
            raise ValueError("index must be sorted ascending (monotonic increasing).")
        if idx.has_duplicates:
            raise ValueError("duplicate timestamps are not allowed.")

        # Extract values as float32 directly
        vals = dfx.to_numpy(dtype=dtype_np, copy=False)  # shape (#rows, C)

        # Vectorized search positions for all anchors on this symbol's index
        end_pos  = idx.searchsorted(anchors_idx,     side="right")  # exclusive end (<= t)
        pos_next = idx.searchsorted(anchors_next_idx, side="left")  # candidate for t + offset

        # Batch loop per anchor (kept for clarity; windows are ragged)
        for i in range(B):
            # --- Window up to and including t (right-aligned in W) ---
            e = int(end_pos[i])  # rows <= t
            s0 = max(0, e - window_size)
            win_len = e - s0
            if win_len > 0:
                # Right-align: last 'win_len' slots get the actual data, left side stays NaN
                W[i, s_idx, -win_len:, :] = vals[s0:e, :]

            # --- Exact y_t at t (only if exact index match) ---
            t = anchors_idx[i]
            if e > 0 and idx[e - 1] == t:
                y_t[i, s_idx, :] = vals[e - 1, :]

            # --- Exact y_{t+1} at t + offset (only if exact index match) ---
            pn = int(pos_next[i])
            t1 = anchors_next_idx[i]
            if pn < len(idx) and pd.notna(t1) and idx[pn] == t1:
                y_p1[i, s_idx, :] = vals[pn, :]

    # Convert to torch tensors (float32 enforced)
    y_tp1_t = torch.from_numpy(y_p1).to(device=dev, dtype=dtype_torch)
    y_t_t   = torch.from_numpy(y_t).to(device=dev, dtype=dtype_torch)
    W_t     = torch.from_numpy(W).to(device=dev, dtype=dtype_torch)

    return y_tp1_t, y_t_t, W_t