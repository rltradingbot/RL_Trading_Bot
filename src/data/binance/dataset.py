# dataset_binance_vision.py
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable

from torch.utils.data import Dataset, DataLoader

from data.binance.constants import (
    FEATURES, EXTRA_RAW_FEATURES, DERIVED_FEATURES, TI_IDX_IN_DERIVED,
    mask_name, elapsed_name, interval_to_minutes
)

# -----------------------------
# Internal helpers (kept here; only build_vision_features is used inside Dataset)
# -----------------------------

def _compute_taker_imbalance(volume: pd.Series, taker_buy_base_asset_volume: Optional[pd.Series]) -> pd.Series:
    """
    (2 * taker_buy_base_asset_volume - volume) / volume, clipped to [-1, 1]
    NaN when volume <= 0 or inputs missing.
    """
    if taker_buy_base_asset_volume is None:
        return pd.Series(np.nan, index=volume.index, dtype=np.float64)
    vol = volume.astype(float)
    tb = taker_buy_base_asset_volume.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ti = (2.0 * tb - vol) / vol
    ti[(~np.isfinite(ti)) | (vol <= 0)] = np.nan
    ti = ti.clip(-1.0, 1.0)
    return ti

def _elapsed_minutes_per_feature(mask: pd.Series, interval_minutes: int) -> pd.Series:
    """
    Per-feature elapsed time (in minutes) since last OBSERVED value, at each timestamp.
    """
    idx = mask.index
    m = mask.astype(bool).to_numpy()
    n = m.shape[0]
    out = np.zeros(n, dtype=np.float64)
    last_obs = None
    consec_miss = 0
    for i in range(n):
        if i == 0:
            out[i] = 0.0
            if m[i]:
                last_obs = 0
                consec_miss = 0
            else:
                last_obs = None
                consec_miss = 0
            continue
        if m[i]:
            if last_obs is None:
                out[i] = i * interval_minutes if consec_miss > 0 else interval_minutes
            else:
                out[i] = (i - last_obs) * interval_minutes
            last_obs = i
            consec_miss = 0
        else:
            consec_miss += 1
            out[i] = consec_miss * interval_minutes
    return pd.Series(out, index=idx, dtype=np.float64)

def _build_vision_features(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """
    Build the 6 engineered features + per-feature mask + per-feature elapsed(log1p minutes).
    Requires df to be already preprocessed EXTERNALLY:
      - UTC timezone-aware DatetimeIndex
      - numeric OHLCV (+ optional taker_buy_base_asset_volume)
      - regular grid established (reindexed)
      - only necessary raw columns may remain
    Returns columns in order:
        [DERIVED_FEATURES] + [f__mask] + [f__elapsed]
    """
    if df.index.tz is None:
        raise ValueError("Index must be timezone-aware (UTC). Run external preprocess before Dataset.")

    df = df.copy()
    # Ensure needed raw columns exist (they may be NaN)
    for c in FEATURES:
        if c not in df.columns:
            df[c] = np.nan
    if EXTRA_RAW_FEATURES[0] not in df.columns:
        df[EXTRA_RAW_FEATURES[0]] = np.nan

    O = df["open"].astype(float)
    H = df["high"].astype(float)
    L = df["low"].astype(float)
    C = df["close"].astype(float)
    V = df["volume"].astype(float)
    TB = df[EXTRA_RAW_FEATURES[0]].astype(float)

    C_prev = C.shift(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r_oc_prev = np.log(O / C_prev)
        r_co      = np.log(C / O)
        r_ho      = np.log(H / O)
        r_lo      = np.log(L / O)

    v_log1p = np.log1p(V)
    taker_imb = _compute_taker_imbalance(V, TB)

    feat_df = pd.DataFrame({
        "r_oc_prev": r_oc_prev,
        "r_co": r_co,
        "r_ho": r_ho,
        "r_lo": r_lo,
        "v_log1p": v_log1p,
        "taker_imbalance": taker_imb,
    }, index=df.index)

    # masks + elapsed
    iv_minutes = interval_to_minutes(interval)
    cols = []
    for f in DERIVED_FEATURES:
        cols.append(f)
    for f in DERIVED_FEATURES:
        m = (~feat_df[f].isna()).astype(np.float32)
        feat_df[mask_name(f)] = m
        cols.append(mask_name(f))
    for f in DERIVED_FEATURES:
        m = feat_df[mask_name(f)]
        elapsed_min = _elapsed_minutes_per_feature(m.astype(bool), iv_minutes)
        feat_df[elapsed_name(f)] = np.log1p(elapsed_min.astype(np.float32))
        cols.append(elapsed_name(f))

    return feat_df[cols].copy()

# -----------------------------
# Dataset
# -----------------------------

@dataclass
class DatasetConfig:
    window_size: int                       # L (sequence length)
    strict_anchor: bool = True             # True: use only timestamps where every combo has at least L rows
    symbol_order: Optional[List[str]] = None  # Desired symbol order; if None, use sorted order
    base_interval: Optional[str] = None       # If None, choose the shortest interval automatically

class BinanceVisionDataset(Dataset):
    """
    Input:
      data[symbol][interval] -> preprocessed pd.DataFrame (EXTERNAL preprocess done!)
        Required:
          - UTC tz-aware DatetimeIndex
          - columns: 'open','high','low','close','volume' (+ optional 'taker_buy_base_asset_volume')
          - already reindexed to regular grid for that interval (no forward-fill)
    Internally applies ONLY feature-building (build_vision_features).

    __getitem__(idx) → if idx is int, use internal anchor index; if idx is pd.Timestamp-like, use that timestamp directly.
    Returns: (X, M, T, meta)
      - X: (L, C) engineered features, where C = 6 * num_combos
           Z-score per-channel over the L window is applied **except** for 'taker_imbalance' channels.
           After normalization, NaNs are set to 0.
      - M: (L, C) missingness masks (0/1) aligned 1:1 with X channels.
      - T: (L, C) per-feature elapsed-time (log1p minutes), NaNs set to 0. (z-score currently disabled)
      - meta: dict(anchor_time, combos, feature_names_6)
    """
    def __init__(self,
                 data: Dict[str, Dict[str, pd.DataFrame]],
                 cfg: DatasetConfig):
        self.raw = data
        self.cfg = cfg

        # 1) Determine symbol order
        symbols_available = list(data.keys())
        if cfg.symbol_order is not None:
            # Keep provided order (filter to available), then append the rest sorted
            provided = [s for s in cfg.symbol_order if s in symbols_available]
            rest = sorted([s for s in symbols_available if s not in provided])
            self.symbols = provided + rest
        else:
            self.symbols = sorted(symbols_available)

        if not self.symbols:
            raise ValueError("No symbols found in input data.")

        # 2) Collect all intervals across symbols and sort by length
        intervals_set = {iv for s in self.symbols for iv in data.get(s, {}).keys()}
        if not intervals_set:
            raise ValueError("No intervals found in input data.")
        self.intervals = sorted(list(intervals_set), key=lambda x: interval_to_minutes(x))

        # 3) Base interval selection (external override or automatic shortest)
        if cfg.base_interval is not None:
            if cfg.base_interval not in intervals_set:
                raise ValueError(f"Requested base_interval '{cfg.base_interval}' not present in input intervals {sorted(list(intervals_set))}.")
            self.base_interval = cfg.base_interval
        else:
            self.base_interval = min(self.intervals, key=lambda x: interval_to_minutes(x))

        # 4) Build engineered-feature DataFrames ONLY (no external preprocessing here)
        self.feature_names: List[str] = DERIVED_FEATURES[:]  # 6
        self.mask_names: List[str] = [mask_name(f) for f in self.feature_names]
        self.elapsed_names: List[str] = [elapsed_name(f) for f in self.feature_names]
        self.required_cols: List[str] = self.feature_names + self.mask_names + self.elapsed_names

        self.df: Dict[str, Dict[str, pd.DataFrame]] = {}
        for sym in self.symbols:
            self.df[sym] = {}
            for iv, df0 in data.get(sym, {}).items():
                # IMPORTANT: df0 must be externally preprocessed
                dfx = _build_vision_features(df0, iv)

                # Ensure required columns exist (if something went missing, keep NaN)
                for col in self.required_cols:
                    if col not in dfx.columns:
                        dfx[col] = np.nan

                self.df[sym][iv] = dfx[self.required_cols].copy()

        # 5) Build actual (symbol, interval) combos in the specified symbol order
        self.combos: List[Tuple[str, str]] = []
        for sym in self.symbols:
            for iv in self.intervals:
                if iv in self.df.get(sym, {}):
                    self.combos.append((sym, iv))
        self.num_combos = len(self.combos)
        if self.num_combos == 0:
            raise ValueError("Found no (symbol, interval) combos after feature-building. Check external preprocessing.")

        # 6) Base interval availability check & anchor candidates
        base_indices: List[pd.DatetimeIndex] = []
        for sym in self.symbols:
            if self.base_interval in self.df[sym]:
                base_indices.append(self.df[sym][self.base_interval].index)
        if not base_indices:
            raise ValueError(f"At least one symbol must have data for the base interval '{self.base_interval}'.")

        anchor_candidates: pd.DatetimeIndex = base_indices[0]
        for idx in base_indices[1:]:
            anchor_candidates = anchor_candidates.union(idx)
        anchor_candidates = anchor_candidates.sort_values()

        # 7) Build anchors (strict vs loose)
        L = cfg.window_size
        if cfg.strict_anchor:
            anchors_list: List[pd.Timestamp] = []
            for t in anchor_candidates:
                ok = True
                for sym, iv in self.combos:
                    idx_pos = self.df[sym][iv].index.searchsorted(t, side="right")
                    if idx_pos < L:
                        ok = False
                        break
                if ok:
                    anchors_list.append(t)
            anchors = anchors_list
        else:
            anchors = list(anchor_candidates)

        if len(anchors) == 0:
            raise ValueError("No usable anchor timestamps. Try strict_anchor=False or reduce window_size.")

        # 8) Drop the most recent anchor (latest timestamp) to allow t+1 labeling
        if len(anchors) >= 1:
            anchors = anchors[:-1]
        if len(anchors) == 0:
            raise ValueError("All anchors were dropped after removing the latest timestamp. Provide more data or relax constraints.")

        self.anchors = anchors

        # 9) Channel indices metadata
        self.channels_per_combo = len(self.feature_names)  # 6
        self.total_feature_channels = self.num_combos * self.channels_per_combo

        self.ti_channel_indices: List[int] = []
        for k in range(self.num_combos):
            base = k * self.channels_per_combo
            self.ti_channel_indices.append(base + TI_IDX_IN_DERIVED)

    def __len__(self) -> int:
        return len(self.anchors)

    def __getitem__(self, idx):
        # idx can be an int (anchor index) or a timestamp-like object
        if isinstance(idx, (int, np.integer)):
            t_anchor = self.anchors[int(idx)]
        else:
            t_anchor = pd.to_datetime(idx, utc=True)

        L = self.cfg.window_size
        C = self.total_feature_channels

        out_feat = np.full((C, L), np.nan, dtype=np.float32)
        out_mask = np.zeros((C, L), dtype=np.float32)          # 0/1 mask
        out_elapsed = np.full((C, L), np.nan, dtype=np.float32)

        for k, (sym, iv) in enumerate(self.combos):
            df = self.df[sym][iv]
            end_pos = df.index.searchsorted(t_anchor, side="right")
            start_pos = max(0, end_pos - L)
            window = df.iloc[start_pos:end_pos]

            w_feat = window[self.feature_names].to_numpy(dtype=np.float32)
            w_mask = window[self.mask_names].to_numpy(dtype=np.float32)
            w_elapsed = window[self.elapsed_names].to_numpy(dtype=np.float32)

            pad = L - w_feat.shape[0]
            if pad > 0:
                w_feat = np.concatenate([np.full((pad, w_feat.shape[1]), np.nan, dtype=np.float32), w_feat], axis=0)
                w_mask = np.concatenate([np.zeros((pad, w_mask.shape[1]), dtype=np.float32), w_mask], axis=0)
                w_elapsed = np.concatenate([np.full((pad, w_elapsed.shape[1]), np.nan, dtype=np.float32), w_elapsed], axis=0)

            ch_start = k * self.channels_per_combo
            ch_end = ch_start + self.channels_per_combo
            out_feat[ch_start:ch_end, :] = w_feat.T  # (6, L)
            out_mask[ch_start:ch_end, :] = w_mask.T
            out_elapsed[ch_start:ch_end, :] = w_elapsed.T

        
        # -------- Z-Score normalization (per-sample, per-channel) --------
        eps = 1e-6
        for ch in range(C):
            if ch in self.ti_channel_indices:
                continue  # taker_imbalance already in [-1, 1]
            vals = out_feat[ch]
            valid = ~np.isnan(vals)
            if valid.any():
                m = vals[valid].mean()
                s = vals[valid].std()
                if not np.isfinite(s) or s < eps:
                    out_feat[ch, valid] = 0.0
                else:
                    out_feat[ch, valid] = (vals[valid] - m) / s
        '''
        #Elapsed z-score intentionally disabled (kept as log1p minutes)
        for ch in range(C):
            vals = out_elapsed[ch]
            valid = ~np.isnan(vals)
            if valid.any():
                m = vals[valid].mean()
                s = vals[valid].std()
                out_elapsed[ch, valid] = 0.0 if (not np.isfinite(s) or s < eps) else (vals[valid] - m) / s
        '''
        np.nan_to_num(out_feat, copy=False, nan=0.0)
        np.nan_to_num(out_elapsed, copy=False, nan=0.0)

        #np.clip(out_feat, -8.0, 8.0, out=out_feat)

        X = torch.from_numpy(out_feat).transpose(0, 1).contiguous()    # (L, C)
        M = torch.from_numpy(out_mask).transpose(0, 1).contiguous()    # (L, C)
        T = torch.from_numpy(out_elapsed).transpose(0, 1).contiguous() # (L, C)

        meta = {
            "anchor_time": t_anchor,
            "combos": self.combos,
            "feature_names_6": self.feature_names,
        }
        return X, M, T, meta


# -----------------------------
# DataLoader & utilities
# -----------------------------

def make_dataloader(dataset: BinanceVisionDataset,
                    batch_size: int = 32,
                    shuffle: bool = True,
                    num_workers: int = 0,
                    pin_memory: bool = True) -> DataLoader:
    """
    Build a DataLoader that returns (X, M, T, metas) per batch,
    where X/M/T are (B, L, C) and metas is a list of dicts.
    """
    def collate_fn(batch):
        xs, ms, ts, metas = zip(*batch)
        X = torch.stack(xs, dim=0)   # (B, L, C)
        M = torch.stack(ms, dim=0)   # (B, L, C)
        T = torch.stack(ts, dim=0)   # (B, L, C)
        return X, M, T, list(metas)

    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      pin_memory=pin_memory,
                      collate_fn=collate_fn,
                      drop_last=False)