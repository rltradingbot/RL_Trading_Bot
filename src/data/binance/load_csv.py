import pandas as pd
from pathlib import Path
from typing import Tuple

# -------------------------------
# 1) Path-based detectors
# -------------------------------

def _detect_market_and_dtype(csv_path: str) -> Tuple[str, str]:
    """
    Infer market (spot/futures_um/futures_cm/option) and dtype (klines/trades/aggtrades) from the file path.
    """
    p = str(csv_path).replace("\\", "/").lower()
    if "/spot/" in p:
        market = "spot"
    elif "/futures/um/" in p or "/futures/usdm/" in p:
        market = "futures_um"
    elif "/futures/cm/" in p or "/futures/coinm/" in p:
        market = "futures_cm"
    elif "/option/" in p or "/options/" in p:
        market = "option"
    else:
        market = None

    dtype = None
    for dt in ("klines", "trades", "aggtrades"):
        if f"/{dt}/" in p:
            dtype = dt
            break

    return market, dtype


def _has_header(csv_path: str) -> bool:
    """
    If the first non-empty line contains any letter, consider it as a header.
    """
    with open(csv_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            return any(c.isalpha() for c in s)
    return False


# -------------------------------
# 2) Header presets (Binance schema)
# -------------------------------

# Spot & USD-M Futures Kline
STD_KLINE = [
    "open_time","open","high","low","close","volume",
    "close_time","quote_asset_volume","number_of_trades",
    "taker_buy_base_asset_volume","taker_buy_quote_asset_volume","ignore"
]

# COIN-M Kline (field names differ)
CM_KLINE = [
    "open_time","open","high","low","close","volume",
    "close_time","base_asset_volume","number_of_trades",
    "taker_buy_volume","taker_buy_base_asset_volume","ignore"
]

# Spot aggTrades (has is_best_match)
SPOT_AGG = [
    "agg_trade_id","price","quantity","first_trade_id",
    "last_trade_id","timestamp","is_buyer_maker","is_best_match"
]

# Futures aggTrades (no is_best_match)
FUT_AGG = [
    "agg_trade_id","price","quantity","first_trade_id",
    "last_trade_id","timestamp","is_buyer_maker"
]

# Spot trades
SPOT_TRADES = ["id","price","qty","quoteQty","time","isBuyerMaker","isBestMatch"]

# USD-M trades
UM_TRADES   = ["id","price","qty","quoteQty","time","isBuyerMaker"]

# COIN-M trades
CM_TRADES   = ["id","price","qty","baseQty","time","isBuyerMaker"]

# Options klines (different schema)
OPT_KLINE = [
    "open","high","low","close","volume","amount",
    "interval","tradeCount","takerVolume","takerAmount",
    "openTime","closeTime"
]

# Options trades
OPT_TRADES = ["id","symbol","price","qty","quoteQty","side","time"]


def get_binance_headers(csv_path: str):
    market, dtype = _detect_market_and_dtype(csv_path)
    if market is None or dtype is None:
        raise ValueError("Path must include market (spot/futures/option) and dtype (klines/trades/aggTrades).")

    if dtype == "klines":
        if market in ("spot", "futures_um"):
            return STD_KLINE
        elif market == "futures_cm":
            return CM_KLINE
        elif market == "option":
            return OPT_KLINE

    if dtype == "trades":
        if market == "spot":
            return SPOT_TRADES
        elif market == "futures_um":
            return UM_TRADES
        elif market == "futures_cm":
            return CM_TRADES
        elif market == "option":
            return OPT_TRADES

    if dtype == "aggtrades":
        if market == "spot":
            return SPOT_AGG
        elif market in ("futures_um", "futures_cm"):
            return FUT_AGG
        elif market == "option":
            raise ValueError("aggTrades dataset is not provided for Options.")

    raise ValueError("Unsupported market/dtype combination or ambiguous path.")


# -------------------------------
# 3) Smart epoch → UTC converter
# -------------------------------

def _infer_time_unit_from_values(s: pd.Series) -> str:
    """
    Guess the epoch unit by magnitude (vectorized).
    - >= 1e15 → microseconds
    - >= 1e12 → milliseconds
    - >= 1e9  → seconds
    Fallback to milliseconds if unclear.
    """
    s_num = pd.to_numeric(s, errors="coerce")
    max_abs = s_num.dropna().abs().max()
    if pd.isna(max_abs):
        return "ms"
    if max_abs >= 1e15:
        return "us"
    if max_abs >= 1e12:
        return "ms"
    if max_abs >= 1e9:
        return "s"
    # extremely small or already human-readable? treat as ms to avoid explosions
    return "ms"


def _to_utc(series: pd.Series) -> pd.Series:
    """
    Convert epoch-like numbers to pandas UTC datetimes, auto-detecting unit.
    """
    unit = _infer_time_unit_from_values(series)
    s_num = pd.to_numeric(series, errors="coerce")
    return pd.to_datetime(s_num, unit=unit, utc=True)


def _add_utc_columns(df: pd.DataFrame, market: str, dtype: str) -> pd.DataFrame:
    """
    Add new UTC datetime columns, keeping originals intact.
    - klines (spot/futures): open_time_utc / close_time_utc
    - klines (options):     openTimeUTC / closeTimeUTC
    - trades:               time_utc
    - aggTrades:            timestamp_utc
    """
    def maybe_add(src_col: str, new_col: str):
        if src_col in df.columns and new_col not in df.columns:
            df[new_col] = _to_utc(df[src_col])

    if dtype == "klines":
        # spot & futures
        maybe_add("open_time",  "open_time_utc")
        maybe_add("close_time", "close_time_utc")
        # options
        maybe_add("openTime",   "openTimeUTC")
        maybe_add("closeTime",  "closeTimeUTC")

    elif dtype == "trades":
        maybe_add("time", "time_utc")

    elif dtype == "aggtrades":
        maybe_add("timestamp", "timestamp_utc")

    return df


# -------------------------------
# 4) Public entrypoint
# -------------------------------

def read_binance_with_header(csv_path: str) -> pd.DataFrame:
    """
    Input: CSV file path (Binance Vision layout; path contains market/dtype words)
    Output: pandas DataFrame with proper headers + extra UTC datetime columns added.
    - Keeps original numeric time columns.
    - Adds human-readable UTC columns next to them.
    - Error messages are in English.
    """
    market, dtype = _detect_market_and_dtype(csv_path)
    if _has_header(csv_path):
        df = pd.read_csv(csv_path)
        if market and dtype:
            df = _add_utc_columns(df, market, dtype)
        return df

    headers = get_binance_headers(csv_path)
    df = pd.read_csv(csv_path, header=None, names=headers)
    df = _add_utc_columns(df, market, dtype)
    return df

from typing import Dict

def load_vision_monthlies_csv(
    symbol: str,
    start_month: str,  # "YYYY-MM"
    end_month: str,    # "YYYY-MM"
    root_dir: str      # ".../raw/spot/monthly/klines"
) -> Dict[str, pd.DataFrame]:
    symbol_dir = Path(root_dir) / symbol
    if not symbol_dir.exists():
        raise FileNotFoundError(f"Symbol dir not found: {symbol_dir}")

    months = pd.period_range(start=start_month, end=end_month, freq="M").strftime("%Y-%m").tolist()

    start_dt = pd.Timestamp(f"{start_month}-01", tz="UTC")
    end_exclusive = pd.Timestamp(f"{end_month}-01", tz="UTC") + pd.offsets.MonthBegin(1)

    out: Dict[str, pd.DataFrame] = {}

    for interval_dir in sorted(p for p in symbol_dir.iterdir() if p.is_dir()):
        interval = interval_dir.name
        frames = []

        for ym in months:
            csv_path = interval_dir / f"{symbol}-{interval}-{ym}.csv"
            if not csv_path.exists():
                continue

            df = read_binance_with_header(str(csv_path))
            frames.append(df)

        if not frames:
            continue

        merged = pd.concat(frames, ignore_index=True)

        time_col_utc = next((c for c in ("close_time_utc", "closeTimeUTC", "open_time_utc", "openTimeUTC", "time_utc", "timestamp_utc") if c in merged.columns), None)
        if time_col_utc is None:
            raise ValueError(f"No UTC time column found for interval {interval}. Check the dataset path/type.")

        merged = merged[(merged[time_col_utc] >= start_dt) & (merged[time_col_utc] < end_exclusive)]
        merged = merged.sort_values(time_col_utc).drop_duplicates(subset=[time_col_utc], keep="last").reset_index(drop=True)

        out[interval] = merged

    return out


if __name__ == '__main__':
    csv_path = '/data/workspace_294/private/aiden/old_experiments/experiments/data/binance/raw/spot/monthly/klines/ADAUSDT/4h/ADAUSDT-4h-2018-06.csv'
    df = read_binance_with_header(csv_path)
    df.to_csv('test.csv', index = False)