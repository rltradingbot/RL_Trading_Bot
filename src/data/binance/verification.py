#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Scan a Binance Vision symbol folder and detect missing/duplicate kline candles
across all existing interval subfolders. Works with mixed epoch units:
seconds (s), milliseconds (ms), and microseconds (us).

Usage:
    python check_missing_klines.py /path/to/SYMBOL
"""

from pathlib import Path
import re
from typing import Dict, List
import pandas as pd

# ---- Interval helpers ------------------------------------------------------

INTERVAL_PATTERN = re.compile(r"^(\d+)([mhdwM])$")

def interval_to_offset(interval: str) -> pd.DateOffset:
    """Convert Binance interval string (e.g., '15m', '4h', '1d', '1w', '1M') to a pandas DateOffset."""
    n, unit = int(INTERVAL_PATTERN.match(interval).group(1)), INTERVAL_PATTERN.match(interval).group(2)
    return {
        "m": pd.DateOffset(minutes=n),
        "h": pd.DateOffset(hours=n),
        "d": pd.DateOffset(days=n),
        "w": pd.DateOffset(weeks=n),
        "M": pd.DateOffset(months=n),
    }[unit]

# ---- CSV reading -----------------------------------------------------------

def infer_epoch_unit(sample: pd.Series) -> str:
    """
    Infer epoch unit by order of magnitude.
    - us: 1e15 ~ 1e16 (16 digits)
    - ms: 1e12 ~ 1e13 (13 digits)
    - s : 1e9  ~ 1e10 (10 digits)
    Returns 'us', 'ms', or 's'.
    """
    med = int(sample.median())
    if med >= 10**14:
        return "us"
    if med >= 10**12:
        return "ms"
    if med >= 10**9:
        return "s"
    # Fallback: assume milliseconds (most common in older Vision dumps)
    return "ms"

def read_open_times(csv_path: Path) -> pd.DatetimeIndex:
    """
    Read only the first column (open_time) as UTC DatetimeIndex.
    Handles seconds, milliseconds, and microseconds automatically.
    Ignores headers by coercing non-numeric values to NaN.
    """
    s = pd.read_csv(
        csv_path,
        header=None,
        usecols=[0],
        dtype=str,
        engine="c",
        low_memory=False,
    ).iloc[:, 0]

    nums = pd.to_numeric(s, errors="coerce").dropna().astype("int64")
    if nums.empty:
        return pd.DatetimeIndex([], tz="UTC")

    # Use a small sample to infer the unit robustly
    sample = nums.head(1000)
    unit = infer_epoch_unit(sample)

    return pd.to_datetime(nums, unit=unit, utc=True)

def read_interval_times(interval_dir: Path) -> pd.DatetimeIndex:
    """Read all CSVs in an interval directory and return unique, sorted open_time index (UTC)."""
    parts: List[pd.DatetimeIndex] = []
    for csv_path in sorted(interval_dir.glob("*.csv")):
        dt_idx = read_open_times(csv_path)
        if not dt_idx.empty:
            parts.append(dt_idx)
    if not parts:
        return pd.DatetimeIndex([], tz="UTC")
    merged = pd.DatetimeIndex(pd.unique(pd.concat(parts)))
    return merged.sort_values()

# ---- Gap detection ---------------------------------------------------------

def expected_range(start: pd.Timestamp, end: pd.Timestamp, offset: pd.DateOffset) -> pd.DatetimeIndex:
    """Build the expected time grid from start to end (inclusive) stepping by `offset`."""
    out = []
    t = start
    # Simple guard to avoid runaway loops if timestamps are corrupted
    hard_cap = 20_000_000  # plenty for years of minute bars
    while t <= end and len(out) < hard_cap:
        out.append(t)
        t = t + offset
    return pd.DatetimeIndex(out)

def check_interval(interval_dir: Path, interval_name: str) -> Dict:
    """Compute missing and duplicate candles for a given interval folder."""
    actual = read_interval_times(interval_dir)
    if actual.empty:
        return {
            "interval": interval_name,
            "files": 0,
            "actual_count": 0,
            "expected_count": 0,
            "missing_count": 0,
            "duplicate_count": 0,
            "missing_examples_utc": [],
            "duplicate_examples_utc": [],
        }

    # Duplicate detection: count per-file occurrences
    dup_counts = {}
    for csv_path in sorted(interval_dir.glob("*.csv")):
        for t in read_open_times(csv_path):
            dup_counts[t] = dup_counts.get(t, 0) + 1
    dups = sorted([t for t, c in dup_counts.items() if c > 1])

    offset = interval_to_offset(interval_name)
    exp = expected_range(actual[0], actual[-1], offset)
    missing = exp.difference(actual)

    return {
        "interval": interval_name,
        "files": len(list(interval_dir.glob("*.csv"))),
        "actual_count": len(actual),
        "expected_count": len(exp),
        "missing_count": len(missing),
        "duplicate_count": len(dups),
        "missing_examples_utc": [ts.isoformat() for ts in missing[:10]],
        "duplicate_examples_utc": [ts.isoformat() for ts in dups[:10]],
    }

# ---- Main scan -------------------------------------------------------------

def scan_symbol_folder(symbol_folder: Path) -> List[Dict]:
    """Scan all valid interval subfolders under a symbol folder."""
    results = []
    for sub in sorted(p for p in symbol_folder.iterdir() if p.is_dir()):
        name = sub.name
        if not INTERVAL_PATTERN.match(name):
            continue
        try:
            results.append(check_interval(sub, name))
        except Exception as e:
            results.append({"interval": name, "error": str(e), "files": len(list(sub.glob('*.csv')))})
    return results

def print_report(results: List[Dict]) -> None:
    """Simple console report."""
    print("\n=== Missing/Duplicate Klines Report ===\n")
    for r in results:
        if "error" in r:
            print(f"[{r['interval']}] ERROR: {r['error']} (files: {r.get('files', 0)})\n")
            continue
        print(f"[{r['interval']}] files={r['files']}, actual={r['actual_count']}, expected={r['expected_count']}")
        print(f"  -> missing={r['missing_count']}, duplicates={r['duplicate_count']}")
        if r["missing_examples_utc"]:
            print("  missing examples (UTC):")
            for ts in r["missing_examples_utc"]:
                print(f"    - {ts}")
        if r["duplicate_examples_utc"]:
            print("  duplicate examples (UTC):")
            for ts in r["duplicate_examples_utc"]:
                print(f"    - {ts}")
        print("")
    total_missing = sum(r.get("missing_count", 0) for r in results if "missing_count" in r)
    total_dups = sum(r.get("duplicate_count", 0) for r in results if "duplicate_count" in r)
    print(f"=== Summary: total_missing={total_missing}, total_duplicates={total_dups} ===\n")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python check_missing_klines.py /path/to/SYMBOL")
        sys.exit(1)
    p = Path(sys.argv[1]).expanduser().resolve()
    if not p.is_dir():
        print(f"Symbol path not found or not a directory: {p}")
        sys.exit(1)
    print_report(scan_symbol_folder(p))