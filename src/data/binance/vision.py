import hashlib
import re
import zipfile
from pathlib import Path
import requests
from typing import Optional

BASE = "https://data.binance.vision/data"

def sha256_file(path: Path, chunk=1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()

def parse_checksum_text(txt: str, expect_filename: str) -> str:
    """
    .CHECKSUM can be either:
      1) "<hex>  filename"
      2) "SHA256 (filename) = <hex>"
    Return the hex digest (lowercase).
    """
    txt = txt.strip()
    # pattern 1
    m = re.search(r"([a-fA-F0-9]{64})\s+(\S+)", txt)
    if m:
        hex_, fname = m.group(1), m.group(2)
        # sometimes fname may be just the basename; sanity check only
        if expect_filename.endswith(Path(fname).name):
            return hex_.lower()
    # pattern 2
    m = re.search(r"SHA256\s*\(\s*([^)]+)\s*\)\s*=\s*([a-fA-F0-9]{64})", txt)
    if m:
        fname, hex_ = m.group(1), m.group(2)
        if expect_filename.endswith(Path(fname).name):
            return hex_.lower()
    # fallback: if it’s just a naked hash
    m = re.search(r"\b([a-fA-F0-9]{64})\b", txt)
    if m:
        return m.group(1).lower()
    raise ValueError("Could not parse CHECKSUM file format.")

def build_urls(market: str, freq: str, dtype: str, symbol: str,
               interval: Optional[str], ym: Optional[str], ymd: Optional[str]) -> tuple[str, str, str]:
    """
    Build the exact remote URL and filenames.
    dtype: "klines" (needs interval) or "trades"/"aggTrades" (no interval)
    freq: "daily" or "monthly"
    """
    market = market.lower()        # "spot", "futures/um", etc. (we’ll keep it simple: spot)
    freq = freq.lower()            # daily | monthly
    dtype = dtype                   # klines | trades | aggTrades
    if dtype == "klines":
        if interval is None:
            raise ValueError("klines requires --interval like 1m, 1h, 1d, 1s...")
        if freq == "monthly":
            if not ym:
                raise ValueError("monthly requires --ym YYYY-MM")
            # e.g. data/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2025-07.zip
            folder = f"{BASE}/{market}/{freq}/{dtype}/{symbol}/{interval}"
            fname = f"{symbol}-{interval}-{ym}.zip"
        else:
            if not ymd:
                raise ValueError("daily requires --ymd YYYY-MM-DD")
            # e.g. data/spot/daily/klines/BTCUSDT/1m/BTCUSDT-1m-2025-08-15.zip
            folder = f"{BASE}/{market}/{freq}/{dtype}/{symbol}/{interval}"
            fname = f"{symbol}-{interval}-{ymd}.zip"
    else:
        # trades or aggTrades
        if freq == "monthly":
            if not ym:
                raise ValueError("monthly requires --ym YYYY-MM")
            folder = f"{BASE}/{market}/{freq}/{dtype}/{symbol}"
            fname = f"{symbol}-{dtype}-{ym}.zip"
        else:
            if not ymd:
                raise ValueError("daily requires --ymd YYYY-MM-DD")
            folder = f"{BASE}/{market}/{freq}/{dtype}/{symbol}"
            fname = f"{symbol}-{dtype}-{ymd}.zip"

    url_zip = f"{folder}/{fname}"
    url_checksum = f"{url_zip}.CHECKSUM"
    return url_zip, url_checksum, fname

def http_download(url: str, out_path: Path, timeout=60):
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with out_path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

def unzip_to(dir_out: Path, zip_path: Path):
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dir_out)