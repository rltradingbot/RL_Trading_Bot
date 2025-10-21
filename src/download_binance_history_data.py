"""
Download Binance historical OHLCV from data.binance.vision (no API),
and verify the zip via its .CHECKSUM (SHA-256).

Quick start examples:
  # monthly 1m candles for BTCUSDT, July 2025
  python vision_dl.py spot monthly klines BTCUSDT --interval 1m --ym 2025-07

  # daily 1h candles for ETHUSDT, 2025-08-15
  python vision_dl.py spot daily klines ETHUSDT --interval 1h --ymd 2025-08-15

Outputs:
  ./downloads/<file>.zip
  ./downloads/<unzipped csv files...>

Notes:
- data is organized by market/frequency/type/symbol[/interval]/file.zip
- after 2025-01-01, SPOT timestamps are in microseconds (per Binance docs).
Requires: pip install requests
"""
import os
import sys
from pathlib import Path
from typing import Optional
from data.binance.vision import build_urls, http_download, parse_checksum_text, sha256_file, unzip_to
from utils._time.expand_date import expand_date_or_month_range

DOWNLOAD_FOLDER_ROOT = '/data/workspace_294/private/aiden/old_experiments/experiments/data/binance/raw' 

def download_binance_vision_dataset(market: str, freq: str, dtype: str, symbol: str, interval:Optional[str], ym:Optional[str], ymd:Optional[str]):
    url_zip, url_checksum, fname = build_urls(
        market, freq, dtype, symbol, interval, ym, ymd
    )

    market_dir = Path(os.path.abspath(os.path.join(DOWNLOAD_FOLDER_ROOT, market)))
    market_dir.mkdir(parents=True, exist_ok=True)

    freq_dir = market_dir / freq
    freq_dir.mkdir(parents=True, exist_ok=True)

    dtype_dir = freq_dir / dtype
    dtype_dir.mkdir(parents=True, exist_ok=True)

    symbol_dir = dtype_dir / symbol
    symbol_dir.mkdir(parents=True, exist_ok=True)

    interval_dir = symbol_dir / interval
    interval_dir.mkdir(parents=True, exist_ok=True)

    zip_path = interval_dir / fname
    checksum_path = interval_dir / (fname + ".CHECKSUM")

    print(f"Downloading:\n  {url_zip}\n  {url_checksum}")

    # 1) download .zip and .CHECKSUM
    http_download(url_zip, zip_path)
    http_download(url_checksum, checksum_path)

    # 2) read expected hash from .CHECKSUM
    expected_hex = parse_checksum_text(checksum_path.read_text(), expect_filename=fname)

    # 3) compute local sha256
    actual_hex = sha256_file(zip_path)
    print(f"Expected SHA256: {expected_hex}")
    print(f"Actual   SHA256: {actual_hex}")

    if actual_hex != expected_hex:
        print("❌ Checksum mismatch — file may be corrupted. Aborting.", file=sys.stderr)
        sys.exit(2)

    print("✅ Checksum verified. Unzipping…")
    unzip_to(interval_dir, zip_path)
    print(f"Done. Files are in: {interval_dir.resolve()}")

def main():
  market = "spot"
  freq = "monthly"      # choices=["daily", "monthly"]
  dtype = "klines"    # choices=["klines", "trades", "aggTrades"]
  interval = "1m"     # 1m,5m,15m,1h,4h,1d
  #ym = "2025-01"      # YYYY-MM
  #ymd = "2025-01-01"  # YYYY-MM-DD

  '''data available
  BTCUSDT 2017-08
  ETHUSDT 2017-08
  XRPUSDT 2018-05
  BNBUSDT 2017-11
  DOGEUSDT 2019-07
  SOLUSDT 2020-08
  TRXUSDT 2018-06
  ADAUSDT 2018-04
  SHIBUSDT 2021-05
  PEPEUSDT 2023-05
  '''

  symbols = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "BNBUSDT", "DOGEUSDT", "SOLUSDT", "TRXUSDT", "ADAUSDT"]
  intervals = ["1m", "5m", "15m", "1h", "4h", "1d"]
  month_from_table = {"BTCUSDT": "2017-08",
                       "ETHUSDT": "2017-08", 
                       "XRPUSDT": "2018-05", 
                       "BNBUSDT": "2017-11", 
                       "DOGEUSDT": "2019-07", 
                       "SOLUSDT": "2020-08",
                       "TRXUSDT": "2018-06",
                       "ADAUSDT": "2018-04",
                       "SHIBUSDT": "2021-05",
                       "PEPEUSDT": "2023-05"}
  
  month_to = "2025-07"
  for symbol in symbols:
      month_from = month_from_table[symbol]
      months = expand_date_or_month_range(month_from, month_to)
      for month in months:
        for interval in intervals:
          download_binance_vision_dataset(market, freq, dtype, symbol, interval, month, None)


if __name__ == "__main__":
    main()
