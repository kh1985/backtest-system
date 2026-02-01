"""
SUI, TRX, HYPE 1年分ダウンロード
"""

import os
import sys
import zipfile
import urllib.request
import urllib.error
from datetime import date, timedelta
from pathlib import Path

SYMBOLS = ["SUIUSDT", "TRXUSDT", "HYPEUSDT"]
TIMEFRAMES = ["1m", "15m", "1h", "4h"]

MONTHLY_START = (2025, 2)
MONTHLY_END = (2025, 12)
DAILY_START = date(2026, 1, 1)
DAILY_END = date(2026, 1, 28)

BASE_DIR = Path(__file__).resolve().parent.parent / "sample_data" / "binance"
MONTHLY_BASE = "https://data.binance.vision/data/spot/monthly/klines"
DAILY_BASE = "https://data.binance.vision/data/spot/daily/klines"


def download_file(url, dest):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
            with open(dest, "wb") as f:
                f.write(data)
        return True
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False
        print(f"  HTTP Error {e.code}: {url}")
        return False
    except Exception as e:
        print(f"  Error: {e}")
        return False


def generate_months(start, end):
    months = []
    y, m = start
    ey, em = end
    while (y, m) <= (ey, em):
        months.append((y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return months


def generate_dates(start, end):
    dates = []
    d = start
    while d <= end:
        dates.append(d)
        d += timedelta(days=1)
    return dates


def download_symbol(symbol):
    output_dir = BASE_DIR / f"{symbol}_1y"
    output_dir.mkdir(parents=True, exist_ok=True)

    months = generate_months(MONTHLY_START, MONTHLY_END)
    dates = generate_dates(DAILY_START, DAILY_END)

    print(f"\n{'='*50}")
    print(f"  {symbol} - 1 Year Download")
    print(f"{'='*50}")

    downloaded = 0
    skipped = 0
    failed = 0

    for tf in TIMEFRAMES:
        tf_dir = output_dir / tf
        tf_dir.mkdir(exist_ok=True)
        print(f"  --- {tf} ---")

        for y, m in months:
            filename = f"{symbol}-{tf}-{y}-{m:02d}.zip"
            dest = tf_dir / filename
            url = f"{MONTHLY_BASE}/{symbol}/{tf}/{filename}"

            if dest.exists():
                skipped += 1
                continue

            ok = download_file(url, dest)
            if ok:
                size_kb = dest.stat().st_size / 1024
                print(f"    {filename} ({size_kb:.0f} KB)")
                downloaded += 1
            else:
                print(f"    {filename} - not found")
                failed += 1

        for d in dates:
            filename = f"{symbol}-{tf}-{d}.zip"
            dest = tf_dir / filename
            url = f"{DAILY_BASE}/{symbol}/{tf}/{filename}"

            if dest.exists():
                skipped += 1
                continue

            ok = download_file(url, dest)
            if ok:
                size_kb = dest.stat().st_size / 1024
                print(f"    {filename} ({size_kb:.0f} KB)")
                downloaded += 1
            else:
                failed += 1

    print(f"  Downloaded: {downloaded} / Skipped: {skipped} / Failed: {failed}")

    print(f"  Merging...")
    merge_zips(symbol, output_dir)

    return downloaded, skipped, failed


def merge_zips(symbol, output_dir):
    for tf in TIMEFRAMES:
        tf_dir = output_dir / tf
        zips = sorted(tf_dir.glob("*.zip"))
        if not zips:
            print(f"    {tf}: No ZIPs found")
            continue

        output_csv = BASE_DIR / f"{symbol}-{tf}-20250201-20260130-merged.csv"
        rows = []

        for zp in zips:
            try:
                with zipfile.ZipFile(zp, "r") as zf:
                    csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
                    for cn in csv_names:
                        with zf.open(cn) as f:
                            text = f.read().decode("utf-8")
                            for line in text.strip().split("\n"):
                                if line and not line[0].isalpha():
                                    rows.append(line)
            except Exception as e:
                print(f"    Error reading {zp.name}: {e}")

        if rows:
            rows.sort(key=lambda r: int(r.split(",")[0]))
            seen = set()
            unique = []
            for r in rows:
                key = r.split(",")[0]
                if key not in seen:
                    seen.add(key)
                    unique.append(r)

            with open(output_csv, "w", newline="") as f:
                f.write("\n".join(unique))

            print(f"    {tf}: {len(unique)} rows -> {output_csv.name}")
        else:
            print(f"    {tf}: No data rows found")


def main():
    print(f"=== Multi-Symbol Download (1 Year) ===")
    print(f"Symbols: {SYMBOLS}")

    total_dl = 0
    total_skip = 0
    total_fail = 0

    for symbol in SYMBOLS:
        dl, skip, fail = download_symbol(symbol)
        total_dl += dl
        total_skip += skip
        total_fail += fail

    print(f"\n{'='*50}")
    print(f"  ALL DONE")
    print(f"  Total Downloaded: {total_dl}")
    print(f"  Total Skipped: {total_skip}")
    print(f"  Total Failed: {total_fail}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
