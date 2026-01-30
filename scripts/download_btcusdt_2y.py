"""
Binance data.binance.vision からBTCUSDT 2年分をダウンロード

対象: BTCUSDT spot
期間: 2024-02 ~ 2026-01（約2年）
TF:   1m, 15m, 1h, 4h

月別ZIP（monthly）+ 直近月は日別ZIP（daily）で取得。
"""

import os
import sys
import zipfile
import urllib.request
import urllib.error
from datetime import date, timedelta
from pathlib import Path

SYMBOL = "BTCUSDT"
TIMEFRAMES = ["1m", "15m", "1h", "4h"]

# 月別データ: 2024-02 ~ 2025-12
MONTHLY_START = (2024, 2)
MONTHLY_END = (2025, 12)

# 日別データ: 2026-01-01 ~ 2026-01-28（直近月、月別ZIPが未公開の場合）
DAILY_START = date(2026, 1, 1)
DAILY_END = date(2026, 1, 28)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "sample_data" / "binance" / "BTCUSDT"

MONTHLY_BASE = "https://data.binance.vision/data/spot/monthly/klines"
DAILY_BASE = "https://data.binance.vision/data/spot/daily/klines"


def download_file(url: str, dest: Path) -> bool:
    """URLからファイルをダウンロード"""
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
    """(year, month) のリストを生成"""
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
    """日付リストを生成"""
    dates = []
    d = start
    while d <= end:
        dates.append(d)
        d += timedelta(days=1)
    return dates


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    months = generate_months(MONTHLY_START, MONTHLY_END)
    dates = generate_dates(DAILY_START, DAILY_END)

    print(f"=== Binance Data Download (2 Years) ===")
    print(f"Symbol: {SYMBOL}")
    print(f"Monthly: {MONTHLY_START[0]}-{MONTHLY_START[1]:02d} ~ {MONTHLY_END[0]}-{MONTHLY_END[1]:02d} ({len(months)} months)")
    print(f"Daily:   {DAILY_START} ~ {DAILY_END} ({len(dates)} days)")
    print(f"Timeframes: {TIMEFRAMES}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    downloaded = 0
    skipped = 0
    failed = 0

    for tf in TIMEFRAMES:
        tf_dir = OUTPUT_DIR / tf
        tf_dir.mkdir(exist_ok=True)
        print(f"--- {tf} ---")

        # 月別ダウンロード
        for y, m in months:
            filename = f"{SYMBOL}-{tf}-{y}-{m:02d}.zip"
            dest = tf_dir / filename
            url = f"{MONTHLY_BASE}/{SYMBOL}/{tf}/{filename}"

            if dest.exists():
                skipped += 1
                continue

            ok = download_file(url, dest)
            if ok:
                size_kb = dest.stat().st_size / 1024
                print(f"  {filename} ({size_kb:.0f} KB)")
                downloaded += 1
            else:
                print(f"  {filename} - not found")
                failed += 1

        # 日別ダウンロード（直近月）
        for d in dates:
            filename = f"{SYMBOL}-{tf}-{d}.zip"
            dest = tf_dir / filename
            url = f"{DAILY_BASE}/{SYMBOL}/{tf}/{filename}"

            if dest.exists():
                skipped += 1
                continue

            ok = download_file(url, dest)
            if ok:
                size_kb = dest.stat().st_size / 1024
                print(f"  {filename} ({size_kb:.0f} KB)")
                downloaded += 1
            else:
                failed += 1

        print()

    print(f"=== Download Done ===")
    print(f"Downloaded: {downloaded}")
    print(f"Skipped: {skipped}")
    print(f"Failed/Not available: {failed}")
    print()

    # マージ
    print("Merging into single CSV per timeframe...")
    _merge_zips()


def _merge_zips():
    """各TFのZIPを展開して1つのCSVに結合"""
    merged_dir = OUTPUT_DIR.parent  # sample_data/binance/

    for tf in TIMEFRAMES:
        tf_dir = OUTPUT_DIR / tf
        zips = sorted(tf_dir.glob("*.zip"))
        if not zips:
            print(f"  {tf}: No ZIPs found")
            continue

        output_csv = merged_dir / f"{SYMBOL}-{tf}-merged.csv"
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
                print(f"  Error reading {zp.name}: {e}")

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

            print(f"  {tf}: {len(unique)} rows -> {output_csv.name}")
        else:
            print(f"  {tf}: No data rows found")


if __name__ == "__main__":
    main()
