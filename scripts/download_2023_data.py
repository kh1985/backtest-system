"""
Binance data.binance.vision から全銘柄の2023年分データをダウンロード

対象: 全30銘柄（現在のinputdataと同じ）
期間: 2023-02 ~ 2024-01（約1年）
TF:   15m, 1h, 4h（1mは除外）

出力先: inputdata/
"""

import os
import sys
import zipfile
import urllib.request
import urllib.error
from datetime import date, timedelta
from pathlib import Path

# 全30銘柄
SYMBOLS = [
    "AAVEUSDT", "ADAUSDT", "APTUSDT", "ARBUSDT", "ATOMUSDT",
    "AVAXUSDT", "BNBUSDT", "BTCUSDT", "DOGEUSDT", "DOTUSDT",
    "ENAUSDT", "ETHUSDT", "FILUSDT", "INJUSDT", "JUPUSDT",
    "LINKUSDT", "LTCUSDT", "NEARUSDT", "ONDOUSDT", "OPUSDT",
    "PEPEUSDT", "RENDERUSDT", "SEIUSDT", "SOLUSDT", "SUIUSDT",
    "TIAUSDT", "TRXUSDT", "UNIUSDT", "WIFUSDT", "XRPUSDT",
]

# 1mを除外（15m, 1h, 4hのみ）
TIMEFRAMES = ["15m", "1h", "4h"]

# 2023-02 ~ 2023-12 は月次、2024-01 は日次
MONTHLY_START = (2023, 2)
MONTHLY_END = (2023, 12)
DAILY_START = date(2024, 1, 1)
DAILY_END = date(2024, 1, 31)

# 出力先
BASE_DIR = Path(__file__).resolve().parent.parent / "inputdata"
MONTHLY_BASE = "https://data.binance.vision/data/spot/monthly/klines"
DAILY_BASE = "https://data.binance.vision/data/spot/daily/klines"

# 期間識別子（ファイル名用）
PERIOD_TAG = "20230201-20240131"


def download_file(url: str, dest: Path) -> bool:
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


def download_symbol(symbol, temp_dir):
    """1銘柄のデータをダウンロード"""
    output_dir = temp_dir / symbol
    output_dir.mkdir(parents=True, exist_ok=True)

    months = generate_months(MONTHLY_START, MONTHLY_END)
    dates = generate_dates(DAILY_START, DAILY_END)

    print(f"\n{'='*50}")
    print(f"  {symbol}")
    print(f"{'='*50}")

    downloaded = 0
    skipped = 0
    failed = 0

    for tf in TIMEFRAMES:
        tf_dir = output_dir / tf
        tf_dir.mkdir(exist_ok=True)
        print(f"  --- {tf} ---")

        # 月次データ
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

        # 日次データ（2024年1月分）
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
    return downloaded, skipped, failed


def merge_zips(symbol, temp_dir):
    """ZIPファイルをマージしてCSV出力"""
    output_dir = temp_dir / symbol

    for tf in TIMEFRAMES:
        tf_dir = output_dir / tf
        zips = sorted(tf_dir.glob("*.zip"))
        if not zips:
            print(f"    {tf}: No ZIPs found")
            continue

        output_csv = BASE_DIR / f"{symbol}-{tf}-{PERIOD_TAG}.csv"
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
            # タイムスタンプでソートして重複除去
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
    import tempfile
    import shutil

    print(f"=== 2023 Data Download (excluding 1m) ===")
    print(f"Symbols: {len(SYMBOLS)} symbols")
    print(f"Period: {MONTHLY_START[0]}-{MONTHLY_START[1]:02d} ~ {DAILY_END}")
    print(f"Timeframes: {TIMEFRAMES}")
    print(f"Output: {BASE_DIR}")

    # 一時ディレクトリを使用
    temp_base = Path(tempfile.mkdtemp(prefix="binance_dl_"))
    print(f"Temp dir: {temp_base}")

    total_dl = 0
    total_skip = 0
    total_fail = 0

    try:
        for i, symbol in enumerate(SYMBOLS):
            print(f"\n[{i+1}/{len(SYMBOLS)}] Processing {symbol}...")
            dl, skip, fail = download_symbol(symbol, temp_base)
            total_dl += dl
            total_skip += skip
            total_fail += fail

            # マージ
            print(f"  Merging...")
            merge_zips(symbol, temp_base)

    finally:
        # 一時ディレクトリ削除（容量節約）
        print(f"\nCleaning up temp dir...")
        shutil.rmtree(temp_base, ignore_errors=True)

    print(f"\n{'='*50}")
    print(f"  ALL DONE")
    print(f"  Total Downloaded: {total_dl}")
    print(f"  Total Skipped: {total_skip}")
    print(f"  Total Failed: {total_fail}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
