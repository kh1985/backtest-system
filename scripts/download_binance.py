"""
Binance data.binance.vision からOHLCVデータをダウンロード

対象: WLDUSDT spot
期間: 2025-12-28 ~ 2026-01-30
TF:   1m, 15m, 1h, 4h
"""

import os
import sys
import urllib.request
import urllib.error
from datetime import date, timedelta
from pathlib import Path

SYMBOL = "WLDUSDT"
BASE_URL = "https://data.binance.vision/data/spot/daily/klines"
TIMEFRAMES = ["1m", "15m", "1h", "4h"]
START_DATE = date(2025, 12, 28)
END_DATE = date(2026, 1, 30)  # inclusive

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "sample_data" / "binance"


def download_file(url: str, dest: Path) -> bool:
    """URLからファイルをダウンロード"""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
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


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 日付リスト生成
    dates = []
    d = START_DATE
    while d <= END_DATE:
        dates.append(d)
        d += timedelta(days=1)

    total_files = len(dates) * len(TIMEFRAMES)
    downloaded = 0
    skipped = 0
    failed = 0

    print(f"=== Binance Data Download ===")
    print(f"Symbol: {SYMBOL}")
    print(f"Period: {START_DATE} ~ {END_DATE} ({len(dates)} days)")
    print(f"Timeframes: {TIMEFRAMES}")
    print(f"Total files to download: {total_files}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    # まず spot を試す。404なら futures/um を試す
    base_urls = [
        f"https://data.binance.vision/data/spot/daily/klines",
        f"https://data.binance.vision/data/futures/um/daily/klines",
    ]

    # 最初のTFの最初の日付で、どちらのURLパターンが有効かチェック
    active_base = None
    test_d = dates[0]
    test_tf = TIMEFRAMES[0]
    for base in base_urls:
        test_url = f"{base}/{SYMBOL}/{test_tf}/{SYMBOL}-{test_tf}-{test_d}.zip"
        print(f"Testing: {test_url}")
        if download_file(test_url, OUTPUT_DIR / "test.zip"):
            active_base = base
            os.remove(OUTPUT_DIR / "test.zip")
            source_type = "spot" if "spot" in base else "futures/um"
            print(f"  -> Found on {source_type}!")
            break
        else:
            print(f"  -> Not found")

    if active_base is None:
        print(f"\nERROR: {SYMBOL} not found on spot or futures.")
        print("Trying alternative symbol names...")
        # futures might use different naming
        sys.exit(1)

    print(f"\nDownloading from: {active_base}")
    print()

    for tf in TIMEFRAMES:
        tf_dir = OUTPUT_DIR / tf
        tf_dir.mkdir(exist_ok=True)
        print(f"--- {tf} ---")

        for d in dates:
            filename = f"{SYMBOL}-{tf}-{d}.zip"
            dest = tf_dir / filename
            url = f"{active_base}/{SYMBOL}/{tf}/{filename}"

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

    print(f"=== Done ===")
    print(f"Downloaded: {downloaded}")
    print(f"Skipped (already exists): {skipped}")
    print(f"Failed/Not available: {failed}")

    # ZIPを展開して1つのCSVに結合するオプション
    print()
    print("Merging ZIPs into single CSV per timeframe...")
    _merge_zips()


def _merge_zips():
    """各TFのZIPを展開して1つのCSVに結合"""
    import zipfile
    import csv

    for tf in TIMEFRAMES:
        tf_dir = OUTPUT_DIR / tf
        zips = sorted(tf_dir.glob("*.zip"))
        if not zips:
            print(f"  {tf}: No ZIPs found")
            continue

        output_csv = OUTPUT_DIR / f"{SYMBOL}-{tf}-merged.csv"
        rows = []

        for zp in zips:
            try:
                with zipfile.ZipFile(zp, "r") as zf:
                    csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
                    for cn in csv_names:
                        with zf.open(cn) as f:
                            text = f.read().decode("utf-8")
                            for line in text.strip().split("\n"):
                                # ヘッダー行をスキップ
                                if line and not line[0].isalpha():
                                    rows.append(line)
            except Exception as e:
                print(f"  Error reading {zp.name}: {e}")

        if rows:
            # ソート（open_time昇順）
            rows.sort(key=lambda r: int(r.split(",")[0]))
            # 重複除去
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
