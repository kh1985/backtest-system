"""
Modal上でBinanceデータを直接ダウンロードしてVolumeに保存するスクリプト

ローカルにCSVを落とす必要なし。Binance → Modal Volume に直接保存。

使い方:
    # デフォルト: 上位5銘柄 × 15m,1h,4h
    modal run scripts/modal_fetch_data.py

    # 銘柄指定
    modal run scripts/modal_fetch_data.py --symbols BTCUSDT,ETHUSDT

    # TF指定
    modal run scripts/modal_fetch_data.py --timeframes 15m,1h

    # 期間指定（月次: YYYY-MM 形式, 日次: YYYY-MM-DD 形式）
    modal run scripts/modal_fetch_data.py --monthly-start 2025-02 --monthly-end 2025-12 --daily-start 2026-01-01 --daily-end 2026-01-30

    # Volume内の既存データ確認
    modal run scripts/modal_fetch_data.py --list
"""

import modal

vol_data = modal.Volume.from_name("prism-data", create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.11")

app = modal.App("prism-fetch-data", image=image)

DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]
DEFAULT_TIMEFRAMES = ["15m", "1h", "4h"]


@app.function(
    cpu=2,
    memory=2048,
    timeout=600,
    volumes={"/data": vol_data},
)
def fetch_symbol(
    symbol: str,
    timeframes: list[str],
    monthly_start: tuple[int, int],
    monthly_end: tuple[int, int],
    daily_start: str,
    daily_end: str,
) -> dict:
    """1銘柄のデータをBinanceからDLしてVolumeに保存"""
    import io
    import zipfile
    import urllib.request
    import urllib.error
    from datetime import date, timedelta
    from pathlib import Path

    MONTHLY_BASE = "https://data.binance.vision/data/spot/monthly/klines"
    DAILY_BASE = "https://data.binance.vision/data/spot/daily/klines"

    def download_bytes(url: str) -> bytes | None:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code != 404:
                print(f"  HTTP Error {e.code}: {url}")
            return None
        except Exception as e:
            print(f"  Error: {e}")
            return None

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

    def generate_dates(start_str, end_str):
        dates = []
        d = date.fromisoformat(start_str)
        end = date.fromisoformat(end_str)
        while d <= end:
            dates.append(d)
            d += timedelta(days=1)
        return dates

    def extract_rows_from_zip(zip_bytes: bytes) -> list[str]:
        rows = []
        with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
            for name in zf.namelist():
                if name.endswith(".csv"):
                    text = zf.read(name).decode("utf-8")
                    for line in text.strip().split("\n"):
                        if line and not line[0].isalpha():
                            rows.append(line)
        return rows

    months = generate_months(monthly_start, monthly_end)
    dates = generate_dates(daily_start, daily_end)

    # 期間ラベル（ファイル名用）
    period_start = f"{monthly_start[0]}{monthly_start[1]:02d}01"
    period_end = daily_end.replace("-", "")
    period_label = f"{period_start}-{period_end}"

    result = {"symbol": symbol, "files": {}, "stats": {}}

    for tf in timeframes:
        all_rows = []
        downloaded = 0
        failed = 0

        # 月次データ
        for y, m in months:
            filename = f"{symbol}-{tf}-{y}-{m:02d}.zip"
            url = f"{MONTHLY_BASE}/{symbol}/{tf}/{filename}"
            data = download_bytes(url)
            if data:
                all_rows.extend(extract_rows_from_zip(data))
                downloaded += 1
            else:
                failed += 1

        # 日次データ
        for d in dates:
            filename = f"{symbol}-{tf}-{d}.zip"
            url = f"{DAILY_BASE}/{symbol}/{tf}/{filename}"
            data = download_bytes(url)
            if data:
                all_rows.extend(extract_rows_from_zip(data))
                downloaded += 1
            else:
                failed += 1

        # ソート＆重複排除
        if all_rows:
            all_rows.sort(key=lambda r: int(r.split(",")[0]))
            seen = set()
            unique = []
            for r in all_rows:
                key = r.split(",")[0]
                if key not in seen:
                    seen.add(key)
                    unique.append(r)

            # Volumeに保存
            csv_name = f"{symbol}-{tf}-{period_label}-merged.csv"
            csv_path = Path(f"/data/{csv_name}")
            csv_path.write_text("\n".join(unique))
            vol_data.commit()

            result["files"][tf] = {
                "name": csv_name,
                "rows": len(unique),
                "downloaded": downloaded,
                "failed": failed,
            }
            print(f"  {symbol} {tf}: {len(unique)} rows ({downloaded} zips)")
        else:
            result["files"][tf] = {
                "name": None,
                "rows": 0,
                "downloaded": downloaded,
                "failed": failed,
            }
            print(f"  {symbol} {tf}: No data")

    return result


@app.function(
    cpu=1,
    memory=512,
    timeout=120,
    volumes={"/data": vol_data},
)
def list_volume_data() -> list[dict]:
    """Volume内のCSVファイル一覧"""
    from pathlib import Path

    data_dir = Path("/data")
    files = []
    for f in sorted(data_dir.glob("*.csv")):
        size_mb = f.stat().st_size / (1024 * 1024)
        files.append({
            "name": f.name,
            "size_mb": round(size_mb, 2),
        })
    return files


@app.local_entrypoint()
def main(
    symbols: str = "",
    timeframes: str = "",
    monthly_start: str = "2025-02",
    monthly_end: str = "2025-12",
    daily_start: str = "2026-01-01",
    daily_end: str = "2026-01-30",
    list: bool = False,
):
    """Binance → Modal Volume に直接データをダウンロード"""

    # --list: 既存データ確認
    if list:
        files = list_volume_data.remote()
        if not files:
            print("Volume にデータなし")
            return
        print(f"Volume内のCSVファイル ({len(files)} 件):")
        total_mb = 0
        for f in files:
            print(f"  {f['name']}  ({f['size_mb']:.1f} MB)")
            total_mb += f["size_mb"]
        print(f"\n合計: {total_mb:.1f} MB")
        return

    # パラメータ解析
    sym_list = [s.strip() for s in symbols.split(",") if s.strip()] if symbols else DEFAULT_SYMBOLS
    tf_list = [t.strip() for t in timeframes.split(",") if t.strip()] if timeframes else DEFAULT_TIMEFRAMES

    ms_parts = monthly_start.split("-")
    me_parts = monthly_end.split("-")
    ms_tuple = (int(ms_parts[0]), int(ms_parts[1]))
    me_tuple = (int(me_parts[0]), int(me_parts[1]))

    print("=" * 60)
    print("  Prism データ取得 (Binance → Modal Volume)")
    print("=" * 60)
    print(f"銘柄: {sym_list}")
    print(f"TF: {tf_list}")
    print(f"期間: {monthly_start} ~ {daily_end}")
    print()

    # 銘柄ごとに並列実行
    handles = []
    for sym in sym_list:
        h = fetch_symbol.spawn(
            symbol=sym,
            timeframes=tf_list,
            monthly_start=ms_tuple,
            monthly_end=me_tuple,
            daily_start=daily_start,
            daily_end=daily_end,
        )
        handles.append((sym, h))
        print(f"[START] {sym}")

    # 結果回収
    total_files = 0
    for sym, h in handles:
        result = h.get()
        for tf, info in result["files"].items():
            if info["name"]:
                total_files += 1
                print(f"  {sym}/{tf}: {info['rows']} rows")

    print()
    print("=" * 60)
    print(f"  完了: {total_files} ファイルを Volume に保存")
    print(f"  確認: modal run scripts/modal_fetch_data.py --list")
    print("=" * 60)
