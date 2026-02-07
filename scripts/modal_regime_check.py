"""
各銘柄×各期間のDual-TF EMAレジーム分布を計算する簡易スクリプト

使い方:
    modal run scripts/modal_regime_check.py
"""

import modal

vol_data = modal.Volume.from_name("prism-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "pandas>=2.0,<3.0",
        "numpy>=1.24,<2.0",
        "numba>=0.59",
        "pyyaml>=6.0",
    )
    .add_local_dir(".", "/app", copy=True, ignore=[
        "inputdata", "results", "sample_data", "cache",
        ".git", "__pycache__", "*.pyc", ".claude",
    ])
)

app = modal.App("prism-regime-check", image=image)


@app.function(
    cpu=2,
    memory=4096,
    timeout=300,
    volumes={"/data": vol_data},
)
def check_regime(symbol: str, period: str) -> dict:
    """1銘柄×1期間のレジーム分布を計算"""
    import sys
    sys.path.insert(0, "/app")

    from pathlib import Path
    from data.binance_loader import BinanceCSVLoader
    from analysis.trend import TrendDetector

    MA_FAST = 20
    MA_SLOW = 50

    loader = BinanceCSVLoader()
    data_dir = Path("/data")

    # 1h + 4h データ読み込み
    htf_path = data_dir / f"{symbol}-1h-{period}-merged.csv"
    super_htf_path = data_dir / f"{symbol}-4h-{period}-merged.csv"
    exec_path = data_dir / f"{symbol}-15m-{period}-merged.csv"

    if not htf_path.exists() or not super_htf_path.exists() or not exec_path.exists():
        return {"symbol": symbol, "period": period, "status": "no_data"}

    htf_ohlcv = loader.load(str(htf_path), symbol=symbol)
    super_htf_ohlcv = loader.load(str(super_htf_path), symbol=symbol)
    exec_ohlcv = loader.load(str(exec_path), symbol=symbol)

    htf_df = htf_ohlcv.df.copy()
    super_htf_df = super_htf_ohlcv.df.copy()
    exec_df = exec_ohlcv.df.copy()

    detector = TrendDetector()

    # Dual-TF EMA レジーム検出
    htf_df = detector.detect_dual_tf_ema(
        htf_df, super_htf_df, fast_period=MA_FAST, slow_period=MA_SLOW
    )

    # 15m実行TFにラベル伝播
    exec_df = TrendDetector.label_execution_tf(exec_df, htf_df)

    # OOS分割: Train 60% / Val 20% / Test 20%
    total = len(exec_df)
    train_end = int(total * 0.6)
    val_end = int(total * 0.8)

    splits = {
        "all": exec_df,
        "train": exec_df.iloc[:train_end],
        "val": exec_df.iloc[train_end:val_end],
        "test": exec_df.iloc[val_end:],
    }

    result = {
        "symbol": symbol,
        "period": period,
        "status": "ok",
        "total_bars": total,
    }

    for split_name, split_df in splits.items():
        n = len(split_df)
        if n == 0:
            continue
        counts = split_df["trend_regime"].value_counts()
        result[f"{split_name}_up"] = round(counts.get("uptrend", 0) / n * 100, 1)
        result[f"{split_name}_down"] = round(counts.get("downtrend", 0) / n * 100, 1)
        result[f"{split_name}_range"] = round(counts.get("range", 0) / n * 100, 1)

    print(f"  {symbol} {period}: test_down={result.get('test_down', 0)}% all_down={result.get('all_down', 0)}%")
    return result


@app.local_entrypoint()
def main(
    symbols: str = "BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,ADAUSDT,AVAXUSDT,DOGEUSDT,DOTUSDT,TRXUSDT",
    periods: str = "20230201-20240131,20240201-20250131,20250201-20260130",
):
    symbol_list = [s.strip() for s in symbols.split(",")]
    period_list = [p.strip() for p in periods.split(",")]

    print("=" * 70)
    print("  Dual-TF EMA レジーム分布チェック")
    print("=" * 70)

    # 全ジョブ投入
    handles = []
    for period in period_list:
        for symbol in symbol_list:
            h = check_regime.spawn(symbol, period)
            handles.append(h)

    # 結果回収
    results = [h.get() for h in handles]

    # 期間ラベル
    period_labels = {
        "20230201-20240131": "2023",
        "20240201-20250131": "2024",
        "20250201-20260130": "2025",
    }

    # Train/Val/Test 区間ごとの Downtrend% を表示
    print()
    print(f"{'銘柄':<12} | {'期間':<6} | {'Train Down':>11} | {'Val Down':>9} | {'Test Down':>10} | {'All Down':>9}")
    print("-" * 75)

    for r in results:
        if r["status"] != "ok":
            continue
        label = period_labels.get(r["period"], r["period"])
        print(
            f"{r['symbol']:<12} | {label:<6} | "
            f"{r.get('train_down', 0):>10.1f}% | "
            f"{r.get('val_down', 0):>8.1f}% | "
            f"{r.get('test_down', 0):>9.1f}% | "
            f"{r.get('all_down', 0):>8.1f}%"
        )

    # 期間別平均
    print()
    print("--- 期間別 Test区間(後半20%) Downtrend% 平均 ---")
    for period in period_list:
        label = period_labels.get(period, period)
        period_results = [r for r in results if r["period"] == period and r["status"] == "ok"]
        if not period_results:
            continue
        avg_test_down = sum(r.get("test_down", 0) for r in period_results) / len(period_results)
        avg_train_down = sum(r.get("train_down", 0) for r in period_results) / len(period_results)
        avg_all_down = sum(r.get("all_down", 0) for r in period_results) / len(period_results)
        print(f"{label}: Train={avg_train_down:.1f}%  Test={avg_test_down:.1f}%  All={avg_all_down:.1f}%")
