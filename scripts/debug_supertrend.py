"""
SuperTrend戦略のデバッグスクリプト

BTCUSDT 2024年 downtrendでの動作を詳細に検証。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.binance_loader import BinanceCSVLoader
from analysis.trend import TrendDetector
from indicators.registry import create_indicator
from strategy.builder import load_strategy_from_dict
from engine.backtest import BacktestEngine
import pandas as pd


def debug_supertrend():
    print("=" * 80)
    print("SuperTrend戦略デバッグ: BTCUSDT 2024年 downtrend")
    print("=" * 80)

    # データ読み込み
    inputdata_dir = Path("/Users/kenjihachiya/Desktop/work/development/backtest-system/inputdata")
    exec_file = inputdata_dir / "BTCUSDT-15m-20240201-20250131-merged.csv"
    htf_file = inputdata_dir / "BTCUSDT-1h-20240201-20250131-merged.csv"
    super_htf_file = inputdata_dir / "BTCUSDT-4h-20240201-20250131-merged.csv"

    if not exec_file.exists():
        print(f"❌ データファイルが見つかりません: {exec_file}")
        return

    loader = BinanceCSVLoader()
    exec_ohlcv = loader.load(str(exec_file), symbol="BTCUSDT")
    htf_ohlcv = loader.load(str(htf_file), symbol="BTCUSDT")
    super_htf_ohlcv = loader.load(str(super_htf_file), symbol="BTCUSDT")

    exec_df = exec_ohlcv.df.copy()
    htf_df = htf_ohlcv.df.copy()
    super_htf_df = super_htf_ohlcv.df.copy()

    print(f"✅ データ読み込み完了")
    print(f"   15m: {len(exec_df)} 行")
    print(f"   1h:  {len(htf_df)} 行")
    print(f"   4h:  {len(super_htf_df)} 行")

    # レジーム検出
    detector = TrendDetector()
    result = detector.detect_dual_tf_ema(
        htf_df, super_htf_df, fast_period=20, slow_period=50
    )
    if isinstance(result, tuple) and len(result) == 3:
        htf_df, super_htf_df, _ = result
    else:
        htf_df, super_htf_df = result, super_htf_df
    exec_df = detector.merge_regime_to_exec(exec_df, htf_df, super_htf_df)

    downtrend_rows = exec_df[exec_df["regime"] == "downtrend"]
    print(f"\n✅ レジーム検出完了")
    print(f"   Downtrend: {len(downtrend_rows)} 行 ({len(downtrend_rows)/len(exec_df)*100:.1f}%)")
    print(f"   Uptrend:   {len(exec_df[exec_df['regime'] == 'uptrend'])} 行")
    print(f"   Range:     {len(exec_df[exec_df['regime'] == 'range'])} 行")

    # SuperTrendインジケーター計算
    period = 7
    multiplier = 2.0
    st_indicator = create_indicator("supertrend", period=period, multiplier=multiplier)
    exec_df = st_indicator.calculate(exec_df)

    st_col = f"supertrend_{period}_{multiplier}"
    print(f"\n✅ SuperTrendインジケーター計算完了")
    print(f"   カラム名: {st_col}")
    print(f"   カラム存在: {st_col in exec_df.columns}")

    if st_col in exec_df.columns:
        st_series = exec_df[st_col]
        print(f"   NaN数: {st_series.isna().sum()} / {len(st_series)} ({st_series.isna().sum()/len(st_series)*100:.1f}%)")
        print(f"   有効値数: {st_series.notna().sum()}")
        print(f"   最小値: {st_series.min():.2f}")
        print(f"   最大値: {st_series.max():.2f}")
        print(f"   平均値: {st_series.mean():.2f}")
    else:
        print(f"   ❌ カラムが見つかりません！")
        print(f"   存在するカラム: {[c for c in exec_df.columns if 'supertrend' in c.lower()]}")
        return

    # Downtrend期間での条件評価
    print(f"\n" + "=" * 80)
    print("Downtrend期間での条件評価")
    print("=" * 80)

    dt_df = exec_df[exec_df["regime"] == "downtrend"].copy()
    dt_df_valid = dt_df[dt_df[st_col].notna()].copy()

    print(f"Downtrend期間: {len(dt_df)} 行")
    print(f"  うち有効（SuperTrend非NaN）: {len(dt_df_valid)} 行 ({len(dt_df_valid)/len(dt_df)*100:.1f}%)")

    if len(dt_df_valid) > 0:
        # close < supertrend を満たす行
        condition_met = dt_df_valid[dt_df_valid["close"] < dt_df_valid[st_col]]
        print(f"\n条件評価: close < supertrend_{period}_{multiplier}")
        print(f"  条件を満たす行: {len(condition_met)} / {len(dt_df_valid)} ({len(condition_met)/len(dt_df_valid)*100:.1f}%)")

        # サンプル値を表示
        print(f"\nサンプル値（最初の10行）:")
        print(dt_df_valid[["datetime", "close", st_col]].head(10).to_string(index=False))

        print(f"\n条件を満たす行（最初の5行）:")
        if len(condition_met) > 0:
            print(condition_met[["datetime", "close", st_col]].head(5).to_string(index=False))
        else:
            print("  ❌ 条件を満たす行が0件です！")

        # 統計
        print(f"\n統計:")
        print(f"  close平均: {dt_df_valid['close'].mean():.2f}")
        print(f"  supertrend平均: {dt_df_valid[st_col].mean():.2f}")
        print(f"  close < supertrend の割合: {len(condition_met)/len(dt_df_valid)*100:.1f}%")

    # テンプレートからバックテスト実行
    print(f"\n" + "=" * 80)
    print("テンプレートバックテスト実行")
    print("=" * 80)

    from optimizer.templates import BUILTIN_TEMPLATES
    template = BUILTIN_TEMPLATES["supertrend_short"]
    configs = template.generate_configs()

    # period=7, multiplier=2.0 の設定を探す
    target_config = None
    for config in configs:
        if config["indicators"][0]["period"] == 7 and config["indicators"][0]["multiplier"] == 2.0:
            target_config = config
            break

    if target_config:
        print(f"✅ テンプレート設定取得成功")
        print(f"   period: {target_config['indicators'][0]['period']}")
        print(f"   multiplier: {target_config['indicators'][0]['multiplier']}")

        strategy = load_strategy_from_dict(target_config)
        engine = BacktestEngine(strategy=strategy, initial_capital=10000.0, commission_pct=0.04)

        # Downtrend期間のみでバックテスト
        result = engine.run(dt_df)

        print(f"\n✅ バックテスト完了")
        print(f"   トレード数: {len(result.trades)}")

        if len(result.trades) > 0:
            print(f"   最初の3トレード:")
            for i, trade in enumerate(result.trades[:3], 1):
                print(f"     {i}. エントリー: {trade.entry_time}, 決済: {trade.exit_time}, PnL: {trade.pnl_pct:.2f}%")
        else:
            print(f"   ❌ トレードが0件です！")

    print(f"\n" + "=" * 80)
    print("デバッグ完了")
    print("=" * 80)


if __name__ == "__main__":
    debug_supertrend()
