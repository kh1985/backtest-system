"""
SuperTrend戦略の簡易デバッグスクリプト
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.binance_loader import BinanceCSVLoader
from indicators.registry import create_indicator
import pandas as pd


def main():
    print("=" * 80)
    print("SuperTrend インジケーター計算テスト")
    print("=" * 80)

    # データ読み込み
    inputdata_dir = Path("/Users/kenjihachiya/Desktop/work/development/backtest-system/inputdata")
    exec_file = inputdata_dir / "BTCUSDT-15m-20240201-20250131-merged.csv"

    if not exec_file.exists():
        print(f"❌ データファイルが見つかりません: {exec_file}")
        return

    loader = BinanceCSVLoader()
    ohlcv = loader.load(str(exec_file), symbol="BTCUSDT")
    df = ohlcv.df.copy()

    print(f"✅ データ読み込み完了: {len(df)} 行")

    # SuperTrendインジケーター計算
    period = 7
    multiplier = 2.0

    print(f"\nSuperTrend計算開始: period={period}, multiplier={multiplier}")
    st_indicator = create_indicator("supertrend", period=period, multiplier=multiplier)
    df = st_indicator.calculate(df)

    st_col = f"supertrend_{period}_{multiplier}"
    st_dir_col = f"supertrend_{period}_{multiplier}_direction"

    print(f"\n✅ SuperTrend計算完了")
    print(f"   カラム名: {st_col}")
    print(f"   方向カラム: {st_dir_col}")
    print(f"   カラム存在: {st_col in df.columns}")

    if st_col not in df.columns:
        print(f"\n❌ カラムが見つかりません！")
        print(f"   生成されたカラム: {[c for c in df.columns if 'supertrend' in c.lower()]}")
        return

    # 統計情報
    st_series = df[st_col]
    close_series = df["close"]

    print(f"\nSuperTrend統計:")
    print(f"   全行数: {len(df)}")
    print(f"   NaN数: {st_series.isna().sum()} ({st_series.isna().sum()/len(df)*100:.1f}%)")
    print(f"   有効値数: {st_series.notna().sum()}")
    print(f"   最小値: {st_series.min():.2f}")
    print(f"   最大値: {st_series.max():.2f}")
    print(f"   平均値: {st_series.mean():.2f}")

    # close vs supertrend の比較
    valid_df = df[st_series.notna()].copy()
    condition_met = valid_df[valid_df["close"] < valid_df[st_col]]

    print(f"\n条件評価: close < supertrend")
    print(f"   有効行数: {len(valid_df)}")
    print(f"   条件を満たす行: {len(condition_met)} ({len(condition_met)/len(valid_df)*100:.1f}%)")

    # サンプル値
    print(f"\nサンプル値（最初の20行）:")
    sample = valid_df[["datetime", "close", st_col, st_dir_col]].head(20)
    print(sample.to_string(index=False))

    # 条件を満たす行のサンプル
    if len(condition_met) > 0:
        print(f"\n条件を満たす行のサンプル（最初の10行）:")
        sample_met = condition_met[["datetime", "close", st_col, st_dir_col]].head(10)
        print(sample_met.to_string(index=False))

        # 統計
        print(f"\n統計:")
        print(f"   全期間でのclose平均: {close_series.mean():.2f}")
        print(f"   全期間でのSuperTrend平均: {st_series.mean():.2f}")
        print(f"   条件を満たす期間でのclose平均: {condition_met['close'].mean():.2f}")
        print(f"   条件を満たす期間でのSuperTrend平均: {condition_met[st_col].mean():.2f}")
    else:
        print(f"\n❌ 条件を満たす行が0件です！")
        print(f"   これは問題です。SuperTrendがずっとcloseより下にある状態です。")

        # 診断
        print(f"\n診断:")
        print(f"   close > supertrend の行数: {len(valid_df[valid_df['close'] > valid_df[st_col]])}")
        print(f"   close == supertrend の行数: {len(valid_df[valid_df['close'] == valid_df[st_col]])}")

        # direction の分布
        if st_dir_col in df.columns:
            dir_series = df[st_dir_col]
            print(f"\n   Direction分布:")
            print(f"     direction=1 (uptrend): {(dir_series == 1).sum()}")
            print(f"     direction=-1 (downtrend): {(dir_series == -1).sum()}")


if __name__ == "__main__":
    main()
