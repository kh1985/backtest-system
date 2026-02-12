#!/usr/bin/env python3
"""Donchian戦略の合成データテスト"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from optimizer.templates import BUILTIN_TEMPLATES
from indicators.registry import create_indicator

def create_synthetic_data():
    """Donchianシグナルが発生する合成データを作成"""
    dates = pd.date_range("2024-01-01", periods=500, freq="15min")

    # ベースとなる価格（下降トレンド）
    base_price = 100.0
    prices = []
    for i in range(500):
        # 徐々に下がっていく + ランダムノイズ
        trend = base_price - (i * 0.1)  # 下降トレンド
        noise = np.random.randn() * 1.0
        price = max(trend + noise, 50.0)  # 最低50まで
        prices.append(price)

    prices = np.array(prices)

    # OHLCV作成
    df = pd.DataFrame({
        "timestamp": dates,
        "open": prices,
        "high": prices * 1.01,
        "low": prices * 0.99,
        "close": prices,
        "volume": np.random.rand(500) * 1000 + 1000,
    })
    df.set_index("timestamp", inplace=True)

    return df

def test_donchian_synthetic():
    """合成データでDonchian戦略をテスト"""

    # 1. テンプレート取得
    template = BUILTIN_TEMPLATES.get("donchian_breakdown_short")
    
    if template is None:
        print(f"❌ donchian_breakdown_short テンプレートが見つかりません")
        return

    print(f"✅ テンプレート取得: {template.name}")

    # 2. 合成データ作成
    df = create_synthetic_data()
    print(f"✅ 合成データ作成完了: {len(df)} rows")
    print(f"   価格範囲: {df['close'].min():.2f} - {df['close'].max():.2f}")

    # 3. Donchianインジケーター計算（period=20）
    period = 20
    indicator = create_indicator("donchian", period=period)
    df = indicator.calculate(df)
    print(f"✅ Donchianインジケーター計算完了")

    # 4. シグナル生成チェック
    donchian_col = f"donchian_lower_{period}"
    
    print(f"\n=== {donchian_col}の統計 ===")
    print(df[donchian_col].describe())

    # NaN除外
    valid_df = df[df[donchian_col].notna()]
    print(f"\n有効データ: {len(valid_df)} / {len(df)} rows")

    # エントリーシグナル（close < donchian_lower）
    signal = valid_df["close"] < valid_df[donchian_col]
    signal_count = signal.sum()
    print(f"\nclose < {donchian_col}の発生回数: {signal_count} / {len(valid_df)} ({signal_count/len(valid_df)*100:.2f}%)")

    if signal_count == 0:
        print("\n❌ 合成データでもシグナルが発生していません！")
        print("\n詳細診断:")
        print(f"close の最小値: {valid_df['close'].min():.2f}")
        print(f"{donchian_col} の最小値: {valid_df[donchian_col].min():.2f}")
        
        # rolling計算の問題を検証
        print(f"\n最初の30行:")
        print(valid_df[["close", "low", donchian_col]].head(30))
        
        # 手動で rolling().min() を計算
        manual_min = valid_df["low"].rolling(window=period).min()
        print(f"\n手動計算 vs インジケーター:")
        comparison = pd.DataFrame({
            "indicator": valid_df[donchian_col],
            "manual": manual_min,
            "diff": valid_df[donchian_col] - manual_min,
        })
        print(comparison.head(30))
        
        if (comparison["diff"].abs() > 0.01).any():
            print("\n⚠️ インジケーター計算に問題があります！")
        else:
            print("\n✅ インジケーター計算は正しいです")
            print("   → rolling().min()がルックアヘッドを含んでいる可能性があります")
            
    else:
        print(f"\n✅ シグナルが発生しました！")
        signal_df = valid_df[signal].head(10)
        print(f"\n最初の10件のシグナル:")
        print(signal_df[["close", "low", donchian_col]])

if __name__ == "__main__":
    test_donchian_synthetic()
