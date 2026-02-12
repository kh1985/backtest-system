#!/usr/bin/env python3
"""Donchian修正後の詳細検証"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from optimizer.templates import BUILTIN_TEMPLATES
from indicators.registry import create_indicator

def create_test_data():
    """テスト用の価格データ（明確なブレイクダウンを含む）"""
    dates = pd.date_range("2024-01-01", periods=100, freq="15min")
    
    # 前半: 安定（100付近）
    # 後半: 急落（80付近）
    prices = []
    for i in range(100):
        if i < 50:
            # 前半: 95-105の範囲で変動
            base = 100.0
            noise = np.random.randn() * 2.0
        else:
            # 後半: 75-85の範囲で変動（明確なブレイクダウン）
            base = 80.0
            noise = np.random.randn() * 2.0
        
        prices.append(max(base + noise, 50.0))
    
    prices = np.array(prices)
    
    df = pd.DataFrame({
        "timestamp": dates,
        "open": prices,
        "high": prices * 1.01,
        "low": prices * 0.99,
        "close": prices,
        "volume": np.random.rand(100) * 1000 + 1000,
    })
    df.set_index("timestamp", inplace=True)
    
    return df

def test_donchian_detailed():
    """Donchian修正後の詳細検証"""
    
    # 1. テストデータ作成
    df = create_test_data()
    print(f"✅ テストデータ作成完了: {len(df)} rows")
    print(f"   前半50本: close平均={df['close'].iloc[:50].mean():.2f}")
    print(f"   後半50本: close平均={df['close'].iloc[50:].mean():.2f}")
    
    # 2. Donchianインジケーター計算（period=20）
    period = 20
    indicator = create_indicator("donchian", period=period)
    df = indicator.calculate(df)
    
    donchian_col = f"donchian_lower_{period}"
    print(f"\n✅ Donchianインジケーター計算完了")
    
    # 3. ルックアヘッド検証
    print(f"\n=== ルックアヘッド検証 ===")
    
    # 任意の行で、donchian_lowerが「現在足を含まない過去20本の最小値」になっているか確認
    test_idx = 50  # 50本目（ブレイクダウン開始直後）
    
    if test_idx >= period:
        current_low = df["low"].iloc[test_idx]
        donchian_value = df[donchian_col].iloc[test_idx]
        
        # 手動計算: 現在足を除外した過去20本の最小値
        past_20_lows = df["low"].iloc[test_idx - period:test_idx]  # [test_idx-20, test_idx) の範囲
        manual_min = past_20_lows.min()
        
        print(f"Index {test_idx} (ブレイクダウン開始点):")
        print(f"  現在足のlow: {current_low:.2f}")
        print(f"  Donchian lower: {donchian_value:.2f}")
        print(f"  手動計算（過去20本のmin）: {manual_min:.2f}")
        print(f"  一致: {'✅' if abs(donchian_value - manual_min) < 0.01 else '❌'}")
        
        if abs(donchian_value - manual_min) < 0.01:
            print("\n✅ ルックアヘッドバイアスは除去されています")
        else:
            print(f"\n❌ 不一致: 差分={abs(donchian_value - manual_min):.4f}")
    
    # 4. シグナル発生確認
    print(f"\n=== シグナル発生確認 ===")
    
    valid_df = df[df[donchian_col].notna()]
    signal = valid_df["close"] < valid_df[donchian_col]
    signal_count = signal.sum()
    
    print(f"close < {donchian_col}の発生回数: {signal_count} / {len(valid_df)} ({signal_count/len(valid_df)*100:.2f}%)")
    
    if signal_count > 0:
        print(f"\n✅ シグナルが発生しました！")
        
        # ブレイクダウンポイントを確認
        signal_df = valid_df[signal]
        print(f"\nシグナル発生箇所:")
        for idx, row in signal_df.iterrows():
            idx_num = df.index.get_loc(idx)
            print(f"  Index {idx_num}: close={row['close']:.2f} < donchian_lower={row[donchian_col]:.2f} (diff={row[donchian_col] - row['close']:.2f})")
            if idx_num >= 50:  # ブレイクダウン開始後
                print(f"    → ✅ ブレイクダウン検出成功")
    else:
        print(f"\n❌ シグナルが発生していません")
    
    # 5. 期待値との比較
    print(f"\n=== 期待されるシグナル発生タイミング ===")
    print(f"前半50本（高値圏）: donchian_lowerも高い → シグナル少ない（期待）")
    print(f"後半50本（安値圏）: closeがdonchian_lowerを下回る → シグナル発生（期待）")
    
    # 前半・後半のシグナル発生率
    df_first_half = valid_df.iloc[:50]
    df_second_half = valid_df.iloc[50:]
    
    if len(df_first_half) > 0:
        signal_first = (df_first_half["close"] < df_first_half[donchian_col]).sum()
        print(f"\n前半シグナル: {signal_first} / {len(df_first_half)} ({signal_first/len(df_first_half)*100:.1f}%)")
    
    if len(df_second_half) > 0:
        signal_second = (df_second_half["close"] < df_second_half[donchian_col]).sum()
        print(f"後半シグナル: {signal_second} / {len(df_second_half)} ({signal_second/len(df_second_half)*100:.1f}%)")
        
        if signal_second > signal_first:
            print("\n✅ 期待通り、後半（ブレイクダウン）でシグナル増加")
        else:
            print("\n⚠️ 期待と異なる: 後半でシグナルが増えていません")

if __name__ == "__main__":
    test_donchian_detailed()
