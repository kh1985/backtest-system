#!/usr/bin/env python3
"""Donchian戦略のローカルテスト"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data.binance_loader import BinanceCSVLoader
from optimizer.templates import BUILTIN_TEMPLATES
from indicators.registry import create_indicator

def test_donchian():
    """Donchian戦略のインジケーター計算とシグナル生成をテスト"""

    # 1. テンプレート取得
    template = BUILTIN_TEMPLATES.get("donchian_breakdown_short")
    
    if template is None:
        print(f"❌ donchian_breakdown_short テンプレートが見つかりません")
        print(f"利用可能なテンプレート: {list(BUILTIN_TEMPLATES.keys())}")
        return

    print(f"✅ テンプレート取得: {template.name}")
    print(f"   説明: {template.description}")
    print(f"   パラメータ範囲: {[(r.name, r.min_val, r.max_val, r.step) for r in template.param_ranges]}")

    # 2. データロード
    loader = BinanceCSVLoader()
    data_path = Path("inputdata/BTCUSDT-15m.zip")

    if not data_path.exists():
        print(f"❌ データファイルが見つかりません: {data_path}")
        return

    df = loader.load(str(data_path))
    print(f"✅ データロード完了: {len(df)} rows")

    # 2024年分に絞る
    df = df[df.index.year == 2024]
    print(f"✅ 期間絞り込み: {len(df)} rows ({df.index[0]} to {df.index[-1]})")

    # 3. 1つのパラメータセットでテスト（period=20）
    period = 20
    print(f"\n=== テストケース: period={period} ===")

    # 4. インジケーター計算
    indicator_configs = template.config_template["indicators"]
    for ind_conf in indicator_configs:
        ind_type = ind_conf["type"]
        ind_params = {k: v for k, v in ind_conf.items() if k != "type"}

        # パラメータ置換
        for k, v in list(ind_params.items()):
            if isinstance(v, str) and v.startswith("{") and v.endswith("}"):
                param_name = v[1:-1]
                if param_name == "period":
                    ind_params[k] = period

        print(f"\nインジケーター: {ind_type} with {ind_params}")

        indicator = create_indicator(ind_type, **ind_params)
        df = indicator.calculate(df)
        print(f"✅ インジケーター計算完了")
        print(f"   生成カラム: {indicator.columns}")

    # 5. シグナル生成チェック
    donchian_col = f"donchian_lower_{period}"
    if donchian_col not in df.columns:
        print(f"❌ カラムが見つかりません: {donchian_col}")
        print(f"   実際のカラム: {[c for c in df.columns if 'donchian' in c]}")
        return

    print(f"\n=== インジケーター分析 ===")
    print(f"{donchian_col}の統計:")
    print(df[donchian_col].describe())

    # NaNチェック
    nan_count = df[donchian_col].isna().sum()
    print(f"\nNaN数: {nan_count} / {len(df)} ({nan_count/len(df)*100:.2f}%)")

    # 6. エントリーシグナルの発生頻度（NaN除外）
    valid_df = df[df[donchian_col].notna()]
    signal = valid_df["close"] < valid_df[donchian_col]
    signal_count = signal.sum()
    print(f"\nclose < {donchian_col}の発生回数: {signal_count} / {len(valid_df)} ({signal_count/len(valid_df)*100:.2f}%)")

    if signal_count == 0:
        print("\n❌ シグナルが一度も発生していません！")
        print("\n問題の詳細診断:")
        print(f"close の範囲: {valid_df['close'].min():.2f} - {valid_df['close'].max():.2f}")
        print(f"{donchian_col} の範囲: {valid_df[donchian_col].min():.2f} - {valid_df[donchian_col].max():.2f}")
        
        # rolling計算の確認
        print(f"\n最初の30行（インジケーター生成後）:")
        print(valid_df[["close", "low", donchian_col]].head(30))

        # 最小値の比較
        min_close = valid_df['close'].min()
        min_donchian = valid_df[donchian_col].min()
        print(f"\nclose の最小値: {min_close:.2f}")
        print(f"{donchian_col} の最小値: {min_donchian:.2f}")
        
        if min_close >= min_donchian:
            print("⚠️ close の最小値が donchian_lower の最小値以上です！")
            print("   → rolling(window=20).min() がルックアヘッドを含んでいる可能性があります")

    else:
        print(f"\n✅ シグナルが発生しました！")
        signal_df = valid_df[signal].head(10)
        print(f"\n最初の10件のシグナル:")
        print(signal_df[["close", "low", donchian_col]])

if __name__ == "__main__":
    test_donchian()
