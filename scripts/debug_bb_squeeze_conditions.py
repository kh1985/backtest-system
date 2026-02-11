"""BB Squeeze Breakout戦略の各条件の動作を詳細検証"""
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from data.binance_loader import BinanceCSVLoader
from strategy.builder import ConfigStrategy
from analysis.trend import TrendDetector
from indicators.registry import create_indicator


def debug_conditions():
    """各条件の通過率を詳細に検証"""

    # データ読み込み
    symbol = "AAVEUSDT"
    execution_tf = "15m"

    print(f"\n=== {symbol} BB Squeeze Conditions Debug ===\n")

    # 15m足データ読み込み
    exec_path = project_root / f"inputdata/{symbol}-{execution_tf}-20250201-20260130-merged.csv"
    loader = BinanceCSVLoader()
    exec_data = loader.load(str(exec_path))
    exec_df = exec_data.df.copy()

    # HTFデータ読み込み
    htf = "1h"
    super_htf = "4h"
    htf_path = project_root / f"inputdata/{symbol}-{htf}-20250201-20260130-merged.csv"
    super_htf_path = project_root / f"inputdata/{symbol}-{super_htf}-20250201-20260130-merged.csv"

    htf_data = loader.load(str(htf_path))
    super_htf_data = loader.load(str(super_htf_path))
    htf_df = htf_data.df
    super_htf_df = super_htf_data.df

    print(f"✓ Loaded data: {len(exec_df)} rows")

    # レジーム検出
    detector = TrendDetector()
    htf_with_regime = detector.detect_dual_tf_ema(htf_df, super_htf_df)

    exec_df['datetime'] = pd.to_datetime(exec_df['datetime'])
    htf_with_regime['datetime'] = pd.to_datetime(htf_with_regime['datetime'])

    exec_df_with_regime = pd.merge_asof(
        exec_df.sort_values('datetime'),
        htf_with_regime[['datetime', 'trend_regime']].sort_values('datetime'),
        on='datetime',
        direction='backward'
    )
    exec_df_with_regime.rename(columns={'trend_regime': 'regime'}, inplace=True)

    # Uptrend期間のみ抽出
    uptrend_df = exec_df_with_regime[exec_df_with_regime['regime'] == 'uptrend'].copy()
    print(f"✓ Uptrend periods: {len(uptrend_df)} rows ({len(uptrend_df)/len(exec_df_with_regime)*100:.1f}%)\n")

    if len(uptrend_df) == 0:
        print("⚠️ No uptrend periods found!")
        return

    # インジケーター計算
    df_with_indicators = uptrend_df.copy()

    # Bollinger Bands
    bb_indicator = create_indicator("bollinger", period=20, std_dev=2.0)
    df_with_indicators = bb_indicator.calculate(df_with_indicators)

    # ADX
    adx_indicator = create_indicator("adx", period=14)
    df_with_indicators = adx_indicator.calculate(df_with_indicators)

    # EMA 5
    ema5_indicator = create_indicator("ema", period=5)
    df_with_indicators = ema5_indicator.calculate(df_with_indicators)

    # EMA 13
    ema13_indicator = create_indicator("ema", period=13)
    df_with_indicators = ema13_indicator.calculate(df_with_indicators)

    # Volume SMA
    volume_sma_indicator = create_indicator("volume_sma", period=20)
    df_with_indicators = volume_sma_indicator.calculate(df_with_indicators)

    print(f"✓ Indicators calculated\n")

    # 必要なカラム確認
    required_cols = ['close', 'volume', 'bb_upper_20', 'bb_lower_20', 'bb_middle_20',
                     'adx_14', 'ema_5', 'ema_13', 'volume_sma_20']

    print(f"=== Column Availability ===")
    for col in required_cols:
        if col in df_with_indicators.columns:
            non_null = df_with_indicators[col].notna().sum()
            print(f"  {col}: {non_null}/{len(df_with_indicators)} valid ({non_null/len(df_with_indicators)*100:.1f}%)")
        else:
            print(f"  {col}: MISSING!")

    # 全カラムが揃っている行のみ
    valid_df = df_with_indicators.dropna(subset=required_cols).copy()
    print(f"\n✓ Valid rows (all columns): {len(valid_df)}/{len(df_with_indicators)} ({len(valid_df)/len(df_with_indicators)*100:.1f}%)\n")

    if len(valid_df) == 0:
        print("⚠️ No valid rows after filtering!")
        return

    # === 各条件の評価 ===
    print(f"=== Condition Evaluation (squeeze_threshold=0.05, adx=20, volume_mult=1.5) ===\n")

    # 1. BB Squeeze条件
    valid_df['bandwidth'] = (valid_df['bb_upper_20'] - valid_df['bb_lower_20']) / valid_df['bb_middle_20']
    cond1 = valid_df['bandwidth'] < 0.05
    print(f"1. BB Squeeze (bandwidth < 0.05):")
    print(f"   Pass: {cond1.sum()}/{len(valid_df)} ({cond1.sum()/len(valid_df)*100:.2f}%)")
    print(f"   Bandwidth stats: min={valid_df['bandwidth'].min():.4f}, "
          f"median={valid_df['bandwidth'].median():.4f}, max={valid_df['bandwidth'].max():.4f}")
    print(f"   Values < 0.05: {(valid_df['bandwidth'] < 0.05).sum()}")
    print(f"   Values < 0.10: {(valid_df['bandwidth'] < 0.10).sum()}")
    print(f"   Values < 0.15: {(valid_df['bandwidth'] < 0.15).sum()}\n")

    # 2. Close > BB Upper条件
    cond2 = valid_df['close'] > valid_df['bb_upper_20']
    print(f"2. Close > BB Upper:")
    print(f"   Pass: {cond2.sum()}/{len(valid_df)} ({cond2.sum()/len(valid_df)*100:.2f}%)\n")

    # 3. ADX >= 20条件
    cond3 = valid_df['adx_14'] >= 20
    print(f"3. ADX >= 20:")
    print(f"   Pass: {cond3.sum()}/{len(valid_df)} ({cond3.sum()/len(valid_df)*100:.2f}%)")
    print(f"   ADX stats: min={valid_df['adx_14'].min():.2f}, "
          f"median={valid_df['adx_14'].median():.2f}, max={valid_df['adx_14'].max():.2f}\n")

    # 4. Volume条件
    valid_df['volume_ratio'] = valid_df['volume'] / valid_df['volume_sma_20']
    cond4 = valid_df['volume_ratio'] >= 1.5
    print(f"4. Volume >= avg * 1.5:")
    print(f"   Pass: {cond4.sum()}/{len(valid_df)} ({cond4.sum()/len(valid_df)*100:.2f}%)")
    print(f"   Volume ratio stats: min={valid_df['volume_ratio'].min():.2f}, "
          f"median={valid_df['volume_ratio'].median():.2f}, max={valid_df['volume_ratio'].max():.2f}\n")

    # 5. EMA State条件
    cond5 = valid_df['ema_5'] > valid_df['ema_13']
    print(f"5. EMA(5) > EMA(13):")
    print(f"   Pass: {cond5.sum()}/{len(valid_df)} ({cond5.sum()/len(valid_df)*100:.2f}%)\n")

    # === 複合条件 ===
    print(f"=== Compound Conditions ===\n")

    all_conds = cond1 & cond2 & cond3 & cond4 & cond5
    print(f"ALL 5 conditions (AND):")
    print(f"   Pass: {all_conds.sum()}/{len(valid_df)} ({all_conds.sum()/len(valid_df)*100:.2f}%)\n")

    # 段階的な絞り込み
    print(f"=== Stepwise Filtering ===\n")
    step1 = cond1
    print(f"After condition 1 (BB Squeeze): {step1.sum()} rows")

    step2 = step1 & cond2
    print(f"After condition 2 (+ Close > BB Upper): {step2.sum()} rows")

    step3 = step2 & cond3
    print(f"After condition 3 (+ ADX >= 20): {step3.sum()} rows")

    step4 = step3 & cond4
    print(f"After condition 4 (+ Volume >= 1.5x): {step4.sum()} rows")

    step5 = step4 & cond5
    print(f"After condition 5 (+ EMA above): {step5.sum()} rows")

    # サンプル表示
    if all_conds.sum() > 0:
        print(f"\n=== Sample Entry Signals (first 5) ===\n")
        sample = valid_df[all_conds].head(5)
        for idx, row in sample.iterrows():
            print(f"Datetime: {row['datetime']}")
            print(f"  close: {row['close']:.2f}, bb_upper: {row['bb_upper_20']:.2f}")
            print(f"  bandwidth: {row['bandwidth']:.4f}")
            print(f"  adx: {row['adx_14']:.2f}")
            print(f"  volume_ratio: {row['volume_ratio']:.2f}")
            print(f"  ema_5: {row['ema_5']:.2f}, ema_13: {row['ema_13']:.2f}\n")
    else:
        print(f"\n⚠️ No entry signals found with ALL conditions!")
        print(f"\nTrying relaxed thresholds:")
        print(f"  - squeeze_threshold=0.10: {(cond1_relaxed := valid_df['bandwidth'] < 0.10).sum()} rows")
        print(f"  - squeeze_threshold=0.15: {(cond1_relaxed2 := valid_df['bandwidth'] < 0.15).sum()} rows")


if __name__ == "__main__":
    debug_conditions()
