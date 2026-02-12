"""
TSMOM戦略のローカルテスト

TSMOMCondition と tsmom_short テンプレートが正常に動作することを確認。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from optimizer.templates import BUILTIN_TEMPLATES
from strategy.builder import ConfigStrategy
from indicators.registry import create_indicator


def test_tsmom_template():
    """tsmom_short テンプレートの動作確認"""
    print("=" * 60)
    print("TSMOM Template Test")
    print("=" * 60)

    # テンプレート取得
    template = BUILTIN_TEMPLATES["tsmom_short"]
    print(f"\n✓ テンプレート取得成功: {template.name}")
    print(f"  Description: {template.description}")

    # パラメータ組み合わせ生成
    configs = list(template.generate_configs())
    print(f"\n✓ パラメータ生成成功: {len(configs)} configs")

    for i, config in enumerate(configs):
        print(f"\n  Config #{i+1}:")
        print(f"    roc_period: {config['indicators'][0]['period']}")
        cond = config['entry_conditions'][0]
        print(f"    threshold: {cond['threshold']}")

    return True


def test_tsmom_condition():
    """TSMOMCondition の動作確認"""
    print("\n" + "=" * 60)
    print("TSMOM Condition Test")
    print("=" * 60)

    # サンプルデータ作成
    dates = pd.date_range("2024-01-01", periods=100, freq="1h")
    df = pd.DataFrame({
        "timestamp": dates,
        "open": 40000 + (pd.Series(range(100)) * 10),
        "high": 40100 + (pd.Series(range(100)) * 10),
        "low": 39900 + (pd.Series(range(100)) * 10),
        "close": 40000 + (pd.Series(range(100)) * 10),
        "volume": 1000000,
    })

    # ROC指標計算
    roc = create_indicator("roc", period=30)
    df = roc.calculate(df)

    print(f"\n✓ ROC計算成功")
    print(f"  ROC(30) range: {df['roc_30'].min():.2f} to {df['roc_30'].max():.2f}")
    print(f"  ROC(30) mean: {df['roc_30'].mean():.2f}")

    # 戦略構築
    template = BUILTIN_TEMPLATES["tsmom_short"]
    configs = list(template.generate_configs())
    config = configs[0]
    strategy = ConfigStrategy(config)

    # Setup: インジケーター計算 + エントリー条件構築
    df = strategy.setup(df)

    # エントリーシグナル計算
    signals = []
    for i in range(1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i-1]
        signal = strategy.check_entry(row, prev_row)
        if signal:
            signals.append(i)

    print(f"\n✓ シグナル計算成功")
    print(f"  エントリーシグナル数: {len(signals)}")

    if signals:
        print(f"\n  最初の3シグナル:")
        for idx in signals[:3]:
            row = df.iloc[idx]
            print(f"    Index {idx}: ROC={row['roc_30']:.2f}")

    return True


def main():
    """メインテスト"""
    try:
        success = True

        # テンプレートテスト
        if not test_tsmom_template():
            success = False

        # 条件テスト
        if not test_tsmom_condition():
            success = False

        print("\n" + "=" * 60)
        if success:
            print("✅ 全テスト成功")
            print("=" * 60)
            return 0
        else:
            print("❌ テスト失敗")
            print("=" * 60)
            return 1

    except Exception as e:
        print(f"\n❌ エラー発生: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
