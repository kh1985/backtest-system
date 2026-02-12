"""
SuperTrend戦略のローカルテスト

SuperTrendインジケーターとConditionが正しく動作することを確認。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.binance_loader import BinanceCSVLoader
from optimizer.templates import BUILTIN_TEMPLATES
from engine.backtest import BacktestEngine
from indicators.registry import create_indicator
from strategy.builder import load_strategy_from_dict
import pandas as pd


def test_supertrend_indicator():
    """SuperTrendインジケーターのテスト"""
    print("=" * 60)
    print("SuperTrendインジケーターのテスト")
    print("=" * 60)

    # サンプルデータを生成
    data = {
        "datetime": pd.date_range("2024-01-01", periods=100, freq="1h"),
        "open": [100 + i * 0.1 for i in range(100)],
        "high": [101 + i * 0.1 for i in range(100)],
        "low": [99 + i * 0.1 for i in range(100)],
        "close": [100.5 + i * 0.1 for i in range(100)],
        "volume": [1000 for _ in range(100)],
    }
    df = pd.DataFrame(data)

    # SuperTrendインジケーターを計算
    st_indicator = create_indicator("supertrend", period=10, multiplier=3.0)
    df = st_indicator.calculate(df)

    # 結果を確認
    print(f"✅ SuperTrend計算完了")
    print(f"   カラム: {st_indicator.columns}")
    print(f"   最初の非NaN行: {df[st_indicator.columns].notna().all(axis=1).idxmax()}")
    print(f"   サンプル値（最後の5行）:")
    print(df[["close", "supertrend_10_3.0", "supertrend_10_3.0_direction"]].tail())

    return True


def test_supertrend_template():
    """supertrend_shortテンプレートのテスト"""
    print("\n" + "=" * 60)
    print("supertrend_shortテンプレートのテスト")
    print("=" * 60)

    template = BUILTIN_TEMPLATES["supertrend_short"]
    print(f"✅ テンプレート取得成功")
    print(f"   名前: {template.name}")
    print(f"   説明: {template.description}")

    configs = template.generate_configs()
    print(f"✅ 設定生成完了: {len(configs)} configs")

    # 最初の設定を表示
    if configs:
        print(f"\n   サンプル設定 (1/{len(configs)}):")
        config = configs[0]
        print(f"     - period: {config['indicators'][0]['period']}")
        print(f"     - multiplier: {config['indicators'][0]['multiplier']}")

    return True


def test_supertrend_backtest():
    """SuperTrend戦略のバックテストテスト"""
    print("\n" + "=" * 60)
    print("SuperTrend戦略のバックテストテスト")
    print("=" * 60)

    # データを読み込む
    inputdata_dir = Path("/Users/kenjihachiya/Desktop/work/development/backtest-system/inputdata")
    test_file = inputdata_dir / "BTCUSDT-15m-20240101-20240630-merged.csv"

    if not test_file.exists():
        print(f"⚠️  テストデータが見つかりません: {test_file}")
        print(f"   スキップします")
        return True

    loader = BinanceCSVLoader()
    ohlcv = loader.load(str(test_file), symbol="BTCUSDT")
    print(f"✅ データ読み込み完了: {len(ohlcv.df)} 行")

    # テンプレートから設定を生成
    template = BUILTIN_TEMPLATES["supertrend_short"]
    configs = template.generate_configs()
    config = configs[0]  # 最初の設定を使用

    # 戦略を構築
    strategy = load_strategy_from_dict(config)
    print(f"✅ 戦略構築完了")
    print(f"   - period: {config['indicators'][0]['period']}")
    print(f"   - multiplier: {config['indicators'][0]['multiplier']}")

    # バックテストを実行
    engine = BacktestEngine(initial_capital=10000.0, commission_pct=0.04)
    result = engine.run(strategy, ohlcv)

    print(f"✅ バックテスト完了")
    print(f"   トレード数: {len(result.trades)}")
    print(f"   最終資産: ${result.portfolio['equity'].iloc[-1]:.2f}")

    if len(result.trades) > 0:
        print(f"\n   最初の3トレード:")
        for i, trade in enumerate(result.trades[:3], 1):
            print(f"     {i}. エントリー: {trade.entry_time}, "
                  f"決済: {trade.exit_time}, "
                  f"PnL: {trade.pnl_pct:.2f}%")
    else:
        print(f"   ⚠️  トレードが発生しませんでした（条件が厳しい可能性）")

    return True


def main():
    """全テストを実行"""
    print("SuperTrend戦略 ローカルテスト開始")
    print("=" * 60)

    try:
        # 1. インジケーターテスト
        test_supertrend_indicator()

        # 2. テンプレートテスト
        test_supertrend_template()

        # 3. バックテストテスト
        test_supertrend_backtest()

        print("\n" + "=" * 60)
        print("✅ 全テスト完了")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ テスト失敗: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
