#!/usr/bin/env python3
"""
BB Squeeze Breakout戦略のローカル動作確認テスト

実行方法:
    python3 test_bb_squeeze.py
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.binance_loader import BinanceCSVLoader
from optimizer.templates import BUILTIN_TEMPLATES
from strategy.builder import ConfigStrategy
from engine.backtest import BacktestEngine
from metrics.calculator import calculate_metrics


def main():
    print("=" * 60)
    print("BB Squeeze Breakout 戦略テスト")
    print("=" * 60)

    # 1. データ読み込み
    print("\n[1/5] データ読み込み中...")
    loader = BinanceCSVLoader()
    data = loader.load(
        "sample_data/binance/AAVEUSDT_1y/15m/AAVEUSDT-15m-2025-02.zip",
        symbol="AAVEUSDT"
    )
    df = data.df
    print(f"データ読み込み完了: {len(df)} bars")

    # 2. テンプレート取得
    print("\n[2/5] テンプレート取得...")
    template_long = BUILTIN_TEMPLATES["bb_squeeze_breakout_long"]
    template_short = BUILTIN_TEMPLATES["bb_squeeze_breakout_short"]

    # 3. 戦略設定生成（パラメータ固定でテスト）
    print("\n[3/5] 戦略設定生成...")

    # ロング戦略（パラメータ1組のみテスト）
    configs_long = template_long.generate_configs()
    print(f"ロング戦略: {len(configs_long)} configs生成")

    # ショート戦略
    configs_short = template_short.generate_configs()
    print(f"ショート戦略: {len(configs_short)} configs生成")

    # 4. バックテスト実行（最初の1つのみ）
    print("\n[4/5] バックテスト実行...")

    # ロング戦略テスト
    print("\n--- bb_squeeze_breakout_long ---")
    config = configs_long[0]
    print(f"Config: {config['name']}")
    print(f"Parameters: {config.get('_params', {})}")

    strategy = ConfigStrategy(config)
    engine = BacktestEngine(strategy)
    result = engine.run(df)

    if result.trades:
        print(f"トレード数: {len(result.trades)}")
        print(f"最初のトレード: entry={result.trades[0].entry_time}, "
              f"exit={result.trades[0].exit_time}, "
              f"pnl={result.trades[0].profit_pct:.2f}%")

        # メトリクス計算
        metrics = calculate_metrics(result.trades, result.portfolio.equity_curve)
        print(f"勝率: {metrics.win_rate:.1f}%")
        print(f"PF: {metrics.profit_factor:.2f}")
        print(f"Total PnL: {metrics.total_profit_pct:.2f}%")
    else:
        print("トレードなし")

    # ショート戦略テスト
    print("\n--- bb_squeeze_breakout_short ---")
    config = configs_short[0]
    print(f"Config: {config['name']}")
    print(f"Parameters: {config.get('_params', {})}")

    strategy = ConfigStrategy(config)
    engine = BacktestEngine(strategy)
    result = engine.run(df)

    if result.trades:
        print(f"トレード数: {len(result.trades)}")
        print(f"最初のトレード: entry={result.trades[0].entry_time}, "
              f"exit={result.trades[0].exit_time}, "
              f"pnl={result.trades[0].profit_pct:.2f}%")

        metrics = calculate_metrics(result.trades, result.portfolio.equity_curve)
        print(f"勝率: {metrics.win_rate:.1f}%")
        print(f"PF: {metrics.profit_factor:.2f}")
        print(f"Total PnL: {metrics.total_profit_pct:.2f}%")
    else:
        print("トレードなし")

    # 5. 完了
    print("\n[5/5] テスト完了")
    print("=" * 60)
    print("✓ BB Squeeze Breakout戦略が正常に動作しました")
    print("=" * 60)


if __name__ == "__main__":
    main()
