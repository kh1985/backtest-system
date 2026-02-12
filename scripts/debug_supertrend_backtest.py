"""
SuperTrend戦略のバックテストデバッグ
テンプレートから戦略を構築してバックテストを実行
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.binance_loader import BinanceCSVLoader
from optimizer.templates import BUILTIN_TEMPLATES
from strategy.builder import load_strategy_from_dict
from engine.backtest import BacktestEngine
import pandas as pd


def main():
    print("=" * 80)
    print("SuperTrend戦略バックテストデバッグ")
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

    # テンプレート取得
    template = BUILTIN_TEMPLATES["supertrend_short"]
    configs = template.generate_configs()

    print(f"\n✅ テンプレート取得: {len(configs)} configs")

    # period=7, multiplier=2.0 の設定を使用
    target_config = None
    for config in configs:
        if config["indicators"][0]["period"] == 7 and config["indicators"][0]["multiplier"] == 2.0:
            target_config = config
            break

    if not target_config:
        print("❌ 対象設定が見つかりません")
        return

    print(f"\n設定詳細:")
    print(f"  テンプレート: {target_config['name']}")
    print(f"  side: {target_config['side']}")
    print(f"  indicators:")
    for ind in target_config["indicators"]:
        print(f"    - type: {ind['type']}, period: {ind['period']}, multiplier: {ind['multiplier']}")
    print(f"  entry_conditions:")
    for cond in target_config["entry_conditions"]:
        print(f"    - type: {cond['type']}, period: {cond['period']}, multiplier: {cond['multiplier']}, direction: {cond['direction']}")

    # 戦略構築
    print(f"\n戦略構築中...")
    strategy = load_strategy_from_dict(target_config)

    print(f"✅ 戦略構築完了")
    print(f"  name: {strategy.name}")
    print(f"  side: {strategy.side}")

    # バックテスト実行
    print(f"\nバックテスト実行中...")
    engine = BacktestEngine(strategy=strategy, initial_capital=10000.0, commission_pct=0.04)
    result = engine.run(df)

    print(f"\n✅ バックテスト完了")
    print(f"  トレード数: {len(result.trades)}")

    if len(result.trades) > 0:
        print(f"\n最初の5トレード:")
        for i, trade in enumerate(result.trades[:5], 1):
            print(f"  {i}. エントリー: {trade.entry_time}, 決済: {trade.exit_time}")
            print(f"     価格: {trade.entry_price:.2f} → {trade.exit_price:.2f}, PnL: {trade.pnl_pct:.2f}%")
    else:
        print(f"\n❌ トレードが0件です！")

        # デバッグ情報
        print(f"\nデバッグ情報:")
        print(f"  result.df に含まれるSuperTrendカラム:")
        st_cols = [c for c in result.df.columns if 'supertrend' in c.lower()]
        print(f"    {st_cols}")

        if st_cols:
            # SuperTrendカラムの統計
            for col in st_cols:
                series = result.df[col]
                print(f"\n  {col}:")
                print(f"    NaN数: {series.isna().sum()}")
                print(f"    有効値数: {series.notna().sum()}")
                if series.notna().sum() > 0:
                    print(f"    最小値: {series.min()}")
                    print(f"    最大値: {series.max()}")

            # close < supertrend の行数
            if "supertrend_7_2.0" in result.df.columns:
                st_col = "supertrend_7_2.0"
                valid_df = result.df[result.df[st_col].notna()]
                condition_met = valid_df[valid_df["close"] < valid_df[st_col]]
                print(f"\n  close < {st_col} を満たす行:")
                print(f"    {len(condition_met)} / {len(valid_df)} ({len(condition_met)/len(valid_df)*100:.1f}%)")


if __name__ == "__main__":
    main()
