"""
トラリピ戦略 Range期間のみバックテスト

Dual-TF EMAでレジーム判定し、Range期間のみでバックテストを実行。
DD改善効果を検証する。

使い方:
    python3 scripts/test_trapreate_range.py --symbol BTCUSDT --template trap_repeat_long
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.binance_loader import BinanceCSVLoader
from analysis.trend import TrendDetector
from optimizer.templates import BUILTIN_TEMPLATES
from strategy.builder import load_strategy_from_dict
from engine.backtest import BacktestEngine
from metrics.calculator import calculate_metrics


def run_range_only_backtest(symbol: str, template_name: str, trap_interval: float, profit_width: float):
    """Range期間のみでバックテスト"""

    # データ読み込み
    inputdata_dir = Path("inputdata")
    period = "20240201-20250131"
    exec_tf = "15m"
    htf = "1h"
    super_htf = "4h"

    exec_path = inputdata_dir / f"{symbol}-{exec_tf}-{period}-merged.csv"
    htf_path = inputdata_dir / f"{symbol}-{htf}-{period}-merged.csv"
    super_htf_path = inputdata_dir / f"{symbol}-{super_htf}-{period}-merged.csv"

    if not exec_path.exists():
        print(f"エラー: データファイルが見つかりません: {exec_path}")
        return None

    loader = BinanceCSVLoader()
    exec_ohlcv = loader.load(str(exec_path), symbol=symbol)
    df = exec_ohlcv.df.copy()

    print(f"データ期間: {df.index[0]} ~ {df.index[-1]} ({len(df)} bars)")

    # レジーム検出（Dual-TF EMA）
    detector = TrendDetector()

    # HTFデータ読み込み（必要な場合）
    if htf_path.exists() and super_htf_path.exists():
        htf_ohlcv = loader.load(str(htf_path), symbol=symbol)
        super_htf_ohlcv = loader.load(str(super_htf_path), symbol=symbol)
        htf_df = htf_ohlcv.df.copy()
        super_htf_df = super_htf_ohlcv.df.copy()

        # Dual-TF EMA検出（HTFとSuper HTFのみ使用）
        htf_df_with_regime = detector.detect_dual_tf_ema(
            htf_df=htf_df,
            super_htf_df=super_htf_df
        )

        # exec_dfにレジームをマージ
        df = pd.merge_asof(
            df.sort_index(),
            htf_df_with_regime[["trend_regime"]].sort_index(),
            left_index=True,
            right_index=True,
            direction="backward"
        )
        df = df.rename(columns={"trend_regime": "regime"})
    else:
        # HTFがない場合は単一TF
        df = detector.detect_ma_cross(df, fast=20, slow=50)
        df["regime"] = df["trend"].apply(lambda x: "uptrend" if x == 1 else "downtrend" if x == -1 else "range")

    # レジーム統計
    total_bars = len(df)
    range_bars = len(df[df["regime"] == "range"])
    uptrend_bars = len(df[df["regime"] == "uptrend"])
    downtrend_bars = len(df[df["regime"] == "downtrend"])

    print(f"\nレジーム分布:")
    print(f"  Range: {range_bars} bars ({range_bars/total_bars*100:.1f}%)")
    print(f"  Uptrend: {uptrend_bars} bars ({uptrend_bars/total_bars*100:.1f}%)")
    print(f"  Downtrend: {downtrend_bars} bars ({downtrend_bars/total_bars*100:.1f}%)")

    # テンプレート取得
    template = BUILTIN_TEMPLATES[template_name]
    configs = template.generate_configs()

    # 指定パラメータのconfig探索
    target_config = None
    for config in configs:
        trap_val = None
        if "entry_conditions" in config:
            for cond in config["entry_conditions"]:
                if cond.get("type") == "trap_grid":
                    trap_val = cond.get("trap_interval_pct", 0)
                    break

        profit_val = config.get("exit", {}).get("take_profit_pct", 0)

        if (abs(trap_val - trap_interval) < 0.01 and
            abs(profit_val - profit_width) < 0.01):
            target_config = config
            break

    if not target_config:
        print(f"エラー: パラメータ設定が見つかりません")
        return None

    # 戦略構築
    strategy = load_strategy_from_dict(target_config)

    # バックテスト実行（全期間）
    print("\n--- 全期間バックテスト ---")
    engine_all = BacktestEngine(strategy=strategy, initial_capital=10000.0, commission_pct=0.04)
    result_all = engine_all.run(df)

    if len(result_all.trades) > 0:
        metrics_all = calculate_metrics(result_all.trades, result_all.portfolio.equity_curve)
        print(f"トレード数: {metrics_all.total_trades}")
        print(f"勝率: {metrics_all.win_rate:.1f}%")
        print(f"PF: {metrics_all.profit_factor:.2f}")
        print(f"総利益: {metrics_all.total_profit_pct:.2f}%")
        print(f"最大DD: {metrics_all.max_drawdown_pct:.2f}%")
    else:
        print("トレード数: 0")
        metrics_all = None

    # Range期間のみでバックテスト
    print("\n--- Range期間のみバックテスト ---")
    df_range = df[df["regime"] == "range"].copy()

    if len(df_range) == 0:
        print("エラー: Range期間が存在しません")
        return None

    print(f"Range期間: {len(df_range)} bars ({len(df_range)/len(df)*100:.1f}%)")

    engine_range = BacktestEngine(strategy=strategy, initial_capital=10000.0, commission_pct=0.04)
    result_range = engine_range.run(df_range)

    if len(result_range.trades) == 0:
        print("⚠️ Range期間でトレード数0件")
        return {
            "all": metrics_all,
            "range": None,
            "range_pct": len(df_range) / len(df) * 100
        }

    metrics_range = calculate_metrics(result_range.trades, result_range.portfolio.equity_curve)

    print(f"トレード数: {metrics_range.total_trades}")
    print(f"勝率: {metrics_range.win_rate:.1f}%")
    print(f"PF: {metrics_range.profit_factor:.2f}")
    print(f"総利益: {metrics_range.total_profit_pct:.2f}%")
    print(f"最大DD: {metrics_range.max_drawdown_pct:.2f}%")

    # 決済タイプ分布
    exit_types = {}
    for trade in result_range.trades:
        etype = trade.exit_type or "UNKNOWN"
        exit_types[etype] = exit_types.get(etype, 0) + 1

    print(f"\n決済タイプ分布:")
    for etype, count in sorted(exit_types.items()):
        print(f"  {etype}: {count} ({count/len(result_range.trades)*100:.1f}%)")

    return {
        "all": metrics_all,
        "range": metrics_range,
        "range_pct": len(df_range) / len(df) * 100
    }


def main():
    parser = argparse.ArgumentParser(description="トラリピ戦略 Range期間のみバックテスト")
    parser.add_argument("--symbol", default="BTCUSDT", help="テスト銘柄")
    parser.add_argument("--template", default="trap_repeat_long",
                       choices=["trap_repeat_long", "trap_repeat_short"],
                       help="テンプレート名")
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"トラリピ戦略 Range期間のみバックテスト")
    print(f"{'='*70}")
    print(f"銘柄: {args.symbol}")
    print(f"テンプレート: {args.template}")
    print(f"期間: 2024-02-01 ~ 2025-01-31")
    print(f"パラメータ: trap_interval=0.5%, profit_width=1.0%")
    print(f"{'='*70}\n")

    results = run_range_only_backtest(
        symbol=args.symbol,
        template_name=args.template,
        trap_interval=0.5,
        profit_width=1.0
    )

    if not results:
        return 1

    # 比較サマリー
    print(f"\n{'='*70}")
    print("比較サマリー")
    print(f"{'='*70}\n")

    if results["all"] and results["range"]:
        print(f"{'指標':<15} {'全期間':<20} {'Range期間のみ':<20} {'変化':<15}")
        print(f"{'-'*70}")

        def format_change(before, after):
            if before == 0:
                return "N/A"
            change = after - before
            pct = (after / before - 1) * 100
            return f"{change:+.2f} ({pct:+.1f}%)"

        print(f"{'トレード数':<15} {results['all'].total_trades:<20} {results['range'].total_trades:<20}")
        print(f"{'勝率':<15} {results['all'].win_rate:<20.1f} {results['range'].win_rate:<20.1f} {format_change(results['all'].win_rate, results['range'].win_rate):<15}")
        print(f"{'PF':<15} {results['all'].profit_factor:<20.2f} {results['range'].profit_factor:<20.2f} {format_change(results['all'].profit_factor, results['range'].profit_factor):<15}")
        print(f"{'総利益%':<15} {results['all'].total_profit_pct:<20.2f} {results['range'].total_profit_pct:<20.2f} {format_change(results['all'].total_profit_pct, results['range'].total_profit_pct):<15}")
        print(f"{'最大DD%':<15} {results['all'].max_drawdown_pct:<20.2f} {results['range'].max_drawdown_pct:<20.2f} {format_change(results['all'].max_drawdown_pct, results['range'].max_drawdown_pct):<15}")

        print(f"\nRange期間割合: {results['range_pct']:.1f}%")

        # DD評価
        print(f"\n{'='*70}")
        if results['range'].max_drawdown_pct < 30.0:
            print("✅ DD目標達成！（<30%）")
            print("→ フェーズ2（30銘柄WFA検証）に進むことを推奨")
        else:
            print("❌ DD目標未達成（>=30%）")
            print("→ パラメータ再調整またはトラリピ設計見直しが必要")
        print(f"{'='*70}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
