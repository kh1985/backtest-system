"""
トラリピ戦略パラメータ感度テスト

trap_interval × profit_widthの全組み合わせでバックテストを実行し、
最適パラメータ範囲を特定する。

使い方:
    python3 scripts/trapreate_sensitivity.py --symbol BTCUSDT --template trap_repeat_long
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.binance_loader import BinanceCSVLoader
from optimizer.templates import BUILTIN_TEMPLATES
from strategy.builder import load_strategy_from_dict
from engine.backtest import BacktestEngine
from metrics.calculator import calculate_metrics


# パラメータ範囲
TRAP_INTERVALS = [0.5, 1.0, 1.5, 2.0]
PROFIT_WIDTHS = [0.3, 0.5, 0.7, 1.0]


def run_backtest(symbol: str, template_name: str, trap_interval: float, profit_width: float) -> Dict:
    """単一パラメータ組み合わせのバックテスト"""
    # テンプレート取得
    template = BUILTIN_TEMPLATES[template_name]

    # パラメータ設定でconfig生成
    configs = template.generate_configs()

    # 指定パラメータに近いconfigを探す
    target_config = None
    for config in configs:
        # entry_conditionsからtrap_intervalを取得
        trap_val = None
        if "entry_conditions" in config:
            for cond in config["entry_conditions"]:
                if cond.get("type") == "trap_grid":
                    trap_val = cond.get("trap_interval_pct", 0)
                    break

        # exitからprofit_widthを取得
        profit_val = config.get("exit", {}).get("take_profit_pct", 0)

        if (abs(trap_val - trap_interval) < 0.01 and
            abs(profit_val - profit_width) < 0.01):
            target_config = config
            break

    if not target_config:
        return None

    # データ読み込み
    inputdata_dir = Path("inputdata")
    period = "20240201-20250131"
    exec_tf = "15m"
    data_path = inputdata_dir / f"{symbol}-{exec_tf}-{period}-merged.csv"

    if not data_path.exists():
        return None

    loader = BinanceCSVLoader()
    ohlcv = loader.load(str(data_path), symbol=symbol)
    df = ohlcv.df.copy()

    # バックテスト実行
    strategy = load_strategy_from_dict(target_config)
    engine = BacktestEngine(strategy=strategy, initial_capital=10000.0, commission_pct=0.04)
    result = engine.run(df)

    # メトリクス計算
    if len(result.trades) == 0:
        return {
            "trap_interval": trap_interval,
            "profit_width": profit_width,
            "total_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "total_pnl_pct": 0.0,
            "max_dd_pct": 0.0,
            "sharpe_ratio": 0.0,
        }

    metrics = calculate_metrics(result.trades, result.portfolio.equity_curve)

    # 決済タイプ分布
    exit_types = {}
    for trade in result.trades:
        etype = trade.exit_type or "UNKNOWN"
        exit_types[etype] = exit_types.get(etype, 0) + 1

    return {
        "trap_interval": trap_interval,
        "profit_width": profit_width,
        "total_trades": metrics.total_trades,
        "win_rate": metrics.win_rate,
        "profit_factor": metrics.profit_factor,
        "total_pnl_pct": metrics.total_profit_pct,
        "max_dd_pct": metrics.max_drawdown_pct,
        "sharpe_ratio": metrics.sharpe_ratio,
        "exit_types": exit_types,
    }


def main():
    parser = argparse.ArgumentParser(description="トラリピ戦略パラメータ感度テスト")
    parser.add_argument("--symbol", default="BTCUSDT", help="テスト銘柄")
    parser.add_argument("--template", default="trap_repeat_long",
                       choices=["trap_repeat_long", "trap_repeat_short"],
                       help="テンプレート名")
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"トラリピ戦略パラメータ感度テスト")
    print(f"{'='*70}")
    print(f"銘柄: {args.symbol}")
    print(f"テンプレート: {args.template}")
    print(f"期間: 2024-02-01 ~ 2025-01-31")
    print(f"{'='*70}\n")

    results = []
    total_tests = len(TRAP_INTERVALS) * len(PROFIT_WIDTHS)
    test_count = 0

    for trap_interval in TRAP_INTERVALS:
        for profit_width in PROFIT_WIDTHS:
            test_count += 1
            print(f"[{test_count}/{total_tests}] trap_interval={trap_interval}%, profit_width={profit_width}% ... ", end="", flush=True)

            result = run_backtest(args.symbol, args.template, trap_interval, profit_width)

            if result:
                results.append(result)
                print(f"✓ (Trades: {result['total_trades']}, WR: {result['win_rate']:.1f}%, PnL: {result['total_pnl_pct']:.2f}%)")
            else:
                print("✗ (Config not found)")

    # 結果サマリー
    print(f"\n{'='*70}")
    print("感度分析結果サマリー")
    print(f"{'='*70}\n")

    # Profit Factor > 1.0 の組み合わせ
    profitable = [r for r in results if r["profit_factor"] > 1.0]
    print(f"【Profit Factor > 1.0】: {len(profitable)}/{len(results)}組み合わせ")
    if profitable:
        for r in sorted(profitable, key=lambda x: x["profit_factor"], reverse=True):
            print(f"  trap={r['trap_interval']}%, profit={r['profit_width']}%: "
                  f"PF={r['profit_factor']:.2f}, WR={r['win_rate']:.1f}%, "
                  f"PnL={r['total_pnl_pct']:.2f}%, DD={r['max_dd_pct']:.2f}%")
    else:
        print("  該当なし")

    print()

    # 勝率 > 70% の組み合わせ
    high_wr = [r for r in results if r["win_rate"] > 70.0]
    print(f"【勝率 > 70%】: {len(high_wr)}/{len(results)}組み合わせ")
    if high_wr:
        for r in sorted(high_wr, key=lambda x: x["win_rate"], reverse=True)[:5]:
            print(f"  trap={r['trap_interval']}%, profit={r['profit_width']}%: "
                  f"WR={r['win_rate']:.1f}%, PF={r['profit_factor']:.2f}, "
                  f"PnL={r['total_pnl_pct']:.2f}%")
    else:
        print("  該当なし")

    print()

    # 最大DD < 20% の組み合わせ
    low_dd = [r for r in results if r["max_dd_pct"] < 20.0]
    print(f"【最大DD < 20%】: {len(low_dd)}/{len(results)}組み合わせ")
    if low_dd:
        for r in sorted(low_dd, key=lambda x: x["max_dd_pct"])[:5]:
            print(f"  trap={r['trap_interval']}%, profit={r['profit_width']}%: "
                  f"DD={r['max_dd_pct']:.2f}%, PF={r['profit_factor']:.2f}, "
                  f"PnL={r['total_pnl_pct']:.2f}%")
    else:
        print("  該当なし")

    print()

    # 総合評価（3指標すべて満たす）
    best = [r for r in results if
            r["profit_factor"] > 1.0 and
            r["win_rate"] > 70.0 and
            r["max_dd_pct"] < 20.0]
    print(f"【総合評価（PF>1.0 & WR>70% & DD<20%）】: {len(best)}/{len(results)}組み合わせ")
    if best:
        print("\n✅ 推奨パラメータ:")
        for r in sorted(best, key=lambda x: x["profit_factor"], reverse=True):
            print(f"  trap_interval={r['trap_interval']}%, profit_width={r['profit_width']}%")
            print(f"    - PF: {r['profit_factor']:.2f}")
            print(f"    - WR: {r['win_rate']:.1f}%")
            print(f"    - PnL: {r['total_pnl_pct']:.2f}%")
            print(f"    - DD: {r['max_dd_pct']:.2f}%")
            print(f"    - Sharpe: {r['sharpe_ratio']:.2f}")
    else:
        print("  ❌ 全条件を満たす組み合わせなし")
        print("\n  代替推奨（PF最大）:")
        if results:
            best_pf = max(results, key=lambda x: x["profit_factor"])
            print(f"    trap_interval={best_pf['trap_interval']}%, profit_width={best_pf['profit_width']}%")
            print(f"    - PF: {best_pf['profit_factor']:.2f}")
            print(f"    - WR: {best_pf['win_rate']:.1f}%")
            print(f"    - PnL: {best_pf['total_pnl_pct']:.2f}%")
            print(f"    - DD: {best_pf['max_dd_pct']:.2f}%")

    print(f"\n{'='*70}\n")

    # JSON保存
    output_file = Path(f"results/sensitivity_{args.template}_{args.symbol}.json")
    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, "w") as f:
        json.dump({
            "symbol": args.symbol,
            "template": args.template,
            "results": results
        }, f, indent=2)

    print(f"結果保存: {output_file}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
