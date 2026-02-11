"""
トラリピ 3m足 vs 15m足 比較テスト

3m足でtrapreate戦略を実行し、15m足との性能比較を行う。

使い方:
    python3 scripts/test_trapreate_3m.py
    python3 scripts/test_trapreate_3m.py --symbol ADAUSDT
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.binance_loader import BinanceCSVLoader
from optimizer.templates import BUILTIN_TEMPLATES
from strategy.builder import load_strategy_from_dict
from engine.backtest import BacktestEngine
from metrics.calculator import calculate_metrics


def test_trapreate_with_tf(symbol: str, template_name: str, timeframe: str):
    """指定TFでトラリピテスト"""
    print(f"\n{'='*60}")
    print(f"テンプレート: {template_name}")
    print(f"銘柄: {symbol}")
    print(f"TF: {timeframe}")
    print(f"{'='*60}\n")

    # テンプレート取得
    template = BUILTIN_TEMPLATES.get(template_name)
    if not template:
        print(f"エラー: テンプレート '{template_name}' が見つかりません")
        return None

    # パラメータ設定（中間値）
    params = {
        "trap_interval": 1.0,  # 1.0%
        "profit_width": 0.5,   # 0.5%
    }

    # 設定生成
    configs = template.generate_configs()
    if not configs:
        print("エラー: 設定生成失敗")
        return None

    # 中間パラメータのconfigを探す
    config = None
    for c in configs:
        c_params = {}
        # entry_conditionsからパラメータ抽出
        if "entry_conditions" in c:
            for cond in c["entry_conditions"]:
                if cond.get("type") == "trap_grid":
                    c_params["trap_interval"] = cond.get("trap_interval_pct", 0)
        # exit設定からパラメータ抽出
        if "exit" in c:
            exit_conf = c["exit"]
            if "take_profit_pct" in exit_conf:
                c_params["profit_width"] = exit_conf["take_profit_pct"]

        if all(abs(c_params.get(k, 0) - v) < 0.01 for k, v in params.items()):
            config = c
            break

    if not config:
        print(f"警告: パラメータ {params} に一致する設定が見つかりません。最初の設定を使用します。")
        config = configs[0]

    # データ読み込み
    inputdata_dir = Path(__file__).parent.parent / "inputdata"

    # TFによってディレクトリとファイルパターンを変更
    if timeframe == "3m":
        data_dir = inputdata_dir / "3m"
        period = "20250201-20260130"  # 2025期間（3m足データ）
    else:
        data_dir = inputdata_dir
        period = "20240201-20250131"  # 2024期間（15m足データ）

    data_path = data_dir / f"{symbol}-{timeframe}-{period}-merged.csv"
    if not data_path.exists():
        print(f"エラー: データファイルが見つかりません: {data_path}")
        return None

    loader = BinanceCSVLoader()
    ohlcv = loader.load(str(data_path), symbol=symbol)
    df = ohlcv.df.copy()

    print(f"データ期間: {df['datetime'].min()} ~ {df['datetime'].max()} ({len(df)} bars)")

    # 戦略構築
    strategy = load_strategy_from_dict(config)

    # バックテスト実行
    engine = BacktestEngine(
        strategy=strategy,
        initial_capital=10000.0,
        commission_pct=0.04,
    )

    result = engine.run(df)

    # 結果表示
    print(f"\n--- バックテスト結果 ---")
    print(f"総トレード数: {len(result.trades)}")

    # トレード数が0の場合は早期リターン
    if len(result.trades) == 0:
        print("⚠️ トレード数が0件です。条件が成立していません。")
        return {
            "trades": 0,
            "win_rate": 0,
            "pf": 0,
            "total_pnl": 0,
            "max_dd": 0,
        }

    # メトリクス計算
    metrics = calculate_metrics(
        result.trades,
        result.portfolio.equity_curve
    )
    print(f"勝率: {metrics.win_rate:.1f}%")
    print(f"Profit Factor: {metrics.profit_factor:.2f}")
    print(f"総利益: {metrics.total_profit_pct:.2f}%")
    print(f"最大DD: {metrics.max_drawdown_pct:.2f}%")

    if metrics.total_trades > 0:
        print(f"\n--- トレード詳細 ---")
        print(f"平均保有期間: {metrics.avg_duration_bars:.1f} bars")
        if result.trades:
            print(f"最初のトレード: {result.trades[0].entry_time}")
            print(f"最後のトレード: {result.trades[-1].exit_time}")
            # 決済タイプ分布
            exit_types = {}
            for trade in result.trades:
                exit_type = trade.exit_type or "UNKNOWN"
                exit_types[exit_type] = exit_types.get(exit_type, 0) + 1
            print(f"\n決済タイプ分布:")
            for etype, count in sorted(exit_types.items()):
                print(f"  {etype}: {count} ({count/len(result.trades)*100:.1f}%)")

    return {
        "trades": metrics.total_trades,
        "win_rate": metrics.win_rate,
        "pf": metrics.profit_factor,
        "total_pnl": metrics.total_profit_pct,
        "max_dd": metrics.max_drawdown_pct,
    }


def main():
    parser = argparse.ArgumentParser(description="トラリピ 3m vs 15m 比較テスト")
    parser.add_argument("--symbol", default="ADAUSDT", help="テスト銘柄")
    args = parser.parse_args()

    template_name = "trap_repeat_long"

    # 15m足テスト
    print("\n" + "="*60)
    print("【15m足】トラリピテスト")
    print("="*60)
    result_15m = test_trapreate_with_tf(args.symbol, template_name, "15m")

    # 3m足テスト
    print("\n" + "="*60)
    print("【3m足】トラリピテスト")
    print("="*60)
    result_3m = test_trapreate_with_tf(args.symbol, template_name, "3m")

    # 比較表示
    if result_15m and result_3m:
        print("\n" + "="*60)
        print("【比較結果】3m足 vs 15m足")
        print("="*60)
        print(f"\n{'指標':<20} {'15m足':<15} {'3m足':<15} {'改善率':<10}")
        print("-" * 60)

        metrics = [
            ("トレード数", "trades", ""),
            ("勝率 (%)", "win_rate", "%"),
            ("Profit Factor", "pf", ""),
            ("総利益 (%)", "total_pnl", "%"),
            ("最大DD (%)", "max_dd", "%"),
        ]

        for label, key, unit in metrics:
            val_15m = result_15m[key]
            val_3m = result_3m[key]

            if val_15m > 0:
                if key == "max_dd":
                    # DDは小さい方が良い
                    improvement = ((val_15m - val_3m) / val_15m * 100)
                    improvement_str = f"{improvement:+.1f}%"
                else:
                    improvement = ((val_3m - val_15m) / val_15m * 100)
                    improvement_str = f"{improvement:+.1f}%"
            else:
                improvement_str = "N/A"

            print(f"{label:<20} {val_15m:<15.2f} {val_3m:<15.2f} {improvement_str:<10}")

        # 判定
        print("\n" + "="*60)
        if result_3m["trades"] > result_15m["trades"] * 2:
            print("✅ 3m足でトレード数が大幅増加（2倍以上）")
        else:
            print("⚠️ 3m足でもトレード数の増加が不十分")

        if result_3m["max_dd"] < result_15m["max_dd"]:
            print("✅ 3m足でDD改善")
        else:
            print("⚠️ 3m足でもDD改善なし")

        if result_3m["total_pnl"] > result_15m["total_pnl"]:
            print("✅ 3m足で総利益改善")
        else:
            print("⚠️ 3m足でも総利益改善なし")
        print("="*60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
