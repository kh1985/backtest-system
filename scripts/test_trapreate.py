"""
トラリピテンプレート動作確認テスト

trap_repeat_long/shortテンプレートの基本動作を検証する。

使い方:
    python3 scripts/test_trapreate.py
    python3 scripts/test_trapreate.py --symbol BTCUSDT
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


def test_trap_repeat_template(symbol: str, template_name: str):
    """トラリピテンプレートのテスト"""
    print(f"\n{'='*60}")
    print(f"テンプレート: {template_name}")
    print(f"銘柄: {symbol}")
    print(f"{'='*60}\n")

    # テンプレート取得
    template = BUILTIN_TEMPLATES.get(template_name)
    if not template:
        print(f"エラー: テンプレート '{template_name}' が見つかりません")
        return False

    # パラメータ設定（中間値）
    params = {
        "trap_interval": 1.0,  # 1.0%
        "profit_width": 0.5,   # 0.5%
    }

    # 設定生成
    configs = template.generate_configs()
    if not configs:
        print("エラー: 設定生成失敗")
        return False

    # 中間パラメータのconfigを探す
    config = None
    for c in configs:
        c_params = {
            k: v for k, v in c.items()
            if k in ["trap_interval", "profit_width"]
        }
        # exit設定からパラメータ抽出
        if "exit" in c:
            exit_conf = c["exit"]
            if "take_profit_pct" in exit_conf:
                c_params["profit_width"] = exit_conf["take_profit_pct"]

        # entry_conditionsからパラメータ抽出
        if "entry_conditions" in c:
            for cond in c["entry_conditions"]:
                if cond.get("type") == "trap_grid":
                    c_params["trap_interval"] = cond.get("trap_interval_pct", 0)

        if all(abs(c_params.get(k, 0) - v) < 0.01 for k, v in params.items()):
            config = c
            break

    if not config:
        print(f"警告: パラメータ {params} に一致する設定が見つかりません。最初の設定を使用します。")
        config = configs[0]

    print(f"使用パラメータ: {config.get('entry_conditions', [{}])[0].get('trap_interval_pct', 'N/A')}% interval, "
          f"{config.get('exit', {}).get('take_profit_pct', 'N/A')}% profit_width")

    # データ読み込み
    inputdata_dir = Path(__file__).parent.parent / "inputdata"
    period = "20240201-20250131"  # 2024期間を使用
    exec_tf = "15m"

    data_path = inputdata_dir / f"{symbol}-{exec_tf}-{period}-merged.csv"
    if not data_path.exists():
        print(f"エラー: データファイルが見つかりません: {data_path}")
        return False

    loader = BinanceCSVLoader()
    ohlcv = loader.load(str(data_path), symbol=symbol)
    df = ohlcv.df.copy()

    print(f"データ期間: {df.index[0]} ~ {df.index[-1]} ({len(df)} bars)")

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
        print(f"\n{'='*60}")
        print(f"テスト結果: ❌ FAIL (トレード0件)")
        print(f"{'='*60}\n")
        return False

    # メトリクス計算
    metrics = calculate_metrics(
        result.trades,
        result.portfolio.equity_curve
    )
    print(f"勝率: {metrics.win_rate:.1f}%")
    print(f"Profit Factor: {metrics.profit_factor:.2f}")
    print(f"総利益: {metrics.total_profit_pct:.2f}%")
    print(f"最大DD: {metrics.max_drawdown_pct:.2f}%")
    print(f"Sharpe比: {metrics.sharpe_ratio:.2f}")

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

    # 検証基準
    success = (
        metrics.total_trades >= 30 and
        metrics.win_rate >= 50.0 and
        metrics.total_profit_pct > 0
    )

    print(f"\n{'='*60}")
    print(f"テスト結果: {'✅ PASS' if success else '❌ FAIL'}")
    print(f"{'='*60}\n")

    return success


def main():
    parser = argparse.ArgumentParser(description="トラリピテンプレート動作確認")
    parser.add_argument("--symbol", default="BTCUSDT", help="テスト銘柄")
    args = parser.parse_args()

    templates = ["trap_repeat_long", "trap_repeat_short"]
    results = {}

    for template_name in templates:
        try:
            success = test_trap_repeat_template(args.symbol, template_name)
            results[template_name] = success
        except Exception as e:
            print(f"\nエラー発生: {template_name}")
            print(f"{type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            results[template_name] = False

    # 最終サマリー
    print(f"\n{'='*60}")
    print("最終結果サマリー")
    print(f"{'='*60}")
    for template_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{template_name}: {status}")

    all_pass = all(results.values())
    print(f"\n全体結果: {'✅ 全てPASS' if all_pass else '❌ 一部FAIL'}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
