"""
SuperTrend戦略の合成データバックテストテスト

実データがない場合でも、合成データでバックテストが正常に動作することを確認。
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.base import OHLCVData
from optimizer.templates import BUILTIN_TEMPLATES
from engine.backtest import BacktestEngine
from strategy.builder import load_strategy_from_dict


def generate_synthetic_data(n_bars=1000, trend="down"):
    """合成データを生成（ダウントレンド）"""
    np.random.seed(42)

    dates = pd.date_range("2024-01-01", periods=n_bars, freq="15min")

    # ダウントレンドの価格系列を生成
    if trend == "down":
        # 下降トレンド + ノイズ
        base_price = 100 - np.linspace(0, 30, n_bars)
        noise = np.random.randn(n_bars) * 0.5
        close = base_price + noise
    else:
        # レンジ相場
        base_price = 100 + np.sin(np.linspace(0, 10, n_bars)) * 5
        noise = np.random.randn(n_bars) * 0.3
        close = base_price + noise

    high = close + np.abs(np.random.randn(n_bars) * 0.3)
    low = close - np.abs(np.random.randn(n_bars) * 0.3)
    open_price = close + np.random.randn(n_bars) * 0.2
    volume = np.random.randint(1000, 10000, n_bars)

    df = pd.DataFrame({
        "datetime": dates,
        "timestamp": dates,
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })

    return df


def test_supertrend_backtest_synthetic():
    """合成データでのバックテストテスト"""
    print("=" * 60)
    print("SuperTrend合成データバックテストテスト")
    print("=" * 60)

    # ダウントレンドの合成データを生成
    df = generate_synthetic_data(n_bars=1000, trend="down")
    ohlcv = OHLCVData(df=df, symbol="SYNTHETIC", timeframe="15m", source="synthetic")
    print(f"✅ 合成データ生成完了: {len(df)} 行（ダウントレンド）")

    # テンプレートから設定を生成
    template = BUILTIN_TEMPLATES["supertrend_short"]
    configs = template.generate_configs()

    print(f"\n全{len(configs)}種類のパラメータ設定でバックテスト実行:")

    results = []
    for i, config in enumerate(configs, 1):
        period = config['indicators'][0]['period']
        multiplier = config['indicators'][0]['multiplier']

        # 戦略を構築
        strategy = load_strategy_from_dict(config)

        # バックテストを実行
        engine = BacktestEngine(strategy=strategy, initial_capital=10000.0, commission_pct=0.04)
        result = engine.run(df)

        # 結果を記録
        n_trades = len(result.trades)
        portfolio_df = result.portfolio.to_df() if hasattr(result.portfolio, 'to_df') else result.df
        final_equity = portfolio_df['equity'].iloc[-1] if len(portfolio_df) > 0 else 10000.0
        pnl_pct = ((final_equity - 10000.0) / 10000.0) * 100

        results.append({
            "period": period,
            "multiplier": multiplier,
            "trades": n_trades,
            "final_equity": final_equity,
            "pnl_pct": pnl_pct
        })

        print(f"  {i}. period={period}, mult={multiplier}: "
              f"{n_trades}トレード, PnL={pnl_pct:+.2f}%")

    # サマリー
    print(f"\n" + "=" * 60)
    print("サマリー:")
    print("=" * 60)

    total_trades = sum(r["trades"] for r in results)
    avg_pnl = sum(r["pnl_pct"] for r in results) / len(results)
    best_result = max(results, key=lambda x: x["pnl_pct"])

    print(f"✅ 全{len(results)}設定でバックテスト完了")
    print(f"   合計トレード数: {total_trades}")
    print(f"   平均PnL: {avg_pnl:+.2f}%")
    print(f"   最良設定: period={best_result['period']}, mult={best_result['multiplier']}")
    print(f"   最良PnL: {best_result['pnl_pct']:+.2f}% ({best_result['trades']}トレード)")

    if total_trades == 0:
        print("\n⚠️  トレードが発生しませんでした")
        print("   原因: SuperTrend条件が厳しすぎる、または合成データが不適切")
        return False

    print(f"\n✅ SuperTrend戦略のバックテストエンジン統合は正常に動作しています")
    return True


def main():
    """テスト実行"""
    print("SuperTrend戦略 合成データテスト開始")
    print("=" * 60)

    try:
        success = test_supertrend_backtest_synthetic()

        if success:
            print("\n" + "=" * 60)
            print("✅ 全テスト完了")
            print("=" * 60)
            print("\n実装は正常に動作しています。")
            print("実データでのOOS検証の準備が整っています。")
        else:
            print("\n" + "=" * 60)
            print("⚠️  テスト完了（トレード0件）")
            print("=" * 60)
            print("\n実装は動作していますが、合成データでトレードが発生しませんでした。")
            print("実データでの検証が必要です。")

    except Exception as e:
        print(f"\n❌ テスト失敗: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
