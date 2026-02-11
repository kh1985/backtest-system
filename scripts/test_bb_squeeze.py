"""BB Squeeze Breakout戦略のローカル再現テスト"""
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from data.binance_loader import BinanceCSVLoader
from strategy.builder import ConfigStrategy
from engine.backtest import BacktestEngine
from analysis.trend import TrendDetector


def test_bb_squeeze_breakout():
    """AAVEUSDTの2025期間uptrendでBB Squeeze Breakout Long戦略をテスト"""

    # データ読み込み
    symbol = "AAVEUSDT"
    execution_tf = "15m"

    print(f"\n=== {symbol} BB Squeeze Breakout Local Test ===\n")

    # 15m足データ読み込み（execution TF）
    exec_path = project_root / f"inputdata/{symbol}-{execution_tf}-20250201-20260130-merged.csv"
    loader = BinanceCSVLoader()
    exec_data = loader.load(str(exec_path))
    exec_df = exec_data.df
    print(f"✓ Loaded {execution_tf} data: {len(exec_df)} rows")

    # HTFデータ読み込み（レジーム検出用）
    htf = "1h"
    super_htf = "4h"
    htf_path = project_root / f"inputdata/{symbol}-{htf}-20250201-20260130-merged.csv"
    super_htf_path = project_root / f"inputdata/{symbol}-{super_htf}-20250201-20260130-merged.csv"

    htf_data = loader.load(str(htf_path))
    super_htf_data = loader.load(str(super_htf_path))
    htf_df = htf_data.df
    super_htf_df = super_htf_data.df
    print(f"✓ Loaded {htf} data: {len(htf_df)} rows")
    print(f"✓ Loaded {super_htf} data: {len(super_htf_df)} rows")

    # レジーム検出（Dual-TF EMA）
    detector = TrendDetector()
    htf_with_regime = detector.detect_dual_tf_ema(htf_df, super_htf_df)

    # HTFのレジームをexecution TFにマージ
    exec_df['datetime'] = pd.to_datetime(exec_df['datetime'])
    htf_with_regime['datetime'] = pd.to_datetime(htf_with_regime['datetime'])

    exec_df_with_regime = pd.merge_asof(
        exec_df.sort_values('datetime'),
        htf_with_regime[['datetime', 'trend_regime']].sort_values('datetime'),
        on='datetime',
        direction='backward'
    )
    exec_df_with_regime.rename(columns={'trend_regime': 'regime'}, inplace=True)
    print(f"✓ Regime detection completed")

    # Uptrend期間のみ抽出
    uptrend_df = exec_df_with_regime[exec_df_with_regime['regime'] == 'uptrend'].copy()
    print(f"✓ Uptrend periods: {len(uptrend_df)} rows ({len(uptrend_df)/len(exec_df_with_regime)*100:.1f}%)")

    if len(uptrend_df) == 0:
        print("⚠️ No uptrend periods found!")
        return

    # 戦略設定（squeeze_threshold=0.05をテスト）
    strategy_config = {
        "name": "bb_squeeze_breakout_long_test",
        "side": "long",
        "indicators": [
            {"type": "bollinger", "period": 20, "std_dev": 2.0},
            {"type": "adx", "period": 14},
            {"type": "ema", "period": 5},
            {"type": "ema", "period": 13},
            {"type": "volume_sma", "period": 20},
        ],
        "entry_conditions": [
            {
                "type": "bb_squeeze",
                "threshold": 0.05,  # Modal結果と同じパラメータ
                "bb_period": 20,
            },
            {
                "type": "column_compare",
                "column_a": "close",
                "operator": ">",
                "column_b": "bb_upper_20",
            },
            {
                "type": "threshold",
                "column": "adx_14",
                "operator": ">=",
                "value": 20,
            },
            {
                "type": "volume",
                "volume_mult": 1.5,
                "volume_period": 20,
            },
            {
                "type": "ema_state",
                "fast_period": 5,
                "slow_period": 13,
                "direction": "above",
            },
        ],
        "entry_logic": "and",
        "exit": {
            "take_profit_pct": 2.0,
            "stop_loss_pct": 2.0,
            "timeout_bars": 100,
        },
    }

    # 戦略ビルド
    strategy = ConfigStrategy(strategy_config)
    print(f"✓ Strategy built: {strategy.name}")

    # バックテスト実行
    engine = BacktestEngine(
        strategy=strategy,
        initial_capital=10000,
        commission_pct=0.001,
        slippage_pct=0.0005
    )

    result = engine.run(uptrend_df)

    # 結果表示
    print(f"\n=== Backtest Results ===")
    print(f"Total Trades: {len(result.trades)}")

    if len(result.trades) == 0:
        print(f"⚠️ No trades generated!")
    else:
        from metrics.calculator import calculate_metrics
        metrics = calculate_metrics(result.trades, result.portfolio)
        print(f"Win Rate: {metrics.win_rate:.2f}%")
        print(f"PnL: {metrics.total_pnl_pct:+.2f}%")
        print(f"Profit Factor: {metrics.profit_factor:.2f}")
        print(f"Max Drawdown: {metrics.max_drawdown:.2f}%")

    # デバッグ情報
    if len(result.trades) == 0:
        print(f"\n⚠️ No trades generated!")
        print(f"Checking data availability:")
        required_cols = ['close', 'volume', 'bb_upper_20', 'bb_lower_20', 'bb_middle_20',
                        'adx_14', 'ema_5', 'ema_13', 'volume_sma_20']
        for col in required_cols:
            if col in uptrend_df.columns:
                non_null = uptrend_df[col].notna().sum()
                print(f"  {col}: {non_null}/{len(uptrend_df)} valid ({non_null/len(uptrend_df)*100:.1f}%)")
            else:
                print(f"  {col}: MISSING!")

        # Sample data inspection
        print(f"\nSample data (first valid row):")
        valid_idx = uptrend_df.dropna(subset=required_cols).index
        if len(valid_idx) > 0:
            sample = uptrend_df.loc[valid_idx[0]]
            print(f"  close: {sample['close']:.2f}")
            print(f"  bb_upper_20: {sample['bb_upper_20']:.2f}")
            print(f"  bb_middle_20: {sample['bb_middle_20']:.2f}")
            print(f"  bb_lower_20: {sample['bb_lower_20']:.2f}")
            bandwidth = (sample['bb_upper_20'] - sample['bb_lower_20']) / sample['bb_middle_20']
            print(f"  BB bandwidth: {bandwidth:.4f} (threshold: 0.05)")
            print(f"  adx_14: {sample['adx_14']:.2f}")
            print(f"  volume: {sample['volume']:.0f}")
            print(f"  volume_sma_20: {sample['volume_sma_20']:.0f}")
            print(f"  ema_5: {sample['ema_5']:.2f}")
            print(f"  ema_13: {sample['ema_13']:.2f}")

    return result


if __name__ == "__main__":
    test_bb_squeeze_breakout()
