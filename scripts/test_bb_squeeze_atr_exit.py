"""BB Squeeze Breakout戦略のATR exit profile検証"""
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
from metrics.calculator import calculate_metrics


def test_with_atr_exit():
    """AAVEUSDTの2025期間uptrendでBB Squeeze Breakout + ATR exitをテスト"""

    # データ読み込み
    symbol = "AAVEUSDT"
    execution_tf = "15m"

    print(f"\n=== {symbol} BB Squeeze Breakout + ATR Exit Test ===\n")

    # 15m足データ読み込み
    exec_path = project_root / f"inputdata/{symbol}-{execution_tf}-20250201-20260130-merged.csv"
    loader = BinanceCSVLoader()
    exec_data = loader.load(str(exec_path))
    exec_df = exec_data.df.copy()

    # HTFデータ読み込み
    htf = "1h"
    super_htf = "4h"
    htf_path = project_root / f"inputdata/{symbol}-{htf}-20250201-20260130-merged.csv"
    super_htf_path = project_root / f"inputdata/{symbol}-{super_htf}-20250201-20260130-merged.csv"

    htf_data = loader.load(str(htf_path))
    super_htf_data = loader.load(str(super_htf_path))
    htf_df = htf_data.df
    super_htf_df = super_htf_data.df

    print(f"✓ Loaded data: {len(exec_df)} rows")

    # レジーム検出
    detector = TrendDetector()
    htf_with_regime = detector.detect_dual_tf_ema(htf_df, super_htf_df)

    exec_df['datetime'] = pd.to_datetime(exec_df['datetime'])
    htf_with_regime['datetime'] = pd.to_datetime(htf_with_regime['datetime'])

    exec_df_with_regime = pd.merge_asof(
        exec_df.sort_values('datetime'),
        htf_with_regime[['datetime', 'trend_regime']].sort_values('datetime'),
        on='datetime',
        direction='backward'
    )
    exec_df_with_regime.rename(columns={'trend_regime': 'regime'}, inplace=True)

    # Uptrend期間のみ抽出
    uptrend_df = exec_df_with_regime[exec_df_with_regime['regime'] == 'uptrend'].copy()
    print(f"✓ Uptrend periods: {len(uptrend_df)} rows ({len(uptrend_df)/len(exec_df_with_regime)*100:.1f}%)\n")

    if len(uptrend_df) == 0:
        print("⚠️ No uptrend periods found!")
        return

    # 価格レンジとATR値の確認（デバッグ用）
    print(f"Price range: {uptrend_df['close'].min():.2f} - {uptrend_df['close'].max():.2f}")
    print(f"Avg close: {uptrend_df['close'].mean():.2f}\n")

    # テストするexit profiles
    exit_profiles = [
        {
            "name": "固定TP/SL（ローカルテスト用）",
            "config": {
                "take_profit_pct": 2.0,
                "stop_loss_pct": 2.0,
                "timeout_bars": 100,
            }
        },
        {
            "name": "atr_tp20_sl10（Modal使用）",
            "config": {
                "use_atr_exit": True,
                "atr_tp_mult": 2.0,
                "atr_sl_mult": 1.0,
                "atr_period": 14,
                "timeout_bars": 100,
            }
        },
        {
            "name": "atr_tp15_sl15（Step 17推奨）",
            "config": {
                "use_atr_exit": True,
                "atr_tp_mult": 1.5,
                "atr_sl_mult": 1.5,
                "atr_period": 14,
                "timeout_bars": 100,
            }
        },
    ]

    for exit_profile in exit_profiles:
        print(f"=== Testing: {exit_profile['name']} ===\n")

        # 戦略設定
        strategy_config = {
            "name": f"bb_squeeze_breakout_long_{exit_profile['name'].replace(' ', '_')}",
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
                    "threshold": 0.05,
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
            "exit": exit_profile['config'],
        }

        # ATR exitの場合、ATRインジケーターを追加
        if exit_profile['config'].get('use_atr_exit'):
            strategy_config['indicators'].append({
                "type": "atr",
                "period": exit_profile['config']['atr_period']
            })

        # 戦略ビルド
        strategy = ConfigStrategy(strategy_config)

        # バックテスト実行
        engine = BacktestEngine(
            strategy=strategy,
            initial_capital=10000,
            commission_pct=0.001,
            slippage_pct=0.0005
        )

        result = engine.run(uptrend_df)

        # ATR値を確認（デバッグ用）
        if 'atr_14' in result.df.columns:
            atr_values = result.df['atr_14'].dropna()
            if len(atr_values) > 0:
                print(f"ATR values: min={atr_values.min():.6f}, max={atr_values.max():.6f}, mean={atr_values.mean():.6f}")
                print(f"ATR as % of price: {atr_values.mean() / result.df['close'].mean() * 100:.2f}%\n")

        # 結果表示
        print(f"Exit config: {exit_profile['config']}")
        print(f"Total Trades: {len(result.trades)}")

        if len(result.trades) > 0:
            # Exit type分布
            exit_types = {}
            total_pnl = 0
            winning_trades = 0
            for trade in result.trades:
                exit_type = trade.exit_type
                exit_types[exit_type] = exit_types.get(exit_type, 0) + 1
                total_pnl += trade.profit_pct
                if trade.profit_pct > 0:
                    winning_trades += 1

            win_rate = (winning_trades / len(result.trades)) * 100 if result.trades else 0
            avg_pnl = total_pnl / len(result.trades) if result.trades else 0

            print(f"Win Rate: {win_rate:.2f}%")
            print(f"Avg PnL per trade: {avg_pnl:+.2f}%")
            print(f"Total PnL: {total_pnl:+.2f}%")
            print(f"Exit types: {exit_types}")

            # サンプルトレード（最初の3件）
            print(f"\nSample trades (first 3):")
            for i, trade in enumerate(result.trades[:3]):
                entry_idx = result.df[result.df['datetime'] == trade.entry_time].index[0] if len(result.df[result.df['datetime'] == trade.entry_time]) > 0 else None

                print(f"  {i+1}. Entry: {trade.entry_time}, Exit: {trade.exit_time}")
                print(f"     Entry price: {trade.entry_price:.4f}, Exit price: {trade.exit_price:.4f}")
                print(f"     PnL: {trade.profit_pct:+.2f}%, Exit type: {trade.exit_type}")

                # ATR exitの場合、エントリー時のATR値を表示
                if exit_profile['config'].get('use_atr_exit') and entry_idx is not None and 'atr_14' in result.df.columns:
                    atr_at_entry = result.df.loc[entry_idx, 'atr_14']
                    tp_mult = exit_profile['config']['atr_tp_mult']
                    sl_mult = exit_profile['config']['atr_sl_mult']
                    expected_tp = trade.entry_price + atr_at_entry * tp_mult
                    expected_sl = trade.entry_price - atr_at_entry * sl_mult

                    # Exit barのhigh/low
                    exit_idx = result.df[result.df['datetime'] == trade.exit_time].index[0] if len(result.df[result.df['datetime'] == trade.exit_time]) > 0 else None
                    if exit_idx is not None:
                        exit_high = result.df.loc[exit_idx, 'high']
                        exit_low = result.df.loc[exit_idx, 'low']

                    print(f"     ATR at entry: {atr_at_entry:.6f} ({atr_at_entry/trade.entry_price*100:.2f}% of price)")
                    print(f"     Expected TP: {expected_tp:.4f} (+{(expected_tp-trade.entry_price)/trade.entry_price*100:.2f}%)")
                    print(f"     Expected SL: {expected_sl:.4f} ({(expected_sl-trade.entry_price)/trade.entry_price*100:.2f}%)")
                    if exit_idx is not None:
                        print(f"     Exit bar H/L: {exit_high:.4f} / {exit_low:.4f}")
                        print(f"     TP reached? {exit_high >= expected_tp}")
        else:
            print(f"⚠️ No trades generated!")

        print()


if __name__ == "__main__":
    test_with_atr_exit()
