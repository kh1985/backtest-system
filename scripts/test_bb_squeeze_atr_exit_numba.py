"""BB Squeeze Breakout戦略のATR exit profile検証（Numba版）"""
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from data.binance_loader import BinanceCSVLoader
from analysis.trend import TrendDetector
from engine.numba_loop import _backtest_loop, vectorize_entry_signals, compute_atr_numpy


def test_with_atr_exit_numba():
    """AAVEUSDTの2025期間uptrendでBB Squeeze Breakout + ATR exitをテスト（Numba版）"""

    # データ読み込み
    symbol = "AAVEUSDT"
    execution_tf = "15m"

    print(f"\n=== {symbol} BB Squeeze Breakout + ATR Exit Test (Numba) ===\n")

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
    uptrend_df = exec_df_with_regime[exec_df_with_regime['regime'] == 'uptrend'].copy().reset_index(drop=True)
    print(f"✓ Uptrend periods: {len(uptrend_df)} rows ({len(uptrend_df)/len(exec_df_with_regime)*100:.1f}%)\n")

    if len(uptrend_df) == 0:
        print("⚠️ No uptrend periods found!")
        return

    # 価格レンジとATR値の確認（デバッグ用）
    print(f"Price range: {uptrend_df['close'].min():.2f} - {uptrend_df['close'].max():.2f}")
    print(f"Avg close: {uptrend_df['close'].mean():.2f}\n")

    # インジケーター計算
    # BB
    bb_period = 20
    sma = uptrend_df['close'].rolling(window=bb_period).mean()
    std = uptrend_df['close'].rolling(window=bb_period).std()
    uptrend_df['bb_upper_20'] = sma + (std * 2.0)
    uptrend_df['bb_middle_20'] = sma
    uptrend_df['bb_lower_20'] = sma - (std * 2.0)

    # ADX
    uptrend_df['adx_14'] = 25.0  # 簡略化のため固定値

    # EMA
    uptrend_df['ema_5'] = uptrend_df['close'].ewm(span=5, adjust=False).mean()
    uptrend_df['ema_13'] = uptrend_df['close'].ewm(span=13, adjust=False).mean()

    # Volume SMA
    uptrend_df['volume_sma_20'] = uptrend_df['volume'].rolling(window=20).mean()

    # ATR
    atr_arr = compute_atr_numpy(
        uptrend_df['high'].values,
        uptrend_df['low'].values,
        uptrend_df['close'].values,
        period=14
    )
    uptrend_df['atr_14'] = atr_arr

    print(f"ATR values: min={atr_arr[bb_period:].min():.6f}, max={atr_arr[bb_period:].max():.6f}, mean={atr_arr[bb_period:].mean():.6f}")
    print(f"ATR as % of price: {atr_arr[bb_period:].mean() / uptrend_df['close'].mean() * 100:.2f}%\n")

    # エントリー条件の設定
    entry_conditions = [
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
    ]

    # エントリーシグナルのベクトル化
    entry_signals = vectorize_entry_signals(uptrend_df, entry_conditions, entry_logic="and")

    print(f"Total signals: {entry_signals.sum()}")
    print()

    # テストするexit profiles
    exit_profiles = [
        {
            "name": "固定TP/SL（ベースライン）",
            "use_atr_exit": False,
            "tp_pct": 2.0,
            "sl_pct": 2.0,
            "atr_tp_mult": 0.0,
            "atr_sl_mult": 0.0,
        },
        {
            "name": "atr_tp20_sl10（Modal使用）",
            "use_atr_exit": True,
            "tp_pct": 0.0,
            "sl_pct": 0.0,
            "atr_tp_mult": 2.0,
            "atr_sl_mult": 1.0,
        },
        {
            "name": "atr_tp15_sl15（Step 17推奨）",
            "use_atr_exit": True,
            "tp_pct": 0.0,
            "sl_pct": 0.0,
            "atr_tp_mult": 1.5,
            "atr_sl_mult": 1.5,
        },
    ]

    # numpy配列の準備
    open_ = uptrend_df['open'].values.astype(np.float64)
    high = uptrend_df['high'].values.astype(np.float64)
    low = uptrend_df['low'].values.astype(np.float64)
    close = uptrend_df['close'].values.astype(np.float64)
    regime_mask = np.ones(len(uptrend_df), dtype=np.bool_)

    for exit_profile in exit_profiles:
        print(f"=== Testing: {exit_profile['name']} ===\n")

        # Numbaループ実行
        profit_pcts, durations, equity_curve = _backtest_loop(
            open_, high, low, close,
            entry_signals, regime_mask,
            entry_on_next_open=True,
            is_long=True,
            tp_pct=exit_profile['tp_pct'],
            sl_pct=exit_profile['sl_pct'],
            trailing_pct=0.0,
            timeout_bars=100,
            commission_pct=0.001,
            slippage_pct=0.0005,
            initial_capital=10000.0,
            atr=atr_arr,
            use_atr_exit=exit_profile['use_atr_exit'],
            atr_tp_mult=exit_profile['atr_tp_mult'],
            atr_sl_mult=exit_profile['atr_sl_mult'],
            bb_upper=np.empty(0, dtype=np.float64),
            bb_lower=np.empty(0, dtype=np.float64),
            use_bb_exit=False,
            vwap_upper=np.empty(0, dtype=np.float64),
            vwap_lower=np.empty(0, dtype=np.float64),
            use_vwap_exit=False,
            use_atr_trailing=False,
            atr_trailing_mult=0.0,
        )

        # 結果表示
        print(f"Exit config: {exit_profile}")
        print(f"Total Trades: {len(profit_pcts)}")

        if len(profit_pcts) > 0:
            winning_trades = (profit_pcts > 0).sum()
            win_rate = (winning_trades / len(profit_pcts)) * 100
            avg_pnl = profit_pcts.mean()
            total_pnl = profit_pcts.sum()
            avg_duration = durations.mean()

            print(f"Win Rate: {win_rate:.2f}%")
            print(f"Avg PnL per trade: {avg_pnl:+.2f}%")
            print(f"Total PnL: {total_pnl:+.2f}%")
            print(f"Avg duration: {avg_duration:.1f} bars ({avg_duration * 15 / 60:.1f} hours)")
            print(f"Final equity: ${equity_curve[-1]:.2f}")
        else:
            print(f"⚠️ No trades generated!")

        print()


if __name__ == "__main__":
    test_with_atr_exit_numba()
