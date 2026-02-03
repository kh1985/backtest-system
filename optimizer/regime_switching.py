"""
レジーム切替バックテスト

3つのレジーム別最適戦略を使って、全期間データ上で
レジームに応じて戦略を切り替えるバックテストを実行する。
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd

from metrics.calculator import BacktestMetrics, calculate_metrics_from_arrays
from optimizer.results import OptimizationEntry

logger = logging.getLogger(__name__)

REGIME_INT_MAP = {"uptrend": 0, "downtrend": 1, "range": 2}
INT_REGIME_MAP = {0: "uptrend", 1: "downtrend", 2: "range"}


@dataclass
class RegimeSwitchingResult:
    """レジーム切替バックテストの結果"""
    overall_metrics: BacktestMetrics
    regime_metrics: Dict[str, BacktestMetrics] = field(default_factory=dict)
    regime_configs: Dict[str, OptimizationEntry] = field(default_factory=dict)
    equity_curve: np.ndarray = field(default_factory=lambda: np.array([]))
    regime_trade_counts: Dict[str, int] = field(default_factory=dict)
    trade_regimes: np.ndarray = field(default_factory=lambda: np.array([]))
    trade_pnls: np.ndarray = field(default_factory=lambda: np.array([]))
    trade_durations: np.ndarray = field(default_factory=lambda: np.array([]))


def _extract_exit_params(config: dict):
    """config から exit パラメータを抽出（ATR対応）"""
    is_long = config.get("side", "long") == "long"
    exit_conf = config.get("exit", {})
    tp = float(exit_conf.get("take_profit_pct", 2.0))
    sl = float(exit_conf.get("stop_loss_pct", 1.0))
    trail = float(exit_conf.get("trailing_stop_pct", 0) or 0)
    timeout = int(exit_conf.get("timeout_bars", 0) or 0)
    use_atr = bool(exit_conf.get("use_atr_exit", False))
    atr_tp_mult = float(exit_conf.get("atr_tp_mult", 0.0))
    atr_sl_mult = float(exit_conf.get("atr_sl_mult", 0.0))
    atr_period = int(exit_conf.get("atr_period", 14))
    return is_long, tp, sl, trail, timeout, use_atr, atr_tp_mult, atr_sl_mult, atr_period


def run_regime_switching_backtest(
    df: pd.DataFrame,
    regime_strategies: Dict[str, OptimizationEntry],
    commission_pct: float = 0.04,
    slippage_pct: float = 0.0,
    initial_capital: float = 10000.0,
    trend_column: str = "trend_regime",
) -> RegimeSwitchingResult:
    """
    レジーム切替バックテストを実行する。

    Args:
        df: 実行TFのDataFrame（trend_regime列付き）
        regime_strategies: {"uptrend": entry, "downtrend": entry, "range": entry}
        commission_pct: 手数料%
        slippage_pct: スリッページ%
        initial_capital: 初期資金
        trend_column: レジーム列名

    Returns:
        RegimeSwitchingResult
    """
    from strategy.builder import ConfigStrategy
    from engine.numba_loop import (
        vectorize_entry_signals,
        _backtest_loop_regime_switching,
        compute_atr_numpy,
    )

    n = len(df)

    # --- regime_array 構築 ---
    regime_array = np.full(n, 2, dtype=np.int64)  # デフォルト = range
    if trend_column in df.columns:
        for regime_str, regime_int in REGIME_INT_MAP.items():
            mask = (df[trend_column] == regime_str).values
            regime_array[mask] = regime_int

    # --- 各レジーム戦略のシグナル生成 ---
    entry_signals = {}
    configs = {}

    for regime in ["uptrend", "downtrend", "range"]:
        if regime in regime_strategies:
            entry = regime_strategies[regime]
            config = entry.config

            strategy = ConfigStrategy(config)
            work_df = strategy.setup(df.copy())

            signals = vectorize_entry_signals(
                work_df,
                config.get("entry_conditions", []),
                config.get("entry_logic", "and"),
            )
            entry_signals[regime] = signals
            configs[regime] = config
            logger.info(
                "regime=%s template=%s signals=%d",
                regime, entry.template_name, int(signals.sum()),
            )
        else:
            entry_signals[regime] = np.zeros(n, dtype=np.bool_)
            configs[regime] = None
            logger.info("regime=%s: no strategy assigned", regime)

    # --- パラメータ抽出 ---
    def _get_params(regime):
        c = configs.get(regime)
        if c is None:
            # is_long, tp, sl, trail, timeout, use_atr, atr_tp, atr_sl, atr_period
            return True, 0.0, 1.0, 0.0, 0, False, 0.0, 0.0, 14
        return _extract_exit_params(c)

    p_up = _get_params("uptrend")
    p_down = _get_params("downtrend")
    p_range = _get_params("range")

    # --- OHLC 配列 ---
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    close = df["close"].values.astype(np.float64)

    # --- ATR配列計算（いずれかのレジームがATRモードの場合） ---
    any_atr = p_up[5] or p_down[5] or p_range[5]
    if any_atr:
        atr_period = max(p_up[8], p_down[8], p_range[8])
        atr_arr = compute_atr_numpy(high, low, close, period=atr_period)
    else:
        atr_arr = np.empty(0, dtype=np.float64)

    # --- Numba 実行 ---
    profit_pcts, durations_arr, equity_curve, trade_regimes = (
        _backtest_loop_regime_switching(
            high, low, close,
            entry_signals["uptrend"],
            entry_signals["downtrend"],
            entry_signals["range"],
            regime_array,
            p_up[0], p_down[0], p_range[0],       # is_long
            p_up[1], p_down[1], p_range[1],       # tp_pct
            p_up[2], p_down[2], p_range[2],       # sl_pct
            p_up[3], p_down[3], p_range[3],       # trailing_pct
            p_up[4], p_down[4], p_range[4],       # timeout_bars
            commission_pct, slippage_pct, initial_capital,
            atr_arr,
            p_up[5], p_down[5], p_range[5],       # use_atr
            p_up[6], p_down[6], p_range[6],       # atr_tp_mult
            p_up[7], p_down[7], p_range[7],       # atr_sl_mult
        )
    )

    # --- 全体メトリクス ---
    overall_metrics = calculate_metrics_from_arrays(profit_pcts, durations_arr, equity_curve)

    # --- レジーム別メトリクス（事後分解） ---
    regime_metrics = {}
    regime_trade_counts = {}

    for regime_int, regime_str in INT_REGIME_MAP.items():
        mask = trade_regimes == regime_int
        r_pnls = profit_pcts[mask]
        r_durs = durations_arr[mask]
        regime_trade_counts[regime_str] = len(r_pnls)

        if len(r_pnls) > 0:
            r_equity = np.empty(len(r_pnls) + 1, dtype=np.float64)
            r_equity[0] = initial_capital
            for j in range(len(r_pnls)):
                r_equity[j + 1] = r_equity[j] * (1.0 + r_pnls[j] / 100.0)
            regime_metrics[regime_str] = calculate_metrics_from_arrays(
                r_pnls, r_durs, r_equity,
            )
        else:
            regime_metrics[regime_str] = calculate_metrics_from_arrays(
                np.array([], dtype=np.float64),
                np.array([], dtype=np.int64),
                np.array([initial_capital], dtype=np.float64),
            )

    return RegimeSwitchingResult(
        overall_metrics=overall_metrics,
        regime_metrics=regime_metrics,
        regime_configs=regime_strategies,
        equity_curve=equity_curve,
        regime_trade_counts=regime_trade_counts,
        trade_regimes=trade_regimes,
        trade_pnls=profit_pcts,
        trade_durations=durations_arr,
    )
