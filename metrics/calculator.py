"""
メトリクス計算

バックテスト結果から各種統計指標を算出する。
"""

from dataclasses import dataclass
from typing import List

import numpy as np

from engine.position import Trade

DEFAULT_BARS_PER_YEAR = 365 * 24 * 4  # 15m


@dataclass
class BacktestMetrics:
    """バックテストメトリクス"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_profit_pct: float
    avg_profit_pct: float
    avg_loss_pct: float
    profit_factor: float
    max_drawdown_pct: float
    sharpe_ratio: float
    avg_duration_bars: float
    best_trade_pct: float
    worst_trade_pct: float
    equity_curve: List[float]
    cumulative_returns: List[float]
    drawdown_series: List[float]


def calculate_metrics(
    trades: List[Trade],
    equity_curve: List[float],
    bars_per_year: int = DEFAULT_BARS_PER_YEAR,
) -> BacktestMetrics:
    """トレードリストとエクイティカーブからメトリクスを算出"""
    if not trades:
        return _empty_metrics(equity_curve)

    profits = [t.profit_pct for t in trades]
    wins = [p for p in profits if p > 0]
    losses = [p for p in profits if p < 0]

    total_win = sum(wins) if wins else 0.0
    total_loss = abs(sum(losses)) if losses else 0.0

    # プロフィットファクター
    pf = total_win / total_loss if total_loss > 0 else float("inf")

    # 最大ドローダウン
    eq = np.array(equity_curve)
    running_max = np.maximum.accumulate(eq)
    drawdown = np.where(
        running_max > 0,
        (running_max - eq) / running_max * 100,
        0.0,
    )
    max_dd = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0

    # シャープレシオ（std が極小の場合に巨大値になるのを防止）
    returns = np.diff(eq) / eq[:-1] if len(eq) > 1 else np.array([])
    sharpe = _compute_annualized_sharpe(returns, bars_per_year)

    # 累積リターン
    cumulative = np.cumsum(profits).tolist()

    return BacktestMetrics(
        total_trades=len(trades),
        winning_trades=len(wins),
        losing_trades=len(losses),
        win_rate=len(wins) / len(trades) * 100 if trades else 0.0,
        total_profit_pct=sum(profits),
        avg_profit_pct=float(np.mean(wins)) if wins else 0.0,
        avg_loss_pct=float(np.mean(losses)) if losses else 0.0,
        profit_factor=pf,
        max_drawdown_pct=max_dd,
        sharpe_ratio=sharpe,
        avg_duration_bars=float(np.mean([t.duration_bars for t in trades])),
        best_trade_pct=max(profits),
        worst_trade_pct=min(profits),
        equity_curve=equity_curve,
        cumulative_returns=cumulative,
        drawdown_series=drawdown.tolist(),
    )


def calculate_metrics_from_arrays(
    profit_pcts: np.ndarray,
    durations: np.ndarray,
    equity_curve: np.ndarray,
    bars_per_year: int = DEFAULT_BARS_PER_YEAR,
) -> BacktestMetrics:
    """numpy配列からメトリクスを算出（Numbaループ用）"""
    if len(profit_pcts) == 0:
        return _empty_metrics(equity_curve.tolist())

    profits = profit_pcts
    wins = profits[profits > 0]
    losses = profits[profits < 0]

    total_win = float(np.sum(wins)) if len(wins) > 0 else 0.0
    total_loss = float(np.abs(np.sum(losses))) if len(losses) > 0 else 0.0

    pf = total_win / total_loss if total_loss > 0 else float("inf")

    running_max = np.maximum.accumulate(equity_curve)
    drawdown = np.where(
        running_max > 0,
        (running_max - equity_curve) / running_max * 100,
        0.0,
    )
    max_dd = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0

    returns = np.diff(equity_curve) / equity_curve[:-1] if len(equity_curve) > 1 else np.array([])
    sharpe = _compute_annualized_sharpe(returns, bars_per_year)

    cumulative = np.cumsum(profits).tolist()

    return BacktestMetrics(
        total_trades=len(profits),
        winning_trades=len(wins),
        losing_trades=len(losses),
        win_rate=len(wins) / len(profits) * 100,
        total_profit_pct=float(np.sum(profits)),
        avg_profit_pct=float(np.mean(wins)) if len(wins) > 0 else 0.0,
        avg_loss_pct=float(np.mean(losses)) if len(losses) > 0 else 0.0,
        profit_factor=pf,
        max_drawdown_pct=max_dd,
        sharpe_ratio=sharpe,
        avg_duration_bars=float(np.mean(durations)),
        best_trade_pct=float(np.max(profits)),
        worst_trade_pct=float(np.min(profits)),
        equity_curve=equity_curve.tolist(),
        cumulative_returns=cumulative,
        drawdown_series=drawdown.tolist(),
    )


def _empty_metrics(equity_curve: List[float] = None) -> BacktestMetrics:
    """トレードが0件の場合のデフォルト"""
    ec = equity_curve or [0.0]
    return BacktestMetrics(
        total_trades=0,
        winning_trades=0,
        losing_trades=0,
        win_rate=0.0,
        total_profit_pct=0.0,
        avg_profit_pct=0.0,
        avg_loss_pct=0.0,
        profit_factor=0.0,
        max_drawdown_pct=0.0,
        sharpe_ratio=0.0,
        avg_duration_bars=0.0,
        best_trade_pct=0.0,
        worst_trade_pct=0.0,
        equity_curve=ec,
        cumulative_returns=[],
        drawdown_series=[],
    )


def _compute_annualized_sharpe(returns: np.ndarray, bars_per_year: int) -> float:
    """バー次元リターンのSharpeを年率化して返す。"""
    if len(returns) <= 1:
        return 0.0
    std = float(np.std(returns))
    if std <= 1e-10:
        return 0.0

    annualizer = np.sqrt(max(int(bars_per_year), 1))
    sharpe = float(np.mean(returns) / std * annualizer)
    if not np.isfinite(sharpe):
        return 0.0
    return sharpe
