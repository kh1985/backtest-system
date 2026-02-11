"""
複合スコア計算 + 過学習警告

バックテスト結果のメトリクスから複合スコアを算出。
各指標を正規化して重み付け合計する。
過学習の疑いがある結果に警告フラグを付与する。
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class ScoringWeights:
    """スコア重み"""
    profit_factor: float = 0.15
    win_rate: float = 0.2
    max_drawdown: float = 0.3
    sharpe_ratio: float = 0.2
    total_return: float = 0.15

    def validate(self) -> bool:
        total = (
            self.profit_factor
            + self.win_rate
            + self.max_drawdown
            + self.sharpe_ratio
            + self.total_return
        )
        return abs(total - 1.0) < 0.01


def calculate_composite_score(
    profit_factor: float,
    win_rate: float,
    max_drawdown_pct: float,
    sharpe_ratio: float,
    total_return_pct: float = 0.0,
    total_trades: Optional[int] = None,
    weights: ScoringWeights = None,
    min_trades_for_confidence: int = 30,
    full_confidence_trades: int = 120,
    min_confidence_factor: float = 0.25,
) -> float:
    """
    複合スコアを算出（0.0 ~ 1.0）

    score = (PF正規化 × w1) + (勝率/100 × w2) + ((1-DD/100) × w3)
          + (Sharpe正規化 × w4) + (Return正規化 × w5)

    - PF: 上限10でクリップ → 0-1に正規化
    - Sharpe: -2~5でクリップ → 0-1に正規化
    - DD: 0~100% → 0-1（低いほど良い）
    - 勝率: 0~100% → 0-1
    - Return: -50~100%でクリップ → 0-1に正規化
    """
    if weights is None:
        weights = ScoringWeights()

    # トレード0件は即スコア0（ランキング汚染防止）
    if total_trades == 0:
        return 0.0

    # PF正規化: クリップ後 0-1
    pf_clipped = np.clip(profit_factor, 0, 10)
    pf_norm = pf_clipped / 10.0

    # 勝率: そのまま0-1
    wr_norm = np.clip(win_rate, 0, 100) / 100.0

    # DD: 低いほど良い → (1 - DD/100)
    dd_norm = 1.0 - np.clip(max_drawdown_pct, 0, 100) / 100.0

    # Sharpe: -2~5にクリップ → 0-1
    sharpe_clipped = np.clip(sharpe_ratio, -2, 5)
    sharpe_norm = (sharpe_clipped + 2) / 7.0  # -2→0, 5→1

    # Return: -50~100%にクリップ → 0-1
    return_clipped = np.clip(total_return_pct, -50, 100)
    return_norm = (return_clipped + 50) / 150.0  # -50→0, 100→1

    raw_score = (
        pf_norm * weights.profit_factor
        + wr_norm * weights.win_rate
        + dd_norm * weights.max_drawdown
        + sharpe_norm * weights.sharpe_ratio
        + return_norm * weights.total_return
    )

    confidence = _trade_confidence_factor(
        total_trades=total_trades,
        min_trades_for_confidence=min_trades_for_confidence,
        full_confidence_trades=full_confidence_trades,
        min_confidence_factor=min_confidence_factor,
    )
    score = raw_score * confidence
    return float(np.clip(score, 0, 1))


def _trade_confidence_factor(
    total_trades: Optional[int],
    min_trades_for_confidence: int,
    full_confidence_trades: int,
    min_confidence_factor: float,
) -> float:
    """トレード件数に応じた信頼度係数を返す。"""
    if total_trades is None:
        return 1.0
    if total_trades <= 0:
        return 0.0

    min_t = max(int(min_trades_for_confidence), 1)
    full_t = max(int(full_confidence_trades), min_t)
    min_factor = float(np.clip(min_confidence_factor, 0.0, 1.0))

    if total_trades >= full_t:
        return 1.0
    if total_trades <= min_t:
        # 閾値未満は強く減衰
        ratio = max(total_trades / min_t, 0.0)
        return min_factor + (1.0 - min_factor) * ratio * 0.5

    # 閾値〜フル信頼の間は線形補間
    ratio = (total_trades - min_t) / max(full_t - min_t, 1)
    return min_factor + (1.0 - min_factor) * (0.5 + 0.5 * ratio)


def detect_overfitting_warnings(
    metrics,
    oos_metrics=None,
) -> List[str]:
    """
    過学習の警告を検出

    Args:
        metrics: Train 期間の BacktestMetrics
        oos_metrics: Test 期間の BacktestMetrics（OOS検証時のみ）

    Returns:
        警告メッセージのリスト
    """
    warnings = []

    # PF > 2.0: 過学習の疑い
    if metrics.profit_factor > 2.0:
        warnings.append(
            f"PF={metrics.profit_factor:.2f} — 過学習の疑い（PF>2.0）"
        )

    # Sharpe > 3.0: 非現実的
    if metrics.sharpe_ratio > 3.0:
        warnings.append(
            f"Sharpe={metrics.sharpe_ratio:.2f} — 非現実的（Sharpe>3.0）"
        )

    # トレード数 < 30: 統計的に不十分
    if metrics.total_trades < 30:
        warnings.append(
            f"{metrics.total_trades}件 — 統計的に不十分（30件未満）"
        )

    # OOS PnL 劣化 > 50%
    if oos_metrics is not None and metrics.total_profit_pct > 0:
        decay = (
            (metrics.total_profit_pct - oos_metrics.total_profit_pct)
            / abs(metrics.total_profit_pct)
            * 100
        )
        if decay > 50:
            warnings.append(
                f"劣化{decay:.0f}% — Train→Test で大幅劣化"
            )

    return warnings
