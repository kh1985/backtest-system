"""
複合スコア計算

バックテスト結果のメトリクスから複合スコアを算出。
各指標を正規化して重み付け合計する。
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class ScoringWeights:
    """スコア重み"""
    profit_factor: float = 0.3
    win_rate: float = 0.3
    max_drawdown: float = 0.2
    sharpe_ratio: float = 0.2

    def validate(self) -> bool:
        total = (
            self.profit_factor
            + self.win_rate
            + self.max_drawdown
            + self.sharpe_ratio
        )
        return abs(total - 1.0) < 0.01


def calculate_composite_score(
    profit_factor: float,
    win_rate: float,
    max_drawdown_pct: float,
    sharpe_ratio: float,
    weights: ScoringWeights = None,
) -> float:
    """
    複合スコアを算出（0.0 ~ 1.0）

    score = (PF正規化 × w1) + (勝率/100 × w2) + ((1-DD/100) × w3) + (Sharpe正規化 × w4)

    - PF: 上限10でクリップ → 0-1に正規化
    - Sharpe: -2~5でクリップ → 0-1に正規化
    - DD: 0~100% → 0-1（低いほど良い）
    - 勝率: 0~100% → 0-1
    """
    if weights is None:
        weights = ScoringWeights()

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

    score = (
        pf_norm * weights.profit_factor
        + wr_norm * weights.win_rate
        + dd_norm * weights.max_drawdown
        + sharpe_norm * weights.sharpe_ratio
    )

    return float(np.clip(score, 0, 1))
