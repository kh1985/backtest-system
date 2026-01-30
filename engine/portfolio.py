"""
ポートフォリオ（資金管理・エクイティカーブ）
"""

from dataclasses import dataclass, field
from typing import List

from .position import Trade


@dataclass
class Portfolio:
    """ポートフォリオ状態の追跡"""
    initial_capital: float
    equity_curve: List[float] = field(default_factory=list)
    current_equity: float = 0.0

    def __post_init__(self):
        self.current_equity = self.initial_capital
        self.equity_curve = [self.initial_capital]

    def close_trade(self, trade: Trade) -> None:
        """トレード決済時にエクイティを更新"""
        pnl = self.current_equity * (trade.profit_pct / 100.0)
        self.current_equity += pnl
        self.equity_curve.append(self.current_equity)

    @property
    def total_return_pct(self) -> float:
        """総リターン（%）"""
        if self.initial_capital == 0:
            return 0.0
        return (
            (self.current_equity - self.initial_capital)
            / self.initial_capital
            * 100
        )
