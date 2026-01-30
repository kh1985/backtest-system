"""
ポジション・トレード管理
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pandas as pd


@dataclass
class Trade:
    """完了済みトレード"""
    entry_time: datetime
    exit_time: datetime
    side: str
    entry_price: float
    exit_price: float
    profit_pct: float
    duration_bars: int
    exit_type: str  # "TP", "SL", "TRAILING", "TIMEOUT", "FORCED"
    reason: str


@dataclass
class Position:
    """オープンポジション"""
    entry_price: float
    entry_time: datetime
    entry_index: int
    side: str  # "long" or "short"
    tp_price: float
    sl_price: float
    trailing_stop_pct: Optional[float] = None
    timeout_bars: Optional[int] = None
    highest_price: float = 0.0
    lowest_price: float = float("inf")
    reason: str = ""

    def __post_init__(self):
        self.highest_price = self.entry_price
        self.lowest_price = self.entry_price

    def check_exit(
        self, row: pd.Series, current_index: int
    ) -> Optional[Trade]:
        """現在のバーで決済条件をチェック"""
        high = row["high"]
        low = row["low"]
        close = row["close"]
        current_time = row.get("datetime", pd.Timestamp.now())

        # トレーリングストップ更新
        if self.trailing_stop_pct:
            if self.side == "long":
                self.highest_price = max(self.highest_price, high)
                trailing_sl = self.highest_price * (
                    1 - self.trailing_stop_pct / 100
                )
                self.sl_price = max(self.sl_price, trailing_sl)
            else:
                self.lowest_price = min(self.lowest_price, low)
                trailing_sl = self.lowest_price * (
                    1 + self.trailing_stop_pct / 100
                )
                self.sl_price = min(self.sl_price, trailing_sl)

        duration = current_index - self.entry_index

        # タイムアウト判定
        if self.timeout_bars and duration >= self.timeout_bars:
            return self._create_trade(
                close, current_time, duration, "TIMEOUT"
            )

        if self.side == "long":
            # TP判定（high >= tp_price）
            if high >= self.tp_price:
                return self._create_trade(
                    self.tp_price, current_time, duration, "TP"
                )
            # SL判定（low <= sl_price）
            if low <= self.sl_price:
                return self._create_trade(
                    self.sl_price, current_time, duration, "SL"
                )
        else:  # short
            # TP判定（low <= tp_price）
            if low <= self.tp_price:
                return self._create_trade(
                    self.tp_price, current_time, duration, "TP"
                )
            # SL判定（high >= sl_price）
            if high >= self.sl_price:
                return self._create_trade(
                    self.sl_price, current_time, duration, "SL"
                )

        return None

    def _create_trade(
        self,
        exit_price: float,
        exit_time: datetime,
        duration: int,
        exit_type: str,
    ) -> Trade:
        if self.side == "long":
            profit_pct = (
                (exit_price - self.entry_price) / self.entry_price * 100
            )
        else:
            profit_pct = (
                (self.entry_price - exit_price) / self.entry_price * 100
            )

        return Trade(
            entry_time=self.entry_time,
            exit_time=exit_time,
            side=self.side,
            entry_price=self.entry_price,
            exit_price=exit_price,
            profit_pct=profit_pct,
            duration_bars=duration,
            exit_type=exit_type,
            reason=self.reason,
        )
