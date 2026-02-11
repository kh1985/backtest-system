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
    exit_slippage_pct: float = 0.0  # 出口スリッページ（%）

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

        # TP/SLを先に評価: タイムアウトバーでもTP/SLが優先
        if self.side == "long":
            tp_hit = high >= self.tp_price
            sl_hit = low <= self.sl_price
            if tp_hit and sl_hit:
                # 同一バーで両方ヒット → 保守的にSL優先
                return self._create_trade(
                    self.sl_price, current_time, duration, "SL"
                )
            if tp_hit:
                return self._create_trade(
                    self.tp_price, current_time, duration, "TP"
                )
            if sl_hit:
                return self._create_trade(
                    self.sl_price, current_time, duration, "SL"
                )
        else:  # short
            tp_hit = low <= self.tp_price
            sl_hit = high >= self.sl_price
            if tp_hit and sl_hit:
                # 同一バーで両方ヒット → 保守的にSL優先
                return self._create_trade(
                    self.sl_price, current_time, duration, "SL"
                )
            if tp_hit:
                return self._create_trade(
                    self.tp_price, current_time, duration, "TP"
                )
            if sl_hit:
                return self._create_trade(
                    self.sl_price, current_time, duration, "SL"
                )

        # タイムアウト判定
        if self.timeout_bars and duration >= self.timeout_bars:
            return self._create_trade(
                close, current_time, duration, "TIMEOUT"
            )

        return None

    def _create_trade(
        self,
        exit_price: float,
        exit_time: datetime,
        duration: int,
        exit_type: str,
    ) -> Trade:
        # 出口スリッページ適用（SL/TRAILING/FORCED/TIMEOUTは不利方向にずれる）
        if self.exit_slippage_pct > 0 and exit_type != "TP":
            if self.side == "long":
                # ロングの損切り → 価格が下にずれる（より不利）
                exit_price *= 1 - self.exit_slippage_pct / 100
            else:
                # ショートの損切り → 価格が上にずれる（より不利）
                exit_price *= 1 + self.exit_slippage_pct / 100

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
