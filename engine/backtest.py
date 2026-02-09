"""
バックテストエンジン

Bar-by-barでシミュレーションを実行するコアエンジン。
"""

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from strategy.base import Strategy
from .position import Position, Trade
from .portfolio import Portfolio


@dataclass
class BacktestResult:
    """バックテスト結果コンテナ"""
    trades: List[Trade]
    portfolio: Portfolio
    strategy_name: str
    df: pd.DataFrame  # インジケーター付きDataFrame


class BacktestEngine:
    """
    コアバックテストエンジン

    - Bar-by-bar反復（ルックアヘッドなし）
    - 1ポジションモデル
    - TP/SLはbar内のhigh/lowで判定
    - 手数料・スリッページ対応
    """

    def __init__(
        self,
        strategy: Strategy,
        initial_capital: float = 10000.0,
        commission_pct: float = 0.0,
        slippage_pct: float = 0.0,
        entry_on_next_open: bool = False,
        exit_slippage_pct: float = 0.0,
    ):
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.entry_on_next_open = entry_on_next_open
        self.exit_slippage_pct = exit_slippage_pct

    def run(self, df: pd.DataFrame) -> BacktestResult:
        """
        バックテストを実行

        1. strategy.setup() でインジケーターを追加
        2. bar-by-barで反復
        3. ポジションあり → 決済判定
        4. ポジションなし → エントリー判定
        5. 未決済ポジションは最終バーで強制決済
        """
        df = df.copy()
        df = self.strategy.setup(df)

        trades: List[Trade] = []
        portfolio = Portfolio(self.initial_capital)
        position: Optional[Position] = None
        pending_signal = None  # entry_on_next_open用

        for i in range(1, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i - 1]

            # 前バーのシグナルを今バーのopenで約定（entry_on_next_open）
            if pending_signal is not None and position is None:
                position = self._open_position(
                    row, i, pending_signal, use_open=True
                )
                pending_signal = None

            # 決済判定
            if position is not None:
                trade = position.check_exit(row, i)
                if trade:
                    # 手数料を適用
                    if self.commission_pct > 0:
                        trade.profit_pct -= self.commission_pct * 2
                    portfolio.close_trade(trade)
                    trades.append(trade)
                    position = None

            # エントリー判定
            if position is None and pending_signal is None:
                signal = self.strategy.check_entry(row, prev_row)
                if signal:
                    if self.entry_on_next_open:
                        pending_signal = signal
                    else:
                        position = self._open_position(row, i, signal)

        # 未決済ポジションの強制決済
        if position is not None:
            trade = self._force_close(position, df.iloc[-1], len(df) - 1)
            if self.commission_pct > 0:
                trade.profit_pct -= self.commission_pct * 2
            portfolio.close_trade(trade)
            trades.append(trade)

        return BacktestResult(
            trades=trades,
            portfolio=portfolio,
            strategy_name=self.strategy.name,
            df=df,
        )

    def _open_position(
        self, row, index, signal, use_open: bool = False
    ) -> Position:
        """ポジションを開く"""
        entry_price = row["open"] if use_open else row["close"]

        # スリッページ適用
        if self.slippage_pct > 0:
            if signal.side.value == "long":
                entry_price *= 1 + self.slippage_pct / 100
            else:
                entry_price *= 1 - self.slippage_pct / 100

        exit_rule = self.strategy.exit_rule

        if signal.side.value == "long":
            tp_price = entry_price * (1 + exit_rule.take_profit_pct / 100)
            sl_price = entry_price * (1 - exit_rule.stop_loss_pct / 100)
        else:
            tp_price = entry_price * (1 - exit_rule.take_profit_pct / 100)
            sl_price = entry_price * (1 + exit_rule.stop_loss_pct / 100)

        return Position(
            entry_price=entry_price,
            entry_time=row.get("datetime", pd.Timestamp.now()),
            entry_index=index,
            side=signal.side.value,
            tp_price=tp_price,
            sl_price=sl_price,
            trailing_stop_pct=exit_rule.trailing_stop_pct,
            timeout_bars=exit_rule.timeout_bars,
            reason=signal.reason,
            exit_slippage_pct=self.exit_slippage_pct,
        )

    def _force_close(self, position, row, index) -> Trade:
        """ポジションを強制決済"""
        close_price = row["close"]
        duration = index - position.entry_index

        if position.side == "long":
            profit_pct = (
                (close_price - position.entry_price)
                / position.entry_price
                * 100
            )
        else:
            profit_pct = (
                (position.entry_price - close_price)
                / position.entry_price
                * 100
            )

        return Trade(
            entry_time=position.entry_time,
            exit_time=row.get("datetime", pd.Timestamp.now()),
            side=position.side,
            entry_price=position.entry_price,
            exit_price=close_price,
            profit_pct=profit_pct,
            duration_bars=duration,
            exit_type="FORCED",
            reason=position.reason,
        )
