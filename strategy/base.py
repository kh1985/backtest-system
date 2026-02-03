"""
戦略の基底クラス・共通型定義
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import pandas as pd


class Side(Enum):
    LONG = "long"
    SHORT = "short"


@dataclass
class Signal:
    """エントリーシグナル"""
    index: int
    side: Side
    reason: str


@dataclass
class ExitRule:
    """決済ルール"""
    take_profit_pct: float = 1.0
    stop_loss_pct: float = 0.5
    trailing_stop_pct: Optional[float] = None
    timeout_bars: Optional[int] = None
    use_atr_exit: bool = False
    atr_tp_mult: float = 0.0
    atr_sl_mult: float = 0.0
    atr_period: int = 14
    use_bb_exit: bool = False
    bb_period: int = 20


class Strategy(ABC):
    """戦略の抽象基底クラス"""

    name: str
    exit_rule: ExitRule

    @abstractmethod
    def setup(self, df: pd.DataFrame) -> pd.DataFrame:
        """DataFrameにインジケーターを追加"""
        ...

    @abstractmethod
    def check_entry(
        self, row: pd.Series, prev_row: pd.Series
    ) -> Optional[Signal]:
        """エントリー条件をチェック"""
        ...
