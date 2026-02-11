"""
データ層の基底クラス・共通型定義
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import pandas as pd


class Timeframe(Enum):
    M1 = "1m"
    M3 = "3m"
    M5 = "5m"
    M15 = "15m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"

    @classmethod
    def from_str(cls, s: str) -> "Timeframe":
        for tf in cls:
            if tf.value == s:
                return tf
        raise ValueError(f"Unknown timeframe: {s}")

    def to_minutes(self) -> int:
        mapping = {
            "1m": 1, "3m": 3, "5m": 5, "15m": 15,
            "1h": 60, "4h": 240, "1d": 1440,
        }
        return mapping[self.value]


@dataclass
class OHLCVData:
    """正規化されたOHLCVデータコンテナ"""
    df: pd.DataFrame  # columns: datetime, open, high, low, close, volume
    symbol: str
    timeframe: Timeframe
    source: str  # "csv" or "exchange:mexc" etc.

    def validate(self) -> bool:
        required = {"datetime", "open", "high", "low", "close", "volume"}
        return required.issubset(set(self.df.columns))

    @property
    def bars(self) -> int:
        return len(self.df)

    @property
    def start_time(self) -> Optional[pd.Timestamp]:
        if self.df.empty:
            return None
        return self.df["datetime"].iloc[0]

    @property
    def end_time(self) -> Optional[pd.Timestamp]:
        if self.df.empty:
            return None
        return self.df["datetime"].iloc[-1]

    def __repr__(self) -> str:
        return (
            f"OHLCVData({self.symbol}, {self.timeframe.value}, "
            f"{self.bars} bars, {self.source})"
        )


class DataSource(ABC):
    """データソースの抽象基底クラス"""

    @abstractmethod
    def load(self, symbol: str, timeframe: Timeframe, **kwargs) -> OHLCVData:
        ...
