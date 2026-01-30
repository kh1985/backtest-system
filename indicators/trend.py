"""
トレンド系インジケーター: SMA, EMA
"""

from typing import List

import pandas as pd

from .base import Indicator


class SMA(Indicator):
    """単純移動平均線"""

    def __init__(self, period: int = 20, source: str = "close"):
        self.period = period
        self.source = source
        self.name = f"sma_{period}"

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.name] = df[self.source].rolling(window=self.period).mean()
        return df

    @property
    def columns(self) -> List[str]:
        return [self.name]

    @property
    def is_overlay(self) -> bool:
        return True


class EMA(Indicator):
    """指数移動平均線"""

    def __init__(self, period: int = 20, source: str = "close"):
        self.period = period
        self.source = source
        self.name = f"ema_{period}"

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.name] = df[self.source].ewm(
            span=self.period, adjust=False
        ).mean()
        return df

    @property
    def columns(self) -> List[str]:
        return [self.name]

    @property
    def is_overlay(self) -> bool:
        return True
