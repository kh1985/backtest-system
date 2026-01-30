"""
ボラティリティ系インジケーター: Bollinger Bands, ATR
"""

from typing import List

import pandas as pd

from .base import Indicator


class BollingerBands(Indicator):
    """ボリンジャーバンド"""

    def __init__(self, period: int = 20, std_dev: float = 2.0):
        self.period = period
        self.std_dev = std_dev
        self.name = "bb"

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        sma = df["close"].rolling(window=self.period).mean()
        std = df["close"].rolling(window=self.period).std()

        df["bb_upper"] = sma + (std * self.std_dev)
        df["bb_middle"] = sma
        df["bb_lower"] = sma - (std * self.std_dev)
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
        return df

    @property
    def columns(self) -> List[str]:
        return ["bb_upper", "bb_middle", "bb_lower", "bb_width"]

    @property
    def is_overlay(self) -> bool:
        return True


class ATR(Indicator):
    """Average True Range"""

    def __init__(self, period: int = 14):
        self.period = period
        self.name = f"atr_{period}"

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()

        true_range = pd.concat(
            [high_low, high_close, low_close], axis=1
        ).max(axis=1)

        df[self.name] = true_range.ewm(
            alpha=1 / self.period, min_periods=self.period
        ).mean()
        return df

    @property
    def columns(self) -> List[str]:
        return [self.name]
