"""
モメンタム系インジケーター: RSI, MACD, Stochastic
"""

from typing import List

import pandas as pd

from .base import Indicator


class RSI(Indicator):
    """Relative Strength Index"""

    def __init__(self, period: int = 14):
        self.period = period
        self.name = f"rsi_{period}"

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        avg_gain = gain.ewm(
            alpha=1 / self.period, min_periods=self.period
        ).mean()
        avg_loss = loss.ewm(
            alpha=1 / self.period, min_periods=self.period
        ).mean()

        rs = avg_gain / avg_loss
        df[self.name] = 100.0 - (100.0 / (1.0 + rs))
        return df

    @property
    def columns(self) -> List[str]:
        return [self.name]


class MACD(Indicator):
    """Moving Average Convergence Divergence"""

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.fast = fast
        self.slow = slow
        self.signal = signal
        self.name = "macd"

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        ema_fast = df["close"].ewm(span=self.fast, adjust=False).mean()
        ema_slow = df["close"].ewm(span=self.slow, adjust=False).mean()

        df["macd_line"] = ema_fast - ema_slow
        df["macd_signal"] = df["macd_line"].ewm(
            span=self.signal, adjust=False
        ).mean()
        df["macd_histogram"] = df["macd_line"] - df["macd_signal"]
        return df

    @property
    def columns(self) -> List[str]:
        return ["macd_line", "macd_signal", "macd_histogram"]


class Stochastic(Indicator):
    """Stochastic Oscillator"""

    def __init__(self, k_period: int = 14, d_period: int = 3):
        self.k_period = k_period
        self.d_period = d_period
        self.name = "stoch"

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        k = self.k_period
        low_min = df["low"].rolling(window=k).min()
        high_max = df["high"].rolling(window=k).max()

        denom = high_max - low_min
        df[f"stoch_k_{k}"] = 100 * (df["close"] - low_min) / denom.replace(0, float("nan"))
        df[f"stoch_d_{k}"] = df[f"stoch_k_{k}"].rolling(window=self.d_period).mean()
        return df

    @property
    def columns(self) -> List[str]:
        k = self.k_period
        return [f"stoch_k_{k}", f"stoch_d_{k}"]
