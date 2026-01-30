"""
ADX (Average Directional Index) インジケーター

+DI / -DI / ADX を計算。Wilder's Smoothing を使用。
"""

from typing import List

import numpy as np
import pandas as pd

from .base import Indicator


class ADX(Indicator):
    """ADX (+DI, -DI, ADX)"""

    name = "adx"

    def __init__(self, period: int = 14):
        self.period = period

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        high = df["high"]
        low = df["low"]
        close = df["close"]

        # +DM / -DM
        up_move = high.diff()
        down_move = -low.diff()

        plus_dm = np.where(
            (up_move > down_move) & (up_move > 0), up_move, 0.0
        )
        minus_dm = np.where(
            (down_move > up_move) & (down_move > 0), down_move, 0.0
        )

        # True Range
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Wilder's Smoothing (期間N)
        atr = self._wilder_smooth(tr, self.period)
        smooth_plus_dm = self._wilder_smooth(
            pd.Series(plus_dm, index=df.index), self.period
        )
        smooth_minus_dm = self._wilder_smooth(
            pd.Series(minus_dm, index=df.index), self.period
        )

        # +DI / -DI
        plus_di = 100 * smooth_plus_dm / atr
        minus_di = 100 * smooth_minus_dm / atr

        # DX → ADX
        di_sum = plus_di + minus_di
        dx = 100 * (plus_di - minus_di).abs() / di_sum.replace(0, np.nan)
        adx = self._wilder_smooth(dx, self.period)

        df[f"plus_di_{self.period}"] = plus_di
        df[f"minus_di_{self.period}"] = minus_di
        df[f"adx_{self.period}"] = adx

        return df

    @staticmethod
    def _wilder_smooth(series: pd.Series, period: int) -> pd.Series:
        """Wilder's Smoothing (EMA with alpha=1/period)"""
        return series.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    @property
    def columns(self) -> List[str]:
        return [
            f"plus_di_{self.period}",
            f"minus_di_{self.period}",
            f"adx_{self.period}",
        ]

    @property
    def is_overlay(self) -> bool:
        return False
