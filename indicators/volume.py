"""
出来高系インジケーター: VWAP, RelativeVolume, VolumeAnalysis
"""

from typing import List

import numpy as np
import pandas as pd

from .base import Indicator


class VWAP(Indicator):
    """Volume Weighted Average Price (日次リセット)"""

    def __init__(self):
        self.name = "vwap"

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        typical_price = (df["high"] + df["low"] + df["close"]) / 3

        # 日次でリセット
        if "datetime" in df.columns:
            date = df["datetime"].dt.date
        else:
            # datetimeカラムがない場合は累積で計算
            date = pd.Series(0, index=df.index)

        # groupby + cumsum でベクトル演算（forループ不要）
        tp_vol = typical_price * df["volume"]
        cum_tp_vol = tp_vol.groupby(date).cumsum()
        cum_vol = df["volume"].groupby(date).cumsum()
        df["vwap"] = cum_tp_vol / cum_vol.replace(0, np.nan)
        return df

    @property
    def columns(self) -> List[str]:
        return ["vwap"]

    @property
    def is_overlay(self) -> bool:
        return True


class RelativeVolume(Indicator):
    """相対出来高 (RVOL)"""

    def __init__(self, period: int = 20):
        self.period = period
        self.name = f"rvol_{period}"

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        avg_vol = df["volume"].rolling(window=self.period).mean()
        df[self.name] = df["volume"] / avg_vol.replace(0, float("nan"))
        return df

    @property
    def columns(self) -> List[str]:
        return [self.name]


class VolumeAnalysis(Indicator):
    """出来高分析（売買圧力分類）"""

    def __init__(self):
        self.name = "vol_analysis"

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        df["is_bullish"] = df["close"] > df["open"]
        df["is_bearish"] = df["close"] < df["open"]

        candle_range = (df["high"] - df["low"]).replace(0, float("nan"))
        df["body_pct"] = (df["close"] - df["open"]).abs() / candle_range
        return df

    @property
    def columns(self) -> List[str]:
        return ["is_bullish", "is_bearish", "body_pct"]
