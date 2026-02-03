"""
出来高系インジケーター: VWAP, RelativeVolume, VolumeAnalysis
"""

from typing import List

import numpy as np
import pandas as pd

from .base import Indicator


class VWAP(Indicator):
    """
    Volume Weighted Average Price (日次リセット + 前日VWAP切替)

    Parameters:
        switch_hour: 当日VWAPに切り替えるUTC時間（デフォルト1 = JST 10:00）
                     0-23の範囲。この時間以降は当日VWAP、以前は前日VWAPを使用。

    Output columns:
        vwap: 当日VWAP（日次リセット）
        vwap_prev: 前日の最終VWAP
        vwap_active: 時間帯に応じて切り替えた値（戦略で使用推奨）
    """

    def __init__(self, switch_hour: int = 1):
        self.name = "vwap"
        self.switch_hour = switch_hour  # UTC時間（1 = JST 10:00）

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        typical_price = (df["high"] + df["low"] + df["close"]) / 3

        # 日次でリセット
        if "datetime" in df.columns:
            date = df["datetime"].dt.date
            hour = df["datetime"].dt.hour
        else:
            # datetimeカラムがない場合は累積で計算（切替なし）
            date = pd.Series(0, index=df.index)
            hour = pd.Series(12, index=df.index)  # 常に切替後とみなす

        # groupby + cumsum でベクトル演算（forループ不要）
        tp_vol = typical_price * df["volume"]
        cum_tp_vol = tp_vol.groupby(date).cumsum()
        cum_vol = df["volume"].groupby(date).cumsum()
        df["vwap"] = cum_tp_vol / cum_vol.replace(0, np.nan)

        # 前日VWAPを計算（各日の最終VWAP値を翌日に引き継ぐ）
        # 日ごとの最終VWAPを取得
        daily_last_vwap = df.groupby(date)["vwap"].transform("last")
        # 1日シフトして前日のVWAPとする
        unique_dates = date.unique()
        date_to_prev_vwap = {}
        prev_vwap = np.nan
        for d in sorted(unique_dates):
            date_to_prev_vwap[d] = prev_vwap
            # この日の最終VWAPを次の日のprev_vwapに
            mask = date == d
            if mask.any():
                prev_vwap = df.loc[mask, "vwap"].iloc[-1]

        df["vwap_prev"] = date.map(date_to_prev_vwap)

        # 切替ロジック: UTC hour < switch_hour なら前日VWAP、以降は当日VWAP
        use_current = hour >= self.switch_hour
        df["vwap_active"] = np.where(
            use_current,
            df["vwap"],
            df["vwap_prev"]
        )

        return df

    @property
    def columns(self) -> List[str]:
        return ["vwap", "vwap_prev", "vwap_active"]

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
