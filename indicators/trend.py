"""
トレンド系インジケーター: SMA, EMA, Parabolic SAR
"""

from typing import List

import numpy as np
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


class ParabolicSAR(Indicator):
    """
    Parabolic SAR (Stop and Reverse)

    トレンドの方向と反転ポイントを示す。
    - SAR < 価格 → 上昇トレンド (sar_trend = 1)
    - SAR > 価格 → 下降トレンド (sar_trend = -1)

    Parameters:
        af_start: 加速係数の初期値 (デフォルト 0.02)
        af_step: 加速係数のステップ (デフォルト 0.02)
        af_max: 加速係数の最大値 (デフォルト 0.2)
    """

    def __init__(
        self, af_start: float = 0.02, af_step: float = 0.02, af_max: float = 0.2
    ):
        self.af_start = af_start
        self.af_step = af_step
        self.af_max = af_max
        self.name = "sar"

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values
        n = len(df)

        sar = np.zeros(n)
        trend = np.zeros(n)  # 1=上昇, -1=下降
        ep = np.zeros(n)  # Extreme Point
        af = np.zeros(n)  # Acceleration Factor

        # 初期化（最初の数本で初期トレンドを決定）
        if n < 2:
            df[self.name] = np.nan
            df["sar_trend"] = 0
            return df

        # 最初のトレンド判定
        if close[1] > close[0]:
            trend[1] = 1  # 上昇
            sar[1] = low[0]
            ep[1] = high[1]
        else:
            trend[1] = -1  # 下降
            sar[1] = high[0]
            ep[1] = low[1]
        af[1] = self.af_start

        # メインループ
        for i in range(2, n):
            prev_sar = sar[i - 1]
            prev_ep = ep[i - 1]
            prev_af = af[i - 1]
            prev_trend = trend[i - 1]

            # SARの計算
            sar[i] = prev_sar + prev_af * (prev_ep - prev_sar)

            # トレンド反転チェック
            if prev_trend == 1:  # 上昇トレンド中
                # SARは直近2本の安値より下に制限
                sar[i] = min(sar[i], low[i - 1], low[i - 2] if i >= 2 else low[i - 1])

                if low[i] < sar[i]:
                    # 反転: 下降トレンドへ
                    trend[i] = -1
                    sar[i] = prev_ep  # EPがSARになる
                    ep[i] = low[i]
                    af[i] = self.af_start
                else:
                    trend[i] = 1
                    if high[i] > prev_ep:
                        ep[i] = high[i]
                        af[i] = min(prev_af + self.af_step, self.af_max)
                    else:
                        ep[i] = prev_ep
                        af[i] = prev_af
            else:  # 下降トレンド中
                # SARは直近2本の高値より上に制限
                sar[i] = max(sar[i], high[i - 1], high[i - 2] if i >= 2 else high[i - 1])

                if high[i] > sar[i]:
                    # 反転: 上昇トレンドへ
                    trend[i] = 1
                    sar[i] = prev_ep  # EPがSARになる
                    ep[i] = high[i]
                    af[i] = self.af_start
                else:
                    trend[i] = -1
                    if low[i] < prev_ep:
                        ep[i] = low[i]
                        af[i] = min(prev_af + self.af_step, self.af_max)
                    else:
                        ep[i] = prev_ep
                        af[i] = prev_af

        df[self.name] = sar
        df["sar_trend"] = trend  # 1=上昇, -1=下降

        return df

    @property
    def columns(self) -> List[str]:
        return [self.name, "sar_trend"]

    @property
    def is_overlay(self) -> bool:
        return True


class DonchianChannel(Indicator):
    """
    Donchian Channel

    過去N本の高値・安値チャネル。タートル流ブレイクアウト戦略で使用。
    - donchian_upper: 過去N本の最高値（ブレイクアウト判定）
    - donchian_lower: 過去N本の最安値（ブレイクダウン判定）
    - donchian_middle: (upper + lower) / 2

    Parameters:
        period: チャネル期間（デフォルト 20）
    """

    def __init__(self, period: int = 20):
        self.period = period
        self.name = f"donchian_{period}"

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        # 過去N本の最高値・最安値（現在足は除外）
        # NOTE: rolling().max()/min()は現在足を含むため、.shift(1)で1本ずらす
        df[f"donchian_upper_{self.period}"] = df["high"].shift(1).rolling(window=self.period).max()
        df[f"donchian_lower_{self.period}"] = df["low"].shift(1).rolling(window=self.period).min()
        df[f"donchian_middle_{self.period}"] = (
            df[f"donchian_upper_{self.period}"] + df[f"donchian_lower_{self.period}"]
        ) / 2.0

        return df

    @property
    def columns(self) -> List[str]:
        return [
            f"donchian_upper_{self.period}",
            f"donchian_lower_{self.period}",
            f"donchian_middle_{self.period}",
        ]

    @property
    def is_overlay(self) -> bool:
        return True
