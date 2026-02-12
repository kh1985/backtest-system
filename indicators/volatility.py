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
        p = self.period
        sma = df["close"].rolling(window=p).mean()
        std = df["close"].rolling(window=p).std()

        df[f"bb_upper_{p}"] = sma + (std * self.std_dev)
        df[f"bb_middle_{p}"] = sma
        df[f"bb_lower_{p}"] = sma - (std * self.std_dev)
        df[f"bb_width_{p}"] = (df[f"bb_upper_{p}"] - df[f"bb_lower_{p}"]) / df[f"bb_middle_{p}"]
        return df

    @property
    def columns(self) -> List[str]:
        p = self.period
        return [f"bb_upper_{p}", f"bb_middle_{p}", f"bb_lower_{p}", f"bb_width_{p}"]

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


class SuperTrend(Indicator):
    """SuperTrend インジケーター

    ATR適応型トレンド指標。価格がSuperTrendの上にある場合は上昇トレンド、
    下にある場合は下降トレンドと判定。

    基本計算式:
    - basic_ub = (high + low) / 2 + multiplier × ATR
    - basic_lb = (high + low) / 2 - multiplier × ATR

    SuperTrendは前足のトレンド状態に応じて上下バンドが切り替わる。
    """

    def __init__(self, period: int = 10, multiplier: float = 3.0):
        self.period = period
        self.multiplier = multiplier
        self.name = f"supertrend_{period}_{multiplier}"

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        import numpy as np

        # ATRを計算（存在しない場合）
        atr_col = f"atr_{self.period}"
        if atr_col not in df.columns:
            high_low = df["high"] - df["low"]
            high_close = (df["high"] - df["close"].shift()).abs()
            low_close = (df["low"] - df["close"].shift()).abs()
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df[atr_col] = true_range.ewm(alpha=1 / self.period, min_periods=self.period).mean()

        # 基本的な上下バンドを計算
        hl2 = (df["high"] + df["low"]) / 2
        basic_ub = hl2 + self.multiplier * df[atr_col]
        basic_lb = hl2 - self.multiplier * df[atr_col]

        # SuperTrendの計算
        final_ub = basic_ub.copy()
        final_lb = basic_lb.copy()
        supertrend = pd.Series(index=df.index, dtype=float)
        direction = pd.Series(index=df.index, dtype=int)

        # 初期値: SuperTrend = basic_ub, direction = 1 (uptrend)
        if len(df) > 0:
            supertrend.iloc[0] = basic_ub.iloc[0]
            direction.iloc[0] = 1

        # ループで計算
        for i in range(1, len(df)):
            # NaNチェック
            if pd.isna(basic_ub.iloc[i]) or pd.isna(basic_lb.iloc[i]):
                supertrend.iloc[i] = np.nan
                direction.iloc[i] = direction.iloc[i-1] if i > 0 else 1
                continue

            # 上バンドの更新
            if pd.isna(final_ub.iloc[i-1]):
                final_ub.iloc[i] = basic_ub.iloc[i]
            elif basic_ub.iloc[i] < final_ub.iloc[i-1] or df["close"].iloc[i-1] > final_ub.iloc[i-1]:
                final_ub.iloc[i] = basic_ub.iloc[i]
            else:
                final_ub.iloc[i] = final_ub.iloc[i-1]

            # 下バンドの更新
            if pd.isna(final_lb.iloc[i-1]):
                final_lb.iloc[i] = basic_lb.iloc[i]
            elif basic_lb.iloc[i] > final_lb.iloc[i-1] or df["close"].iloc[i-1] < final_lb.iloc[i-1]:
                final_lb.iloc[i] = basic_lb.iloc[i]
            else:
                final_lb.iloc[i] = final_lb.iloc[i-1]

            # トレンド方向とSuperTrendの決定
            if direction.iloc[i-1] == 1:  # 前足がuptrend
                if df["close"].iloc[i] <= final_lb.iloc[i]:
                    supertrend.iloc[i] = final_ub.iloc[i]
                    direction.iloc[i] = -1
                else:
                    supertrend.iloc[i] = final_lb.iloc[i]
                    direction.iloc[i] = 1
            else:  # 前足がdowntrend
                if df["close"].iloc[i] >= final_ub.iloc[i]:
                    supertrend.iloc[i] = final_lb.iloc[i]
                    direction.iloc[i] = 1
                else:
                    supertrend.iloc[i] = final_ub.iloc[i]
                    direction.iloc[i] = -1

        df[self.name] = supertrend
        df[f"{self.name}_direction"] = direction

        return df

    @property
    def columns(self) -> List[str]:
        return [self.name, f"{self.name}_direction"]

    @property
    def is_overlay(self) -> bool:
        return True
