"""
ローソク足パターン事前計算インジケーター

やがみ式ノウハウで必要な事前計算カラムをベクトル演算で生成。
- RallyCalc: 直近N本での上昇幅（ReversalHigh条件用）
- WickSpikeCalc: 過去N本の上髭スパイク情報（WickFill条件用）
"""

from typing import List

import numpy as np
import pandas as pd

from .base import Indicator


class RallyCalc(Indicator):
    """直近N本での上昇幅を事前計算

    _rally_N = high - close.shift(N) で急上昇を検出。
    ReversalHighCondition が参照する。
    """

    def __init__(self, lookback: int = 3):
        self.lookback = lookback
        self.name = f"_rally_{lookback}"

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.name] = df["high"] - df["close"].shift(self.lookback)
        return df

    @property
    def columns(self) -> List[str]:
        return [self.name]


class WickSpikeCalc(Indicator):
    """過去N本の上髭スパイク情報を事前計算

    直近N本の中で最も上髭比率が高い足を探し、その高値とbody上端を記録。
    WickFillCondition が参照する。

    出力カラム:
    - _spike_high_N: 上髭スパイク足の高値
    - _spike_body_top_N: 上髭スパイク足の実体上端（= max(open, close)）
    """

    def __init__(self, lookback: int = 3, wick_ratio: float = 1.5):
        self.lookback = lookback
        self.wick_ratio = wick_ratio
        self.name = f"_wick_spike_{lookback}"

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        body_top = np.maximum(df["open"].values, df["close"].values)
        body_bot = np.minimum(df["open"].values, df["close"].values)
        body = body_top - body_bot
        upper_wick = df["high"].values - body_top

        # 上髭が実体のwick_ratio倍以上 → スパイク足
        # body == 0 のときは十字線、上髭があればスパイクとみなす
        is_spike = upper_wick > np.where(body > 0, body * self.wick_ratio, 0)

        # 各行から lookback 本前までの範囲でスパイク足の高値とbody_topを取得
        n = len(df)
        spike_high = np.full(n, np.nan)
        spike_body = np.full(n, np.nan)

        # ベクトル化: shift(1)~shift(lookback) の中でスパイク条件を満たす
        # 最も直近のスパイク足を採用
        for shift in range(1, self.lookback + 1):
            if shift >= n:
                break
            shifted_spike = np.roll(is_spike, shift)
            shifted_spike[:shift] = False
            shifted_high = np.roll(df["high"].values, shift)
            shifted_high[:shift] = np.nan
            shifted_body_top = np.roll(body_top, shift)
            shifted_body_top[:shift] = np.nan

            # まだスパイクが見つかっていない行にのみ上書き
            mask = shifted_spike & np.isnan(spike_high)
            spike_high[mask] = shifted_high[mask]
            spike_body[mask] = shifted_body_top[mask]

        df[f"_spike_high_{self.lookback}"] = spike_high
        df[f"_spike_body_top_{self.lookback}"] = spike_body
        return df

    @property
    def columns(self) -> List[str]:
        return [
            f"_spike_high_{self.lookback}",
            f"_spike_body_top_{self.lookback}",
        ]
