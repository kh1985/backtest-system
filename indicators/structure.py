"""
価格構造（HH/LL）によるトレンド検出

ダウ理論ベース:
- 上昇トレンド: HH (Higher High) + HL (Higher Low)
- 下降トレンド: LH (Lower High) + LL (Lower Low)
- トレンド転換: 押し安値/戻り高値のブレイク
"""

from typing import List

import numpy as np
import pandas as pd

from .base import Indicator


class SwingStructure(Indicator):
    """
    スイング構造によるトレンド検出（ダウ理論ベース）

    出力カラム:
    - swing_trend: トレンド方向 (1=上昇, -1=下降, 0=レンジ)
    - swing_high: 直近の戻り高値（下降中に更新）
    - swing_low: 直近の押し安値（上昇中に更新）
    - swing_break: ブレイクシグナル (1=上昇転換, -1=下降転換, 0=なし)

    トレンド判定ロジック:
    - 上昇: HH（Higher High）+ HL（Higher Low）の継続
    - 下降: LH（Lower High）+ LL（Lower Low）の継続
    - レンジ: HH/LL混在（例: 上昇中にLLが出現、でも押し安値ブレイクせず）
    - 転換: 押し安値/戻り高値のブレイク

    Parameters:
        min_swing_bars: スイングポイント認識に必要な最小バー数（デフォルト: 3）
        atr_filter: ATRフィルター倍率（ノイズ除去用、0で無効、デフォルト: 0.5）
    """

    def __init__(self, min_swing_bars: int = 3, atr_filter: float = 0.5):
        self.min_swing_bars = min_swing_bars
        self.atr_filter = atr_filter
        self.name = "swing"

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values
        n = len(df)

        # ATR計算（ノイズフィルター用）
        if self.atr_filter > 0:
            tr = np.maximum(
                high - low,
                np.maximum(
                    np.abs(high - np.roll(close, 1)),
                    np.abs(low - np.roll(close, 1))
                )
            )
            tr[0] = high[0] - low[0]
            atr = pd.Series(tr).rolling(14).mean().values
        else:
            atr = np.zeros(n)

        # 出力配列
        trend = np.zeros(n)  # 1=上昇, -1=下降, 0=不明
        swing_high = np.full(n, np.nan)  # 戻り高値
        swing_low = np.full(n, np.nan)   # 押し安値
        swing_break = np.zeros(n)  # ブレイクシグナル

        # 状態変数
        current_trend = 0
        last_swing_high = np.nan  # 下降中の戻り高値
        last_swing_low = np.nan   # 上昇中の押し安値

        # 初期スイング検出用
        highest_since_low = high[0]
        lowest_since_high = low[0]
        highest_idx = 0
        lowest_idx = 0
        bars_since_high = 0
        bars_since_low = 0

        for i in range(1, n):
            min_move = atr[i] * self.atr_filter if self.atr_filter > 0 and not np.isnan(atr[i]) else 0

            # 高値・安値の更新追跡
            if high[i] > highest_since_low:
                highest_since_low = high[i]
                highest_idx = i
                bars_since_high = 0
            else:
                bars_since_high += 1

            if low[i] < lowest_since_high:
                lowest_since_high = low[i]
                lowest_idx = i
                bars_since_low = 0
            else:
                bars_since_low += 1

            # === 初期トレンド判定（まだトレンドが決まっていない場合）===
            if current_trend == 0:
                # 十分なスイングが形成されたら初期トレンドを決定
                if bars_since_high >= self.min_swing_bars and bars_since_low >= self.min_swing_bars:
                    if highest_idx > lowest_idx:
                        # 安値→高値の順 → 上昇トレンド開始
                        current_trend = 1
                        last_swing_low = lowest_since_high
                        lowest_since_high = low[i]
                        lowest_idx = i
                    else:
                        # 高値→安値の順 → 下降トレンド開始
                        current_trend = -1
                        last_swing_high = highest_since_low
                        highest_since_low = high[i]
                        highest_idx = i

            # === 上昇トレンド中 ===
            elif current_trend == 1:
                # 押し安値を更新（新しいスイングローが形成されたら）
                if bars_since_low >= self.min_swing_bars and lowest_since_high > last_swing_low:
                    # 新しいHL（Higher Low）が形成された
                    last_swing_low = lowest_since_high
                    lowest_since_high = low[i]
                    lowest_idx = i

                # LL出現チェック（ブレイク前の構造変化）
                if bars_since_low >= self.min_swing_bars and lowest_since_high < last_swing_low:
                    # LL（Lower Low）が出現 → まだブレイクしてなければレンジ
                    if close[i] >= last_swing_low - min_move:
                        current_trend = 0  # レンジへ移行
                        swing_break[i] = 0

                # トレンド転換チェック: 押し安値を下抜け
                if close[i] < last_swing_low - min_move:
                    current_trend = -1
                    swing_break[i] = -1  # 下降転換シグナル
                    last_swing_high = highest_since_low
                    highest_since_low = high[i]
                    highest_idx = i

            # === 下降トレンド中 ===
            elif current_trend == -1:
                # 戻り高値を更新（新しいスイングハイが形成されたら）
                if bars_since_high >= self.min_swing_bars and highest_since_low < last_swing_high:
                    # 新しいLH（Lower High）が形成された
                    last_swing_high = highest_since_low
                    highest_since_low = high[i]
                    highest_idx = i

                # HH出現チェック（ブレイク前の構造変化）
                if bars_since_high >= self.min_swing_bars and highest_since_low > last_swing_high:
                    # HH（Higher High）が出現 → まだブレイクしてなければレンジ
                    if close[i] <= last_swing_high + min_move:
                        current_trend = 0  # レンジへ移行
                        swing_break[i] = 0

                # トレンド転換チェック: 戻り高値を上抜け
                if close[i] > last_swing_high + min_move:
                    current_trend = 1
                    swing_break[i] = 1  # 上昇転換シグナル
                    last_swing_low = lowest_since_high
                    lowest_since_high = low[i]
                    lowest_idx = i

            # === レンジ中 ===
            elif current_trend == 0:
                # 戻り高値を上抜けたら上昇トレンドへ
                if not np.isnan(last_swing_high) and close[i] > last_swing_high + min_move:
                    current_trend = 1
                    swing_break[i] = 1
                    last_swing_low = lowest_since_high
                    lowest_since_high = low[i]
                    lowest_idx = i
                # 押し安値を下抜けたら下降トレンドへ
                elif not np.isnan(last_swing_low) and close[i] < last_swing_low - min_move:
                    current_trend = -1
                    swing_break[i] = -1
                    last_swing_high = highest_since_low
                    highest_since_low = high[i]
                    highest_idx = i

            # 結果を記録
            trend[i] = current_trend
            swing_high[i] = last_swing_high
            swing_low[i] = last_swing_low

        df["swing_trend"] = trend.astype(int)
        df["swing_high"] = swing_high
        df["swing_low"] = swing_low
        df["swing_break"] = swing_break.astype(int)

        return df

    @property
    def columns(self) -> List[str]:
        return ["swing_trend", "swing_high", "swing_low", "swing_break"]

    @property
    def is_overlay(self) -> bool:
        return False


class TrendStructure(Indicator):
    """
    シンプル版: スイングブレイクによるトレンド判定

    直近N本の高値/安値をブレイクしたらトレンド転換とみなす。
    HH/LLの厳密な判定ではなく、シンプルなブレイクアウト方式。

    出力カラム:
    - structure_trend: トレンド方向 (1=上昇, -1=下降, 0=レンジ)
    - structure_high: 直近N本の高値
    - structure_low: 直近N本の安値

    Parameters:
        lookback: 高値/安値を見る期間（デフォルト: 20）
    """

    def __init__(self, lookback: int = 20):
        self.lookback = lookback
        self.name = "structure"

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        high = df["high"]
        low = df["low"]
        close = df["close"]

        # N本の高値/安値
        rolling_high = high.rolling(self.lookback).max()
        rolling_low = low.rolling(self.lookback).min()

        # トレンド判定
        # 終値が直近高値を超えたら上昇、直近安値を割ったら下降
        trend = pd.Series(0, index=df.index)

        # 高値ブレイク → 上昇
        trend = trend.where(~(close > rolling_high.shift(1)), 1)
        # 安値ブレイク → 下降
        trend = trend.where(~(close < rolling_low.shift(1)), -1)

        # トレンドを維持（一度決まったら次のブレイクまで継続）
        trend = trend.replace(0, np.nan).ffill().fillna(0)

        df[f"structure_trend_{self.lookback}"] = trend.astype(int)
        df[f"structure_high_{self.lookback}"] = rolling_high
        df[f"structure_low_{self.lookback}"] = rolling_low

        return df

    @property
    def columns(self) -> List[str]:
        return [
            f"structure_trend_{self.lookback}",
            f"structure_high_{self.lookback}",
            f"structure_low_{self.lookback}",
        ]

    @property
    def is_overlay(self) -> bool:
        return False
