"""
トレンドレジーム検出

上位タイムフレーム（1h/15m等）のOHLCVから
Uptrend / Downtrend / Range の3レジームに分類。

MA Cross方式とADX方式をサポート。
ルックアヘッドバイアス防止のため、pd.merge_asofでforward-fill。
"""

from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd


class TrendRegime(Enum):
    """トレンドレジーム"""
    UPTREND = "uptrend"
    DOWNTREND = "downtrend"
    RANGE = "range"


class TrendDetector:
    """トレンドレジーム検出器"""

    def detect_ma_cross(
        self,
        df: pd.DataFrame,
        fast_period: int = 20,
        slow_period: int = 50,
        range_threshold_pct: float = 0.1,
    ) -> pd.DataFrame:
        """
        MA Cross方式でトレンドを判定

        - SMA_fast > SMA_slow → Uptrend
        - SMA_fast < SMA_slow → Downtrend
        - |SMA_fast - SMA_slow| / close < threshold → Range

        Args:
            df: OHLCV DataFrame (datetime, close必須)
            fast_period: 短期MA期間
            slow_period: 長期MA期間
            range_threshold_pct: レンジ判定の閾値（%）

        Returns:
            DataFrame with 'trend_regime' column added
        """
        df = df.copy()
        close = df["close"]

        sma_fast = close.rolling(window=fast_period).mean()
        sma_slow = close.rolling(window=slow_period).mean()

        diff_pct = ((sma_fast - sma_slow) / close * 100).abs()

        regime = pd.Series(TrendRegime.RANGE.value, index=df.index)
        regime[sma_fast > sma_slow] = TrendRegime.UPTREND.value
        regime[sma_fast < sma_slow] = TrendRegime.DOWNTREND.value
        regime[diff_pct < range_threshold_pct] = TrendRegime.RANGE.value

        df["trend_regime"] = regime
        df["trend_sma_fast"] = sma_fast
        df["trend_sma_slow"] = sma_slow

        return df

    def detect_adx(
        self,
        df: pd.DataFrame,
        adx_period: int = 14,
        trend_threshold: float = 25.0,
        range_threshold: float = 20.0,
    ) -> pd.DataFrame:
        """
        ADX方式でトレンドを判定

        - ADX > trend_threshold → Trending (+DI > -DI = Up, 逆 = Down)
        - ADX < range_threshold → Range
        - それ以外 → 前回の値を維持

        Args:
            df: OHLCV DataFrame
            adx_period: ADX期間
            trend_threshold: トレンド判定閾値
            range_threshold: レンジ判定閾値

        Returns:
            DataFrame with 'trend_regime' column added
        """
        from indicators.adx import ADX as ADXIndicator

        df = df.copy()
        adx_ind = ADXIndicator(period=adx_period)
        df = adx_ind.calculate(df)

        adx_col = f"adx_{adx_period}"
        plus_di_col = f"plus_di_{adx_period}"
        minus_di_col = f"minus_di_{adx_period}"

        adx_vals = df[adx_col]
        plus_di = df[plus_di_col]
        minus_di = df[minus_di_col]

        regime = pd.Series(TrendRegime.RANGE.value, index=df.index)

        # ADX > trend_threshold → トレンド（DI方向で判定）
        trending = adx_vals > trend_threshold
        regime[trending & (plus_di > minus_di)] = TrendRegime.UPTREND.value
        regime[trending & (plus_di <= minus_di)] = TrendRegime.DOWNTREND.value

        # ADX < range_threshold → レンジ
        regime[adx_vals < range_threshold] = TrendRegime.RANGE.value

        df["trend_regime"] = regime

        return df

    def detect_combined(
        self,
        df: pd.DataFrame,
        ma_fast: int = 20,
        ma_slow: int = 50,
        adx_period: int = 14,
        adx_trend_threshold: float = 25.0,
        adx_range_threshold: float = 20.0,
    ) -> pd.DataFrame:
        """
        MA CrossとADXを組み合わせた判定

        - MA Crossで方向を判定
        - ADXでトレンドの強さを確認
        - ADX < range_threshold → Range（MA Crossの結果を上書き）

        Returns:
            DataFrame with 'trend_regime' column added
        """
        df = self.detect_ma_cross(df, ma_fast, ma_slow)
        ma_regime = df["trend_regime"].copy()

        from indicators.adx import ADX as ADXIndicator
        adx_ind = ADXIndicator(period=adx_period)
        df = adx_ind.calculate(df)

        adx_col = f"adx_{adx_period}"
        adx_vals = df[adx_col]

        # ADXが低い場合はRangeに上書き
        df["trend_regime"] = ma_regime
        df.loc[adx_vals < adx_range_threshold, "trend_regime"] = TrendRegime.RANGE.value

        return df

    @staticmethod
    def label_execution_tf(
        exec_df: pd.DataFrame,
        htf_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        上位TFのトレンドラベルを実行TFにforward-fill

        ルックアヘッドバイアス防止:
        上位TFのバーが確定した時点のラベルを、次の上位TFバーまでの
        全実行TFバーに割り当てる。

        Args:
            exec_df: 実行TFのDataFrame (datetime必須)
            htf_df: 上位TFのDataFrame (datetime, trend_regime必須)

        Returns:
            exec_df に trend_regime カラムが追加されたDataFrame
        """
        exec_df = exec_df.copy()
        htf_labels = htf_df[["datetime", "trend_regime"]].copy()
        htf_labels = htf_labels.sort_values("datetime")

        exec_df = exec_df.sort_values("datetime")

        # merge_asof: 実行TFの各バーに、直近の確定済みHTFラベルを割り当て
        merged = pd.merge_asof(
            exec_df[["datetime"]],
            htf_labels,
            on="datetime",
            direction="backward",
        )

        exec_df["trend_regime"] = merged["trend_regime"].values

        # NaN（HTFデータ開始前）はRangeで埋める
        exec_df["trend_regime"] = exec_df["trend_regime"].fillna(
            TrendRegime.RANGE.value
        )

        return exec_df
