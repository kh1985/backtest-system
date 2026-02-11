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

    @staticmethod
    def _apply_no_lookahead_shift(
        regime: pd.Series,
        neutral_value: str = TrendRegime.RANGE.value,
    ) -> pd.Series:
        # 1バーシフト: HTFバーのclose確定後に初めてレジームが利用可能
        return regime.shift(1).fillna(neutral_value)

    def detect_ma_cross(
        self,
        df: pd.DataFrame,
        fast_period: int = 20,
        slow_period: int = 50,
        range_threshold_pct: float = 0.1,
        apply_shift: bool = True,
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

        regime_series = pd.Series(regime, index=df.index)
        if apply_shift:
            regime_series = self._apply_no_lookahead_shift(regime_series)

        df["trend_regime"] = regime_series
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

        regime_series = self._apply_no_lookahead_shift(regime)
        df["trend_regime"] = regime_series

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
        df = self.detect_ma_cross(df, ma_fast, ma_slow, apply_shift=False)
        ma_regime = df["trend_regime"].copy()

        from indicators.adx import ADX as ADXIndicator
        adx_ind = ADXIndicator(period=adx_period)
        df = adx_ind.calculate(df)

        adx_col = f"adx_{adx_period}"
        adx_vals = df[adx_col]

        # ADXが低い場合はRangeに上書き
        df["trend_regime"] = ma_regime
        df.loc[adx_vals < adx_range_threshold, "trend_regime"] = TrendRegime.RANGE.value
        df["trend_regime"] = self._apply_no_lookahead_shift(df["trend_regime"])

        return df

    def detect_dual_tf_ema(
        self,
        htf_df: pd.DataFrame,
        super_htf_df: pd.DataFrame,
        fast_period: int = 20,
        slow_period: int = 50,
    ) -> pd.DataFrame:
        """
        Dual-TF EMA方式でトレンドを判定

        4h と 1h の EMA fast/slow の方向が一致した場合のみトレンド。
        不一致はすべて Range。

        - 4h up AND 1h up → Uptrend
        - 4h down AND 1h down → Downtrend
        - それ以外 → Range

        Args:
            htf_df: 1h OHLCV DataFrame (datetime, close必須)
            super_htf_df: 4h OHLCV DataFrame (datetime, close必須)
            fast_period: 短期EMA期間
            slow_period: 長期EMA期間

        Returns:
            htf_df に 'trend_regime' カラムが追加されたDataFrame
        """
        htf_df = htf_df.copy()
        super_htf_df = super_htf_df.copy()

        # --- 4h: EMA計算 & 方向判定 ---
        s_close = super_htf_df["close"]
        s_ema_fast = s_close.ewm(span=fast_period, adjust=False).mean()
        s_ema_slow = s_close.ewm(span=slow_period, adjust=False).mean()
        # up=1, down=-1
        super_dir = np.where(s_ema_fast > s_ema_slow, 1, -1)
        super_htf_df["_trend_dir"] = super_dir
        # 4hバーのclose確定後に初めて方向が利用可能（merge_asof前にシフト）
        super_htf_df["_trend_dir"] = super_htf_df["_trend_dir"].shift(1)

        # --- 1h: EMA計算 & 方向判定 ---
        h_close = htf_df["close"]
        h_ema_fast = h_close.ewm(span=fast_period, adjust=False).mean()
        h_ema_slow = h_close.ewm(span=slow_period, adjust=False).mean()
        htf_dir = np.where(h_ema_fast > h_ema_slow, 1, -1)
        htf_df["_trend_dir"] = htf_dir

        # --- 4h方向を1hにforward-fill (merge_asof) ---
        super_labels = super_htf_df[["datetime", "_trend_dir"]].copy()
        super_labels = super_labels.rename(columns={"_trend_dir": "_super_dir"})
        super_labels = super_labels.sort_values("datetime")

        htf_df = htf_df.sort_values("datetime").reset_index(drop=True)
        merged = pd.merge_asof(
            htf_df[["datetime"]],
            super_labels,
            on="datetime",
            direction="backward",
        )
        htf_df["_super_dir"] = merged["_super_dir"].values

        # --- レジーム判定: 両TF一致 → トレンド, 不一致 → Range ---
        htf_dir_col = htf_df["_trend_dir"].values
        super_dir_col = htf_df["_super_dir"].values

        regime = np.full(len(htf_df), TrendRegime.RANGE.value, dtype=object)
        both_up = (htf_dir_col == 1) & (super_dir_col == 1)
        both_down = (htf_dir_col == -1) & (super_dir_col == -1)
        regime[both_up] = TrendRegime.UPTREND.value
        regime[both_down] = TrendRegime.DOWNTREND.value

        regime_series = pd.Series(regime, index=htf_df.index)
        htf_df["trend_regime"] = self._apply_no_lookahead_shift(regime_series)
        htf_df["trend_ema_fast"] = h_ema_fast
        htf_df["trend_ema_slow"] = h_ema_slow

        # 一時カラム削除
        htf_df.drop(columns=["_trend_dir", "_super_dir"], inplace=True)

        return htf_df

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
