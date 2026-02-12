"""
条件ブロック（Condition Building Blocks）

戦略のエントリー条件を構成する再利用可能なパーツ。
"""

from abc import ABC, abstractmethod
from typing import List

import pandas as pd


class Condition(ABC):
    """条件の抽象基底クラス"""

    @abstractmethod
    def evaluate(
        self, row: pd.Series, prev_row: pd.Series = None
    ) -> bool:
        ...

    @abstractmethod
    def describe(self) -> str:
        ...


class ThresholdCondition(Condition):
    """閾値条件: column operator value (例: rsi_14 < 30)"""

    OPERATORS = {
        ">": lambda a, b: a > b,
        "<": lambda a, b: a < b,
        ">=": lambda a, b: a >= b,
        "<=": lambda a, b: a <= b,
        "==": lambda a, b: a == b,
    }

    def __init__(self, column: str, operator: str, value: float):
        if operator not in self.OPERATORS:
            raise ValueError(f"Unknown operator: {operator}")
        self.column = column
        self.operator = operator
        self.value = value

    def evaluate(self, row, prev_row=None) -> bool:
        col_val = row.get(self.column)
        if col_val is None or pd.isna(col_val):
            return False
        return self.OPERATORS[self.operator](col_val, self.value)

    def describe(self) -> str:
        return f"{self.column} {self.operator} {self.value}"


class CrossoverCondition(Condition):
    """クロスオーバー条件: fast_col crosses above/below slow_col"""

    def __init__(
        self, fast_col: str, slow_col: str, direction: str = "above"
    ):
        if direction not in ("above", "below"):
            raise ValueError(f"direction must be 'above' or 'below'")
        self.fast_col = fast_col
        self.slow_col = slow_col
        self.direction = direction

    def evaluate(self, row, prev_row=None) -> bool:
        if prev_row is None:
            return False
        prev_fast = prev_row.get(self.fast_col)
        prev_slow = prev_row.get(self.slow_col)
        cur_fast = row.get(self.fast_col)
        cur_slow = row.get(self.slow_col)

        if any(pd.isna(v) for v in [prev_fast, prev_slow, cur_fast, cur_slow] if v is not None):
            return False

        if self.direction == "above":
            return prev_fast <= prev_slow and cur_fast > cur_slow
        else:
            return prev_fast >= prev_slow and cur_fast < cur_slow

    def describe(self) -> str:
        return f"{self.fast_col} crosses {self.direction} {self.slow_col}"


class CandlePatternCondition(Condition):
    """ローソク足パターン条件: bearish / bullish"""

    def __init__(self, pattern: str):
        if pattern not in ("bearish", "bullish"):
            raise ValueError(f"pattern must be 'bearish' or 'bullish'")
        self.pattern = pattern

    def evaluate(self, row, prev_row=None) -> bool:
        if self.pattern == "bearish":
            return row["close"] < row["open"]
        return row["close"] > row["open"]

    def describe(self) -> str:
        return f"candle is {self.pattern}"


class ColumnCompareCondition(Condition):
    """カラム間比較条件: column_a operator column_b (例: plus_di_14 > minus_di_14)"""

    OPERATORS = {
        ">": lambda a, b: a > b,
        "<": lambda a, b: a < b,
        ">=": lambda a, b: a >= b,
        "<=": lambda a, b: a <= b,
        "==": lambda a, b: a == b,
    }

    def __init__(self, column_a: str, operator: str, column_b: str):
        if operator not in self.OPERATORS:
            raise ValueError(f"Unknown operator: {operator}")
        self.column_a = column_a
        self.operator = operator
        self.column_b = column_b

    def evaluate(self, row, prev_row=None) -> bool:
        val_a = row.get(self.column_a)
        val_b = row.get(self.column_b)
        if val_a is None or val_b is None:
            return False
        if pd.isna(val_a) or pd.isna(val_b):
            return False
        return self.OPERATORS[self.operator](val_a, val_b)

    def describe(self) -> str:
        return f"{self.column_a} {self.operator} {self.column_b}"


class BBSqueezeCondition(Condition):
    """ボリンジャーバンドスクイーズ条件: bandwidth < threshold"""

    def __init__(self, squeeze_threshold: float, bb_period: int = 20):
        self.squeeze_threshold = squeeze_threshold
        self.bb_period = bb_period

    def evaluate(self, row, prev_row=None) -> bool:
        bb_upper = row.get(f"bb_upper_{self.bb_period}")
        bb_lower = row.get(f"bb_lower_{self.bb_period}")
        bb_middle = row.get(f"bb_middle_{self.bb_period}")

        if any(v is None or pd.isna(v) for v in [bb_upper, bb_lower, bb_middle]):
            return False

        if bb_middle == 0:
            return False

        bandwidth = (bb_upper - bb_lower) / bb_middle
        return bandwidth < self.squeeze_threshold

    def describe(self) -> str:
        return f"BB bandwidth < {self.squeeze_threshold}"


class VolumeCondition(Condition):
    """出来高条件: volume >= avg_volume * multiplier"""

    def __init__(self, volume_mult: float, volume_period: int = 20):
        self.volume_mult = volume_mult
        self.volume_period = volume_period

    def evaluate(self, row, prev_row=None) -> bool:
        volume = row.get("volume")
        avg_volume = row.get(f"volume_sma_{self.volume_period}")

        if any(v is None or pd.isna(v) for v in [volume, avg_volume]):
            return False

        if avg_volume == 0:
            return False

        return volume >= avg_volume * self.volume_mult

    def describe(self) -> str:
        return f"volume >= avg * {self.volume_mult}"


class EMAStateCondition(Condition):
    """EMA状態条件: EMA fast vs EMA slow の大小関係"""

    def __init__(self, fast_period: int, slow_period: int, direction: str):
        if direction not in ("above", "below"):
            raise ValueError(f"direction must be 'above' or 'below'")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.direction = direction

    def evaluate(self, row, prev_row=None) -> bool:
        ema_fast = row.get(f"ema_{self.fast_period}")
        ema_slow = row.get(f"ema_{self.slow_period}")

        if any(v is None or pd.isna(v) for v in [ema_fast, ema_slow]):
            return False

        if self.direction == "above":
            return ema_fast > ema_slow
        else:
            return ema_fast < ema_slow

    def describe(self) -> str:
        return f"EMA({self.fast_period}) {self.direction} EMA({self.slow_period})"


class TrapGridCondition(Condition):
    """
    トラップグリッド条件

    レンジ内に等間隔でトラップを配置し、価格が接触したらエントリー。
    トラリピ（トラップリピートイフダン）戦略用。
    """

    def __init__(
        self,
        trap_interval_pct: float,
        range_source: str = "bb",
        range_low: float = None,
        range_high: float = None,
        side: str = "long",
        bb_period: int = 20,
    ):
        self.trap_interval_pct = trap_interval_pct
        self.range_source = range_source
        self.range_low = range_low
        self.range_high = range_high
        self.side = side
        self.bb_period = bb_period
        self.traps = []

    def _calculate_traps(self, row: pd.Series):
        """現在のレンジ範囲からトラップ価格リストを計算"""
        import numpy as np

        if self.range_source == "bb":
            low = row.get(f"bb_lower_{self.bb_period}")
            high = row.get(f"bb_upper_{self.bb_period}")
            if low is None or high is None or pd.isna(low) or pd.isna(high):
                return []
        elif self.range_source == "fixed":
            if self.range_low is None or self.range_high is None:
                return []
            low = self.range_low
            high = self.range_high
        elif self.range_source == "atr":
            center = row.get("close")
            atr = row.get("atr")
            if center is None or atr is None or pd.isna(center) or pd.isna(atr):
                return []
            width = atr * 2.0
            low = center - width / 2
            high = center + width / 2
        else:
            return []

        if low >= high:
            return []

        interval = (high - low) * self.trap_interval_pct / 100
        if interval <= 0:
            return []

        traps = np.arange(low, high + interval, interval)
        return traps.tolist()

    def evaluate(self, row, prev_row=None) -> bool:
        """価格がいずれかのトラップに接触したか判定"""
        if prev_row is None:
            return False

        self.traps = self._calculate_traps(row)
        if not self.traps:
            return False

        close = row.get("close")
        low = row.get("low")
        high = row.get("high")
        prev_close = prev_row.get("close")

        if any(v is None or pd.isna(v) for v in [close, low, high, prev_close]):
            return False

        for trap in self.traps:
            if self.side == "long":
                # 価格がトラップを下から上に通過、またはbar内で接触
                if prev_close < trap <= close:
                    return True
                if low <= trap <= high:
                    return True
            else:  # short
                # 価格がトラップを上から下に通過
                if prev_close > trap >= close:
                    return True
                if low <= trap <= high:
                    return True

        return False

    def describe(self) -> str:
        return f"TrapGrid({self.range_source}, interval={self.trap_interval_pct}%)"


class TimeBasedCondition(Condition):
    """時間帯フィルター条件: start_hour <= JST時刻 < end_hour"""

    def __init__(self, start_hour: int, end_hour: int):
        if not (0 <= start_hour <= 23):
            raise ValueError(f"start_hour must be 0-23, got {start_hour}")
        if not (0 <= end_hour <= 23):
            raise ValueError(f"end_hour must be 0-23, got {end_hour}")
        self.start_hour = start_hour
        self.end_hour = end_hour

    def evaluate(self, row, prev_row=None) -> bool:
        # datetime または timestamp カラムを取得（両方に対応）
        timestamp = row.get("timestamp") or row.get("datetime")
        if timestamp is None or pd.isna(timestamp):
            return False

        # UTC → JST変換（UTC+9）
        jst_time = timestamp + pd.Timedelta(hours=9)
        current_hour = jst_time.hour

        # 日付跨ぎ対応（例: start=14, end=0 → 14:00-23:59）
        if self.start_hour <= self.end_hour:
            # 通常範囲（例: start=9, end=15 → 9:00-14:59）
            return self.start_hour <= current_hour < self.end_hour
        else:
            # 日付跨ぎ範囲（例: start=14, end=1 → 14:00-23:59 + 0:00-0:59）
            return current_hour >= self.start_hour or current_hour < self.end_hour

    def describe(self) -> str:
        if self.start_hour <= self.end_hour:
            return f"JST時刻が{self.start_hour}時～{self.end_hour}時未満"
        else:
            return f"JST時刻が{self.start_hour}時～翌{self.end_hour}時未満"


class RSIConnorsCondition(Condition):
    """
    RSI(2) Connors式ショート条件

    Larry Connors氏のRSI(2)戦略:
    - ベアラリーの頂点（RSI(2)が高値圏）でショートエントリー
    - SMA(200)下でトレンド方向確認
    """

    def __init__(self, sma_period: int = 200, rsi_threshold: int = 95):
        self.sma_period = sma_period
        self.rsi_threshold = rsi_threshold

    def evaluate(self, row, prev_row=None) -> bool:
        close = row.get("close")
        sma = row.get(f"sma_{self.sma_period}")
        rsi_2 = row.get("rsi_2")

        if any(v is None or pd.isna(v) for v in [close, sma, rsi_2]):
            return False

        # close < SMA(200): 下降トレンド確認
        # RSI(2) > threshold: ベアラリー頂点（売られすぎではなく買われすぎ）
        return close < sma and rsi_2 > self.rsi_threshold

    def describe(self) -> str:
        return f"RSI(2) Connors: close < SMA({self.sma_period}) AND RSI(2) > {self.rsi_threshold}"


class DonchianCondition(Condition):
    """
    Donchian Channel ブレイクダウン条件

    close が過去N本の最安値を下回ったらショートエントリー（タートル流）
    """

    def __init__(self, period: int = 20):
        """
        Args:
            period: Donchian Channel の期間（デフォルト: 20）
        """
        self.period = period

    def evaluate(self, row, prev_row=None) -> bool:
        """
        close が donchian_lower（過去N本の最安値）を下回っているかチェック

        Args:
            row: 現在行
            prev_row: 前行（未使用）

        Returns:
            bool: ブレイクダウン条件を満たすか
        """
        col_name = f"donchian_lower_{self.period}"
        donchian_lower = row.get(col_name)
        close = row.get("close")

        if donchian_lower is None or close is None or pd.isna(donchian_lower) or pd.isna(close):
            return False

        return close < donchian_lower

    def describe(self) -> str:
        return f"close < Donchian Lower({self.period})"


class TSMOMCondition(Condition):
    """
    TSMOM (Time Series Momentum) 条件

    ROC (Rate of Change) が閾値より低い場合にショートシグナル。
    AQR「クライシスアルファ」戦略の基本コンセプト。
    """

    def __init__(self, roc_period: int = 30, threshold: float = 0.0):
        self.roc_period = roc_period
        self.threshold = threshold

    def evaluate(self, row, prev_row=None) -> bool:
        roc = row.get(f"roc_{self.roc_period}")
        if roc is None or pd.isna(roc):
            return False
        return roc < self.threshold

    def describe(self) -> str:
        return f"ROC({self.roc_period}) < {self.threshold}"


class SuperTrendCondition(Condition):
    """
    SuperTrend 条件

    ATR適応型トレンド指標。close < supertrend でダウントレンド判定（ショート条件）、
    close > supertrend でアップトレンド判定（ロング条件）。

    パラメータ:
    - period: ATR計算期間（デフォルト: 10）
    - multiplier: ATRの乗数（デフォルト: 3.0）
    - direction: "below"（ショート）または "above"（ロング）
    """

    def __init__(self, period: int = 10, multiplier: float = 3.0, direction: str = "below"):
        if direction not in ("above", "below"):
            raise ValueError(f"direction must be 'above' or 'below', got {direction}")
        self.period = period
        self.multiplier = multiplier
        self.direction = direction

    def evaluate(self, row, prev_row=None) -> bool:
        # SuperTrendインジケーターのカラム名を構築
        st_col = f"supertrend_{self.period}_{self.multiplier}"
        supertrend = row.get(st_col)
        close = row.get("close")

        if supertrend is None or close is None:
            return False
        if pd.isna(supertrend) or pd.isna(close):
            return False

        if self.direction == "below":
            return close < supertrend
        else:
            return close > supertrend

    def describe(self) -> str:
        direction_str = "下" if self.direction == "below" else "上"
        return f"close が SuperTrend({self.period}, {self.multiplier}) の{direction_str}"


class CompoundCondition(Condition):
    """複合条件: AND / OR"""

    def __init__(self, conditions: List[Condition], logic: str = "and"):
        if logic not in ("and", "or"):
            raise ValueError(f"logic must be 'and' or 'or'")
        self.conditions = conditions
        self.logic = logic

    def evaluate(self, row, prev_row=None) -> bool:
        results = [c.evaluate(row, prev_row) for c in self.conditions]
        if self.logic == "and":
            return all(results)
        return any(results)

    def describe(self) -> str:
        joiner = " AND " if self.logic == "and" else " OR "
        parts = [c.describe() for c in self.conditions]
        return f"({joiner.join(parts)})"


class MultiLayerVolumeSpikeCondition(Condition):
    """多層Volume Spike検出（3層フィルター）

    Layer 1: Volume Spike（volume >= avg × threshold）
    Layer 2: Price Drop（価格下落 >= threshold%）
    Layer 3: Consecutive Bearish（連続陰線）
    """

    def __init__(
        self,
        spike_threshold: float = 2.5,
        price_drop_pct: float = 2.0,
        consecutive_bars: int = 2,
        volume_period: int = 20,
    ):
        self.spike_threshold = spike_threshold
        self.price_drop_pct = price_drop_pct
        self.consecutive_bars = consecutive_bars
        self.volume_period = volume_period

    def evaluate(self, row, prev_row=None) -> bool:
        # Layer 1: Volume Spike
        vol_avg_col = f"volume_sma_{self.volume_period}"
        vol_avg = row.get(vol_avg_col)
        volume = row.get("volume")

        if vol_avg is None or volume is None or pd.isna(vol_avg) or pd.isna(volume):
            return False

        if volume < vol_avg * self.spike_threshold:
            return False

        # Layer 2: Price Drop（陰線で下落率チェック）
        open_price = row.get("open")
        close_price = row.get("close")

        if open_price is None or close_price is None:
            return False
        if pd.isna(open_price) or pd.isna(close_price):
            return False
        if open_price == 0:
            return False

        # 陰線であることを確認
        if close_price >= open_price:
            return False

        price_drop = (close_price - open_price) / open_price * 100
        if price_drop >= -self.price_drop_pct:
            return False

        # Layer 3: Consecutive Bearish（連続陰線チェック）
        # consecutive_bars=1の場合は現在のbarのみ（既に陰線確認済み）
        if self.consecutive_bars == 1:
            return True

        # consecutive_bars >= 2の場合、過去のbarもチェック
        # 注: DataFrameレベルでの事前計算が必要
        # ここでは簡易版として、prev_rowのみチェック
        if self.consecutive_bars >= 2 and prev_row is not None:
            prev_open = prev_row.get("open")
            prev_close = prev_row.get("close")
            if prev_open is None or prev_close is None:
                return False
            if pd.isna(prev_open) or pd.isna(prev_close):
                return False
            if prev_close >= prev_open:
                return False

        return True

    def describe(self) -> str:
        return (
            f"MultiLayerVolumeSpike(spike={self.spike_threshold}x, "
            f"drop={self.price_drop_pct}%, bars={self.consecutive_bars})"
        )


class VolumeAccelerationCondition(Condition):
    """Volume加速度検出（急激な出来高増加率の変化）

    Volume変化率の変化率（2次微分）を検出
    vol_change[t] = volume[t] / volume[t-1]
    vol_accel[t] = vol_change[t] / vol_change[t-lookback]
    """

    def __init__(
        self,
        accel_threshold: float = 1.5,
        lookback: int = 3,
    ):
        self.accel_threshold = accel_threshold
        self.lookback = lookback

    def evaluate(self, row, prev_row=None) -> bool:
        # Volume加速度（事前計算カラムを使用）
        vol_accel_col = f"volume_accel_{self.lookback}"
        vol_accel = row.get(vol_accel_col)

        if vol_accel is None or pd.isna(vol_accel):
            return False

        return vol_accel >= self.accel_threshold

    def describe(self) -> str:
        return f"VolumeAccel(threshold={self.accel_threshold}x, lookback={self.lookback})"


class PriceVolumeDivergenceCondition(Condition):
    """価格とVolumeの逆行検出（急落 + 出来高急増）

    価格は下落しているが、Volumeは増加している状態を検出
    """

    def __init__(
        self,
        price_change_threshold: float = -1.5,
        volume_change_threshold: float = 2.0,
        period: int = 3,
    ):
        self.price_threshold = price_change_threshold
        self.vol_threshold = volume_change_threshold
        self.period = period

    def evaluate(self, row, prev_row=None) -> bool:
        # 価格変化率（事前計算カラムを使用）
        price_change_col = f"close_pct_change_{self.period}"
        price_change = row.get(price_change_col)

        # Volume変化率（事前計算カラムを使用）
        vol_change_col = f"volume_ratio_{self.period}"
        vol_change = row.get(vol_change_col)

        if price_change is None or vol_change is None:
            return False
        if pd.isna(price_change) or pd.isna(vol_change):
            return False

        # 価格は下落 AND Volumeは増加
        return (
            price_change <= self.price_threshold and
            vol_change >= self.vol_threshold
        )

    def describe(self) -> str:
        return (
            f"PriceVolDivergence(price<={self.price_threshold}%, "
            f"vol>={self.vol_threshold}x, period={self.period})"
        )
