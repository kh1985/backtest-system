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
