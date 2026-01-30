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
