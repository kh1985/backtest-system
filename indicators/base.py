"""
インジケーター基底クラス
"""

from abc import ABC, abstractmethod
from typing import List

import pandas as pd


class Indicator(ABC):
    """インジケーターの抽象基底クラス"""

    name: str

    @abstractmethod
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        DataFrameにインジケーターカラムを追加して返す

        カラム名の規則: {indicator}_{period} (例: sma_20, rsi_14)
        """
        ...

    @property
    @abstractmethod
    def columns(self) -> List[str]:
        """このインジケーターが追加するカラム名のリスト"""
        ...

    @property
    def is_overlay(self) -> bool:
        """価格チャートに重ねて表示するかどうか（デフォルト: False）"""
        return False
