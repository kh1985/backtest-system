"""
インジケーターレジストリ

名前からインジケータークラスを生成するファクトリ。
"""

from typing import Dict, Type

from .base import Indicator
from .trend import SMA, EMA
from .momentum import RSI, MACD, Stochastic
from .volatility import BollingerBands, ATR
from .volume import VWAP, RelativeVolume, VolumeAnalysis
from .adx import ADX


INDICATOR_REGISTRY: Dict[str, Type[Indicator]] = {
    "sma": SMA,
    "ema": EMA,
    "rsi": RSI,
    "macd": MACD,
    "stochastic": Stochastic,
    "bollinger": BollingerBands,
    "atr": ATR,
    "vwap": VWAP,
    "rvol": RelativeVolume,
    "volume_analysis": VolumeAnalysis,
    "adx": ADX,
}

# UI表示用の情報
INDICATOR_INFO = {
    "sma": {"label": "SMA (単純移動平均)", "params": {"period": 20, "source": "close"}},
    "ema": {"label": "EMA (指数移動平均)", "params": {"period": 20, "source": "close"}},
    "rsi": {"label": "RSI", "params": {"period": 14}},
    "macd": {"label": "MACD", "params": {"fast": 12, "slow": 26, "signal": 9}},
    "stochastic": {"label": "Stochastic", "params": {"k_period": 14, "d_period": 3}},
    "bollinger": {"label": "Bollinger Bands", "params": {"period": 20, "std_dev": 2.0}},
    "atr": {"label": "ATR", "params": {"period": 14}},
    "vwap": {"label": "VWAP", "params": {}},
    "rvol": {"label": "RVOL (相対出来高)", "params": {"period": 20}},
    "volume_analysis": {"label": "Volume Analysis", "params": {}},
    "adx": {"label": "ADX (+DI/-DI)", "params": {"period": 14}},
}


def create_indicator(name: str, **params) -> Indicator:
    """名前とパラメータからインジケーターを生成"""
    if name not in INDICATOR_REGISTRY:
        available = ", ".join(INDICATOR_REGISTRY.keys())
        raise ValueError(
            f"Unknown indicator: '{name}'. Available: {available}"
        )
    cls = INDICATOR_REGISTRY[name]
    return cls(**params)


def list_indicators() -> list:
    """利用可能なインジケーター名のリスト"""
    return list(INDICATOR_REGISTRY.keys())
