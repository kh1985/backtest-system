"""
戦略ビルダー

YAML設定ファイルやdict（UI入力）から戦略オブジェクトを構築する。
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
import pandas as pd

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from indicators.registry import create_indicator
from strategy.base import ExitRule, Signal, Side, Strategy
from strategy.conditions import (
    CandlePatternCondition,
    ColumnCompareCondition,
    CompoundCondition,
    Condition,
    CrossoverCondition,
    ThresholdCondition,
)


class ConfigStrategy(Strategy):
    """設定（YAML/dict）から構築される戦略"""

    def __init__(self, config: Dict[str, Any]):
        self.name = config["name"]
        self.side = Side(config.get("side", "long"))
        exit_conf = config.get("exit", {})
        self.exit_rule = ExitRule(
            take_profit_pct=exit_conf.get("take_profit_pct", 0.0),
            stop_loss_pct=exit_conf.get("stop_loss_pct", 0.0),
            trailing_stop_pct=exit_conf.get("trailing_stop_pct"),
            timeout_bars=exit_conf.get("timeout_bars"),
            # ATRベースexit
            use_atr_exit=exit_conf.get("use_atr_exit", False),
            atr_tp_mult=exit_conf.get("atr_tp_mult", 0.0),
            atr_sl_mult=exit_conf.get("atr_sl_mult", 0.0),
            atr_period=exit_conf.get("atr_period", 14),
            # ATRベーストレーリング
            use_atr_trailing=exit_conf.get("use_atr_trailing", False),
            atr_trailing_mult=exit_conf.get("atr_trailing_mult", 0.0),
            # VWAP/BB exit
            use_vwap_exit=exit_conf.get("use_vwap_exit", False),
            vwap_band=exit_conf.get("vwap_band", 1),
            use_bb_exit=exit_conf.get("use_bb_exit", False),
            bb_period=exit_conf.get("bb_period", 20),
        )
        self._indicator_configs = [
            dict(ic) for ic in config.get("indicators", [])
        ]
        self._condition_configs = config.get("entry_conditions", [])
        self._entry_logic = config.get("entry_logic", "and")
        self._indicators = []
        self._entry_condition: Optional[Condition] = None

    def setup(self, df: pd.DataFrame) -> pd.DataFrame:
        """インジケーターを計算し、エントリー条件を構築"""
        for ind_conf in self._indicator_configs:
            conf = dict(ind_conf)
            name = conf.pop("type")
            indicator = create_indicator(name, **conf)
            self._indicators.append(indicator)
            df = indicator.calculate(df)

        self._entry_condition = self._build_condition(
            self._condition_configs
        )
        return df

    def check_entry(
        self, row: pd.Series, prev_row: pd.Series
    ) -> Optional[Signal]:
        """エントリー条件を評価"""
        if self._entry_condition and self._entry_condition.evaluate(
            row, prev_row
        ):
            return Signal(
                index=int(row.name) if hasattr(row, 'name') else 0,
                side=self.side,
                reason=self._entry_condition.describe(),
            )
        return None

    def _build_condition(
        self, config_list: List[Dict]
    ) -> Optional[Condition]:
        if not config_list:
            return None

        conditions = []
        for c in config_list:
            ctype = c["type"]
            if ctype == "threshold":
                conditions.append(
                    ThresholdCondition(c["column"], c["operator"], c["value"])
                )
            elif ctype == "crossover":
                conditions.append(
                    CrossoverCondition(
                        c["fast"], c["slow"], c.get("direction", "above")
                    )
                )
            elif ctype == "column_compare":
                conditions.append(
                    ColumnCompareCondition(
                        c["column_a"], c["operator"], c["column_b"]
                    )
                )
            elif ctype == "candle":
                conditions.append(CandlePatternCondition(c["pattern"]))

        if len(conditions) == 1:
            return conditions[0]
        return CompoundCondition(conditions, self._entry_logic)


def load_strategy_from_yaml(path: str) -> ConfigStrategy:
    """YAMLファイルから戦略を読み込む"""
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return ConfigStrategy(config)


def load_strategy_from_dict(config: Dict[str, Any]) -> ConfigStrategy:
    """dictから戦略を構築（UI用）"""
    return ConfigStrategy(config)
