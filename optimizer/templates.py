"""
戦略テンプレート

12種の組み込みテンプレート（ロング6種＋ショート6種）とパラメータ範囲定義。
プレースホルダー {param} で変数パラメータを定義し、
generate_configs() で全組み合わせ（直積）を自動生成。
"""

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Dict, List, Optional
import copy


@dataclass
class ParameterRange:
    """パラメータの探索範囲"""
    name: str
    min_val: float
    max_val: float
    step: float
    param_type: str = "int"  # "int" or "float"

    def values(self) -> List:
        """探索値のリストを生成"""
        vals = []
        v = self.min_val
        while v <= self.max_val + 1e-9:
            if self.param_type == "int":
                vals.append(int(v))
            else:
                vals.append(round(v, 4))
            v += self.step
        return vals


@dataclass
class StrategyTemplate:
    """戦略テンプレート"""
    name: str
    description: str
    config_template: Dict[str, Any]
    param_ranges: List[ParameterRange] = field(default_factory=list)

    def generate_configs(
        self,
        custom_ranges: Optional[Dict[str, ParameterRange]] = None,
    ) -> List[Dict[str, Any]]:
        """
        パラメータ範囲の直積で全設定を生成

        Args:
            custom_ranges: カスタムパラメータ範囲（UI設定で上書き用）

        Returns:
            ConfigStrategy用のconfig dictリスト
        """
        ranges = {r.name: r for r in self.param_ranges}
        if custom_ranges:
            ranges.update(custom_ranges)

        if not ranges:
            return [copy.deepcopy(self.config_template)]

        param_names = list(ranges.keys())
        param_values = [ranges[n].values() for n in param_names]

        configs = []
        for combo in product(*param_values):
            params = dict(zip(param_names, combo))
            config = self._apply_params(params)
            # 名前にパラメータを含める
            param_str = "_".join(f"{k}{v}" for k, v in params.items())
            config["name"] = f"{self.name}_{param_str}"
            # メタ情報を付加（GridSearchOptimizerが利用）
            config["_template_name"] = self.name
            config["_params"] = dict(params)
            configs.append(config)

        return configs

    def _apply_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """テンプレートのプレースホルダーをパラメータ値で置換"""
        config = copy.deepcopy(self.config_template)
        return self._replace_recursive(config, params)

    def _replace_recursive(self, obj, params):
        if isinstance(obj, dict):
            return {k: self._replace_recursive(v, params) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._replace_recursive(item, params) for item in obj]
        elif isinstance(obj, str):
            # 完全一致 "{param}" → 値そのものに置換（型を保持）
            if obj.startswith("{") and obj.endswith("}"):
                key = obj[1:-1]
                if key in params:
                    return params[key]
            # 部分一致 "sma_{fast_period}" → "sma_10" に文字列置換
            result = obj
            for key, val in params.items():
                result = result.replace("{" + key + "}", str(val))
            return result
        return obj

    def combination_count(
        self,
        custom_ranges: Optional[Dict[str, ParameterRange]] = None,
    ) -> int:
        """組み合わせ数を計算"""
        ranges = {r.name: r for r in self.param_ranges}
        if custom_ranges:
            ranges.update(custom_ranges)
        if not ranges:
            return 1
        count = 1
        for r in ranges.values():
            count *= len(r.values())
        return count


# =====================================================================
# 12種の組み込みテンプレート（ロング6種 + ショート6種）
# =====================================================================

BUILTIN_TEMPLATES: Dict[str, StrategyTemplate] = {}


def _register(template: StrategyTemplate):
    BUILTIN_TEMPLATES[template.name] = template


# 1. MA Crossover
_register(StrategyTemplate(
    name="ma_crossover",
    description="SMA fast/slow クロスでエントリー",
    config_template={
        "name": "ma_crossover",
        "side": "long",
        "indicators": [
            {"type": "sma", "period": "{fast_period}", "source": "close"},
            {"type": "sma", "period": "{slow_period}", "source": "close"},
        ],
        "entry_conditions": [
            {
                "type": "crossover",
                "fast": "sma_{fast_period}",
                "slow": "sma_{slow_period}",
                "direction": "above",
            },
        ],
        "entry_logic": "and",
        "exit": {
            "take_profit_pct": 2.0,
            "stop_loss_pct": 1.0,
        },
    },
    param_ranges=[
        ParameterRange("fast_period", 10, 30, 5, "int"),
        ParameterRange("slow_period", 40, 80, 10, "int"),
    ],
))

# 2. RSI Reversal
_register(StrategyTemplate(
    name="rsi_reversal",
    description="RSI売られすぎからの反発でエントリー",
    config_template={
        "name": "rsi_reversal",
        "side": "long",
        "indicators": [
            {"type": "rsi", "period": "{rsi_period}"},
        ],
        "entry_conditions": [
            {
                "type": "threshold",
                "column": "rsi_{rsi_period}",
                "operator": "<",
                "value": "{rsi_threshold}",
            },
        ],
        "entry_logic": "and",
        "exit": {
            "take_profit_pct": 2.0,
            "stop_loss_pct": 1.0,
        },
    },
    param_ranges=[
        ParameterRange("rsi_period", 10, 30, 5, "int"),
        ParameterRange("rsi_threshold", 20, 35, 5, "int"),
    ],
))

# 3. BB Bounce
_register(StrategyTemplate(
    name="bb_bounce",
    description="Bollinger Band下限タッチ + RSI で反発エントリー",
    config_template={
        "name": "bb_bounce",
        "side": "long",
        "indicators": [
            {"type": "bollinger", "period": "{bb_period}", "std_dev": 2.0},
            {"type": "rsi", "period": 14},
        ],
        "entry_conditions": [
            {
                "type": "column_compare",
                "column_a": "close",
                "operator": "<=",
                "column_b": "bb_lower_{bb_period}",
            },
            {
                "type": "threshold",
                "column": "rsi_14",
                "operator": "<",
                "value": "{rsi_threshold}",
            },
        ],
        "entry_logic": "and",
        "exit": {
            "take_profit_pct": 2.0,
            "stop_loss_pct": 1.0,
        },
    },
    param_ranges=[
        ParameterRange("bb_period", 15, 25, 5, "int"),
        ParameterRange("rsi_threshold", 25, 40, 5, "int"),
    ],
))

# 4. MACD Signal
_register(StrategyTemplate(
    name="macd_signal",
    description="MACDラインがシグナルラインを上抜け",
    config_template={
        "name": "macd_signal",
        "side": "long",
        "indicators": [
            {"type": "macd", "fast": "{macd_fast}", "slow": "{macd_slow}", "signal": 9},
        ],
        "entry_conditions": [
            {
                "type": "crossover",
                "fast": "macd_line",
                "slow": "macd_signal",
                "direction": "above",
            },
        ],
        "entry_logic": "and",
        "exit": {
            "take_profit_pct": 2.0,
            "stop_loss_pct": 1.0,
        },
    },
    param_ranges=[
        ParameterRange("macd_fast", 8, 16, 2, "int"),
        ParameterRange("macd_slow", 20, 30, 5, "int"),
    ],
))

# 5. Volume Spike
_register(StrategyTemplate(
    name="volume_spike",
    description="出来高急増 + 陰線からの反転でエントリー",
    config_template={
        "name": "volume_spike",
        "side": "long",
        "indicators": [
            {"type": "rvol", "period": "{rvol_period}"},
        ],
        "entry_conditions": [
            {
                "type": "threshold",
                "column": "rvol_{rvol_period}",
                "operator": ">",
                "value": "{rvol_threshold}",
            },
            {
                "type": "candle",
                "pattern": "bullish",
            },
        ],
        "entry_logic": "and",
        "exit": {
            "take_profit_pct": 2.0,
            "stop_loss_pct": 1.0,
        },
    },
    param_ranges=[
        ParameterRange("rvol_period", 10, 30, 10, "int"),
        ParameterRange("rvol_threshold", 2, 6, 1, "int"),
    ],
))

# 6. Stochastic Reversal
_register(StrategyTemplate(
    name="stochastic_reversal",
    description="Stochastic K/D 売られすぎゾーンでのクロス",
    config_template={
        "name": "stochastic_reversal",
        "side": "long",
        "indicators": [
            {"type": "stochastic", "k_period": "{k_period}", "d_period": 3},
        ],
        "entry_conditions": [
            {
                "type": "threshold",
                "column": "stoch_k_{k_period}",
                "operator": "<",
                "value": "{oversold}",
            },
            {
                "type": "crossover",
                "fast": "stoch_k_{k_period}",
                "slow": "stoch_d_{k_period}",
                "direction": "above",
            },
        ],
        "entry_logic": "and",
        "exit": {
            "take_profit_pct": 2.0,
            "stop_loss_pct": 1.0,
        },
    },
    param_ranges=[
        ParameterRange("k_period", 10, 20, 5, "int"),
        ParameterRange("oversold", 15, 30, 5, "int"),
    ],
))


# =====================================================================
# ショートテンプレート（ロング版のミラー）
# =====================================================================

# 7. MA Crossover Short
_register(StrategyTemplate(
    name="ma_crossover_short",
    description="SMA fast/slow 下抜けでショートエントリー",
    config_template={
        "name": "ma_crossover_short",
        "side": "short",
        "indicators": [
            {"type": "sma", "period": "{fast_period}", "source": "close"},
            {"type": "sma", "period": "{slow_period}", "source": "close"},
        ],
        "entry_conditions": [
            {
                "type": "crossover",
                "fast": "sma_{fast_period}",
                "slow": "sma_{slow_period}",
                "direction": "below",
            },
        ],
        "entry_logic": "and",
        "exit": {
            "take_profit_pct": 2.0,
            "stop_loss_pct": 1.0,
        },
    },
    param_ranges=[
        ParameterRange("fast_period", 10, 30, 5, "int"),
        ParameterRange("slow_period", 40, 80, 10, "int"),
    ],
))

# 8. RSI Reversal Short
_register(StrategyTemplate(
    name="rsi_reversal_short",
    description="RSI買われすぎからの反落でショートエントリー",
    config_template={
        "name": "rsi_reversal_short",
        "side": "short",
        "indicators": [
            {"type": "rsi", "period": "{rsi_period}"},
        ],
        "entry_conditions": [
            {
                "type": "threshold",
                "column": "rsi_{rsi_period}",
                "operator": ">",
                "value": "{rsi_threshold}",
            },
        ],
        "entry_logic": "and",
        "exit": {
            "take_profit_pct": 2.0,
            "stop_loss_pct": 1.0,
        },
    },
    param_ranges=[
        ParameterRange("rsi_period", 10, 30, 5, "int"),
        ParameterRange("rsi_threshold", 65, 80, 5, "int"),
    ],
))

# 9. BB Bounce Short
_register(StrategyTemplate(
    name="bb_bounce_short",
    description="Bollinger Band上限タッチ + RSI高でショートエントリー",
    config_template={
        "name": "bb_bounce_short",
        "side": "short",
        "indicators": [
            {"type": "bollinger", "period": "{bb_period}", "std_dev": 2.0},
            {"type": "rsi", "period": 14},
        ],
        "entry_conditions": [
            {
                "type": "column_compare",
                "column_a": "close",
                "operator": ">=",
                "column_b": "bb_upper_{bb_period}",
            },
            {
                "type": "threshold",
                "column": "rsi_14",
                "operator": ">",
                "value": "{rsi_threshold}",
            },
        ],
        "entry_logic": "and",
        "exit": {
            "take_profit_pct": 2.0,
            "stop_loss_pct": 1.0,
        },
    },
    param_ranges=[
        ParameterRange("bb_period", 15, 25, 5, "int"),
        ParameterRange("rsi_threshold", 60, 75, 5, "int"),
    ],
))

# 10. MACD Signal Short
_register(StrategyTemplate(
    name="macd_signal_short",
    description="MACDラインがシグナルラインを下抜け",
    config_template={
        "name": "macd_signal_short",
        "side": "short",
        "indicators": [
            {"type": "macd", "fast": "{macd_fast}", "slow": "{macd_slow}", "signal": 9},
        ],
        "entry_conditions": [
            {
                "type": "crossover",
                "fast": "macd_line",
                "slow": "macd_signal",
                "direction": "below",
            },
        ],
        "entry_logic": "and",
        "exit": {
            "take_profit_pct": 2.0,
            "stop_loss_pct": 1.0,
        },
    },
    param_ranges=[
        ParameterRange("macd_fast", 8, 16, 2, "int"),
        ParameterRange("macd_slow", 20, 30, 5, "int"),
    ],
))

# 11. Volume Spike Short
_register(StrategyTemplate(
    name="volume_spike_short",
    description="出来高急増 + 陽線からの反落でショートエントリー",
    config_template={
        "name": "volume_spike_short",
        "side": "short",
        "indicators": [
            {"type": "rvol", "period": "{rvol_period}"},
        ],
        "entry_conditions": [
            {
                "type": "threshold",
                "column": "rvol_{rvol_period}",
                "operator": ">",
                "value": "{rvol_threshold}",
            },
            {
                "type": "candle",
                "pattern": "bearish",
            },
        ],
        "entry_logic": "and",
        "exit": {
            "take_profit_pct": 2.0,
            "stop_loss_pct": 1.0,
        },
    },
    param_ranges=[
        ParameterRange("rvol_period", 10, 30, 10, "int"),
        ParameterRange("rvol_threshold", 2, 6, 1, "int"),
    ],
))

# 12. Stochastic Reversal Short
_register(StrategyTemplate(
    name="stochastic_reversal_short",
    description="Stochastic K/D 買われすぎゾーンでのクロス",
    config_template={
        "name": "stochastic_reversal_short",
        "side": "short",
        "indicators": [
            {"type": "stochastic", "k_period": "{k_period}", "d_period": 3},
        ],
        "entry_conditions": [
            {
                "type": "threshold",
                "column": "stoch_k_{k_period}",
                "operator": ">",
                "value": "{overbought}",
            },
            {
                "type": "crossover",
                "fast": "stoch_k_{k_period}",
                "slow": "stoch_d_{k_period}",
                "direction": "below",
            },
        ],
        "entry_logic": "and",
        "exit": {
            "take_profit_pct": 2.0,
            "stop_loss_pct": 1.0,
        },
    },
    param_ranges=[
        ParameterRange("k_period", 10, 20, 5, "int"),
        ParameterRange("overbought", 70, 85, 5, "int"),
    ],
))
