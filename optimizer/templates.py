"""
戦略テンプレート

25種の組み込みテンプレート:
- 基本テンプレート（ロング7種 + ショート7種 = 14種）
- VWAP戦略（6種）:
  - vwap_touch_long/short: VWAPタッチ（押し目/戻り）
  - vwap_upper1_long / vwap_lower1_short: ±1σバンド順張り（トレンドフォロー）
  - vwap_2sigma_long/short: ±2σバンド逆張り（レンジ向け）
- VP戦略（1種）: vp_pullback_long
- 複合テンプレート（4種）:
  - rsi_volume_reversal: RSI売られすぎ + 出来高急増 → ロング
  - rsi_volume_reversal_short: RSI買われすぎ + 出来高急増 → ショート
  - rsi_bb_reversal: RSI売られすぎ + BB下限タッチ → ロング
  - rsi_bb_reversal_short: RSI買われすぎ + BB上限タッチ → ショート

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
        exit_profiles: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        パラメータ範囲の直積で全設定を生成

        Args:
            custom_ranges: カスタムパラメータ範囲（UI設定で上書き用）
            exit_profiles: exit 戦略プロファイルのリスト。
                指定時はエントリー条件 x exit条件 の直積を生成。
                各プロファイルは {"name": "...", "take_profit_pct": ..., ...} 形式。

        Returns:
            ConfigStrategy用のconfig dictリスト
        """
        ranges = {r.name: r for r in self.param_ranges}
        if custom_ranges:
            ranges.update(custom_ranges)

        if not ranges:
            entry_configs = [copy.deepcopy(self.config_template)]
        else:
            param_names = list(ranges.keys())
            param_values = [ranges[n].values() for n in param_names]

            entry_configs = []
            for combo in product(*param_values):
                params = dict(zip(param_names, combo))
                config = self._apply_params(params)
                param_str = "_".join(f"{k}{v}" for k, v in params.items())
                config["name"] = f"{self.name}_{param_str}"
                config["_template_name"] = self.name
                config["_params"] = dict(params)
                entry_configs.append(config)

        if not exit_profiles:
            # exit_profiles 未指定 → 従来動作（テンプレート内蔵の exit を使用）
            return entry_configs

        # exit_profiles × entry_configs の直積
        final_configs = []
        for config in entry_configs:
            for ep in exit_profiles:
                c = copy.deepcopy(config)
                # exit セクションをプロファイルで上書き
                exit_conf = {k: v for k, v in ep.items() if k != "name"}
                c["exit"] = exit_conf
                c["_exit_profile"] = ep.get("name", "unknown")
                # 名前にexit profileを含める
                c["name"] = f"{config['name']}_{ep.get('name', '')}"
                final_configs.append(c)

        return final_configs

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

# 13. Trend Pullback Long（上昇トレンド押し目買い）
_register(StrategyTemplate(
    name="trend_pullback_long",
    description="上昇トレンドの押し目（RSI一時下落）でロング、トレーリングストップで利益延伸",
    config_template={
        "name": "trend_pullback_long",
        "side": "long",
        "indicators": [
            {"type": "rsi", "period": "{rsi_period}"},
        ],
        "entry_conditions": [
            {
                "type": "threshold",
                "column": "rsi_{rsi_period}",
                "operator": ">=",
                "value": "{rsi_low}",
            },
            {
                "type": "threshold",
                "column": "rsi_{rsi_period}",
                "operator": "<=",
                "value": "{rsi_high}",
            },
        ],
        "entry_logic": "and",
        "exit": {
            "take_profit_pct": 0,
            "stop_loss_pct": "{sl_pct}",
            "trailing_stop_pct": "{trailing_pct}",
        },
    },
    param_ranges=[
        ParameterRange("rsi_period", 14, 14, 1, "int"),
        ParameterRange("rsi_low", 30, 40, 5, "int"),
        ParameterRange("rsi_high", 45, 55, 5, "int"),
        ParameterRange("trailing_pct", 1.0, 3.0, 0.5, "float"),
        ParameterRange("sl_pct", 1.0, 2.0, 0.5, "float"),
    ],
))

# 14. Trend Pullback Short（下落トレンド戻り売り）
_register(StrategyTemplate(
    name="trend_pullback_short",
    description="下落トレンドの戻り（RSI一時上昇）でショート、トレーリングストップで利益延伸",
    config_template={
        "name": "trend_pullback_short",
        "side": "short",
        "indicators": [
            {"type": "rsi", "period": "{rsi_period}"},
        ],
        "entry_conditions": [
            {
                "type": "threshold",
                "column": "rsi_{rsi_period}",
                "operator": ">=",
                "value": "{rsi_low}",
            },
            {
                "type": "threshold",
                "column": "rsi_{rsi_period}",
                "operator": "<=",
                "value": "{rsi_high}",
            },
        ],
        "entry_logic": "and",
        "exit": {
            "take_profit_pct": 0,
            "stop_loss_pct": "{sl_pct}",
            "trailing_stop_pct": "{trailing_pct}",
        },
    },
    param_ranges=[
        ParameterRange("rsi_period", 14, 14, 1, "int"),
        ParameterRange("rsi_low", 45, 55, 5, "int"),
        ParameterRange("rsi_high", 60, 70, 5, "int"),
        ParameterRange("trailing_pct", 1.0, 3.0, 0.5, "float"),
        ParameterRange("sl_pct", 1.0, 2.0, 0.5, "float"),
    ],
))


# =====================================================================
# VWAP戦略（6パターン）
# - VWAPタッチ: トレンド相場の押し目買い/戻り売り
# - 1σバンド順張り: 強いトレンドで押し/戻りを待たずにエントリー
# - 2σバンド逆張り: レンジ相場の逆張り
# =====================================================================

# 15. VWAP Touch Long（VWAPタッチでロング）
_register(StrategyTemplate(
    name="vwap_touch_long",
    description="価格がVWAPラインに到達したらロング（トレンド押し目買い向け）",
    config_template={
        "name": "vwap_touch_long",
        "side": "long",
        "indicators": [
            {"type": "vwap", "switch_hour": 1},
        ],
        "entry_conditions": [
            {
                "type": "column_compare",
                "column_a": "close",
                "operator": "<=",
                "column_b": "vwap_active",
            },
        ],
        "entry_logic": "and",
        "exit": {
            "take_profit_pct": 2.0,
            "stop_loss_pct": 1.0,
        },
    },
    param_ranges=[],
))

# 16. VWAP Touch Short（VWAPタッチでショート）
_register(StrategyTemplate(
    name="vwap_touch_short",
    description="価格がVWAPラインに到達したらショート（トレンド戻り売り向け）",
    config_template={
        "name": "vwap_touch_short",
        "side": "short",
        "indicators": [
            {"type": "vwap", "switch_hour": 1},
        ],
        "entry_conditions": [
            {
                "type": "column_compare",
                "column_a": "close",
                "operator": ">=",
                "column_b": "vwap_active",
            },
        ],
        "entry_logic": "and",
        "exit": {
            "take_profit_pct": 2.0,
            "stop_loss_pct": 1.0,
        },
    },
    param_ranges=[],
))

# 17. VWAP Upper 1σ Long（上限1σでロング - 上昇トレンド順張り）
_register(StrategyTemplate(
    name="vwap_upper1_long",
    description="価格がVWAP+1σに到達したらロング（上昇トレンド順張り、押しを待たない）",
    config_template={
        "name": "vwap_upper1_long",
        "side": "long",
        "indicators": [
            {"type": "vwap", "switch_hour": 1},
        ],
        "entry_conditions": [
            {
                "type": "column_compare",
                "column_a": "close",
                "operator": ">=",
                "column_b": "vwap_upper1_active",
            },
        ],
        "entry_logic": "and",
        "exit": {
            "take_profit_pct": 2.0,
            "stop_loss_pct": 1.0,
        },
    },
    param_ranges=[],
))

# 18. VWAP Lower 1σ Short（下限1σでショート - 下落トレンド順張り）
_register(StrategyTemplate(
    name="vwap_lower1_short",
    description="価格がVWAP-1σに到達したらショート（下落トレンド順張り、戻りを待たない）",
    config_template={
        "name": "vwap_lower1_short",
        "side": "short",
        "indicators": [
            {"type": "vwap", "switch_hour": 1},
        ],
        "entry_conditions": [
            {
                "type": "column_compare",
                "column_a": "close",
                "operator": "<=",
                "column_b": "vwap_lower1_active",
            },
        ],
        "entry_logic": "and",
        "exit": {
            "take_profit_pct": 2.0,
            "stop_loss_pct": 1.0,
        },
    },
    param_ranges=[],
))

# 19. VWAP 2σ Long（下限2σでロング）
_register(StrategyTemplate(
    name="vwap_2sigma_long",
    description="価格がVWAP-2σバンドに到達したらロング（レンジ逆張り向け）",
    config_template={
        "name": "vwap_2sigma_long",
        "side": "long",
        "indicators": [
            {"type": "vwap", "switch_hour": 1},
        ],
        "entry_conditions": [
            {
                "type": "column_compare",
                "column_a": "close",
                "operator": "<=",
                "column_b": "vwap_lower2_active",
            },
        ],
        "entry_logic": "and",
        "exit": {
            "take_profit_pct": 2.0,
            "stop_loss_pct": 1.0,
        },
    },
    param_ranges=[],
))

# 20. VWAP 2σ Short（上限2σでショート）
_register(StrategyTemplate(
    name="vwap_2sigma_short",
    description="価格がVWAP+2σバンドに到達したらショート（レンジ逆張り向け）",
    config_template={
        "name": "vwap_2sigma_short",
        "side": "short",
        "indicators": [
            {"type": "vwap", "switch_hour": 1},
        ],
        "entry_conditions": [
            {
                "type": "column_compare",
                "column_a": "close",
                "operator": ">=",
                "column_b": "vwap_upper2_active",
            },
        ],
        "entry_logic": "and",
        "exit": {
            "take_profit_pct": 2.0,
            "stop_loss_pct": 1.0,
        },
    },
    param_ranges=[],
))


# =====================================================================
# Volume Profile戦略
# =====================================================================

# 21. VP Pullback Long（Volume Profile未テストLVN初タッチでロング）
# exit_profiles と組み合わせて使う（ATR/BB/固定%等）
_register(StrategyTemplate(
    name="vp_pullback_long",
    description="上昇トレンドでVolume ProfileのLVN（出来高の谷）に初タッチしたらロング（押し目買い）",
    config_template={
        "name": "vp_pullback_long",
        "side": "long",
        "indicators": [
            {"type": "volume_profile", "n_bins": "{n_bins}", "smoothing": 3, "touch_tolerance": "{touch_tolerance}"},
        ],
        "entry_conditions": [
            {
                "type": "threshold",
                "column": "vp_lvn_first_touch",
                "operator": "==",
                "value": 1,
            },
        ],
        "entry_logic": "and",
        "exit": {
            "take_profit_pct": 2.0,
            "stop_loss_pct": 1.0,
        },
    },
    param_ranges=[
        ParameterRange("n_bins", 30, 70, 20, "int"),
        ParameterRange("touch_tolerance", 0.3, 0.7, 0.2, "float"),
    ],
))


# =====================================================================
# 複合テンプレート（downtrend横断検証用）
# インジケーター期間は固定、探索は閾値のみ → 過学習を抑制
# =====================================================================

# 22. RSI + Volume Reversal Long
# RSI売られすぎ + 出来高急増 + 陽線 → 反転ロング
_register(StrategyTemplate(
    name="rsi_volume_reversal",
    description="RSI売られすぎ + 出来高急増 + 陽線反転でロング（下落トレンド逆張り）",
    config_template={
        "name": "rsi_volume_reversal",
        "side": "long",
        "indicators": [
            {"type": "rsi", "period": 14},
            {"type": "rvol", "period": 20},
        ],
        "entry_conditions": [
            {
                "type": "threshold",
                "column": "rsi_14",
                "operator": "<",
                "value": "{rsi_threshold}",
            },
            {
                "type": "threshold",
                "column": "rvol_20",
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
        ParameterRange("rsi_threshold", 25, 35, 5, "int"),
        ParameterRange("rvol_threshold", 2, 4, 1, "int"),
    ],
))

# 23. RSI + Volume Reversal Short
# RSI買われすぎ + 出来高急増 + 陰線 → 戻り売りショート
_register(StrategyTemplate(
    name="rsi_volume_reversal_short",
    description="RSI買われすぎ + 出来高急増 + 陰線反転でショート（下落トレンド順張り）",
    config_template={
        "name": "rsi_volume_reversal_short",
        "side": "short",
        "indicators": [
            {"type": "rsi", "period": 14},
            {"type": "rvol", "period": 20},
        ],
        "entry_conditions": [
            {
                "type": "threshold",
                "column": "rsi_14",
                "operator": ">",
                "value": "{rsi_threshold}",
            },
            {
                "type": "threshold",
                "column": "rvol_20",
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
        ParameterRange("rsi_threshold", 65, 75, 5, "int"),
        ParameterRange("rvol_threshold", 2, 4, 1, "int"),
    ],
))

# 24. RSI + BB Reversal Long
# RSI売られすぎ + BB下限タッチ → ダブル逆張り確認ロング
_register(StrategyTemplate(
    name="rsi_bb_reversal",
    description="RSI売られすぎ + ボリンジャーバンド下限タッチでロング（ダブル逆張り確認）",
    config_template={
        "name": "rsi_bb_reversal",
        "side": "long",
        "indicators": [
            {"type": "rsi", "period": 14},
            {"type": "bollinger", "period": 20, "std_dev": 2.0},
        ],
        "entry_conditions": [
            {
                "type": "threshold",
                "column": "rsi_14",
                "operator": "<",
                "value": "{rsi_threshold}",
            },
            {
                "type": "column_compare",
                "column_a": "close",
                "operator": "<",
                "column_b": "bb_lower_20",
            },
        ],
        "entry_logic": "and",
        "exit": {
            "take_profit_pct": 2.0,
            "stop_loss_pct": 1.0,
        },
    },
    param_ranges=[
        ParameterRange("rsi_threshold", 25, 35, 5, "int"),
    ],
))

# 25. RSI + BB Reversal Short
# RSI買われすぎ + BB上限タッチ → ダブル逆張り確認ショート
# Step 7b: RSI閾値4段階（σ=2.0固定 — σ拡張は逆効果と判明）
_register(StrategyTemplate(
    name="rsi_bb_reversal_short",
    description="RSI買われすぎ + ボリンジャーバンド上限タッチでショート（ダブル逆張り確認）",
    config_template={
        "name": "rsi_bb_reversal_short",
        "side": "short",
        "indicators": [
            {"type": "rsi", "period": 14},
            {"type": "bollinger", "period": 20, "std_dev": 2.0},
        ],
        "entry_conditions": [
            {
                "type": "threshold",
                "column": "rsi_14",
                "operator": ">",
                "value": "{rsi_threshold}",
            },
            {
                "type": "column_compare",
                "column_a": "close",
                "operator": ">",
                "column_b": "bb_upper_20",
            },
        ],
        "entry_logic": "and",
        "exit": {
            "take_profit_pct": 2.0,
            "stop_loss_pct": 1.0,
        },
    },
    param_ranges=[
        ParameterRange("rsi_threshold", 60, 75, 5, "int"),
    ],
))


# =====================================================================
# 複合テンプレート 第2弾（Step 5: 組み合わせ探索）
# =====================================================================

# 26. Stochastic + BB Reversal Long
# Stoch K 売られすぎ + BB下限タッチ → ロング
_register(StrategyTemplate(
    name="stoch_bb_reversal",
    description="Stochastic売られすぎ + BB下限タッチでロング",
    config_template={
        "name": "stoch_bb_reversal",
        "side": "long",
        "indicators": [
            {"type": "stochastic", "k_period": 14, "d_period": 3},
            {"type": "bollinger", "period": 20, "std_dev": 2.0},
        ],
        "entry_conditions": [
            {
                "type": "threshold",
                "column": "stoch_k_14",
                "operator": "<",
                "value": "{stoch_threshold}",
            },
            {
                "type": "column_compare",
                "column_a": "close",
                "operator": "<",
                "column_b": "bb_lower_20",
            },
        ],
        "entry_logic": "and",
        "exit": {
            "take_profit_pct": 2.0,
            "stop_loss_pct": 1.0,
        },
    },
    param_ranges=[
        ParameterRange("stoch_threshold", 20, 30, 5, "int"),
    ],
))

# 27. Stochastic + BB Reversal Short
# Stoch K 買われすぎ + BB上限タッチ → ショート
_register(StrategyTemplate(
    name="stoch_bb_reversal_short",
    description="Stochastic買われすぎ + BB上限タッチでショート",
    config_template={
        "name": "stoch_bb_reversal_short",
        "side": "short",
        "indicators": [
            {"type": "stochastic", "k_period": 14, "d_period": 3},
            {"type": "bollinger", "period": 20, "std_dev": 2.0},
        ],
        "entry_conditions": [
            {
                "type": "threshold",
                "column": "stoch_k_14",
                "operator": ">",
                "value": "{stoch_threshold}",
            },
            {
                "type": "column_compare",
                "column_a": "close",
                "operator": ">",
                "column_b": "bb_upper_20",
            },
        ],
        "entry_logic": "and",
        "exit": {
            "take_profit_pct": 2.0,
            "stop_loss_pct": 1.0,
        },
    },
    param_ranges=[
        ParameterRange("stoch_threshold", 70, 80, 5, "int"),
    ],
))

# 28. MACD + BB Reversal Long
# MACD弱気圏（histogram < 0）+ BB下限タッチ → 反転ロング
_register(StrategyTemplate(
    name="macd_bb_reversal",
    description="MACD弱気圏 + BB下限タッチでロング（反転狙い）",
    config_template={
        "name": "macd_bb_reversal",
        "side": "long",
        "indicators": [
            {"type": "macd", "fast": 12, "slow": 26, "signal": 9},
            {"type": "bollinger", "period": 20, "std_dev": 2.0},
        ],
        "entry_conditions": [
            {
                "type": "threshold",
                "column": "macd_histogram",
                "operator": "<",
                "value": 0,
            },
            {
                "type": "column_compare",
                "column_a": "close",
                "operator": "<",
                "column_b": "bb_lower_20",
            },
        ],
        "entry_logic": "and",
        "exit": {
            "take_profit_pct": 2.0,
            "stop_loss_pct": 1.0,
        },
    },
    param_ranges=[],
))

# 29. MACD + BB Reversal Short
# MACD強気圏（histogram > 0）+ BB上限タッチ → 反転ショート
_register(StrategyTemplate(
    name="macd_bb_reversal_short",
    description="MACD強気圏 + BB上限タッチでショート（反転狙い）",
    config_template={
        "name": "macd_bb_reversal_short",
        "side": "short",
        "indicators": [
            {"type": "macd", "fast": 12, "slow": 26, "signal": 9},
            {"type": "bollinger", "period": 20, "std_dev": 2.0},
        ],
        "entry_conditions": [
            {
                "type": "threshold",
                "column": "macd_histogram",
                "operator": ">",
                "value": 0,
            },
            {
                "type": "column_compare",
                "column_a": "close",
                "operator": ">",
                "column_b": "bb_upper_20",
            },
        ],
        "entry_logic": "and",
        "exit": {
            "take_profit_pct": 2.0,
            "stop_loss_pct": 1.0,
        },
    },
    param_ranges=[],
))

# 30. RSI + MACD Reversal Long
# RSI売られすぎ + MACDクロスオーバー → ロング
_register(StrategyTemplate(
    name="rsi_macd_reversal",
    description="RSI売られすぎ + MACDゴールデンクロスでロング",
    config_template={
        "name": "rsi_macd_reversal",
        "side": "long",
        "indicators": [
            {"type": "rsi", "period": 14},
            {"type": "macd", "fast": 12, "slow": 26, "signal": 9},
        ],
        "entry_conditions": [
            {
                "type": "threshold",
                "column": "rsi_14",
                "operator": "<",
                "value": "{rsi_threshold}",
            },
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
        ParameterRange("rsi_threshold", 25, 35, 5, "int"),
    ],
))

# 31. RSI + MACD Reversal Short
# RSI買われすぎ + MACDデッドクロス → ショート
_register(StrategyTemplate(
    name="rsi_macd_reversal_short",
    description="RSI買われすぎ + MACDデッドクロスでショート",
    config_template={
        "name": "rsi_macd_reversal_short",
        "side": "short",
        "indicators": [
            {"type": "rsi", "period": 14},
            {"type": "macd", "fast": 12, "slow": 26, "signal": 9},
        ],
        "entry_conditions": [
            {
                "type": "threshold",
                "column": "rsi_14",
                "operator": ">",
                "value": "{rsi_threshold}",
            },
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
        ParameterRange("rsi_threshold", 65, 75, 5, "int"),
    ],
))

# 32. BB + Volume Reversal Long
# BB下限タッチ + 出来高急増 → ロング
_register(StrategyTemplate(
    name="bb_volume_reversal",
    description="BB下限タッチ + 出来高急増でロング",
    config_template={
        "name": "bb_volume_reversal",
        "side": "long",
        "indicators": [
            {"type": "bollinger", "period": 20, "std_dev": 2.0},
            {"type": "rvol", "period": 20},
        ],
        "entry_conditions": [
            {
                "type": "column_compare",
                "column_a": "close",
                "operator": "<",
                "column_b": "bb_lower_20",
            },
            {
                "type": "threshold",
                "column": "rvol_20",
                "operator": ">",
                "value": "{rvol_threshold}",
            },
        ],
        "entry_logic": "and",
        "exit": {
            "take_profit_pct": 2.0,
            "stop_loss_pct": 1.0,
        },
    },
    param_ranges=[
        ParameterRange("rvol_threshold", 2, 4, 1, "int"),
    ],
))

# 33. BB + Volume Reversal Short
# BB上限タッチ + 出来高急増 → ショート
# Step 7b: RVOL閾値4段階（σ=2.0固定、RVOL 2.0除去 — PASS率25.7%で低すぎ）
_register(StrategyTemplate(
    name="bb_volume_reversal_short",
    description="BB上限タッチ + 出来高急増でショート",
    config_template={
        "name": "bb_volume_reversal_short",
        "side": "short",
        "indicators": [
            {"type": "bollinger", "period": 20, "std_dev": 2.0},
            {"type": "rvol", "period": 20},
        ],
        "entry_conditions": [
            {
                "type": "column_compare",
                "column_a": "close",
                "operator": ">",
                "column_b": "bb_upper_20",
            },
            {
                "type": "threshold",
                "column": "rvol_20",
                "operator": ">",
                "value": "{rvol_threshold}",
            },
        ],
        "entry_logic": "and",
        "exit": {
            "take_profit_pct": 2.0,
            "stop_loss_pct": 1.0,
        },
    },
    param_ranges=[
        ParameterRange("rvol_threshold", 2.5, 4.0, 0.5, "float"),
    ],
))


# =====================================================================
# Step 7b: 新規テンプレート（3重複合 + ロング版）
# =====================================================================

# 34. RSI + BB + Volume Reversal Short（3重複合ショート）
# RSI買われすぎ + BB上限タッチ + 出来高急増 → ショート
_register(StrategyTemplate(
    name="rsi_bb_volume_reversal_short",
    description="RSI買われすぎ + BB上限タッチ + 出来高急増でショート（3重フィルタ）",
    config_template={
        "name": "rsi_bb_volume_reversal_short",
        "side": "short",
        "indicators": [
            {"type": "rsi", "period": 14},
            {"type": "bollinger", "period": 20, "std_dev": 2.0},
            {"type": "rvol", "period": 20},
        ],
        "entry_conditions": [
            {
                "type": "threshold",
                "column": "rsi_14",
                "operator": ">",
                "value": "{rsi_threshold}",
            },
            {
                "type": "column_compare",
                "column_a": "close",
                "operator": ">",
                "column_b": "bb_upper_20",
            },
            {
                "type": "threshold",
                "column": "rvol_20",
                "operator": ">",
                "value": "{rvol_threshold}",
            },
        ],
        "entry_logic": "and",
        "exit": {
            "take_profit_pct": 2.0,
            "stop_loss_pct": 1.0,
        },
    },
    param_ranges=[
        ParameterRange("rsi_threshold", 65, 75, 5, "int"),
        ParameterRange("rvol_threshold", 2.5, 3.5, 0.5, "float"),
    ],
))

# 35. RSI + BB Reversal Long
# RSI売られすぎ + BB下限タッチ → ロング（σ=2.0固定）
# Step 8b: RSI閾値を31/35の2値に絞り込み — RSI<29は過学習トラップと判明
# Step 8a結果: RSI<29=6.7%, RSI<31=44.4%, RSI<35=66.7% → 31/35のみ有効
_register(StrategyTemplate(
    name="rsi_bb_reversal_long",
    description="RSI売られすぎ + ボリンジャーバンド下限タッチでロング（ダブル逆張り確認）",
    config_template={
        "name": "rsi_bb_reversal_long",
        "side": "long",
        "indicators": [
            {"type": "rsi", "period": 14},
            {"type": "bollinger", "period": 20, "std_dev": 2.0},
        ],
        "entry_conditions": [
            {
                "type": "threshold",
                "column": "rsi_14",
                "operator": "<",
                "value": "{rsi_threshold}",
            },
            {
                "type": "column_compare",
                "column_a": "close",
                "operator": "<",
                "column_b": "bb_lower_20",
            },
        ],
        "entry_logic": "and",
        "exit": {
            "take_profit_pct": 2.0,
            "stop_loss_pct": 1.0,
        },
    },
    param_ranges=[
        ParameterRange("rsi_threshold", 31, 35, 4, "int"),
    ],
))

# 36. BB + Volume Reversal Long
# BB下限タッチ + 出来高急増 → ロング（σ=2.0固定）
_register(StrategyTemplate(
    name="bb_volume_reversal_long",
    description="BB下限タッチ + 出来高急増でロング",
    config_template={
        "name": "bb_volume_reversal_long",
        "side": "long",
        "indicators": [
            {"type": "bollinger", "period": 20, "std_dev": 2.0},
            {"type": "rvol", "period": 20},
        ],
        "entry_conditions": [
            {
                "type": "column_compare",
                "column_a": "close",
                "operator": "<",
                "column_b": "bb_lower_20",
            },
            {
                "type": "threshold",
                "column": "rvol_20",
                "operator": ">",
                "value": "{rvol_threshold}",
            },
        ],
        "entry_logic": "and",
        "exit": {
            "take_profit_pct": 2.0,
            "stop_loss_pct": 1.0,
        },
    },
    param_ranges=[
        ParameterRange("rvol_threshold", 2.5, 3.5, 0.5, "float"),
    ],
))

# 37. RSI + BB Reversal Long (RSI=35 固定)
# Step 9-1: 探索空間をexit 3択のみに縮小し最終確認
_register(StrategyTemplate(
    name="rsi_bb_long_f35",
    description="RSI<35 + BB下限タッチでロング（RSI=35固定、exit最小探索）",
    config_template={
        "name": "rsi_bb_long_f35",
        "side": "long",
        "indicators": [
            {"type": "rsi", "period": 14},
            {"type": "bollinger", "period": 20, "std_dev": 2.0},
        ],
        "entry_conditions": [
            {
                "type": "threshold",
                "column": "rsi_14",
                "operator": "<",
                "value": 35,
            },
            {
                "type": "column_compare",
                "column_a": "close",
                "operator": "<",
                "column_b": "bb_lower_20",
            },
        ],
        "entry_logic": "and",
        "exit": {
            "take_profit_pct": 2.0,
            "stop_loss_pct": 1.0,
        },
    },
    param_ranges=[],
))

# 38. RSI + BB Reversal Short (RSI閾値6値細分化)
# Step 9-2: ショート版のRSI閾値探索（ロング版Step 8aの対称テスト）
_register(StrategyTemplate(
    name="rsi_bb_short_rsi_test",
    description="RSI買われすぎ + BB上限タッチでショート（RSI閾値6値探索）",
    config_template={
        "name": "rsi_bb_short_rsi_test",
        "side": "short",
        "indicators": [
            {"type": "rsi", "period": 14},
            {"type": "bollinger", "period": 20, "std_dev": 2.0},
        ],
        "entry_conditions": [
            {
                "type": "threshold",
                "column": "rsi_14",
                "operator": ">",
                "value": "{rsi_threshold}",
            },
            {
                "type": "column_compare",
                "column_a": "close",
                "operator": ">",
                "column_b": "bb_upper_20",
            },
        ],
        "entry_logic": "and",
        "exit": {
            "take_profit_pct": 2.0,
            "stop_loss_pct": 1.0,
        },
    },
    param_ranges=[
        ParameterRange("rsi_threshold", 65, 67, 2, "int"),
    ],
))

# 39. RSI + BB Reversal Short (RSI=65 固定 - Range向け)
# Step 10-1: ショート版もRSI固定でPASS率向上を確認
_register(StrategyTemplate(
    name="rsi_bb_short_f65",
    description="RSI>65 + BB上限タッチでショート（RSI=65固定、exit最小探索）",
    config_template={
        "name": "rsi_bb_short_f65",
        "side": "short",
        "indicators": [
            {"type": "rsi", "period": 14},
            {"type": "bollinger", "period": 20, "std_dev": 2.0},
        ],
        "entry_conditions": [
            {
                "type": "threshold",
                "column": "rsi_14",
                "operator": ">",
                "value": 65,
            },
            {
                "type": "column_compare",
                "column_a": "close",
                "operator": ">",
                "column_b": "bb_upper_20",
            },
        ],
        "entry_logic": "and",
        "exit": {
            "take_profit_pct": 2.0,
            "stop_loss_pct": 1.0,
        },
    },
    param_ranges=[],
))

# 40. RSI + BB Reversal Short (RSI=67 固定 - Downtrend向け)
# Step 10-2: downtrend で40%→50%に届くか最終確認
_register(StrategyTemplate(
    name="rsi_bb_short_f67",
    description="RSI>67 + BB上限タッチでショート（RSI=67固定、exit最小探索）",
    config_template={
        "name": "rsi_bb_short_f67",
        "side": "short",
        "indicators": [
            {"type": "rsi", "period": 14},
            {"type": "bollinger", "period": 20, "std_dev": 2.0},
        ],
        "entry_conditions": [
            {
                "type": "threshold",
                "column": "rsi_14",
                "operator": ">",
                "value": 67,
            },
            {
                "type": "column_compare",
                "column_a": "close",
                "operator": ">",
                "column_b": "bb_upper_20",
            },
        ],
        "entry_logic": "and",
        "exit": {
            "take_profit_pct": 2.0,
            "stop_loss_pct": 1.0,
        },
    },
    param_ranges=[],
))


# =====================================================================
# Step 13: 2025弱uptrend対応 — 新エントリーパターン探索
# 現行 rsi_bb_long_f35 (RSI<35 + BB下限) は2025で信号不足
# → 条件緩和/別ロジックのBB系ロングを4種テスト
# 全てパラメータ固定、探索空間最小（exit 3択のみ）
# =====================================================================

# 41. RSI + BB Long (RSI=40 固定 — 閾値緩和版)
# f35では2025の弱uptrendでRSI<35到達が少ない → RSI<40に緩和
_register(StrategyTemplate(
    name="rsi_bb_long_f40",
    description="RSI<40 + BB下限タッチでロング（RSI閾値緩和版、弱uptrend対応）",
    config_template={
        "name": "rsi_bb_long_f40",
        "side": "long",
        "indicators": [
            {"type": "rsi", "period": 14},
            {"type": "bollinger", "period": 20, "std_dev": 2.0},
        ],
        "entry_conditions": [
            {
                "type": "threshold",
                "column": "rsi_14",
                "operator": "<",
                "value": 40,
            },
            {
                "type": "column_compare",
                "column_a": "close",
                "operator": "<",
                "column_b": "bb_lower_20",
            },
        ],
        "entry_logic": "and",
        "exit": {
            "take_profit_pct": 2.0,
            "stop_loss_pct": 1.0,
        },
    },
    param_ranges=[],
))

# 42. BB Middle + RSI Long（中間バンド押し目買い）
# BB下限ではなくBB中間（SMA20）への押し目 + RSI<45 → 弱uptrendでも頻繁にシグナル
_register(StrategyTemplate(
    name="bb_middle_rsi_long",
    description="close < BB中間バンド + RSI<45でロング（弱uptrend押し目買い）",
    config_template={
        "name": "bb_middle_rsi_long",
        "side": "long",
        "indicators": [
            {"type": "rsi", "period": 14},
            {"type": "bollinger", "period": 20, "std_dev": 2.0},
        ],
        "entry_conditions": [
            {
                "type": "threshold",
                "column": "rsi_14",
                "operator": "<",
                "value": 45,
            },
            {
                "type": "column_compare",
                "column_a": "close",
                "operator": "<",
                "column_b": "bb_middle_20",
            },
        ],
        "entry_logic": "and",
        "exit": {
            "take_profit_pct": 2.0,
            "stop_loss_pct": 1.0,
        },
    },
    param_ranges=[],
))

# 43. EMA Cross + BB Long（モメンタム + バリュー）
# EMA(9)がEMA(21)を上抜け + close < BB中間 → 上昇モメンタム確認 + 割安
_register(StrategyTemplate(
    name="ema_cross_bb_long",
    description="EMA(9)がEMA(21)を上抜け + close < BB中間でロング（モメンタム+バリュー）",
    config_template={
        "name": "ema_cross_bb_long",
        "side": "long",
        "indicators": [
            {"type": "ema", "period": 9},
            {"type": "ema", "period": 21},
            {"type": "bollinger", "period": 20, "std_dev": 2.0},
        ],
        "entry_conditions": [
            {
                "type": "crossover",
                "fast": "ema_9",
                "slow": "ema_21",
                "direction": "above",
            },
            {
                "type": "column_compare",
                "column_a": "close",
                "operator": "<",
                "column_b": "bb_middle_20",
            },
        ],
        "entry_logic": "and",
        "exit": {
            "take_profit_pct": 2.0,
            "stop_loss_pct": 1.0,
        },
    },
    param_ranges=[],
))

# 44. BB Squeeze Long（ボラティリティ収縮→ブレイクアウト）
# BB bandwidth < 閾値（低ボラ=スクイーズ）+ close > BB中間 → ブレイクアウト初動
_register(StrategyTemplate(
    name="bb_squeeze_long",
    description="BB幅収縮（スクイーズ）+ close > BB中間でロング（ブレイクアウト初動）",
    config_template={
        "name": "bb_squeeze_long",
        "side": "long",
        "indicators": [
            {"type": "bollinger", "period": 20, "std_dev": 2.0},
        ],
        "entry_conditions": [
            {
                "type": "threshold",
                "column": "bb_width_20",
                "operator": "<",
                "value": 0.04,
            },
            {
                "type": "column_compare",
                "column_a": "close",
                "operator": ">",
                "column_b": "bb_middle_20",
            },
        ],
        "entry_logic": "and",
        "exit": {
            "take_profit_pct": 2.0,
            "stop_loss_pct": 1.0,
        },
    },
    param_ranges=[],
))

# --- Step 14: Downtrend Short 固定テンプレート ---

# 45. Volume Spike Short (固定: rvol_period=20, rvol_threshold=3)
# OOS 61% PASS in downtrend。パラメータ固定でWFA安定性を検証
_register(StrategyTemplate(
    name="volume_spike_short_dt",
    description="出来高急増(RVOL>3) + 陰線でショート（DT順張り、パラメータ固定）",
    config_template={
        "name": "volume_spike_short_dt",
        "side": "short",
        "indicators": [
            {"type": "rvol", "period": 20},
        ],
        "entry_conditions": [
            {
                "type": "threshold",
                "column": "rvol_20",
                "operator": ">",
                "value": 3,
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
    param_ranges=[],
))

# 46. RSI + BB Upper Short (RSI>65固定、DT用)
# rsi_bb_reversal_shortのRSI=65固定版。RSI>70が最多勝だが65も21勝で汎用性高い
_register(StrategyTemplate(
    name="rsi_bb_short_dt65",
    description="RSI>65 + BB上限タッチでショート（DT戻り売り、RSI=65固定）",
    config_template={
        "name": "rsi_bb_short_dt65",
        "side": "short",
        "indicators": [
            {"type": "rsi", "period": 14},
            {"type": "bollinger", "period": 20, "std_dev": 2.0},
        ],
        "entry_conditions": [
            {
                "type": "threshold",
                "column": "rsi_14",
                "operator": ">",
                "value": 65,
            },
            {
                "type": "column_compare",
                "column_a": "close",
                "operator": ">",
                "column_b": "bb_upper_20",
            },
        ],
        "entry_logic": "and",
        "exit": {
            "take_profit_pct": 2.0,
            "stop_loss_pct": 1.0,
        },
    },
    param_ranges=[],
))

# 47. BB Upper + Volume Short (固定: rvol_threshold=2.5)
# bb_volume_reversal_shortのrvol=2.5固定版。46% OOS PASS
_register(StrategyTemplate(
    name="bb_vol_short_dt",
    description="BB上限タッチ + 出来高急増(RVOL>2.5)でショート（DT固定）",
    config_template={
        "name": "bb_vol_short_dt",
        "side": "short",
        "indicators": [
            {"type": "bollinger", "period": 20, "std_dev": 2.0},
            {"type": "rvol", "period": 20},
        ],
        "entry_conditions": [
            {
                "type": "column_compare",
                "column_a": "close",
                "operator": ">",
                "column_b": "bb_upper_20",
            },
            {
                "type": "threshold",
                "column": "rvol_20",
                "operator": ">",
                "value": 2.5,
            },
        ],
        "entry_logic": "and",
        "exit": {
            "take_profit_pct": 2.0,
            "stop_loss_pct": 1.0,
        },
    },
    param_ranges=[],
))

# --- Step 14b: DT Short 試行錯誤テンプレート ---

# 48. RSI>60 + BB上限ショート（DT中の弱い戻りでもエントリ）
_register(StrategyTemplate(
    name="rsi_bb_short_f60",
    description="RSI>60 + BB上限タッチでショート（DT弱戻りキャッチ）",
    config_template={
        "name": "rsi_bb_short_f60",
        "side": "short",
        "indicators": [
            {"type": "rsi", "period": 14},
            {"type": "bollinger", "period": 20, "std_dev": 2.0},
        ],
        "entry_conditions": [
            {"type": "threshold", "column": "rsi_14", "operator": ">", "value": 60},
            {"type": "column_compare", "column_a": "close", "operator": ">", "column_b": "bb_upper_20"},
        ],
        "entry_logic": "and",
        "exit": {"take_profit_pct": 2.0, "stop_loss_pct": 1.0},
    },
    param_ranges=[],
))

# 49. RSI>55 + BB中間タッチでショート（DT中はBB中間が抵抗線として機能）
_register(StrategyTemplate(
    name="rsi_bb_mid_short_f55",
    description="RSI>55 + close > BB中間でショート（DT戻りのBB中間抵抗）",
    config_template={
        "name": "rsi_bb_mid_short_f55",
        "side": "short",
        "indicators": [
            {"type": "rsi", "period": 14},
            {"type": "bollinger", "period": 20, "std_dev": 2.0},
        ],
        "entry_conditions": [
            {"type": "threshold", "column": "rsi_14", "operator": ">", "value": 55},
            {"type": "column_compare", "column_a": "close", "operator": ">", "column_b": "bb_middle_20"},
        ],
        "entry_logic": "and",
        "exit": {"take_profit_pct": 2.0, "stop_loss_pct": 1.0},
    },
    param_ranges=[],
))

# 50. RSI>50 + BB中間ショート（最も緩い条件 — DT中のニュートラル域からの下落）
_register(StrategyTemplate(
    name="rsi_bb_mid_short_f50",
    description="RSI>50 + close > BB中間でショート（DT中の中立域反落）",
    config_template={
        "name": "rsi_bb_mid_short_f50",
        "side": "short",
        "indicators": [
            {"type": "rsi", "period": 14},
            {"type": "bollinger", "period": 20, "std_dev": 2.0},
        ],
        "entry_conditions": [
            {"type": "threshold", "column": "rsi_14", "operator": ">", "value": 50},
            {"type": "column_compare", "column_a": "close", "operator": ">", "column_b": "bb_middle_20"},
        ],
        "entry_logic": "and",
        "exit": {"take_profit_pct": 2.0, "stop_loss_pct": 1.0},
    },
    param_ranges=[],
))

# 51. EMA(9) < EMA(21) + close > BB中間ショート（トレンド確認 + 戻り）
_register(StrategyTemplate(
    name="ema_cross_bb_short",
    description="EMA(9)<EMA(21) + close > BB中間でショート（DT確認+戻り売り）",
    config_template={
        "name": "ema_cross_bb_short",
        "side": "short",
        "indicators": [
            {"type": "ema", "period": 9},
            {"type": "ema", "period": 21},
            {"type": "bollinger", "period": 20, "std_dev": 2.0},
        ],
        "entry_conditions": [
            {"type": "crossover", "fast": "ema_9", "slow": "ema_21", "direction": "below"},
            {"type": "column_compare", "column_a": "close", "operator": ">", "column_b": "bb_middle_20"},
        ],
        "entry_logic": "and",
        "exit": {"take_profit_pct": 2.0, "stop_loss_pct": 1.0},
    },
    param_ranges=[],
))

# 52. RSI>60 + BB中間ショート（RSI60 + BB中間 = DT最適コンボ仮説）
_register(StrategyTemplate(
    name="rsi_bb_mid_short_f60",
    description="RSI>60 + close > BB中間でショート（DT戻り売りのスイートスポット仮説）",
    config_template={
        "name": "rsi_bb_mid_short_f60",
        "side": "short",
        "indicators": [
            {"type": "rsi", "period": 14},
            {"type": "bollinger", "period": 20, "std_dev": 2.0},
        ],
        "entry_conditions": [
            {"type": "threshold", "column": "rsi_14", "operator": ">", "value": 60},
            {"type": "column_compare", "column_a": "close", "operator": ">", "column_b": "bb_middle_20"},
        ],
        "entry_logic": "and",
        "exit": {"take_profit_pct": 2.0, "stop_loss_pct": 1.0},
    },
    param_ranges=[],
))

# 53. EMAクロスのみショート（BB無し → EMA確認の効果を分離テスト）
_register(StrategyTemplate(
    name="ema_cross_short_only",
    description="EMA(9)<EMA(21)クロスのみでショート（EMA確認効果の分離検証）",
    config_template={
        "name": "ema_cross_short_only",
        "side": "short",
        "indicators": [
            {"type": "ema", "period": 9},
            {"type": "ema", "period": 21},
        ],
        "entry_conditions": [
            {"type": "crossover", "fast": "ema_9", "slow": "ema_21", "direction": "below"},
        ],
        "entry_logic": "and",
        "exit": {"take_profit_pct": 2.0, "stop_loss_pct": 1.0},
    },
    param_ranges=[],
))

# 54. EMA(5) < EMA(13) + BB中間ショート（高速EMAバリエーション）
_register(StrategyTemplate(
    name="ema_fast_cross_bb_short",
    description="EMA(5)<EMA(13) + close > BB中間でショート（高速EMAでDT早期エントリー）",
    config_template={
        "name": "ema_fast_cross_bb_short",
        "side": "short",
        "indicators": [
            {"type": "ema", "period": 5},
            {"type": "ema", "period": 13},
            {"type": "bollinger", "period": 20, "std_dev": 2.0},
        ],
        "entry_conditions": [
            {"type": "crossover", "fast": "ema_5", "slow": "ema_13", "direction": "below"},
            {"type": "column_compare", "column_a": "close", "operator": ">", "column_b": "bb_middle_20"},
        ],
        "entry_logic": "and",
        "exit": {"take_profit_pct": 2.0, "stop_loss_pct": 1.0},
    },
    param_ranges=[],
))

# 55. EMA(9)<EMA(21) + RSI>55 + BB中間ショート（トリプル確認）
_register(StrategyTemplate(
    name="ema_rsi_bb_short",
    description="EMA(9)<EMA(21) + RSI>55 + close > BB中間（トリプル確認で精度向上仮説）",
    config_template={
        "name": "ema_rsi_bb_short",
        "side": "short",
        "indicators": [
            {"type": "ema", "period": 9},
            {"type": "ema", "period": 21},
            {"type": "rsi", "period": 14},
            {"type": "bollinger", "period": 20, "std_dev": 2.0},
        ],
        "entry_conditions": [
            {"type": "crossover", "fast": "ema_9", "slow": "ema_21", "direction": "below"},
            {"type": "threshold", "column": "rsi_14", "operator": ">", "value": 55},
            {"type": "column_compare", "column_a": "close", "operator": ">", "column_b": "bb_middle_20"},
        ],
        "entry_logic": "and",
        "exit": {"take_profit_pct": 2.0, "stop_loss_pct": 1.0},
    },
    param_ranges=[],
))

# =====================================================================
# VP Deep Dive テンプレート（Step 14: VP複合条件探索）
# vp_pullback_long のトレード数不足を解決するため、
# パラメータ緩和・複合条件・POC/HVN活用・ショート版の6種を追加
# =====================================================================

# 56. VP Pullback Wide（パラメータ緩和版）
# n_bins小＝粗いプロファイル＝LVN検出しやすい、tolerance大＝タッチ判定緩い
# break_margin小＝ブレイク判定緩い、min_bars_after_break小＝再タッチまでの待機短い
_register(StrategyTemplate(
    name="vp_pullback_wide",
    description="VP LVN初タッチ（パラメータ緩和版: 粗いビン・広いタッチ判定でシグナル頻度UP）",
    config_template={
        "name": "vp_pullback_wide",
        "side": "long",
        "indicators": [
            {
                "type": "volume_profile",
                "n_bins": "{n_bins}",
                "smoothing": 3,
                "touch_tolerance": "{touch_tolerance}",
                "break_margin": "{break_margin}",
                "min_bars_after_break": "{min_bars_after_break}",
            },
        ],
        "entry_conditions": [
            {
                "type": "threshold",
                "column": "vp_lvn_first_touch",
                "operator": "==",
                "value": 1,
            },
        ],
        "entry_logic": "and",
        "exit": {"take_profit_pct": 2.0, "stop_loss_pct": 1.0},
    },
    param_ranges=[
        ParameterRange("n_bins", 20, 40, 10, "int"),
        ParameterRange("touch_tolerance", 1.0, 2.0, 0.5, "float"),
        ParameterRange("break_margin", 1.0, 3.0, 1.0, "float"),
        ParameterRange("min_bars_after_break", 3, 7, 2, "int"),
    ],
))

# 57. VP + RSI Pullback Long（LVN初タッチ + RSI売られすぎ）
# VP単体のシグナルにRSIフィルタを追加し、押し目の質を向上
_register(StrategyTemplate(
    name="vp_rsi_pullback_long",
    description="VP LVN初タッチ + RSI<40 でロング（VP押し目 + RSI売られすぎの複合）",
    config_template={
        "name": "vp_rsi_pullback_long",
        "side": "long",
        "indicators": [
            {
                "type": "volume_profile",
                "n_bins": "{n_bins}",
                "smoothing": 3,
                "touch_tolerance": "{touch_tolerance}",
                "break_margin": "{break_margin}",
                "min_bars_after_break": "{min_bars_after_break}",
            },
            {"type": "rsi", "period": 14},
        ],
        "entry_conditions": [
            {
                "type": "threshold",
                "column": "vp_lvn_first_touch",
                "operator": "==",
                "value": 1,
            },
            {
                "type": "threshold",
                "column": "rsi_14",
                "operator": "<",
                "value": "{rsi_threshold}",
            },
        ],
        "entry_logic": "and",
        "exit": {"take_profit_pct": 2.0, "stop_loss_pct": 1.0},
    },
    param_ranges=[
        ParameterRange("n_bins", 20, 40, 10, "int"),
        ParameterRange("touch_tolerance", 1.0, 2.0, 0.5, "float"),
        ParameterRange("break_margin", 1.0, 3.0, 1.0, "float"),
        ParameterRange("min_bars_after_break", 3, 7, 2, "int"),
        ParameterRange("rsi_threshold", 35, 45, 5, "int"),
    ],
))

# 58. VP + BB Pullback Long（LVN初タッチ + BB下半分）
# BBミドルより下 = 平均以下の価格でVPシグナルが出た場合のみエントリー
_register(StrategyTemplate(
    name="vp_bb_pullback_long",
    description="VP LVN初タッチ + close < BB中間でロング（VP押し目 + BB下半分フィルタ）",
    config_template={
        "name": "vp_bb_pullback_long",
        "side": "long",
        "indicators": [
            {
                "type": "volume_profile",
                "n_bins": "{n_bins}",
                "smoothing": 3,
                "touch_tolerance": "{touch_tolerance}",
                "break_margin": "{break_margin}",
                "min_bars_after_break": "{min_bars_after_break}",
            },
            {"type": "bollinger", "period": 20, "std_dev": 2.0},
        ],
        "entry_conditions": [
            {
                "type": "threshold",
                "column": "vp_lvn_first_touch",
                "operator": "==",
                "value": 1,
            },
            {
                "type": "column_compare",
                "column_a": "close",
                "operator": "<",
                "column_b": "bb_middle_20",
            },
        ],
        "entry_logic": "and",
        "exit": {"take_profit_pct": 2.0, "stop_loss_pct": 1.0},
    },
    param_ranges=[
        ParameterRange("n_bins", 20, 40, 10, "int"),
        ParameterRange("touch_tolerance", 1.0, 2.0, 0.5, "float"),
        ParameterRange("break_margin", 1.0, 3.0, 1.0, "float"),
        ParameterRange("min_bars_after_break", 3, 7, 2, "int"),
    ],
))

# 59. VP POC + RSI Long（POC近辺 + RSI売られすぎ）
# first_touchに依存しない新しいVPシグナル: POCへの接近をエントリーに使用
# POCは最大出来高価格 = 強力なサポート/レジスタンス
_register(StrategyTemplate(
    name="vp_poc_rsi_long",
    description="close <= VP POC + RSI<45 でロング（POCサポート + RSI押し目の複合）",
    config_template={
        "name": "vp_poc_rsi_long",
        "side": "long",
        "indicators": [
            {
                "type": "volume_profile",
                "n_bins": "{n_bins}",
                "smoothing": 3,
                "touch_tolerance": 0.5,
            },
            {"type": "rsi", "period": 14},
        ],
        "entry_conditions": [
            {
                "type": "column_compare",
                "column_a": "close",
                "operator": "<=",
                "column_b": "vp_poc",
            },
            {
                "type": "threshold",
                "column": "rsi_14",
                "operator": "<",
                "value": "{rsi_threshold}",
            },
        ],
        "entry_logic": "and",
        "exit": {"take_profit_pct": 2.0, "stop_loss_pct": 1.0},
    },
    param_ranges=[
        ParameterRange("n_bins", 30, 50, 10, "int"),
        ParameterRange("rsi_threshold", 35, 45, 5, "int"),
    ],
))

# 60. VP HVN + RSI Long（HVNサポート + RSI売られすぎ）
# HVN = 出来高集中ゾーン = 強いサポート。POCより柔軟で複数検出される
_register(StrategyTemplate(
    name="vp_hvn_rsi_long",
    description="close <= VP HVN_1 + RSI<45 でロング（HVNサポート + RSI押し目の複合）",
    config_template={
        "name": "vp_hvn_rsi_long",
        "side": "long",
        "indicators": [
            {
                "type": "volume_profile",
                "n_bins": "{n_bins}",
                "smoothing": 3,
                "touch_tolerance": 0.5,
            },
            {"type": "rsi", "period": 14},
        ],
        "entry_conditions": [
            {
                "type": "column_compare",
                "column_a": "close",
                "operator": "<=",
                "column_b": "vp_hvn_1",
            },
            {
                "type": "threshold",
                "column": "rsi_14",
                "operator": "<",
                "value": "{rsi_threshold}",
            },
        ],
        "entry_logic": "and",
        "exit": {"take_profit_pct": 2.0, "stop_loss_pct": 1.0},
    },
    param_ranges=[
        ParameterRange("n_bins", 30, 50, 10, "int"),
        ParameterRange("rsi_threshold", 35, 45, 5, "int"),
    ],
))

# 61. VP Pullback Short（LVN初タッチ ショート版）
# LVNを上に突き抜けた後の戻りでショート（レジスタンスとして機能する仮説）
_register(StrategyTemplate(
    name="vp_pullback_short",
    description="VP LVN初タッチでショート（LVNレジスタンス仮説: DT向け）",
    config_template={
        "name": "vp_pullback_short",
        "side": "short",
        "indicators": [
            {
                "type": "volume_profile",
                "n_bins": "{n_bins}",
                "smoothing": 3,
                "touch_tolerance": "{touch_tolerance}",
                "break_margin": "{break_margin}",
                "min_bars_after_break": "{min_bars_after_break}",
            },
        ],
        "entry_conditions": [
            {
                "type": "threshold",
                "column": "vp_lvn_first_touch",
                "operator": "==",
                "value": 1,
            },
        ],
        "entry_logic": "and",
        "exit": {"take_profit_pct": 2.0, "stop_loss_pct": 1.0},
    },
    param_ranges=[
        ParameterRange("n_bins", 20, 40, 10, "int"),
        ParameterRange("touch_tolerance", 1.0, 2.0, 0.5, "float"),
        ParameterRange("break_margin", 1.0, 3.0, 1.0, "float"),
        ParameterRange("min_bars_after_break", 3, 7, 2, "int"),
    ],
))

# 62. ADX + BB Long (Uptrend向け — トレンド強度フィルタ)
# ADXが高い = トレンドが明確 → BB下限タッチからの反発が信頼性高い
_register(StrategyTemplate(
    name="adx_bb_long",
    description="ADX>=閾値（強トレンド確認） + BB下限タッチでロング",
    config_template={
        "name": "adx_bb_long",
        "side": "long",
        "indicators": [
            {"type": "adx", "period": 14},
            {"type": "bollinger", "period": 20, "std_dev": 2.0},
        ],
        "entry_conditions": [
            {
                "type": "threshold",
                "column": "adx_14",
                "operator": ">=",
                "value": "{adx_threshold}",
            },
            {
                "type": "column_compare",
                "column_a": "close",
                "operator": "<",
                "column_b": "bb_lower_20",
            },
        ],
        "entry_logic": "and",
        "exit": {
            "take_profit_pct": 2.0,
            "stop_loss_pct": 1.0,
        },
    },
    param_ranges=[
        ParameterRange("adx_threshold", 20, 30, 5, "int"),
    ],
))

# 63. ADX + BB Short (Downtrend向け — トレンド強度フィルタ)
# ADXが高い = トレンドが明確 → BB上限タッチからの反落が信頼性高い
_register(StrategyTemplate(
    name="adx_bb_short",
    description="ADX>=閾値（強トレンド確認） + BB上限タッチでショート",
    config_template={
        "name": "adx_bb_short",
        "side": "short",
        "indicators": [
            {"type": "adx", "period": 14},
            {"type": "bollinger", "period": 20, "std_dev": 2.0},
        ],
        "entry_conditions": [
            {
                "type": "threshold",
                "column": "adx_14",
                "operator": ">=",
                "value": "{adx_threshold}",
            },
            {
                "type": "column_compare",
                "column_a": "close",
                "operator": ">",
                "column_b": "bb_upper_20",
            },
        ],
        "entry_logic": "and",
        "exit": {
            "take_profit_pct": 2.0,
            "stop_loss_pct": 1.0,
        },
    },
    param_ranges=[
        ParameterRange("adx_threshold", 20, 30, 5, "int"),
    ],
))

# 64. EMA(5)>EMA(13) + BB中間ロング（DT成功パターンのUT完全ミラー）
# DT版 ema_fast_cross_bb_short (ROBUST 24%) のロング反転版
_register(StrategyTemplate(
    name="ema_fast_cross_bb_long",
    description="EMA(5)>EMA(13) + close < BB中間でロング（高速EMAでUT押し目エントリー）",
    config_template={
        "name": "ema_fast_cross_bb_long",
        "side": "long",
        "indicators": [
            {"type": "ema", "period": 5},
            {"type": "ema", "period": 13},
            {"type": "bollinger", "period": 20, "std_dev": 2.0},
        ],
        "entry_conditions": [
            {"type": "crossover", "fast": "ema_5", "slow": "ema_13", "direction": "above"},
            {"type": "column_compare", "column_a": "close", "operator": "<", "column_b": "bb_middle_20"},
        ],
        "entry_logic": "and",
        "exit": {"take_profit_pct": 2.0, "stop_loss_pct": 1.0},
    },
    param_ranges=[],
))

# 65. EMA(5)>EMA(13) + BB下限ロング（押し目が深い版）— crossover版はトレード不足で全滅
# UT向け：上昇モメンタム確認 + BB下限タッチの深い押し目
_register(StrategyTemplate(
    name="ema_fast_bb_lower_long",
    description="EMA(5)>EMA(13) + close < BB下限でロング（深い押し目版）",
    config_template={
        "name": "ema_fast_bb_lower_long",
        "side": "long",
        "indicators": [
            {"type": "ema", "period": 5},
            {"type": "ema", "period": 13},
            {"type": "bollinger", "period": 20, "std_dev": 2.0},
        ],
        "entry_conditions": [
            {"type": "crossover", "fast": "ema_5", "slow": "ema_13", "direction": "above"},
            {"type": "column_compare", "column_a": "close", "operator": "<", "column_b": "bb_lower_20"},
        ],
        "entry_logic": "and",
        "exit": {"take_profit_pct": 2.0, "stop_loss_pct": 1.0},
    },
    param_ranges=[],
))

# 66. EMA(5)>EMA(13)状態確認 + BB下限ロング（状態型 — crossover瞬間ではない）
# crossover版はトレード不足→状態確認に変更。UT向け押し目
_register(StrategyTemplate(
    name="ema_state_bb_lower_long",
    description="EMA(5)>EMA(13)状態 + close < BB下限でロング（状態型で十分なトレード数確保）",
    config_template={
        "name": "ema_state_bb_lower_long",
        "side": "long",
        "indicators": [
            {"type": "ema", "period": 5},
            {"type": "ema", "period": 13},
            {"type": "bollinger", "period": 20, "std_dev": 2.0},
        ],
        "entry_conditions": [
            {"type": "column_compare", "column_a": "ema_5", "operator": ">", "column_b": "ema_13"},
            {"type": "column_compare", "column_a": "close", "operator": "<", "column_b": "bb_lower_20"},
        ],
        "entry_logic": "and",
        "exit": {"take_profit_pct": 2.0, "stop_loss_pct": 1.0},
    },
    param_ranges=[],
))

# 67. EMA(5)>EMA(13)状態確認 + BB中間ロング（DT成功パターンの完全ミラー — 状態型）
# 結果: UT向けPnLマイナス。EMAモメンタム確認はUT側では機能しない
_register(StrategyTemplate(
    name="ema_state_bb_mid_long",
    description="EMA(5)>EMA(13)状態 + close < BB中間でロング（DT成功パターンのUTミラー）",
    config_template={
        "name": "ema_state_bb_mid_long",
        "side": "long",
        "indicators": [
            {"type": "ema", "period": 5},
            {"type": "ema", "period": 13},
            {"type": "bollinger", "period": 20, "std_dev": 2.0},
        ],
        "entry_conditions": [
            {"type": "column_compare", "column_a": "ema_5", "operator": ">", "column_b": "ema_13"},
            {"type": "column_compare", "column_a": "close", "operator": "<", "column_b": "bb_middle_20"},
        ],
        "entry_logic": "and",
        "exit": {"take_profit_pct": 2.0, "stop_loss_pct": 1.0},
    },
    param_ranges=[],
))

# 68. +DI > -DI 状態確認 + BB下限ロング（方向性DIフィルタ）
# ADXは強度、+DI/-DIは方向性。+DI>-DI = 上昇方向確認 + BB下限押し目
_register(StrategyTemplate(
    name="di_bb_lower_long",
    description="+DI>-DI（上昇方向確認） + close < BB下限でロング",
    config_template={
        "name": "di_bb_lower_long",
        "side": "long",
        "indicators": [
            {"type": "adx", "period": 14},
            {"type": "bollinger", "period": 20, "std_dev": 2.0},
        ],
        "entry_conditions": [
            {"type": "column_compare", "column_a": "plus_di_14", "operator": ">", "column_b": "minus_di_14"},
            {"type": "column_compare", "column_a": "close", "operator": "<", "column_b": "bb_lower_20"},
        ],
        "entry_logic": "and",
        "exit": {"take_profit_pct": 2.0, "stop_loss_pct": 1.0},
    },
    param_ranges=[],
))

# 69. -DI > +DI 状態確認 + BB上限ショート（方向性DIフィルタ DT版）
# -DI>+DI = 下降方向確認 + BB上限からの反落
_register(StrategyTemplate(
    name="di_bb_upper_short",
    description="-DI>+DI（下降方向確認） + close > BB上限でショート",
    config_template={
        "name": "di_bb_upper_short",
        "side": "short",
        "indicators": [
            {"type": "adx", "period": 14},
            {"type": "bollinger", "period": 20, "std_dev": 2.0},
        ],
        "entry_conditions": [
            {"type": "column_compare", "column_a": "minus_di_14", "operator": ">", "column_b": "plus_di_14"},
            {"type": "column_compare", "column_a": "close", "operator": ">", "column_b": "bb_upper_20"},
        ],
        "entry_logic": "and",
        "exit": {"take_profit_pct": 2.0, "stop_loss_pct": 1.0},
    },
    param_ranges=[],
))

