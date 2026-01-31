"""
オプティマイザ自動分析

最適化結果をルールベースで分析し、品質警告・戦略評価・アクション推奨を
テキスト出力する。Streamlit には依存しない純粋関数群。
"""

import math
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from optimizer.results import OptimizationResultSet, OptimizationEntry


# ============================================================
# データ構造
# ============================================================

class InsightLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    SUCCESS = "success"
    ERROR = "error"


@dataclass
class AnalysisInsight:
    """1つの分析結果"""
    level: InsightLevel
    category: str  # "quality_warning", "strategy_quality", "action", "cross_pattern", "cross_risk", "cross_action"
    message: str


# ============================================================
# 閾値定数
# ============================================================

TRADES_STAT_WEAK_THRESHOLD = 10  # これ未満は統計的に弱い
DD_HIGH_THRESHOLD = 20.0         # DDがこれ以上で高DD警告
DD_MULTI_RISK_THRESHOLD = 15.0   # 複数銘柄でこれ以上なら累積リスク警告
SHARPE_EXCELLENT_THRESHOLD = 2.0 # シャープ比がこれ以上で優秀
PNL_CV_HIGH_THRESHOLD = 1.0     # PnL変動係数がこれ以上で分散が大きい
PARAM_SIMILARITY_THRESHOLD = 80.0  # パラメータ類似度がこれ以上で共通運用可
DOMINANCE_HIGH_THRESHOLD = 0.7   # テンプレート支配率がこれ以上で汎用的
DOMINANCE_LOW_THRESHOLD = 0.5    # テンプレート支配率がこれ未満で銘柄依存
PARAM_CV_CONVERGED = 0.2        # CV がこれ以下でパラメータ収束
PARAM_CV_DIVERGED = 0.5         # CV がこれ以上でパラメータ発散
REGIME_VIABILITY_HIGH = 0.8     # 採用率がこれ以上でレジーム有効
REGIME_VIABILITY_LOW = 0.5      # 採用率がこれ未満でレジーム微妙


# ============================================================
# ヘルパー
# ============================================================

def _is_viable(entry: "OptimizationEntry") -> bool:
    """採用基準の判定"""
    pf = entry.metrics.profit_factor
    pnl = entry.metrics.total_profit_pct
    trades = entry.metrics.total_trades
    return pf > 1.0 and pnl > 0 and trades >= 5


def _get_regime_bests(
    result_set: "OptimizationResultSet",
) -> Dict[str, dict]:
    """全レジームのベスト+viability を取得"""
    regimes = sorted(set(e.trend_regime for e in result_set.entries))
    result = {}
    for regime in regimes:
        regime_set = result_set.filter_regime(regime)
        best = regime_set.best
        if best:
            result[regime] = {
                "entry": best,
                "is_viable": _is_viable(best),
            }
    return result


def _get_entry_side(entry: "OptimizationEntry") -> str:
    """エントリーの売買方向を取得（long/short）"""
    side = entry.config.get("side", "")
    if side:
        return side
    # config が空の場合はテンプレート名から推定
    if "_short" in entry.template_name.lower():
        return "short"
    return "long"


def _calculate_param_similarity(
    bests: List[dict],
) -> Dict[str, float]:
    """同一テンプレート使用エントリー間のパラメータ類似度を計算"""
    all_params: Dict[str, List[float]] = {}
    for b in bests:
        for k, v in b["entry"].params.items():
            if k not in all_params:
                all_params[k] = []
            try:
                all_params[k].append(float(v))
            except (ValueError, TypeError):
                pass

    result = {}
    for param_name, values in all_params.items():
        if len(values) < 2:
            continue
        mean_val = np.mean(values)
        std_val = np.std(values)
        cv = std_val / abs(mean_val) if abs(mean_val) > 1e-10 else 0
        result[param_name] = max(0, (1 - cv)) * 100

    return result


# ============================================================
# 単一結果分析
# ============================================================

def analyze_single_result(
    result_set: "OptimizationResultSet",
    viable_strategies: Dict[str, "OptimizationEntry"],
) -> List[AnalysisInsight]:
    """単一結果セットの自動分析"""
    insights: List[AnalysisInsight] = []

    regime_bests = _get_regime_bests(result_set)

    insights.extend(_check_quality_warnings(regime_bests, viable_strategies))
    insights.extend(_evaluate_strategy_quality(regime_bests, viable_strategies))
    insights.extend(_recommend_actions(regime_bests, viable_strategies))

    return insights


def _check_quality_warnings(
    regime_bests: Dict[str, dict],
    viable_strategies: Dict[str, "OptimizationEntry"],
) -> List[AnalysisInsight]:
    """品質警告"""
    insights: List[AnalysisInsight] = []

    # W5: 全レジーム不採用
    if not viable_strategies and regime_bests:
        insights.append(AnalysisInsight(
            level=InsightLevel.ERROR,
            category="quality_warning",
            message="全レジームで採用基準を満たす戦略がありません。パラメータ範囲やテンプレートの見直しを推奨します。",
        ))
        return insights  # 他の警告は不要

    for regime, entry in viable_strategies.items():
        m = entry.metrics
        regime_label = {"uptrend": "Uptrend", "downtrend": "Downtrend", "range": "Range"}.get(regime, regime)

        # W1: 取引数が統計的に弱い
        if 5 <= m.total_trades < TRADES_STAT_WEAK_THRESHOLD:
            insights.append(AnalysisInsight(
                level=InsightLevel.WARNING,
                category="quality_warning",
                message=f"{regime_label}のベスト戦略は取引数{m.total_trades}件で統計的に弱い（10件以上推奨）",
            ))

        # W2: PF = Infinity
        if math.isinf(m.profit_factor):
            insights.append(AnalysisInsight(
                level=InsightLevel.WARNING,
                category="quality_warning",
                message=f"{regime_label}のベスト戦略はPF=Infinity（負けトレード0件）。過学習リスクあり",
            ))

        # W3: 勝率100%
        if m.win_rate >= 100.0:
            insights.append(AnalysisInsight(
                level=InsightLevel.WARNING,
                category="quality_warning",
                message=f"{regime_label}のベスト戦略は勝率100%。過学習の兆候",
            ))

        # W4: 最大DD=0%
        if m.max_drawdown_pct == 0.0 and m.total_trades >= 1:
            insights.append(AnalysisInsight(
                level=InsightLevel.WARNING,
                category="quality_warning",
                message=f"{regime_label}のベスト戦略は最大DD=0%。ポジションサイズが小さすぎるか取引数が不足",
            ))

    return insights


def _evaluate_strategy_quality(
    regime_bests: Dict[str, dict],
    viable_strategies: Dict[str, "OptimizationEntry"],
) -> List[AnalysisInsight]:
    """戦略品質評価"""
    insights: List[AnalysisInsight] = []

    if not viable_strategies:
        return insights

    all_regimes = list(regime_bests.keys())

    # Q3/Q4: レジームカバレッジ
    if len(viable_strategies) == len(all_regimes) and len(all_regimes) >= 2:
        insights.append(AnalysisInsight(
            level=InsightLevel.SUCCESS,
            category="strategy_quality",
            message=f"全{len(all_regimes)}レジームで採用可能な戦略あり。堅牢性が高い",
        ))
    elif len(viable_strategies) < len(all_regimes):
        not_viable = [r for r in all_regimes if r not in viable_strategies]
        labels = [{"uptrend": "Uptrend", "downtrend": "Downtrend", "range": "Range"}.get(r, r) for r in not_viable]
        insights.append(AnalysisInsight(
            level=InsightLevel.WARNING,
            category="strategy_quality",
            message=f"{', '.join(labels)}は採用基準未達。このレジームでの運用は見送り推奨",
        ))

    # Q1/Q2: テンプレート統一性
    templates_used = set(v.template_name for v in viable_strategies.values())
    if len(templates_used) == 1 and len(viable_strategies) >= 2:
        tpl = next(iter(templates_used))
        insights.append(AnalysisInsight(
            level=InsightLevel.SUCCESS,
            category="strategy_quality",
            message=f"全レジームで{tpl}が最良。テンプレートの統一性が高い",
        ))
    elif len(templates_used) >= 2:
        tpl_list = ", ".join(sorted(templates_used))
        insights.append(AnalysisInsight(
            level=InsightLevel.INFO,
            category="strategy_quality",
            message=f"レジームごとに最適テンプレートが異なる（{tpl_list}）。レジーム別に個別設定が必要",
        ))

    return insights


def _recommend_actions(
    regime_bests: Dict[str, dict],
    viable_strategies: Dict[str, "OptimizationEntry"],
) -> List[AnalysisInsight]:
    """アクション推奨"""
    insights: List[AnalysisInsight] = []

    if not viable_strategies:
        return insights

    viable_regimes = set(viable_strategies.keys())

    # A1: uptrend限定有効
    if viable_regimes == {"uptrend"}:
        side = _get_entry_side(viable_strategies["uptrend"])
        if side == "long":
            insights.append(AnalysisInsight(
                level=InsightLevel.INFO,
                category="action",
                message="Uptrendでのみ有効な戦略。上昇相場限定のLong only運用を推奨",
            ))

    # A2: downtrend限定有効
    if viable_regimes == {"downtrend"}:
        side = _get_entry_side(viable_strategies["downtrend"])
        if side == "short":
            insights.append(AnalysisInsight(
                level=InsightLevel.INFO,
                category="action",
                message="Downtrendでのみ有効な戦略。下降相場限定のShort only運用を推奨",
            ))

    # A3: 全レジームプラス
    all_positive = all(
        e.metrics.total_profit_pct > 0 for e in viable_strategies.values()
    )
    if all_positive and len(viable_strategies) >= 2:
        insights.append(AnalysisInsight(
            level=InsightLevel.SUCCESS,
            category="action",
            message="全レジームで利益率プラス。常時稼働可能",
        ))

    # A4: シャープ比優秀
    for regime, entry in viable_strategies.items():
        if entry.metrics.sharpe_ratio >= SHARPE_EXCELLENT_THRESHOLD:
            regime_label = {"uptrend": "Uptrend", "downtrend": "Downtrend", "range": "Range"}.get(regime, regime)
            insights.append(AnalysisInsight(
                level=InsightLevel.SUCCESS,
                category="action",
                message=f"{regime_label}でシャープ比{entry.metrics.sharpe_ratio:.2f}。リスクリターン比が優秀",
            ))

    # A5: DD高い
    for regime, entry in viable_strategies.items():
        if entry.metrics.max_drawdown_pct > DD_HIGH_THRESHOLD:
            regime_label = {"uptrend": "Uptrend", "downtrend": "Downtrend", "range": "Range"}.get(regime, regime)
            insights.append(AnalysisInsight(
                level=InsightLevel.WARNING,
                category="action",
                message=f"{regime_label}の最大DDが{entry.metrics.max_drawdown_pct:.1f}%。資金管理に注意",
            ))

    return insights


# ============================================================
# 比較分析
# ============================================================

def analyze_comparison(
    comparison_results: List["OptimizationResultSet"],
) -> List[AnalysisInsight]:
    """複数銘柄の横断分析"""
    if len(comparison_results) < 2:
        return []

    insights: List[AnalysisInsight] = []
    insights.extend(_detect_cross_patterns(comparison_results))
    insights.extend(_evaluate_cross_risk(comparison_results))
    insights.extend(_recommend_cross_actions(comparison_results))

    return insights


def _collect_regime_data(
    comparison_results: List["OptimizationResultSet"],
) -> Dict[str, List[dict]]:
    """レジーム別に各銘柄のベスト情報を収集"""
    all_regimes = sorted(set(
        e.trend_regime
        for rs in comparison_results
        for e in rs.entries
    ))

    regime_data: Dict[str, List[dict]] = {}
    for regime in all_regimes:
        bests = []
        for rs in comparison_results:
            regime_set = rs.filter_regime(regime)
            best = regime_set.best
            if best:
                bests.append({
                    "symbol": rs.symbol,
                    "entry": best,
                    "is_viable": _is_viable(best),
                })
        regime_data[regime] = bests

    return regime_data


_REGIME_LABELS = {
    "uptrend": "Uptrend",
    "downtrend": "Downtrend",
    "range": "Range",
}


def _detect_cross_patterns(
    comparison_results: List["OptimizationResultSet"],
) -> List[AnalysisInsight]:
    """パターン検出"""
    insights: List[AnalysisInsight] = []
    n_symbols = len(comparison_results)
    regime_data = _collect_regime_data(comparison_results)

    for regime, bests in regime_data.items():
        label = _REGIME_LABELS.get(regime, regime)

        if not bests:
            continue

        viable_bests = [b for b in bests if b["is_viable"]]
        not_viable_bests = [b for b in bests if not b["is_viable"]]

        # P3: 全銘柄で不採用
        if len(not_viable_bests) == len(bests) and len(bests) >= 2:
            insights.append(AnalysisInsight(
                level=InsightLevel.INFO,
                category="cross_pattern",
                message=f"{label}は全銘柄で不採用。このレジームではポジションを取らない方針が妥当",
            ))
            continue

        # P4: 特定銘柄だけ不採用
        if 0 < len(not_viable_bests) < len(bests):
            symbols = ", ".join(b["symbol"] for b in not_viable_bests)
            insights.append(AnalysisInsight(
                level=InsightLevel.WARNING,
                category="cross_pattern",
                message=f"{label}で{symbols}のみ不採用。この銘柄はこのレジームでの運用を見送り",
            ))

        # テンプレート分析（viable のみ）
        if len(viable_bests) >= 2:
            templates = [b["entry"].template_name for b in viable_bests]
            counter = Counter(templates)
            most_common_tpl, most_common_count = counter.most_common(1)[0]

            # P1: 全銘柄同一テンプレート
            if most_common_count == len(viable_bests):
                insights.append(AnalysisInsight(
                    level=InsightLevel.SUCCESS,
                    category="cross_pattern",
                    message=f"{label}で全{len(viable_bests)}銘柄が{most_common_tpl}をベスト選択。銘柄横断で有効（汎用性高い）",
                ))
            else:
                # P2: 特定銘柄だけ別テンプレート
                outliers = [b["symbol"] for b in viable_bests if b["entry"].template_name != most_common_tpl]
                if outliers:
                    insights.append(AnalysisInsight(
                        level=InsightLevel.WARNING,
                        category="cross_pattern",
                        message=f"{label}で{', '.join(outliers)}は{most_common_tpl}以外を選択。個別対応が必要",
                    ))

    return insights


def _evaluate_cross_risk(
    comparison_results: List["OptimizationResultSet"],
) -> List[AnalysisInsight]:
    """リスク評価"""
    insights: List[AnalysisInsight] = []
    regime_data = _collect_regime_data(comparison_results)

    for regime, bests in regime_data.items():
        label = _REGIME_LABELS.get(regime, regime)
        viable_bests = [b for b in bests if b["is_viable"]]

        if not viable_bests:
            continue

        # R1: 取引数不足銘柄
        for b in viable_bests:
            trades = b["entry"].metrics.total_trades
            if trades < TRADES_STAT_WEAK_THRESHOLD:
                insights.append(AnalysisInsight(
                    level=InsightLevel.WARNING,
                    category="cross_risk",
                    message=f"{b['symbol']}の{label}は取引数{trades}件。統計的信頼性が低い",
                ))

        # R2: PnL分散が大きい
        if len(viable_bests) >= 2:
            pnls = [b["entry"].metrics.total_profit_pct for b in viable_bests]
            mean_pnl = np.mean(pnls)
            std_pnl = np.std(pnls)
            cv = std_pnl / abs(mean_pnl) if abs(mean_pnl) > 1e-10 else 0

            if cv > PNL_CV_HIGH_THRESHOLD:
                insights.append(AnalysisInsight(
                    level=InsightLevel.WARNING,
                    category="cross_risk",
                    message=f"{label}で銘柄間のPnL分散が大きい（CV={cv:.2f}）。銘柄による成果の差が大きい",
                ))

        # R3: DD集中
        high_dd_symbols = [
            b["symbol"] for b in viable_bests
            if b["entry"].metrics.max_drawdown_pct > DD_MULTI_RISK_THRESHOLD
        ]
        if len(high_dd_symbols) >= 2:
            insights.append(AnalysisInsight(
                level=InsightLevel.WARNING,
                category="cross_risk",
                message=f"{label}で{len(high_dd_symbols)}銘柄が最大DD>{DD_MULTI_RISK_THRESHOLD}%（{', '.join(high_dd_symbols)}）。同時運用時のリスク累積に注意",
            ))

    return insights


def _recommend_cross_actions(
    comparison_results: List["OptimizationResultSet"],
) -> List[AnalysisInsight]:
    """推奨アクション"""
    insights: List[AnalysisInsight] = []
    n_symbols = len(comparison_results)
    regime_data = _collect_regime_data(comparison_results)

    for regime, bests in regime_data.items():
        label = _REGIME_LABELS.get(regime, regime)
        viable_bests = [b for b in bests if b["is_viable"]]

        if len(viable_bests) < 2:
            continue

        templates = [b["entry"].template_name for b in viable_bests]
        counter = Counter(templates)
        most_common_tpl, most_common_count = counter.most_common(1)[0]

        # CA1: 複数銘柄同時運用可
        if most_common_count == len(viable_bests) and len(viable_bests) == n_symbols:
            insights.append(AnalysisInsight(
                level=InsightLevel.SUCCESS,
                category="cross_action",
                message=f"{label}: {n_symbols}銘柄全てで{most_common_tpl}がベスト。複数銘柄同時運用可能",
            ))

            # CA2/CA3: パラメータ類似度
            same_tpl_bests = [b for b in viable_bests if b["entry"].template_name == most_common_tpl]
            if len(same_tpl_bests) >= 2:
                similarities = _calculate_param_similarity(same_tpl_bests)
                if similarities:
                    avg_similarity = np.mean(list(similarities.values()))

                    if avg_similarity >= PARAM_SIMILARITY_THRESHOLD:
                        insights.append(AnalysisInsight(
                            level=InsightLevel.SUCCESS,
                            category="cross_action",
                            message=f"{label}: パラメータ類似度{avg_similarity:.0f}%。共通パラメータでの運用可能",
                        ))
                    else:
                        # 乖離している銘柄を特定
                        insights.append(AnalysisInsight(
                            level=InsightLevel.INFO,
                            category="cross_action",
                            message=f"{label}: パラメータ類似度{avg_similarity:.0f}%。銘柄別のパラメータ調整が必要",
                        ))

    # CA4: 銘柄選定推奨（レジーム横断）
    symbol_viable_count: Dict[str, int] = {}
    total_regimes = len(regime_data)
    for rs in comparison_results:
        count = 0
        for regime, bests in regime_data.items():
            for b in bests:
                if b["symbol"] == rs.symbol and b["is_viable"]:
                    count += 1
        symbol_viable_count[rs.symbol] = count

    if total_regimes >= 2:
        counts = list(symbol_viable_count.values())
        if max(counts) > min(counts):
            best_symbol = max(symbol_viable_count, key=symbol_viable_count.get)
            worst_symbol = min(symbol_viable_count, key=symbol_viable_count.get)
            insights.append(AnalysisInsight(
                level=InsightLevel.INFO,
                category="cross_action",
                message=f"{best_symbol}は{symbol_viable_count[best_symbol]}/{total_regimes}レジームで採用可。"
                        f"{worst_symbol}（{symbol_viable_count[worst_symbol]}/{total_regimes}）より運用適性が高い",
            ))

    return insights


# ============================================================
# メタ分析（大量結果の統計集約）
# ============================================================

def analyze_meta(
    comparison_results: List["OptimizationResultSet"],
) -> List[AnalysisInsight]:
    """
    10〜20件以上の最適化結果を統計的にメタ分析する。

    - テンプレート支配率
    - パラメータ収束
    - レジーム別採用率
    - 外れ値銘柄検出
    """
    if len(comparison_results) < 3:
        return []

    insights: List[AnalysisInsight] = []
    regime_data = _collect_regime_data(comparison_results)

    insights.extend(_analyze_template_dominance(regime_data))
    insights.extend(_analyze_param_convergence(regime_data))
    insights.extend(_analyze_regime_viability(regime_data, len(comparison_results)))
    insights.extend(_analyze_outlier_symbols(regime_data))

    return insights


def _analyze_template_dominance(
    regime_data: Dict[str, List[dict]],
) -> List[AnalysisInsight]:
    """テンプレート支配率分析"""
    insights: List[AnalysisInsight] = []

    for regime, bests in regime_data.items():
        label = _REGIME_LABELS.get(regime, regime)
        viable_bests = [b for b in bests if b["is_viable"]]

        if len(viable_bests) < 3:
            continue

        templates = [b["entry"].template_name for b in viable_bests]
        counter = Counter(templates)
        most_common_tpl, most_common_count = counter.most_common(1)[0]
        dominance = most_common_count / len(viable_bests)

        if dominance >= DOMINANCE_HIGH_THRESHOLD:
            insights.append(AnalysisInsight(
                level=InsightLevel.SUCCESS,
                category="meta_dominance",
                message=f"{label}: {len(viable_bests)}件中{most_common_count}件で{most_common_tpl}が最強"
                        f"（支配率{dominance:.0%}）。汎用的に有効なテンプレート",
            ))
        elif dominance < DOMINANCE_LOW_THRESHOLD:
            top_templates = counter.most_common(3)
            tpl_desc = "、".join(f"{t}({c}件)" for t, c in top_templates)
            insights.append(AnalysisInsight(
                level=InsightLevel.WARNING,
                category="meta_dominance",
                message=f"{label}: テンプレート支配率が低い（{dominance:.0%}）。"
                        f"上位: {tpl_desc}。銘柄によって最適戦略が異なる",
            ))
        else:
            insights.append(AnalysisInsight(
                level=InsightLevel.INFO,
                category="meta_dominance",
                message=f"{label}: {most_common_tpl}が{most_common_count}/{len(viable_bests)}件でベスト"
                        f"（支配率{dominance:.0%}）。やや優勢だが、他テンプレートも競合",
            ))

    return insights


def _analyze_param_convergence(
    regime_data: Dict[str, List[dict]],
) -> List[AnalysisInsight]:
    """パラメータ収束分析"""
    insights: List[AnalysisInsight] = []

    for regime, bests in regime_data.items():
        label = _REGIME_LABELS.get(regime, regime)
        viable_bests = [b for b in bests if b["is_viable"]]

        if len(viable_bests) < 3:
            continue

        # 最頻テンプレートを特定
        templates = [b["entry"].template_name for b in viable_bests]
        counter = Counter(templates)
        most_common_tpl, most_common_count = counter.most_common(1)[0]

        if most_common_count < 3:
            continue

        # 該当テンプレートのパラメータを収集
        same_tpl_bests = [b for b in viable_bests if b["entry"].template_name == most_common_tpl]
        param_stats: Dict[str, dict] = {}

        for b in same_tpl_bests:
            for k, v in b["entry"].params.items():
                try:
                    val = float(v)
                    if k not in param_stats:
                        param_stats[k] = {"values": []}
                    param_stats[k]["values"].append(val)
                except (ValueError, TypeError):
                    pass

        converged_params = []
        diverged_params = []

        for param_name, info in param_stats.items():
            values = info["values"]
            if len(values) < 3:
                continue
            mean_val = np.mean(values)
            std_val = np.std(values)
            cv = std_val / abs(mean_val) if abs(mean_val) > 1e-10 else 0

            if cv <= PARAM_CV_CONVERGED:
                converged_params.append(f"{param_name}={mean_val:.1f}")
            elif cv >= PARAM_CV_DIVERGED:
                diverged_params.append(f"{param_name}(CV={cv:.2f})")

        if converged_params:
            insights.append(AnalysisInsight(
                level=InsightLevel.SUCCESS,
                category="meta_convergence",
                message=f"{label}/{most_common_tpl}: 収束パラメータ {', '.join(converged_params)}。"
                        f"共通パラメータでの運用可能",
            ))

        if diverged_params:
            insights.append(AnalysisInsight(
                level=InsightLevel.WARNING,
                category="meta_convergence",
                message=f"{label}/{most_common_tpl}: 発散パラメータ {', '.join(diverged_params)}。"
                        f"銘柄別のパラメータ調整が必要",
            ))

    return insights


def _analyze_regime_viability(
    regime_data: Dict[str, List[dict]],
    total_symbols: int,
) -> List[AnalysisInsight]:
    """レジーム別採用率分析"""
    insights: List[AnalysisInsight] = []

    for regime, bests in regime_data.items():
        label = _REGIME_LABELS.get(regime, regime)
        viable_count = sum(1 for b in bests if b["is_viable"])
        rate = viable_count / total_symbols if total_symbols > 0 else 0

        if rate >= REGIME_VIABILITY_HIGH:
            insights.append(AnalysisInsight(
                level=InsightLevel.SUCCESS,
                category="meta_viability",
                message=f"{label}: {viable_count}/{total_symbols}銘柄で採用可（{rate:.0%}）。"
                        f"高い汎用性",
            ))
        elif rate < REGIME_VIABILITY_LOW:
            insights.append(AnalysisInsight(
                level=InsightLevel.WARNING,
                category="meta_viability",
                message=f"{label}: {viable_count}/{total_symbols}銘柄で採用可（{rate:.0%}）。"
                        f"このレジームでの運用は限定的",
            ))
        else:
            insights.append(AnalysisInsight(
                level=InsightLevel.INFO,
                category="meta_viability",
                message=f"{label}: {viable_count}/{total_symbols}銘柄で採用可（{rate:.0%}）",
            ))

    return insights


def _analyze_outlier_symbols(
    regime_data: Dict[str, List[dict]],
) -> List[AnalysisInsight]:
    """外れ値銘柄検出"""
    insights: List[AnalysisInsight] = []

    for regime, bests in regime_data.items():
        label = _REGIME_LABELS.get(regime, regime)
        viable_bests = [b for b in bests if b["is_viable"]]

        if len(viable_bests) < 4:
            continue

        # 最頻テンプレートと異なるものを外れ値候補とする
        templates = [b["entry"].template_name for b in viable_bests]
        counter = Counter(templates)
        most_common_tpl, most_common_count = counter.most_common(1)[0]
        dominance = most_common_count / len(viable_bests)

        if dominance >= DOMINANCE_LOW_THRESHOLD:
            outlier_symbols = [
                b["symbol"] for b in viable_bests
                if b["entry"].template_name != most_common_tpl
            ]
            if outlier_symbols and len(outlier_symbols) <= 3:
                insights.append(AnalysisInsight(
                    level=InsightLevel.INFO,
                    category="meta_outlier",
                    message=f"{label}: {', '.join(outlier_symbols)}が"
                            f"{most_common_tpl}以外を選択（外れ値候補）。個別検証推奨",
                ))

        # スコアの外れ値検出（IQR法）
        scores = [b["entry"].composite_score for b in viable_bests]
        if len(scores) >= 5:
            q1 = np.percentile(scores, 25)
            q3 = np.percentile(scores, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr

            low_outliers = [
                b["symbol"] for b in viable_bests
                if b["entry"].composite_score < lower_bound
            ]
            if low_outliers:
                insights.append(AnalysisInsight(
                    level=InsightLevel.WARNING,
                    category="meta_outlier",
                    message=f"{label}: {', '.join(low_outliers)}のスコアが他銘柄より著しく低い"
                            f"（下限: {lower_bound:.3f}）。運用を見送るか要検証",
                ))

    return insights
