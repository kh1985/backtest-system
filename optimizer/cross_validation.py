"""
Multi-Symbol Cross-Validation

複数銘柄のバッチ最適化結果を集約し、戦略の汎用性を判定する。

フロー:
  1. 各銘柄のベスト戦略をレジーム毎に収集
  2. テンプレート単位でグルーピング
  3. 何銘柄でPnL正だったかをカウントし判定を付与
"""

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from .results import OptimizationResultSet

logger = logging.getLogger(__name__)


class CrossValidationVerdict:
    """クロスバリデーション判定"""
    ALL_PASS = "ALL_PASS"
    MAJORITY = "MAJORITY"
    MINORITY = "MINORITY"
    FAIL = "FAIL"


@dataclass
class StrategyProfile:
    """1つの戦略テンプレートのクロスバリデーション結果"""
    template_name: str
    symbols_passed: List[str] = field(default_factory=list)
    symbols_failed: List[str] = field(default_factory=list)
    avg_pnl: float = 0.0
    avg_sharpe: float = 0.0
    avg_win_rate: float = 0.0
    pnl_std: float = 0.0
    verdict: str = CrossValidationVerdict.FAIL

    @property
    def pass_rate(self) -> float:
        total = len(self.symbols_passed) + len(self.symbols_failed)
        return len(self.symbols_passed) / total if total > 0 else 0.0


@dataclass
class RegimeCrossResult:
    """1レジームのクロスバリデーション結果"""
    regime: str
    strategy_profiles: List[StrategyProfile] = field(default_factory=list)
    dominant_template: str = ""
    dominant_template_ratio: float = 0.0


@dataclass
class CrossValidationResult:
    """クロスバリデーション全体の結果"""
    regime_results: Dict[str, RegimeCrossResult] = field(default_factory=dict)
    symbols: List[str] = field(default_factory=list)
    n_symbols: int = 0


def run_cross_validation(
    batch_results: List[OptimizationResultSet],
    target_regimes: List[str],
    pnl_threshold: float = 0.0,
) -> CrossValidationResult:
    """
    バッチ最適化結果からクロスバリデーションを実行。

    Args:
        batch_results: 各銘柄の OptimizationResultSet リスト
        target_regimes: 対象レジーム
        pnl_threshold: PnL の PASS 閾値（デフォルト 0.0）

    Returns:
        CrossValidationResult
    """
    symbols = [rs.symbol for rs in batch_results]
    result = CrossValidationResult(
        symbols=symbols,
        n_symbols=len(symbols),
    )

    for regime in target_regimes:
        regime_result = _analyze_regime(batch_results, regime, pnl_threshold)
        result.regime_results[regime] = regime_result

    return result


def _analyze_regime(
    batch_results: List[OptimizationResultSet],
    regime: str,
    pnl_threshold: float,
) -> RegimeCrossResult:
    """1レジームのクロスバリデーション分析"""
    # 各銘柄のベスト戦略を収集（同一シンボルは後勝ち）
    symbol_bests: Dict[str, tuple] = {}  # symbol -> entry

    for rs in batch_results:
        regime_set = rs.filter_regime(regime)
        best = regime_set.best_by_pnl
        if best:
            symbol_bests[rs.symbol] = best

    if not symbol_bests:
        return RegimeCrossResult(regime=regime)

    # 重複排除後のシンボルからカウント
    template_counter: Counter = Counter()
    for symbol, entry in symbol_bests.items():
        template_counter[entry.template_name] += 1

    # テンプレート単位でグルーピング
    template_groups: Dict[str, List[tuple]] = {}
    for symbol, entry in symbol_bests.items():
        key = entry.template_name
        if key not in template_groups:
            template_groups[key] = []
        template_groups[key].append((symbol, entry))

    profiles = []
    for template_name, symbol_entries in template_groups.items():
        profile = StrategyProfile(template_name=template_name)

        pnls = []
        sharpes = []
        win_rates = []

        for symbol, entry in symbol_entries:
            pnl = entry.metrics.total_profit_pct
            pnls.append(pnl)
            sharpes.append(entry.metrics.sharpe_ratio)
            win_rates.append(entry.metrics.win_rate)

            if pnl > pnl_threshold:
                profile.symbols_passed.append(symbol)
            else:
                profile.symbols_failed.append(symbol)

        profile.avg_pnl = float(np.mean(pnls))
        profile.avg_sharpe = float(np.mean(sharpes))
        profile.avg_win_rate = float(np.mean(win_rates))
        profile.pnl_std = float(np.std(pnls)) if len(pnls) > 1 else 0.0

        # Verdict
        rate = profile.pass_rate
        if rate == 1.0:
            profile.verdict = CrossValidationVerdict.ALL_PASS
        elif rate >= 0.5:
            profile.verdict = CrossValidationVerdict.MAJORITY
        elif rate > 0.0:
            profile.verdict = CrossValidationVerdict.MINORITY
        else:
            profile.verdict = CrossValidationVerdict.FAIL

        profiles.append(profile)

    # avg_pnl で降順ソート
    profiles.sort(key=lambda p: p.avg_pnl, reverse=True)

    # 最頻テンプレート
    dominant = ""
    dominant_ratio = 0.0
    if template_counter:
        dominant, dominant_count = template_counter.most_common(1)[0]
        dominant_ratio = dominant_count / len(symbol_bests)

    return RegimeCrossResult(
        regime=regime,
        strategy_profiles=profiles,
        dominant_template=dominant,
        dominant_template_ratio=dominant_ratio,
    )
