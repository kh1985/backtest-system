"""
Walk-Forward Analysis (WFA)

Anchored WFA でデータをスライドさせながら複数回の IS/OOS 検証を行い、
戦略の時間的一貫性を確認する。

フロー:
  1. compute_fold_ranges() でフォールド区間を計算
  2. 各フォールドで IS 期間のグリッドサーチ → OOS 評価
  3. 全フォールドの OOS 結果を集約して WFE / CR 等を算出

Anchored 方式: IS の開始は常に 0（学習データが段階的に拡大）
"""

import copy
import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd

from .grid import GridSearchOptimizer
from .results import OptimizationEntry, OptimizationResultSet

logger = logging.getLogger(__name__)


@dataclass
class WFAConfig:
    """Walk-Forward Analysis 設定"""
    n_folds: int = 5
    min_is_pct: float = 0.4
    use_validation: bool = True
    val_pct_within_is: float = 0.2
    top_n_for_val: int = 10
    min_trades_for_val: int = 30

    def compute_fold_ranges(
        self, n: int,
    ) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        各フォールドの (IS範囲, OOS範囲) を計算。

        Anchored WFA:
          Fold k: IS = [0, min_is + step*k], OOS = [min_is + step*k, min_is + step*(k+1)]

        Returns:
            [(is_range, oos_range), ...]
        """
        min_is_bars = int(n * self.min_is_pct)
        remaining = n - min_is_bars
        step = remaining // self.n_folds

        if step < 1:
            raise ValueError(
                f"データ不足: {n}本に対して min_is_pct={self.min_is_pct}, "
                f"n_folds={self.n_folds} ではOOSステップが0以下"
            )

        folds = []
        for k in range(self.n_folds):
            is_end = min_is_bars + step * k
            oos_start = is_end
            oos_end = min(is_end + step, n)
            if oos_start >= oos_end:
                break
            folds.append(((0, is_end), (oos_start, oos_end)))

        return folds


@dataclass
class WFAFoldResult:
    """1フォールドの結果"""
    fold_index: int
    is_range: Tuple[int, int]
    oos_range: Tuple[int, int]
    # レジーム毎: IS で選ばれた戦略
    selected_strategy: Dict[str, OptimizationEntry] = field(default_factory=dict)
    # レジーム毎: OOS 評価結果
    oos_results: Dict[str, OptimizationEntry] = field(default_factory=dict)


@dataclass
class WFAResultSet:
    """WFA 全体の結果"""
    config: WFAConfig
    folds: List[WFAFoldResult] = field(default_factory=list)

    # 集約メトリクス（レジーム毎）
    wfe: Dict[str, float] = field(default_factory=dict)
    consistency_ratio: Dict[str, float] = field(default_factory=dict)
    stitched_oos_pnl: Dict[str, float] = field(default_factory=dict)
    strategy_stability: Dict[str, float] = field(default_factory=dict)

    # 参照情報
    symbol: str = ""
    execution_tf: str = ""
    total_bars: int = 0

    # 最終フォールドの IS 結果（結果ビュー用）
    final_train_results: Optional[OptimizationResultSet] = None


def _rebuild_configs_from_entries(
    entries: List[OptimizationEntry],
) -> List[Dict[str, Any]]:
    """OptimizationEntry から再実行用 config リストを構築"""
    configs = []
    for entry in entries:
        config = copy.deepcopy(entry.config)
        config["_template_name"] = entry.template_name
        config["_params"] = copy.deepcopy(entry.params)
        configs.append(config)
    return configs


def run_walk_forward_analysis(
    df: pd.DataFrame,
    all_configs: List[Dict[str, Any]],
    target_regimes: List[str],
    wfa_config: WFAConfig,
    optimizer: GridSearchOptimizer,
    trend_column: str = "trend_regime",
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    n_workers: int = 1,
) -> WFAResultSet:
    """
    Walk-Forward Analysis を実行。

    各フォールドで:
    1. IS 期間でグリッドサーチ（use_validation=True なら IS 内 Train/Val 分割）
    2. IS-best を OOS 期間で評価
    3. 全フォールド結果を集約

    Args:
        df: インジケーター・トレンドラベル付与済み DataFrame
        all_configs: 全テンプレート x パラメータの config リスト
        target_regimes: 対象レジーム
        wfa_config: WFA 設定
        optimizer: GridSearchOptimizer インスタンス
        trend_column: トレンドレジームカラム名
        progress_callback: 進捗コールバック
        n_workers: 並列ワーカー数

    Returns:
        WFAResultSet
    """
    n = len(df)
    fold_ranges = wfa_config.compute_fold_ranges(n)
    n_folds_actual = len(fold_ranges)

    logger.info(
        f"WFA開始: {n}本, {n_folds_actual}フォールド, "
        f"regimes={target_regimes}"
    )

    result_set = WFAResultSet(
        config=wfa_config,
        total_bars=n,
    )

    # 進捗管理: IS + OOS で 2 ステップ/フォールド
    total_steps = n_folds_actual * 2
    current_step = 0

    last_is_results = None

    for fold_idx, (is_range, oos_range) in enumerate(fold_ranges):
        logger.info(
            f"Fold {fold_idx + 1}/{n_folds_actual}: "
            f"IS[{is_range[0]}:{is_range[1]}] "
            f"OOS[{oos_range[0]}:{oos_range[1]}]"
        )

        # --- IS フェーズ ---
        if progress_callback:
            progress_callback(
                current_step, total_steps,
                f"[Fold {fold_idx + 1}/{n_folds_actual}] IS最適化中...",
            )

        is_last_fold = (fold_idx == n_folds_actual - 1)
        selected, full_is_results = _run_is_phase(
            df=df,
            all_configs=all_configs,
            target_regimes=target_regimes,
            is_range=is_range,
            wfa_config=wfa_config,
            optimizer=optimizer,
            trend_column=trend_column,
            n_workers=n_workers,
            return_full_results=is_last_fold,
        )

        # 最終フォールドの IS 結果を保持（結果ビュー表示用）
        if is_last_fold:
            last_is_results = full_is_results

        current_step += 1

        # --- OOS フェーズ ---
        if progress_callback:
            progress_callback(
                current_step, total_steps,
                f"[Fold {fold_idx + 1}/{n_folds_actual}] OOS評価中...",
            )

        oos_results: Dict[str, OptimizationEntry] = {}
        for regime, is_best in selected.items():
            oos_configs = _rebuild_configs_from_entries([is_best])
            oos_result = optimizer.run(
                df=df,
                configs=oos_configs,
                target_regimes=[regime],
                trend_column=trend_column,
                n_workers=1,
                data_range=oos_range,
            )
            if oos_result.best:
                oos_results[regime] = oos_result.best

        current_step += 1

        fold_result = WFAFoldResult(
            fold_index=fold_idx,
            is_range=is_range,
            oos_range=oos_range,
            selected_strategy=selected,
            oos_results=oos_results,
        )
        result_set.folds.append(fold_result)

    result_set.final_train_results = last_is_results

    # 集約メトリクス計算
    _compute_aggregate_metrics(result_set, target_regimes)

    if progress_callback:
        progress_callback(total_steps, total_steps, "WFA完了")

    return result_set


def _run_is_phase(
    df: pd.DataFrame,
    all_configs: List[Dict[str, Any]],
    target_regimes: List[str],
    is_range: Tuple[int, int],
    wfa_config: WFAConfig,
    optimizer: GridSearchOptimizer,
    trend_column: str,
    n_workers: int,
    return_full_results: bool = False,
) -> Tuple[Dict[str, OptimizationEntry], Optional[OptimizationResultSet]]:
    """IS フェーズ: グリッドサーチ → ベスト選択

    Returns:
        (selected, full_results): selected はレジーム毎のベスト、
        full_results は return_full_results=True の場合のみ IS 全結果。
    """
    if wfa_config.use_validation:
        return _run_is_with_validation(
            df, all_configs, target_regimes, is_range,
            wfa_config, optimizer, trend_column, n_workers,
            return_full_results=return_full_results,
        )
    else:
        return _run_is_direct(
            df, all_configs, target_regimes, is_range,
            optimizer, trend_column, n_workers,
            return_full_results=return_full_results,
        )


def _run_is_direct(
    df: pd.DataFrame,
    all_configs: List[Dict[str, Any]],
    target_regimes: List[str],
    is_range: Tuple[int, int],
    optimizer: GridSearchOptimizer,
    trend_column: str,
    n_workers: int,
    return_full_results: bool = False,
) -> Tuple[Dict[str, OptimizationEntry], Optional[OptimizationResultSet]]:
    """IS 全体でグリッドサーチ → ベスト直接採用"""
    is_results = optimizer.run(
        df=df,
        configs=copy.deepcopy(all_configs),
        target_regimes=target_regimes,
        trend_column=trend_column,
        n_workers=n_workers,
        data_range=is_range,
    )
    selected = {}
    for regime in target_regimes:
        best = is_results.filter_regime(regime).best
        if best:
            selected[regime] = best
    return selected, is_results if return_full_results else None


def _run_is_with_validation(
    df: pd.DataFrame,
    all_configs: List[Dict[str, Any]],
    target_regimes: List[str],
    is_range: Tuple[int, int],
    wfa_config: WFAConfig,
    optimizer: GridSearchOptimizer,
    trend_column: str,
    n_workers: int,
    return_full_results: bool = False,
) -> Tuple[Dict[str, OptimizationEntry], Optional[OptimizationResultSet]]:
    """IS 内を Train/Val に分割してベスト選択"""
    is_n = is_range[1] - is_range[0]
    train_end_within_is = int(is_n * (1.0 - wfa_config.val_pct_within_is))

    train_range = (is_range[0], is_range[0] + train_end_within_is)
    val_range = (is_range[0] + train_end_within_is, is_range[1])

    # Phase 1: Train
    train_results = optimizer.run(
        df=df,
        configs=copy.deepcopy(all_configs),
        target_regimes=target_regimes,
        trend_column=trend_column,
        n_workers=n_workers,
        data_range=train_range,
    )

    # Phase 2: Validation (top-N per regime)
    selected = {}
    for regime in target_regimes:
        regime_results = train_results.filter_regime(regime)
        ranked = regime_results.ranked()

        # min_trades フィルタ
        min_t = wfa_config.min_trades_for_val
        if min_t > 0:
            filtered = [
                e for e in ranked
                if e.metrics.total_trades >= min_t
            ]
            if not filtered:
                filtered = ranked
        else:
            filtered = ranked

        top_entries = filtered[:wfa_config.top_n_for_val]
        if not top_entries:
            continue

        val_configs = _rebuild_configs_from_entries(top_entries)
        val_result = optimizer.run(
            df=df,
            configs=val_configs,
            target_regimes=[regime],
            trend_column=trend_column,
            n_workers=1,
            data_range=val_range,
        )
        if val_result.best:
            selected[regime] = val_result.best

    # return_full_results 時は Train 結果を返す（IS の大部分をカバー）
    return selected, train_results if return_full_results else None


def _compute_aggregate_metrics(
    result_set: WFAResultSet,
    target_regimes: List[str],
):
    """フォールド結果から集約メトリクスを計算"""
    for regime in target_regimes:
        is_pnls = []
        oos_pnls = []
        strategy_names = []
        total_folds = len(result_set.folds)

        for fold in result_set.folds:
            is_entry = fold.selected_strategy.get(regime)
            oos_entry = fold.oos_results.get(regime)

            if is_entry:
                is_pnls.append(is_entry.metrics.total_profit_pct)
                strategy_names.append(
                    f"{is_entry.template_name}_{is_entry.param_str}"
                )

            if oos_entry:
                oos_pnls.append(oos_entry.metrics.total_profit_pct)

        # WFE: mean(OOS) / mean(IS)
        if is_pnls and oos_pnls:
            mean_is = float(np.mean(is_pnls))
            mean_oos = float(np.mean(oos_pnls))
            result_set.wfe[regime] = (
                mean_oos / mean_is if abs(mean_is) > 1e-10 else 0.0
            )

        # Consistency Ratio
        if total_folds > 0:
            positive_count = sum(1 for p in oos_pnls if p > 0)
            # CR分母: トレード無しフォールドも含めた全フォールド数を使用
            result_set.consistency_ratio[regime] = (
                positive_count / total_folds
            )

        if oos_pnls:
            # OOS PnL: フォールド間リターンを複利計算で集計
            compounded_return = np.prod([1.0 + (p / 100.0) for p in oos_pnls]) - 1.0
            result_set.stitched_oos_pnl[regime] = float(compounded_return * 100.0)

        # Strategy Stability: 最頻戦略の割合
        if strategy_names:
            counter = Counter(strategy_names)
            most_common_count = counter.most_common(1)[0][1]
            result_set.strategy_stability[regime] = (
                most_common_count / len(strategy_names)
            )
