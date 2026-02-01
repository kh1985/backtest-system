"""
OOS（Out-of-Sample）検証

データを Train / Validation / Test に3分割し、
グリッドサーチの過学習を検出する。

フロー:
  1. Train (60%) でグリッドサーチ → 全configランキング
  2. Validation (20%) で Top-N を再評価 → Val ベスト選択
  3. Test (20%) で最終評価（1回のみ）→ OOS 結果

インジケーターは全データで事前計算し、バックテストループのみ
指定範囲で実行する（warm-up 問題を回避）。
"""

import copy
import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Any, Tuple

import pandas as pd

from .grid import GridSearchOptimizer
from .results import OptimizationEntry, OptimizationResultSet
from .scoring import ScoringWeights, detect_overfitting_warnings

logger = logging.getLogger(__name__)


@dataclass
class DataSplitConfig:
    """データ分割設定"""
    train_pct: float = 0.6
    val_pct: float = 0.2
    top_n_for_val: int = 10

    @property
    def test_pct(self) -> float:
        return round(1.0 - self.train_pct - self.val_pct, 2)

    def compute_indices(self, n: int) -> Tuple[int, int]:
        """(train_end, val_end) を返す"""
        train_end = int(n * self.train_pct)
        val_end = int(n * (self.train_pct + self.val_pct))
        return train_end, val_end


@dataclass
class ValidatedResultSet:
    """OOS 検証済み結果セット"""

    # グリッドサーチ全結果（Train 期間）
    train_results: OptimizationResultSet

    # Validation で選ばれたベスト（レジーム毎）
    val_best: Dict[str, OptimizationEntry] = field(default_factory=dict)

    # Test の最終評価（レジーム毎）
    test_results: Dict[str, OptimizationEntry] = field(default_factory=dict)

    # 分割設定
    split_config: DataSplitConfig = field(default_factory=DataSplitConfig)

    # 分割インデックス
    train_end: int = 0
    val_end: int = 0
    total_bars: int = 0

    # 過学習警告
    overfitting_warnings: List[str] = field(default_factory=list)


def _rebuild_configs_from_entries(
    entries: List[OptimizationEntry],
) -> List[Dict[str, Any]]:
    """OptimizationEntry から再実行用の config リストを構築"""
    configs = []
    for entry in entries:
        config = copy.deepcopy(entry.config)
        config["_template_name"] = entry.template_name
        config["_params"] = copy.deepcopy(entry.params)
        configs.append(config)
    return configs


def run_validated_optimization(
    df: pd.DataFrame,
    all_configs: List[Dict[str, Any]],
    target_regimes: List[str],
    split_config: DataSplitConfig,
    optimizer: GridSearchOptimizer,
    trend_column: str = "trend_regime",
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    n_workers: int = 1,
) -> ValidatedResultSet:
    """
    3分割 OOS 検証付き最適化を実行

    Args:
        df: 実行TFのDataFrame（インジケーター・トレンドラベル付与済み）
        all_configs: 全テンプレート × パラメータの config リスト
        target_regimes: 対象レジーム
        split_config: 分割設定
        optimizer: GridSearchOptimizer インスタンス
        trend_column: トレンドレジームカラム名
        progress_callback: 進捗コールバック
        n_workers: 並列ワーカー数

    Returns:
        ValidatedResultSet
    """
    n = len(df)
    train_end, val_end = split_config.compute_indices(n)

    logger.info(
        f"OOS検証: 全{n}本 → Train[0:{train_end}] "
        f"Val[{train_end}:{val_end}] Test[{val_end}:{n}]"
    )

    # 進捗の3段階管理
    total_configs = len(all_configs) * len(target_regimes)
    val_n = split_config.top_n_for_val * len(target_regimes)
    test_n = len(target_regimes)
    grand_total = total_configs + val_n + test_n
    phase_offset = 0

    def _make_phase_callback(offset: int, label: str):
        """フェーズ別の進捗コールバックを生成"""
        def cb(current, total, desc):
            if progress_callback:
                progress_callback(
                    offset + current,
                    grand_total,
                    f"[{label}] {desc}",
                )
        return cb

    # ========== Phase 1: Train ==========
    train_configs = copy.deepcopy(all_configs)
    train_results = optimizer.run(
        df=df,
        configs=train_configs,
        target_regimes=target_regimes,
        trend_column=trend_column,
        progress_callback=_make_phase_callback(0, "Train"),
        n_workers=n_workers,
        data_range=(0, train_end),
    )
    phase_offset = total_configs

    # ========== Phase 2: Validation ==========
    val_best: Dict[str, OptimizationEntry] = {}

    for regime in target_regimes:
        regime_results = train_results.filter_regime(regime)
        top_entries = regime_results.ranked()[:split_config.top_n_for_val]

        if not top_entries:
            continue

        val_configs = _rebuild_configs_from_entries(top_entries)
        val_result_set = optimizer.run(
            df=df,
            configs=val_configs,
            target_regimes=[regime],
            trend_column=trend_column,
            progress_callback=_make_phase_callback(phase_offset, "Val"),
            n_workers=1,  # 少数なので逐次で十分
            data_range=(train_end, val_end),
        )
        phase_offset += len(val_configs)

        if val_result_set.best:
            val_best[regime] = val_result_set.best

    # ========== Phase 3: Test ==========
    test_results: Dict[str, OptimizationEntry] = {}

    for regime, val_entry in val_best.items():
        test_configs = _rebuild_configs_from_entries([val_entry])
        test_result_set = optimizer.run(
            df=df,
            configs=test_configs,
            target_regimes=[regime],
            trend_column=trend_column,
            progress_callback=_make_phase_callback(phase_offset, "Test"),
            n_workers=1,
            data_range=(val_end, n),
        )
        phase_offset += 1

        if test_result_set.best:
            test_results[regime] = test_result_set.best

    # ========== 警告生成 ==========
    warnings = []
    for regime in target_regimes:
        train_entry = train_results.filter_regime(regime).best
        test_entry = test_results.get(regime)

        if train_entry:
            w = detect_overfitting_warnings(
                train_entry.metrics,
                test_entry.metrics if test_entry else None,
            )
            for msg in w:
                warnings.append(f"[{regime}] {msg}")

    if progress_callback:
        progress_callback(grand_total, grand_total, "完了")

    return ValidatedResultSet(
        train_results=train_results,
        val_best=val_best,
        test_results=test_results,
        split_config=split_config,
        train_end=train_end,
        val_end=val_end,
        total_bars=n,
        overfitting_warnings=warnings,
    )
