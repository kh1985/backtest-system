"""
グリッドサーチエンジン

テンプレート × パラメータ × トレンドレジーム の全組み合わせを
自動バックテストし、複合スコアでランキングする。
並列実行（ProcessPoolExecutor）対応。
"""

import hashlib
import json
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed, BrokenExecutor
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Any, Tuple

import pandas as pd

import numpy as np

from strategy.builder import ConfigStrategy
from engine.backtest import BacktestEngine, BacktestResult
from engine.portfolio import Portfolio
from engine.numba_loop import _backtest_loop, vectorize_entry_signals, HAS_NUMBA
from metrics.calculator import calculate_metrics, calculate_metrics_from_arrays
from analysis.trend import TrendRegime
from .scoring import ScoringWeights, calculate_composite_score
from .results import OptimizationEntry, OptimizationResultSet

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 並列実行用データクラス・モジュールレベル関数
# （spawn コンテキストで pickle 可能にするためクラス外に定義）
# ---------------------------------------------------------------------------

@dataclass
class _BacktestTask:
    """ワーカーに渡す1回分のバックテストタスク"""
    task_id: int
    config: Dict[str, Any]
    template_name: str
    params: Dict[str, Any]
    target_regime: str
    trend_column: str
    cache_key: str
    initial_capital: float
    commission_pct: float
    slippage_pct: float
    scoring_weights: ScoringWeights
    data_range: Optional[Tuple[int, int]] = None


# ワーカープロセス内で保持する事前計算済みDataFrame
_worker_precomputed: Dict[str, pd.DataFrame] = {}


def _init_worker(precomputed: Dict[str, pd.DataFrame]):
    """ワーカープロセスの初期化。事前計算済みDataFrameを受け取る。"""
    global _worker_precomputed
    _worker_precomputed = precomputed


def _run_single_task(task: _BacktestTask) -> Optional[OptimizationEntry]:
    """
    1つのバックテストタスクを実行するワーカー関数。

    GridSearchOptimizer の軽量インスタンスを生成し、
    事前計算済みDFをキャッシュに入れて _run_single() を呼ぶ。
    """
    try:
        global _worker_precomputed

        optimizer = GridSearchOptimizer(
            initial_capital=task.initial_capital,
            commission_pct=task.commission_pct,
            slippage_pct=task.slippage_pct,
            scoring_weights=task.scoring_weights,
        )

        # 事前計算済みDFをキャッシュに入れる（_run_single でキャッシュヒットする）
        precomputed_df = _worker_precomputed[task.cache_key]
        optimizer._indicator_cache.put(task.cache_key, precomputed_df)

        entry = optimizer._run_single(
            df=precomputed_df,  # キャッシュヒットするので df.copy() は不要
            config=task.config,
            template_name=task.template_name,
            params=task.params,
            target_regime=task.target_regime,
            trend_column=task.trend_column,
            data_range=task.data_range,
        )

        # メモリ削減: ワーカーからメインプロセスへの転送量を減らす
        if entry.backtest_result is not None:
            entry.backtest_result.df = None

        return entry
    except Exception as e:
        logger.debug(f"Worker task {task.task_id} failed: {e}")
        return None


class IndicatorCache:
    """
    インジケーター計算結果のキャッシュ

    同じインジケーター設定は1回だけ計算し、結果を再利用。
    キーはインジケーター設定のMD5ハッシュ。
    """

    def __init__(self):
        self._cache: Dict[str, pd.DataFrame] = {}

    def get_key(self, indicator_configs: List[Dict]) -> str:
        """インジケーター設定からキャッシュキーを生成"""
        serialized = json.dumps(indicator_configs, sort_keys=True)
        return hashlib.md5(serialized.encode()).hexdigest()

    def get(self, key: str) -> Optional[pd.DataFrame]:
        return self._cache.get(key)

    def put(self, key: str, df: pd.DataFrame):
        self._cache[key] = df

    def clear(self):
        self._cache.clear()


class GridSearchOptimizer:
    """
    グリッドサーチ最適化エンジン

    各(テンプレート × パラメータ組み合わせ × トレンドレジーム)に対して:
    1. テンプレートからconfig生成
    2. ConfigStrategy構築
    3. インジケーター計算（キャッシュ済みならスキップ）
    4. トレンドフィルタ付きbar-by-barループ
    5. メトリクス算出 → 複合スコア計算
    6. OptimizationEntryに記録
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        commission_pct: float = 0.04,
        slippage_pct: float = 0.0,
        scoring_weights: ScoringWeights = None,
        top_n_results: int = 20,
    ):
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.scoring_weights = scoring_weights or ScoringWeights()
        self.top_n_results = top_n_results
        self._indicator_cache = IndicatorCache()

    def run(
        self,
        df: pd.DataFrame,
        configs: List[Dict[str, Any]],
        target_regimes: List[str],
        trend_column: str = "trend_regime",
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        n_workers: int = 1,
        data_range: Optional[Tuple[int, int]] = None,
    ) -> OptimizationResultSet:
        """
        グリッドサーチを実行

        Args:
            df: 実行TFのDataFrame（trend_regimeカラム付き）
            configs: ConfigStrategy用configのリスト
                     各configに "_template_name" と "_params" を含む
            target_regimes: 対象レジーム ["uptrend", "downtrend", "range"]
            trend_column: トレンドレジームカラム名
            progress_callback: 進捗コールバック(current, total, description)
            n_workers: 並列ワーカー数（1=逐次、2+=並列）
            data_range: バックテスト範囲 (start_idx, end_idx)。
                        Noneの場合は全データ。インジケーターは全データで計算し、
                        バックテストループのみ指定範囲で実行。

        Returns:
            OptimizationResultSet
        """
        if n_workers <= 1:
            return self._run_sequential(
                df, configs, target_regimes, trend_column,
                progress_callback, data_range,
            )
        else:
            return self._run_parallel(
                df, configs, target_regimes, trend_column,
                progress_callback, n_workers, data_range,
            )

    def _run_sequential(
        self,
        df: pd.DataFrame,
        configs: List[Dict[str, Any]],
        target_regimes: List[str],
        trend_column: str,
        progress_callback: Optional[Callable[[int, int, str], None]],
        data_range: Optional[Tuple[int, int]] = None,
    ) -> OptimizationResultSet:
        """逐次実行（既存動作と同一）"""
        result_set = OptimizationResultSet()

        total = len(configs) * len(target_regimes)
        current = 0

        for config in configs:
            template_name = config.pop("_template_name", config.get("name", "unknown"))
            params = config.pop("_params", {})

            for regime in target_regimes:
                current += 1
                desc = f"{template_name} ({regime})"

                if progress_callback:
                    progress_callback(current, total, desc)

                try:
                    entry = self._run_single(
                        df=df,
                        config=config,
                        template_name=template_name,
                        params=params,
                        target_regime=regime,
                        trend_column=trend_column,
                        data_range=data_range,
                    )
                    result_set.add(entry)
                except Exception:
                    pass

        self._trim_results(result_set)
        return result_set

    def _run_parallel(
        self,
        df: pd.DataFrame,
        configs: List[Dict[str, Any]],
        target_regimes: List[str],
        trend_column: str,
        progress_callback: Optional[Callable[[int, int, str], None]],
        n_workers: int,
        data_range: Optional[Tuple[int, int]] = None,
    ) -> OptimizationResultSet:
        """並列実行（ProcessPoolExecutor）"""
        result_set = OptimizationResultSet()

        # Phase 1: config からメタ情報を抽出（pop は破壊的なので先に行う）
        task_configs = []
        for config in configs:
            config_copy = dict(config)
            template_name = config_copy.pop(
                "_template_name", config_copy.get("name", "unknown")
            )
            params = config_copy.pop("_params", {})
            task_configs.append((config_copy, template_name, params))

        # Phase 2: ユニークなインジケーター設定のDataFrameを事前計算
        if progress_callback:
            progress_callback(0, 1, "インジケーター事前計算中...")

        precomputed: Dict[str, pd.DataFrame] = {}
        cache = IndicatorCache()

        for config_copy, _, _ in task_configs:
            indicator_configs = config_copy.get("indicators", [])
            cache_key = cache.get_key(indicator_configs)

            if cache_key not in precomputed:
                strategy = ConfigStrategy(config_copy)
                work_df = strategy.setup(df.copy())
                precomputed[cache_key] = work_df

        logger.info(
            f"事前計算完了: {len(precomputed)} ユニークインジケーター設定"
        )

        # Phase 3: タスクリスト生成
        tasks: List[_BacktestTask] = []
        task_id = 0
        for config_copy, template_name, params in task_configs:
            indicator_configs = config_copy.get("indicators", [])
            cache_key = cache.get_key(indicator_configs)

            for regime in target_regimes:
                tasks.append(_BacktestTask(
                    task_id=task_id,
                    config=config_copy,
                    template_name=template_name,
                    params=params,
                    target_regime=regime,
                    trend_column=trend_column,
                    cache_key=cache_key,
                    initial_capital=self.initial_capital,
                    commission_pct=self.commission_pct,
                    slippage_pct=self.slippage_pct,
                    scoring_weights=self.scoring_weights,
                    data_range=data_range,
                ))
                task_id += 1

        total = len(tasks)
        completed = 0

        # Phase 4: ProcessPoolExecutor で並列実行
        ctx = multiprocessing.get_context("spawn")

        try:
            with ProcessPoolExecutor(
                max_workers=n_workers,
                mp_context=ctx,
                initializer=_init_worker,
                initargs=(precomputed,),
            ) as executor:
                future_to_task = {
                    executor.submit(_run_single_task, task): task
                    for task in tasks
                }

                for future in as_completed(future_to_task):
                    completed += 1
                    task = future_to_task[future]
                    desc = f"{task.template_name} ({task.target_regime})"

                    if progress_callback:
                        progress_callback(completed, total, desc)

                    try:
                        entry = future.result()
                        if entry is not None:
                            result_set.add(entry)
                    except Exception as e:
                        logger.debug(f"Task failed: {desc} - {e}")

        except BrokenExecutor:
            logger.error("ワーカープロセスがクラッシュしました")

        self._trim_results(result_set)
        return result_set

    def _run_single(
        self,
        df: pd.DataFrame,
        config: Dict[str, Any],
        template_name: str,
        params: Dict[str, Any],
        target_regime: str,
        trend_column: str,
        data_range: Optional[Tuple[int, int]] = None,
    ) -> OptimizationEntry:
        """1つの組み合わせでバックテストを実行"""
        strategy = ConfigStrategy(config)

        # インジケーター計算（キャッシュ利用）— 全データで計算
        indicator_configs = config.get("indicators", [])
        cache_key = self._indicator_cache.get_key(indicator_configs)
        cached_df = self._indicator_cache.get(cache_key)

        if cached_df is not None:
            work_df = cached_df.copy()
            strategy._entry_condition = strategy._build_condition(
                config.get("entry_conditions", [])
            )
        else:
            work_df = strategy.setup(df.copy())
            self._indicator_cache.put(cache_key, work_df)

        # --- Numba 高速パス ---
        return self._run_numba(
            work_df, config, strategy, template_name, params,
            target_regime, trend_column, data_range,
        )

    def _run_numba(
        self,
        work_df: pd.DataFrame,
        config: Dict[str, Any],
        strategy: ConfigStrategy,
        template_name: str,
        params: Dict[str, Any],
        target_regime: str,
        trend_column: str,
        data_range: Optional[Tuple[int, int]] = None,
    ) -> OptimizationEntry:
        """ベクトル化シグナル + Numba JIT ループでバックテスト"""
        # エントリーシグナルをベクトル演算で一括計算（全データで）
        entry_signals = vectorize_entry_signals(
            work_df,
            config.get("entry_conditions", []),
            config.get("entry_logic", "and"),
        )

        # レジームマスク（全データで）
        if target_regime == "all":
            regime_mask = np.ones(len(work_df), dtype=np.bool_)
        else:
            if trend_column in work_df.columns:
                regime_mask = (work_df[trend_column] == target_regime).values.astype(np.bool_)
            else:
                regime_mask = np.ones(len(work_df), dtype=np.bool_)

        # numpy 配列抽出（全データ）
        high = work_df["high"].values.astype(np.float64)
        low = work_df["low"].values.astype(np.float64)
        close = work_df["close"].values.astype(np.float64)

        # data_range が指定されている場合、バックテスト範囲をスライス
        if data_range is not None:
            start, end = data_range
            high = high[start:end]
            low = low[start:end]
            close = close[start:end]
            entry_signals = entry_signals[start:end]
            regime_mask = regime_mask[start:end]

        is_long = config.get("side", "long") == "long"
        exit_conf = config.get("exit", {})
        tp_pct = float(exit_conf.get("take_profit_pct", 2.0))
        sl_pct = float(exit_conf.get("stop_loss_pct", 1.0))
        trailing_pct = float(exit_conf.get("trailing_stop_pct", 0) or 0)
        timeout_bars = int(exit_conf.get("timeout_bars", 0) or 0)

        # Numba JIT ループ実行
        profit_pcts, durations, equity_curve = _backtest_loop(
            high, low, close,
            entry_signals, regime_mask,
            is_long, tp_pct, sl_pct,
            trailing_pct, timeout_bars,
            self.commission_pct, self.slippage_pct,
            self.initial_capital,
        )

        # メトリクス算出（numpy 配列版）
        metrics = calculate_metrics_from_arrays(
            profit_pcts, durations, equity_curve,
        )

        # 複合スコア
        score = calculate_composite_score(
            profit_factor=metrics.profit_factor,
            win_rate=metrics.win_rate,
            max_drawdown_pct=metrics.max_drawdown_pct,
            sharpe_ratio=metrics.sharpe_ratio,
            total_return_pct=metrics.total_profit_pct,
            weights=self.scoring_weights,
        )

        # 最小限の BacktestResult を構築（エクイティカーブ描画用）
        portfolio = Portfolio(self.initial_capital)
        portfolio.equity_curve = equity_curve.tolist()
        portfolio.current_equity = float(equity_curve[-1]) if len(equity_curve) > 0 else self.initial_capital

        result = BacktestResult(
            trades=[],
            portfolio=portfolio,
            strategy_name=template_name,
            df=None,
        )

        return OptimizationEntry(
            template_name=template_name,
            params=params,
            trend_regime=target_regime,
            config=config,
            metrics=metrics,
            composite_score=score,
            backtest_result=result,
        )

    def _trim_results(self, result_set: OptimizationResultSet):
        """上位N件以外のBacktestResultをクリア（メモリ節約）"""
        ranked = result_set.ranked()
        for i, entry in enumerate(ranked):
            if i >= self.top_n_results:
                entry.backtest_result = None
