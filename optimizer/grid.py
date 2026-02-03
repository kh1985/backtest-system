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
import multiprocessing.shared_memory as shm
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed, BrokenExecutor
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Any, Tuple

import pandas as pd

import numpy as np

from strategy.builder import ConfigStrategy
from engine.backtest import BacktestEngine, BacktestResult
from engine.portfolio import Portfolio
from engine.numba_loop import (
    _backtest_loop, vectorize_entry_signals, compute_atr_numpy, HAS_NUMBA,
)
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


# ---------------------------------------------------------------------------
# 共有メモリ管理
# ---------------------------------------------------------------------------

# 文字列カラム → float エンコード用の固定マッピング
_REGIME_ENCODE = {"uptrend": 1.0, "downtrend": -1.0, "range": 0.0}
_REGIME_DECODE = {v: k for k, v in _REGIME_ENCODE.items()}


def _encode_df_to_float(df: pd.DataFrame) -> Tuple[np.ndarray, List[str], Dict[str, Dict[str, float]]]:
    """DataFrameを全float64のnumpy配列に変換。文字列列はfloatエンコード。"""
    columns = list(df.columns)
    str_mappings: Dict[str, Dict[str, float]] = {}

    arrays = []
    for col in columns:
        series = df[col]
        if series.dtype == object or str(series.dtype) == "object":
            # 文字列カラム: 固定マッピングで変換
            mapping = _REGIME_ENCODE
            encoded = series.map(mapping).fillna(-999.0).values.astype(np.float64)
            str_mappings[col] = mapping
            arrays.append(encoded)
        else:
            arrays.append(series.values.astype(np.float64))

    arr = np.column_stack(arrays) if arrays else np.empty((len(df), 0), dtype=np.float64)
    return arr, columns, str_mappings


def _decode_df_from_array(
    arr: np.ndarray,
    columns: List[str],
    str_mappings: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    """numpy配列 + メタデータからDataFrameを復元。"""
    df = pd.DataFrame(arr, columns=columns)
    for col, mapping in str_mappings.items():
        reverse = {v: k for k, v in mapping.items()}
        df[col] = df[col].map(reverse)
    return df


@dataclass
class _SharedDFMeta:
    """SharedMemory上の1つのDataFrame情報（pickle可能）"""
    shm_name: str
    shape: Tuple[int, int]
    columns: List[str]
    str_mappings: Dict[str, Dict[str, float]]


class SharedPrecomputedStore:
    """precomputed辞書を共有メモリで管理するストア"""

    def __init__(self):
        self._blocks: List[shm.SharedMemory] = []
        self.metadata: Dict[str, _SharedDFMeta] = {}

    @classmethod
    def create(cls, precomputed: Dict[str, pd.DataFrame]) -> "SharedPrecomputedStore":
        """precomputed辞書からSharedMemoryを作成"""
        store = cls()
        for key, df in precomputed.items():
            arr, columns, str_mappings = _encode_df_to_float(df)
            arr = np.ascontiguousarray(arr)

            block = shm.SharedMemory(create=True, size=arr.nbytes)
            shared_arr = np.ndarray(arr.shape, dtype=np.float64, buffer=block.buf)
            np.copyto(shared_arr, arr)

            store._blocks.append(block)
            store.metadata[key] = _SharedDFMeta(
                shm_name=block.name,
                shape=arr.shape,
                columns=columns,
                str_mappings=str_mappings,
            )
        return store

    def cleanup(self):
        """SharedMemoryの解放とアンリンク"""
        for block in self._blocks:
            try:
                block.close()
                block.unlink()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# ワーカープロセス用グローバル変数・関数
# ---------------------------------------------------------------------------

# レガシー: pickle 直接渡し用（逐次実行フォールバック等）
_worker_precomputed: Dict[str, pd.DataFrame] = {}

# 共有メモリ用
_worker_shm_blocks: List[shm.SharedMemory] = []
_worker_shm_meta: Dict[str, _SharedDFMeta] = {}

# ワーカー内キャッシュ（同一cache_key/configの再計算を回避）
_worker_cached_df_key: str = ""
_worker_cached_df: Optional[pd.DataFrame] = None
_worker_cached_signals_key: str = ""
_worker_cached_signals: Optional[np.ndarray] = None


def _init_worker(precomputed: Dict[str, pd.DataFrame]):
    """レガシー: 事前計算済みDataFrameをpickleで受け取る。"""
    global _worker_precomputed
    _worker_precomputed = precomputed


def _init_worker_shared(metadata: Dict[str, _SharedDFMeta]):
    """共有メモリ版ワーカー初期化。メタデータのみ受け取り、SharedMemoryにアタッチ。"""
    global _worker_shm_blocks, _worker_shm_meta
    global _worker_cached_df_key, _worker_cached_df
    global _worker_cached_signals_key, _worker_cached_signals
    _worker_shm_meta = metadata
    _worker_shm_blocks = []
    for meta in metadata.values():
        block = shm.SharedMemory(name=meta.shm_name)
        _worker_shm_blocks.append(block)
    # ワーカー内キャッシュをリセット
    _worker_cached_df_key = ""
    _worker_cached_df = None
    _worker_cached_signals_key = ""
    _worker_cached_signals = None


def _reconstruct_df(cache_key: str) -> pd.DataFrame:
    """共有メモリからDataFrameを1つ復元（コピーあり = ワーカー専用メモリ）"""
    meta = _worker_shm_meta[cache_key]
    # アタッチ済みブロックから numpy view を取得
    block = shm.SharedMemory(name=meta.shm_name)
    shared_arr = np.ndarray(meta.shape, dtype=np.float64, buffer=block.buf)
    # コピーしてワーカー独自メモリに配置（タスク完了後GC対象）
    local_arr = shared_arr.copy()
    block.close()  # view 不要になったらクローズ（unlink はしない）
    return _decode_df_from_array(local_arr, meta.columns, meta.str_mappings)


def _make_signals_key(cache_key: str, config: Dict[str, Any]) -> str:
    """entry_signals キャッシュ用のキーを生成"""
    entry_conds = config.get("entry_conditions", [])
    entry_logic = config.get("entry_logic", "and")
    raw = f"{cache_key}:{json.dumps(entry_conds, sort_keys=True)}:{entry_logic}"
    return hashlib.md5(raw.encode()).hexdigest()


def _run_single_task(task: _BacktestTask) -> Optional[OptimizationEntry]:
    """
    1つのバックテストタスクを実行するワーカー関数。

    共有メモリモード: _worker_shm_meta があれば共有メモリから復元。
    レガシーモード: _worker_precomputed から直接取得。

    ワーカー内キャッシュ:
      - 同一 cache_key → DataFrame 復元をスキップ
      - 同一 entry_conditions → entry_signals 計算をスキップ
    """
    global _worker_cached_df_key, _worker_cached_df
    global _worker_cached_signals_key, _worker_cached_signals

    try:
        # --- DataFrame 取得（キャッシュ対応）---
        if _worker_shm_meta:
            if task.cache_key == _worker_cached_df_key and _worker_cached_df is not None:
                work_df = _worker_cached_df
            else:
                work_df = _reconstruct_df(task.cache_key)
                _worker_cached_df_key = task.cache_key
                _worker_cached_df = work_df
        else:
            global _worker_precomputed
            work_df = _worker_precomputed[task.cache_key]

        # --- entry_signals 取得（キャッシュ対応）---
        signals_key = _make_signals_key(task.cache_key, task.config)
        if signals_key == _worker_cached_signals_key and _worker_cached_signals is not None:
            entry_signals = _worker_cached_signals
        else:
            entry_signals = vectorize_entry_signals(
                work_df,
                task.config.get("entry_conditions", []),
                task.config.get("entry_logic", "and"),
            )
            _worker_cached_signals_key = signals_key
            _worker_cached_signals = entry_signals

        optimizer = GridSearchOptimizer(
            initial_capital=task.initial_capital,
            commission_pct=task.commission_pct,
            slippage_pct=task.slippage_pct,
            scoring_weights=task.scoring_weights,
        )

        if _worker_shm_meta:
            entry = optimizer._run_single(
                df=work_df,
                config=task.config,
                template_name=task.template_name,
                params=task.params,
                target_regime=task.target_regime,
                trend_column=task.trend_column,
                data_range=task.data_range,
                precomputed_work_df=work_df,
                precomputed_entry_signals=entry_signals,
            )
        else:
            optimizer._indicator_cache.put(task.cache_key, work_df)
            entry = optimizer._run_single(
                df=work_df,
                config=task.config,
                template_name=task.template_name,
                params=task.params,
                target_regime=task.target_regime,
                trend_column=task.trend_column,
                data_range=task.data_range,
                precomputed_entry_signals=entry_signals,
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
    インジケーター計算結果のキャッシュ（LRU制限付き）

    同じインジケーター設定は1回だけ計算し、結果を再利用。
    キーはインジケーター設定のMD5ハッシュ。
    maxsize を超えると最も古いエントリーを削除。
    """

    def __init__(self, maxsize: int = 64):
        self._cache: OrderedDict[str, pd.DataFrame] = OrderedDict()
        self._maxsize = maxsize

    def get_key(self, indicator_configs: List[Dict]) -> str:
        """インジケーター設定からキャッシュキーを生成"""
        serialized = json.dumps(indicator_configs, sort_keys=True)
        return hashlib.md5(serialized.encode()).hexdigest()

    def get(self, key: str) -> Optional[pd.DataFrame]:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, key: str, df: pd.DataFrame):
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = df
        while len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)

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

        # タスクを cache_key でソート（ワーカー内キャッシュのヒット率向上）
        tasks.sort(key=lambda t: t.cache_key)

        total = len(tasks)
        completed = 0

        # Phase 4: 共有メモリに変換 → ProcessPoolExecutor で並列実行
        ctx = multiprocessing.get_context("spawn")
        shared_store = SharedPrecomputedStore.create(precomputed)

        # precomputed 辞書をメインプロセスから解放（共有メモリに移行済み）
        del precomputed

        try:
            with ProcessPoolExecutor(
                max_workers=n_workers,
                mp_context=ctx,
                initializer=_init_worker_shared,
                initargs=(shared_store.metadata,),
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
        finally:
            shared_store.cleanup()

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
        precomputed_work_df: Optional[pd.DataFrame] = None,
        precomputed_entry_signals: Optional[np.ndarray] = None,
    ) -> OptimizationEntry:
        """1つの組み合わせでバックテストを実行

        Args:
            precomputed_work_df: 共有メモリから復元済みのDF。指定時はキャッシュ/コピーをスキップ。
            precomputed_entry_signals: 事前計算済みのentry_signals。指定時はvectorize_entry_signalsをスキップ。
        """
        strategy = ConfigStrategy(config)

        if precomputed_work_df is not None:
            # 共有メモリモード: 復元済みDFをそのまま使う（既にコピー済み）
            work_df = precomputed_work_df
            strategy._entry_condition = strategy._build_condition(
                config.get("entry_conditions", [])
            )
        else:
            # 通常モード: キャッシュ利用 — 全データで計算
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
            precomputed_entry_signals=precomputed_entry_signals,
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
        precomputed_entry_signals: Optional[np.ndarray] = None,
    ) -> OptimizationEntry:
        """ベクトル化シグナル + Numba JIT ループでバックテスト"""
        # エントリーシグナルをベクトル演算で一括計算（全データで）
        if precomputed_entry_signals is not None:
            entry_signals = precomputed_entry_signals
        else:
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

        is_long = config.get("side", "long") == "long"
        exit_conf = config.get("exit", {})
        tp_pct = float(exit_conf.get("take_profit_pct", 2.0))
        sl_pct = float(exit_conf.get("stop_loss_pct", 1.0))
        trailing_pct = float(exit_conf.get("trailing_stop_pct", 0) or 0)
        timeout_bars = int(exit_conf.get("timeout_bars", 0) or 0)

        # ATRベースexit設定
        use_atr_exit = bool(exit_conf.get("use_atr_exit", False))
        atr_tp_mult = float(exit_conf.get("atr_tp_mult", 0.0))
        atr_sl_mult = float(exit_conf.get("atr_sl_mult", 0.0))
        atr_period = int(exit_conf.get("atr_period", 14))

        # BB帯exit設定
        use_bb_exit = bool(exit_conf.get("use_bb_exit", False))
        bb_period = int(exit_conf.get("bb_period", 20))

        # ATR配列計算（全データで計算 → data_rangeと一緒にスライス）
        if use_atr_exit:
            atr_arr = compute_atr_numpy(high, low, close, period=atr_period)
        else:
            atr_arr = np.empty(0, dtype=np.float64)

        # BB配列計算（全データで計算 → data_rangeと一緒にスライス）
        if use_bb_exit:
            bb_upper_col = f"bb_upper_{bb_period}"
            bb_lower_col = f"bb_lower_{bb_period}"
            if bb_upper_col in work_df.columns and bb_lower_col in work_df.columns:
                bb_upper_arr = work_df[bb_upper_col].fillna(0).values.astype(np.float64)
                bb_lower_arr = work_df[bb_lower_col].fillna(0).values.astype(np.float64)
            else:
                # BB計算がまだの場合、ここで計算
                sma = work_df["close"].rolling(window=bb_period).mean()
                std = work_df["close"].rolling(window=bb_period).std()
                bb_upper_arr = (sma + std * 2.0).fillna(0).values.astype(np.float64)
                bb_lower_arr = (sma - std * 2.0).fillna(0).values.astype(np.float64)
        else:
            bb_upper_arr = np.empty(0, dtype=np.float64)
            bb_lower_arr = np.empty(0, dtype=np.float64)

        # data_range が指定されている場合、バックテスト範囲をスライス
        if data_range is not None:
            start, end = data_range
            high = high[start:end]
            low = low[start:end]
            close = close[start:end]
            entry_signals = entry_signals[start:end]
            regime_mask = regime_mask[start:end]
            if use_atr_exit:
                atr_arr = atr_arr[start:end]
            if use_bb_exit:
                bb_upper_arr = bb_upper_arr[start:end]
                bb_lower_arr = bb_lower_arr[start:end]

        # Numba JIT ループ実行
        profit_pcts, durations, equity_curve = _backtest_loop(
            high, low, close,
            entry_signals, regime_mask,
            is_long, tp_pct, sl_pct,
            trailing_pct, timeout_bars,
            self.commission_pct, self.slippage_pct,
            self.initial_capital,
            atr_arr, use_atr_exit, atr_tp_mult, atr_sl_mult,
            bb_upper_arr, bb_lower_arr, use_bb_exit,
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
        """上位N件以外のBacktestResult・時系列データをクリア（メモリ節約）"""
        ranked = result_set.ranked()
        for i, entry in enumerate(ranked):
            if i >= self.top_n_results:
                entry.backtest_result = None
                # 時系列データもクリア（スカラーメトリクスは保持）
                entry.metrics.equity_curve = []
                entry.metrics.cumulative_returns = []
                entry.metrics.drawdown_series = []
