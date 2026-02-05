"""
バッチ自動最適化スクリプト v2

全TF組み合わせ x 全銘柄 x 全期間 x 全exit戦略 を
OOS 3分割検証付きで自動一括実行し、
自動ランキング + Markdownレポートを生成する。

使い方:
    # フル実行（全TF x 全銘柄 x 全期間 x OOS x 全exit profiles）
    python scripts/batch_optimize.py

    # 特定TFのみ
    python scripts/batch_optimize.py --tf-combos 15m:1h,1h:4h

    # OOSなし（高速テスト用）
    python scripts/batch_optimize.py --no-oos

    # 特定exit profileのみ
    python scripts/batch_optimize.py --exit-profiles fixed

    # 小規模テスト（2銘柄 x 1期間 x 1TF）
    python scripts/batch_optimize.py --symbols BTCUSDT,ETHUSDT --periods 20250201-20260130 --tf-combos 15m:1h

    # exit profiles なし（テンプレート内蔵のexitのみ）
    python scripts/batch_optimize.py --exit-profiles none
"""

import argparse
import copy
import json
import logging
import math
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional, Tuple

# プロジェクトルートをパスに追加
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import pandas as pd

from data.binance_loader import BinanceCSVLoader
from data.base import OHLCVData
from analysis.trend import TrendDetector, TrendRegime
from optimizer.grid import GridSearchOptimizer
from optimizer.templates import BUILTIN_TEMPLATES
from optimizer.results import OptimizationResultSet, OptimizationEntry
from optimizer.scoring import ScoringWeights
from optimizer.validation import (
    DataSplitConfig,
    ValidatedResultSet,
    run_validated_optimization,
)
from optimizer.exit_profiles import get_profiles, ALL_PROFILES

# =====================================================================
# 設定
# =====================================================================

# TF組み合わせ（exec_tf, htf）
TF_COMBOS = [
    ("1m", "15m"),    # 超短期スキャル
    ("1m", "1h"),     # 短期スキャル
    ("15m", "1h"),    # 中期（標準）
    ("15m", "4h"),    # 中期スイング
    ("1h", "4h"),     # スイング
    ("1h", "1d"),     # 長期スイング
]

TREND_METHOD = "ma_cross"
MA_FAST = 20
MA_SLOW = 50
ADX_PERIOD = 14
ADX_TREND_TH = 25.0
ADX_RANGE_TH = 20.0

INITIAL_CAPITAL = 10000.0
COMMISSION_PCT = 0.04
SLIPPAGE_PCT = 0.0
N_WORKERS = 4
TOP_N_RESULTS = 20

TARGET_REGIMES = ["uptrend", "downtrend", "range"]

# OOS設定
OOS_TRAIN_PCT = 0.6
OOS_VAL_PCT = 0.2
OOS_TOP_N_FOR_VAL = 10

# ランキング設定
RANKING_MIN_SYMBOLS = 3
RANKING_MIN_OOS_PASS_RATE = 0.5

# データディレクトリ
INPUTDATA_DIR = PROJECT_DIR / "inputdata"
RESULTS_DIR = PROJECT_DIR / "results" / "batch"

# 期間定義
PERIODS = [
    "20240201-20250131",
    "20250201-20260130",
]

# =====================================================================
# ログ設定
# =====================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _safe_float(v: float) -> float:
    if math.isinf(v) or math.isnan(v):
        return 9999.0 if v > 0 else -9999.0 if v < 0 else 0.0
    return v


# =====================================================================
# データスキャン・ロード
# =====================================================================

def scan_available_data(
    inputdata_dir: Path,
    tf_combos: List[Tuple[str, str]],
    periods: List[str],
) -> Dict[str, Dict[Tuple[str, str], List[str]]]:
    """
    利用可能な銘柄をスキャン

    Returns:
        {period: {(exec_tf, htf): [symbol, ...]}}
    """
    available = {}
    for period in periods:
        available[period] = {}
        for exec_tf, htf in tf_combos:
            symbols = set()
            for f in inputdata_dir.glob(f"*-{exec_tf}-{period}-merged.csv"):
                symbol = f.name.split(f"-{exec_tf}")[0]
                htf_file = inputdata_dir / f"{symbol}-{htf}-{period}-merged.csv"
                if htf_file.exists():
                    symbols.add(symbol)
            available[period][(exec_tf, htf)] = sorted(symbols)
    return available


def load_symbol_data(
    symbol: str, period: str, exec_tf: str, htf: str,
    inputdata_dir: Path,
) -> Dict[str, OHLCVData]:
    loader = BinanceCSVLoader()
    tf_dict = {}
    exec_path = inputdata_dir / f"{symbol}-{exec_tf}-{period}-merged.csv"
    htf_path = inputdata_dir / f"{symbol}-{htf}-{period}-merged.csv"
    tf_dict[exec_tf] = loader.load(str(exec_path), symbol=symbol)
    tf_dict[htf] = loader.load(str(htf_path), symbol=symbol)
    return tf_dict


# =====================================================================
# 単一銘柄最適化
# =====================================================================

def prepare_exec_df(
    tf_dict: Dict[str, OHLCVData],
    exec_tf: str,
    htf: str,
) -> pd.DataFrame:
    """実行TFのDataFrameにトレンドラベルを付与"""
    exec_ohlcv = tf_dict[exec_tf]
    exec_df = exec_ohlcv.df.copy()

    if htf and htf in tf_dict:
        htf_ohlcv = tf_dict[htf]
        htf_df = htf_ohlcv.df.copy()
        detector = TrendDetector()
        htf_df = detector.detect_ma_cross(
            htf_df, fast_period=MA_FAST, slow_period=MA_SLOW
        )
        exec_df = TrendDetector.label_execution_tf(exec_df, htf_df)
    else:
        exec_df["trend_regime"] = TrendRegime.RANGE.value

    return exec_df


def generate_all_configs(
    exit_profiles: Optional[List[Dict[str, Any]]] = None,
    template_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """全テンプレート x exit_profiles の config リストを生成

    Args:
        exit_profiles: Exit戦略プロファイル
        template_filter: テンプレートフィルター（"vwap", "ma", "rsi"など。カンマ区切りで複数指定可）
    """
    all_configs = []

    # フィルター処理
    filter_patterns = None
    if template_filter:
        filter_patterns = [p.strip().lower() for p in template_filter.split(",")]

    for tname, template in BUILTIN_TEMPLATES.items():
        # フィルターが指定されている場合、マッチするテンプレートのみ
        if filter_patterns:
            if not any(p in tname.lower() for p in filter_patterns):
                continue
        configs = template.generate_configs(exit_profiles=exit_profiles)
        all_configs.extend(configs)
    return all_configs


def optimize_single(
    exec_df: pd.DataFrame,
    all_configs: List[Dict[str, Any]],
    n_workers: int = N_WORKERS,
) -> OptimizationResultSet:
    """グリッドサーチのみ（OOSなし）"""
    optimizer = GridSearchOptimizer(
        initial_capital=INITIAL_CAPITAL,
        commission_pct=COMMISSION_PCT,
        slippage_pct=SLIPPAGE_PCT,
        top_n_results=TOP_N_RESULTS,
    )
    configs = copy.deepcopy(all_configs)
    result_set = optimizer.run(
        df=exec_df,
        configs=configs,
        target_regimes=TARGET_REGIMES,
        n_workers=n_workers,
        progress_callback=_log_progress,
    )
    return result_set


def _log_progress(completed: int, total: int, desc: str) -> None:
    """グリッドサーチの進捗をログ出力（10%刻み）"""
    if total <= 0:
        return
    pct = completed * 100 // total
    # 10%刻み or 最後に出力
    if completed == total or (pct % 10 == 0 and (completed - 1) * 100 // total < pct):
        logger.info(f"  進捗: {completed}/{total} ({pct}%) - {desc}")


def optimize_single_oos(
    exec_df: pd.DataFrame,
    all_configs: List[Dict[str, Any]],
    n_workers: int = N_WORKERS,
) -> ValidatedResultSet:
    """OOS 3分割検証付き最適化"""
    optimizer = GridSearchOptimizer(
        initial_capital=INITIAL_CAPITAL,
        commission_pct=COMMISSION_PCT,
        slippage_pct=SLIPPAGE_PCT,
        top_n_results=TOP_N_RESULTS,
    )
    split_config = DataSplitConfig(
        train_pct=OOS_TRAIN_PCT,
        val_pct=OOS_VAL_PCT,
        top_n_for_val=OOS_TOP_N_FOR_VAL,
    )
    configs = copy.deepcopy(all_configs)
    validated = run_validated_optimization(
        df=exec_df,
        all_configs=configs,
        target_regimes=TARGET_REGIMES,
        split_config=split_config,
        optimizer=optimizer,
        n_workers=n_workers,
        progress_callback=_log_progress,
    )
    return validated


# =====================================================================
# 結果保存
# =====================================================================

def _entry_to_dict(e: OptimizationEntry) -> Dict[str, Any]:
    """OptimizationEntry を JSON 用辞書に変換"""
    return {
        "template": e.template_name,
        "params": e.params,
        "regime": e.trend_regime,
        "exit_profile": e.config.get("_exit_profile", "default"),
        "score": round(_safe_float(e.composite_score), 4),
        "metrics": {
            "trades": e.metrics.total_trades,
            "win_rate": round(_safe_float(e.metrics.win_rate), 1),
            "profit_factor": round(_safe_float(e.metrics.profit_factor), 2),
            "total_pnl": round(_safe_float(e.metrics.total_profit_pct), 2),
            "max_dd": round(_safe_float(e.metrics.max_drawdown_pct), 2),
            "sharpe": round(_safe_float(e.metrics.sharpe_ratio), 2),
        },
    }


def save_result(
    result: Any,
    symbol: str,
    period: str,
    exec_tf: str,
    htf: str,
    use_oos: bool,
    output_dir: Path,
) -> Path:
    """最適化結果をJSONで保存"""
    data: Dict[str, Any] = {
        "symbol": symbol,
        "period": period,
        "execution_tf": exec_tf,
        "htf": htf,
        "oos": use_oos,
    }

    if use_oos and isinstance(result, ValidatedResultSet):
        data["train_results"] = [
            _entry_to_dict(e)
            for e in result.train_results.ranked()[:TOP_N_RESULTS]
        ]
        data["test_results"] = {}
        for regime, entry in result.test_results.items():
            data["test_results"][regime] = _entry_to_dict(entry)
        data["val_best"] = {}
        for regime, entry in result.val_best.items():
            data["val_best"][regime] = _entry_to_dict(entry)
        data["warnings"] = result.overfitting_warnings
        data["total_combinations"] = result.train_results.total_combinations
    else:
        result_set = result
        data["total_combinations"] = result_set.total_combinations
        data["results"] = [
            _entry_to_dict(e)
            for e in result_set.ranked()[:TOP_N_RESULTS]
        ]

    fname = f"{symbol}_{period}_{exec_tf}_{htf}.json"
    json_path = output_dir / fname
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return json_path


# =====================================================================
# Phase 1: 全銘柄 x 全TF x 全期間の最適化
# =====================================================================

def run_all_optimizations(
    available: Dict[str, Dict[Tuple[str, str], List[str]]],
    use_oos: bool,
    exit_profiles: Optional[List[Dict[str, Any]]],
    output_dir: Path,
    n_workers: int = N_WORKERS,
    force: bool = False,
    template_filter: Optional[str] = None,
) -> Dict[tuple, Dict[str, Any]]:
    """
    全組み合わせの最適化を実行

    Returns:
        {(period, exec_tf, htf, symbol): {"result": ..., "oos": bool}}
    """
    opt_dir = output_dir / "optimization"
    opt_dir.mkdir(parents=True, exist_ok=True)

    all_configs = generate_all_configs(exit_profiles, template_filter)
    config_count = len(all_configs)
    logger.info(
        f"Config数: {config_count} "
        f"({config_count * len(TARGET_REGIMES)} 組み合わせ/銘柄)"
    )

    all_results: Dict[tuple, Dict[str, Any]] = {}
    total_jobs = sum(
        len(symbols)
        for period_data in available.values()
        for symbols in period_data.values()
    )
    current_job = 0

    for period in sorted(available.keys()):
        for (exec_tf, htf), symbols in sorted(available[period].items()):
            if not symbols:
                continue
            logger.info(
                f"\n--- {exec_tf}/{htf} | {period} ({len(symbols)} 銘柄) ---"
            )

            for symbol in symbols:
                current_job += 1
                key = (period, exec_tf, htf, symbol)
                logger.info(f"[{current_job}/{total_jobs}] {symbol}")

                # 既存結果チェック
                fname = f"{symbol}_{period}_{exec_tf}_{htf}.json"
                existing = opt_dir / fname
                if existing.exists() and not force:
                    logger.info("  -> スキップ（既存結果あり）")
                    all_results[key] = {"file": str(existing), "oos": use_oos}
                    continue

                try:
                    t0 = time.time()
                    tf_dict = load_symbol_data(
                        symbol, period, exec_tf, htf, INPUTDATA_DIR,
                    )
                    exec_df = prepare_exec_df(tf_dict, exec_tf, htf)
                    bars = len(exec_df)
                    logger.info(f"  データ: {bars} bars")

                    if use_oos:
                        result = optimize_single_oos(
                            exec_df, all_configs, n_workers,
                        )
                        combos = result.train_results.total_combinations
                    else:
                        result = optimize_single(
                            exec_df, all_configs, n_workers,
                        )
                        combos = result.total_combinations

                    elapsed = time.time() - t0
                    logger.info(
                        f"  完了: {combos} 組み合わせ ({elapsed:.1f}s)"
                    )

                    save_result(
                        result, symbol, period, exec_tf, htf,
                        use_oos, opt_dir,
                    )
                    all_results[key] = {"result": result, "oos": use_oos}

                except Exception as e:
                    logger.error(f"  エラー: {e}")
                    continue

    return all_results


# =====================================================================
# Phase 2: 自動ランキング
# =====================================================================

def _extract_oos_results(
    all_results: Dict[tuple, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    全結果から OOS テスト結果（or Train ベスト）を抽出し、
    ランキング用のフラットリストに変換
    """
    rows: List[Dict[str, Any]] = []

    for key, data in all_results.items():
        period, exec_tf, htf, symbol = key
        result = data.get("result")
        is_oos = data.get("oos", False)

        if result is None:
            # ファイルのみ（analyze-only 用、未実装）
            continue

        if is_oos and isinstance(result, ValidatedResultSet):
            # OOS結果: テスト通過した戦略
            # train結果を辞書化してO(1)ルックアップ（線形探索を排除）
            train_maps: Dict[str, Dict[tuple, Any]] = {}
            for regime in TARGET_REGIMES:
                entries = result.train_results.filter_regime(regime).entries
                train_maps[regime] = {
                    (e.template_name, tuple(sorted(e.params.items()))): e
                    for e in entries
                }

            for regime, test_entry in result.test_results.items():
                # 辞書から直接取得
                params_key = tuple(sorted(test_entry.params.items()))
                train_entry = train_maps.get(regime, {}).get(
                    (test_entry.template_name, params_key)
                )

                rows.append({
                    "period": period,
                    "exec_tf": exec_tf,
                    "htf": htf,
                    "symbol": symbol,
                    "regime": regime,
                    "template": test_entry.template_name,
                    "params": test_entry.params,
                    "exit_profile": test_entry.config.get(
                        "_exit_profile", "default"
                    ),
                    "oos_pnl": _safe_float(
                        test_entry.metrics.total_profit_pct
                    ),
                    "oos_trades": test_entry.metrics.total_trades,
                    "oos_win_rate": _safe_float(test_entry.metrics.win_rate),
                    "oos_sharpe": _safe_float(test_entry.metrics.sharpe_ratio),
                    "oos_pf": _safe_float(
                        test_entry.metrics.profit_factor
                    ),
                    "train_pnl": _safe_float(
                        train_entry.metrics.total_profit_pct
                    ) if train_entry else 0,
                    "oos_pass": test_entry.metrics.total_profit_pct > 0,
                    "data_source": "oos_test",
                })

            # OOS テスト結果がないレジームの Train ベストも記録
            for regime in TARGET_REGIMES:
                if regime in result.test_results:
                    continue
                regime_set = result.train_results.filter_regime(regime)
                if not regime_set.best:
                    continue
                best = regime_set.best
                rows.append({
                    "period": period,
                    "exec_tf": exec_tf,
                    "htf": htf,
                    "symbol": symbol,
                    "regime": regime,
                    "template": best.template_name,
                    "params": best.params,
                    "exit_profile": best.config.get(
                        "_exit_profile", "default"
                    ),
                    "oos_pnl": _safe_float(
                        best.metrics.total_profit_pct
                    ),
                    "oos_trades": best.metrics.total_trades,
                    "oos_win_rate": _safe_float(best.metrics.win_rate),
                    "oos_sharpe": _safe_float(best.metrics.sharpe_ratio),
                    "oos_pf": _safe_float(best.metrics.profit_factor),
                    "train_pnl": _safe_float(
                        best.metrics.total_profit_pct
                    ),
                    "oos_pass": False,
                    "data_source": "train_only",
                })

        elif not is_oos and isinstance(result, OptimizationResultSet):
            # Train 結果のみ（OOSなし）
            for regime in TARGET_REGIMES:
                regime_set = result.filter_regime(regime)
                if not regime_set.best:
                    continue
                best = regime_set.best
                rows.append({
                    "period": period,
                    "exec_tf": exec_tf,
                    "htf": htf,
                    "symbol": symbol,
                    "regime": regime,
                    "template": best.template_name,
                    "params": best.params,
                    "exit_profile": best.config.get(
                        "_exit_profile", "default"
                    ),
                    "oos_pnl": _safe_float(
                        best.metrics.total_profit_pct
                    ),
                    "oos_trades": best.metrics.total_trades,
                    "oos_win_rate": _safe_float(best.metrics.win_rate),
                    "oos_sharpe": _safe_float(best.metrics.sharpe_ratio),
                    "oos_pf": _safe_float(best.metrics.profit_factor),
                    "train_pnl": _safe_float(
                        best.metrics.total_profit_pct
                    ),
                    "oos_pass": False,
                    "data_source": "train_only",
                })

    return rows


def auto_rank(
    all_results: Dict[tuple, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    全結果を自動ランキング

    戦略を (regime, template, exit_profile, exec_tf, htf) でグルーピングし、
    以下のスコアでランキング:
    - OOS通過率 x 0.4
    - 平均OOS PnL（正規化） x 0.3
    - 期間一貫性 x 0.2
    - 銘柄カバー率 x 0.1
    """
    rows = _extract_oos_results(all_results)
    if not rows:
        return {
            "strategies": [], "by_tf": {},
            "by_exit": {}, "by_regime": {},
        }

    # --- 戦略グルーピング ---
    groups: Dict[tuple, List[Dict]] = defaultdict(list)
    for r in rows:
        key = (
            r["regime"], r["template"], r["exit_profile"],
            r["exec_tf"], r["htf"],
        )
        groups[key].append(r)

    # --- 全銘柄セット（カバー率計算用） ---
    all_symbols = set()
    for key in all_results:
        all_symbols.add(key[3])

    # --- 各グループのスコア計算 ---
    strategies = []
    for gkey, entries in groups.items():
        regime, template, exit_profile, exec_tf, htf = gkey
        n_symbols = len(set(e["symbol"] for e in entries))
        n_entries = len(entries)

        oos_passes = [e for e in entries if e["oos_pass"]]
        oos_pass_rate = len(oos_passes) / max(n_entries, 1)

        pnls = [e["oos_pnl"] for e in entries]
        avg_pnl = sum(pnls) / max(len(pnls), 1)
        med_pnl = median(pnls) if pnls else 0

        # 期間一貫性
        periods_seen = set(e["period"] for e in entries)
        period_consistency = len(periods_seen) / max(len(PERIODS), 1)

        # 両期間でプラスの銘柄数
        symbol_period: Dict[str, Dict[str, float]] = defaultdict(dict)
        for e in entries:
            symbol_period[e["symbol"]][e["period"]] = e["oos_pnl"]

        both_positive = 0
        both_count = 0
        for sym, period_pnl in symbol_period.items():
            if len(period_pnl) >= 2:
                both_count += 1
                if all(p > 0 for p in period_pnl.values()):
                    both_positive += 1

        # 銘柄カバー率
        symbol_coverage = n_symbols / max(len(all_symbols), 1)

        # 総合スコア
        score = (
            oos_pass_rate * 0.4
            + min(max(avg_pnl / 10.0, 0), 1.0) * 0.3
            + period_consistency * 0.2
            + symbol_coverage * 0.1
        )

        strategies.append({
            "regime": regime,
            "template": template,
            "exit_profile": exit_profile,
            "exec_tf": exec_tf,
            "htf": htf,
            "score": round(score, 4),
            "oos_pass_rate": round(oos_pass_rate, 3),
            "avg_pnl": round(avg_pnl, 2),
            "median_pnl": round(med_pnl, 2),
            "symbol_count": n_symbols,
            "period_consistency": round(period_consistency, 2),
            "both_periods_positive": both_positive,
            "both_periods_count": both_count,
            "entries": n_entries,
        })

    strategies.sort(key=lambda x: -x["score"])

    # --- TF別集計 ---
    by_tf: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"pass_rate": 0, "avg_pnl": 0.0, "count": 0}
    )
    for r in rows:
        tf_key = f"{r['exec_tf']}/{r['htf']}"
        by_tf[tf_key]["count"] += 1
        by_tf[tf_key]["avg_pnl"] += r["oos_pnl"]
        if r["oos_pass"]:
            by_tf[tf_key]["pass_rate"] += 1

    for tf_key in by_tf:
        n = by_tf[tf_key]["count"]
        by_tf[tf_key]["avg_pnl"] = round(
            by_tf[tf_key]["avg_pnl"] / max(n, 1), 2
        )
        by_tf[tf_key]["pass_rate"] = round(
            by_tf[tf_key]["pass_rate"] / max(n, 1), 3
        )

    # --- Exit profile別集計 ---
    by_exit: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"pass_rate": 0, "avg_pnl": 0.0, "count": 0}
    )
    for r in rows:
        ep = r["exit_profile"]
        by_exit[ep]["count"] += 1
        by_exit[ep]["avg_pnl"] += r["oos_pnl"]
        if r["oos_pass"]:
            by_exit[ep]["pass_rate"] += 1

    for ep in by_exit:
        n = by_exit[ep]["count"]
        by_exit[ep]["avg_pnl"] = round(
            by_exit[ep]["avg_pnl"] / max(n, 1), 2
        )
        by_exit[ep]["pass_rate"] = round(
            by_exit[ep]["pass_rate"] / max(n, 1), 3
        )

    # --- レジーム別集計 ---
    by_regime: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"pass_rate": 0, "avg_pnl": 0.0, "count": 0}
    )
    for r in rows:
        regime = r["regime"]
        by_regime[regime]["count"] += 1
        by_regime[regime]["avg_pnl"] += r["oos_pnl"]
        if r["oos_pass"]:
            by_regime[regime]["pass_rate"] += 1

    for regime in by_regime:
        n = by_regime[regime]["count"]
        by_regime[regime]["avg_pnl"] = round(
            by_regime[regime]["avg_pnl"] / max(n, 1), 2
        )
        by_regime[regime]["pass_rate"] = round(
            by_regime[regime]["pass_rate"] / max(n, 1), 3
        )

    return {
        "strategies": strategies,
        "by_tf": dict(by_tf),
        "by_exit": dict(by_exit),
        "by_regime": dict(by_regime),
    }


# =====================================================================
# Phase 3: Markdown レポート生成
# =====================================================================

def generate_report(
    ranking: Dict[str, Any],
    config: Dict[str, Any],
    output_dir: Path,
    total_time: float,
) -> Path:
    """Markdown レポートを生成"""
    lines: List[str] = []

    lines.append("# バッチ自動最適化レポート")
    lines.append("")
    lines.append(
        f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    lines.append("")

    # --- サマリー ---
    lines.append("## サマリー")
    lines.append("")
    tf_combos = config.get("tf_combos", [])
    periods = config.get("periods", [])
    exit_mode = config.get("exit_mode", "all")
    use_oos = config.get("use_oos", True)
    lines.append(
        f"- **TF組み合わせ**: {len(tf_combos)} "
        f"({', '.join(f'{e}/{h}' for e, h in tf_combos)})"
    )
    lines.append(f"- **期間**: {', '.join(periods)}")
    lines.append(f"- **Exit profiles**: {exit_mode}")
    lines.append(
        f"- **OOS検証**: "
        f"{'ON (Train 60% / Val 20% / Test 20%)' if use_oos else 'OFF'}"
    )
    lines.append(
        f"- **実行時間**: {total_time:.0f}s ({total_time / 60:.1f}min)"
    )
    lines.append("")

    strategies = ranking.get("strategies", [])
    by_tf = ranking.get("by_tf", {})
    by_exit = ranking.get("by_exit", {})
    by_regime = ranking.get("by_regime", {})

    # --- 推奨戦略ランキング ---
    lines.append("## 推奨戦略ランキング (Top 20)")
    lines.append("")

    recommended = [
        s for s in strategies
        if s["oos_pass_rate"] >= RANKING_MIN_OOS_PASS_RATE
        and s["symbol_count"] >= RANKING_MIN_SYMBOLS
    ]

    if recommended:
        lines.append(
            "| Rank | TF | Regime | Template | Exit | "
            "OOS通過率 | 平均PnL | 銘柄数 | 期間一貫性 | Score |"
        )
        lines.append(
            "|------|------|--------|----------|------|"
            "---------|---------|--------|----------|-------|"
        )
        for i, s in enumerate(recommended[:20]):
            lines.append(
                f"| {i + 1} | {s['exec_tf']}/{s['htf']} | "
                f"{s['regime']} | {s['template']} | "
                f"{s['exit_profile']} | {s['oos_pass_rate']:.0%} | "
                f"{s['avg_pnl']:+.1f}% | {s['symbol_count']} | "
                f"{s['period_consistency']:.0%} | {s['score']:.3f} |"
            )
        lines.append("")
    else:
        lines.append(
            "(OOS通過率 >= 50% かつ 3銘柄以上の戦略なし)"
        )
        lines.append("")

    # --- Exit戦略比較 ---
    lines.append("## Exit戦略比較")
    lines.append("")
    if by_exit:
        lines.append(
            "| Exit Profile | OOS通過率 | 平均PnL | サンプル数 |"
        )
        lines.append(
            "|-------------|---------|---------|----------|"
        )
        for ep, stats in sorted(
            by_exit.items(), key=lambda x: -x[1]["pass_rate"]
        ):
            lines.append(
                f"| {ep} | {stats['pass_rate']:.0%} | "
                f"{stats['avg_pnl']:+.1f}% | {stats['count']} |"
            )
        lines.append("")

    # --- TF別比較 ---
    lines.append("## タイムフレーム別比較")
    lines.append("")
    if by_tf:
        lines.append("| TF | OOS通過率 | 平均PnL | サンプル数 |")
        lines.append("|----|---------|---------|----------|")
        for tf_key, stats in sorted(
            by_tf.items(), key=lambda x: -x[1]["pass_rate"]
        ):
            lines.append(
                f"| {tf_key} | {stats['pass_rate']:.0%} | "
                f"{stats['avg_pnl']:+.1f}% | {stats['count']} |"
            )
        lines.append("")

    # --- レジーム別比較 ---
    lines.append("## レジーム別比較")
    lines.append("")
    if by_regime:
        lines.append("| Regime | OOS通過率 | 平均PnL | サンプル数 |")
        lines.append("|--------|---------|---------|----------|")
        for regime, stats in sorted(
            by_regime.items(), key=lambda x: -x[1]["pass_rate"]
        ):
            lines.append(
                f"| {regime} | {stats['pass_rate']:.0%} | "
                f"{stats['avg_pnl']:+.1f}% | {stats['count']} |"
            )
        lines.append("")

    # --- レジーム別詳細 ---
    for regime in TARGET_REGIMES:
        regime_strategies = [
            s for s in strategies if s["regime"] == regime
        ]
        if not regime_strategies:
            continue

        lines.append(f"### {regime.upper()}")
        lines.append("")

        top_regime = regime_strategies[:10]
        lines.append(
            "| Template | Exit | TF | OOS通過率 | "
            "平均PnL | 中央PnL | 銘柄数 |"
        )
        lines.append(
            "|----------|------|----|---------|"
            "---------|---------|--------|"
        )
        for s in top_regime:
            lines.append(
                f"| {s['template']} | {s['exit_profile']} | "
                f"{s['exec_tf']}/{s['htf']} | "
                f"{s['oos_pass_rate']:.0%} | "
                f"{s['avg_pnl']:+.1f}% | "
                f"{s['median_pnl']:+.1f}% | {s['symbol_count']} |"
            )
        lines.append("")

    # --- 過学習警告 ---
    overfitting = [
        s for s in strategies
        if s["oos_pass_rate"] == 0 and s["entries"] >= 5
    ]
    if overfitting:
        lines.append("## 過学習警告")
        lines.append("")
        lines.append(
            "以下の戦略はOOS通過率0%（過学習の可能性大）:"
        )
        lines.append("")
        for s in overfitting[:10]:
            lines.append(
                f"- {s['regime']} / {s['template']} / "
                f"{s['exit_profile']} ({s['exec_tf']}/{s['htf']}): "
                f"平均PnL {s['avg_pnl']:+.1f}%, {s['entries']}件"
            )
        lines.append("")

    # --- フッター ---
    lines.append("---")
    lines.append(
        f"Generated by batch_optimize.py v2 | "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    report_text = "\n".join(lines)

    # Markdown保存
    report_path = output_dir / "report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    logger.info(f"レポート保存: {report_path}")

    # ランキングJSON保存
    ranking_path = output_dir / "ranking.json"
    with open(ranking_path, "w", encoding="utf-8") as f:
        json.dump(ranking, f, ensure_ascii=False, indent=2)
    logger.info(f"ランキング保存: {ranking_path}")

    # config.json
    config_path = output_dir / "config.json"
    config_save: Dict[str, Any] = {}
    for k, v in config.items():
        if isinstance(v, Path):
            config_save[k] = str(v)
        elif isinstance(v, list) and v and isinstance(v[0], tuple):
            config_save[k] = [list(t) for t in v]
        else:
            config_save[k] = v
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_save, f, ensure_ascii=False, indent=2)

    # コンソールにもサマリー出力
    print("\n" + "=" * 70)
    print("  推奨戦略 Top 5")
    print("=" * 70)
    for i, s in enumerate(recommended[:5]):
        print(
            f"  #{i + 1} {s['exec_tf']}/{s['htf']} | {s['regime']} | "
            f"{s['template']} | {s['exit_profile']} | "
            f"OOS {s['oos_pass_rate']:.0%} | PnL {s['avg_pnl']:+.1f}%"
        )
    if not recommended:
        print("  (推奨戦略なし)")
    print("=" * 70)

    return report_path


# =====================================================================
# メイン
# =====================================================================

def parse_tf_combos(arg: str) -> List[Tuple[str, str]]:
    """TF組み合わせ文字列をパース"""
    if arg == "all":
        return list(TF_COMBOS)
    combos = []
    for item in arg.split(","):
        parts = item.strip().split(":")
        if len(parts) == 2:
            combos.append((parts[0], parts[1]))
    return combos


def main():
    parser = argparse.ArgumentParser(
        description="バッチ自動最適化スクリプト v2"
    )
    parser.add_argument(
        "--symbols", type=str, default="",
        help="対象銘柄（カンマ区切り）",
    )
    parser.add_argument(
        "--periods", type=str, default="",
        help="対象期間（カンマ区切り）",
    )
    parser.add_argument(
        "--tf-combos", type=str, default="all",
        help="TF組み合わせ（all or 15m:1h,1h:4h）",
    )
    parser.add_argument(
        "--workers", type=int, default=N_WORKERS,
        help=f"並列ワーカー数（デフォルト: {N_WORKERS}）",
    )
    parser.add_argument(
        "--oos", action="store_true", default=True,
        help="OOS 3分割検証を有効化（デフォルト: ON）",
    )
    parser.add_argument(
        "--no-oos", action="store_true",
        help="OOS検証なし",
    )
    parser.add_argument(
        "--exit-profiles", type=str, default="all",
        help="Exit profiles（all / fixed / atr / no_sl / hybrid / none）",
    )
    parser.add_argument(
        "--templates", type=str, default="",
        help="テンプレートフィルター（vwap, ma, rsi など。カンマ区切りで複数指定可）",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="既存結果を上書き",
    )
    args = parser.parse_args()

    # OOS設定
    use_oos = not args.no_oos

    # TF組み合わせ
    tf_combos = parse_tf_combos(args.tf_combos)
    if not tf_combos:
        logger.error("有効なTF組み合わせがありません")
        sys.exit(1)

    # 期間
    periods = PERIODS
    if args.periods:
        periods = [p.strip() for p in args.periods.split(",")]

    # Exit profiles
    exit_profiles: Optional[List[Dict[str, Any]]] = None
    if args.exit_profiles != "none":
        exit_profiles = get_profiles(args.exit_profiles)

    # Template filter
    template_filter = args.templates if args.templates else None

    # 設定辞書
    config: Dict[str, Any] = {
        "tf_combos": tf_combos,
        "periods": periods,
        "use_oos": use_oos,
        "exit_mode": args.exit_profiles,
        "template_filter": template_filter,
        "n_workers": args.workers,
        "trend_method": TREND_METHOD,
        "ma_fast": MA_FAST,
        "ma_slow": MA_SLOW,
        "initial_capital": INITIAL_CAPITAL,
        "commission_pct": COMMISSION_PCT,
        "slippage_pct": SLIPPAGE_PCT,
    }

    logger.info("=" * 60)
    logger.info("  バッチ自動最適化 v2")
    logger.info("=" * 60)
    logger.info(
        f"TF: {', '.join(f'{e}/{h}' for e, h in tf_combos)}"
    )
    logger.info(f"期間: {', '.join(periods)}")
    logger.info(f"OOS: {'ON' if use_oos else 'OFF'}")
    if template_filter:
        logger.info(f"Templates: {template_filter}")
    logger.info(
        f"Exit profiles: {args.exit_profiles} "
        f"({len(exit_profiles) if exit_profiles else 0} 種)"
    )
    logger.info(f"Workers: {args.workers}")

    # データスキャン
    logger.info("\nデータスキャン中...")
    available = scan_available_data(INPUTDATA_DIR, tf_combos, periods)

    # 銘柄フィルタ
    if args.symbols:
        filter_syms = set(s.strip() for s in args.symbols.split(","))
        for period in available:
            for tf_key in available[period]:
                available[period][tf_key] = [
                    s for s in available[period][tf_key]
                    if s in filter_syms
                ]

    # サマリー表示
    total_jobs = 0
    for period in sorted(available.keys()):
        for (exec_tf, htf), symbols in sorted(available[period].items()):
            n = len(symbols)
            if n > 0:
                logger.info(f"  {exec_tf}/{htf} | {period}: {n} 銘柄")
            total_jobs += n

    logger.info(f"合計: {total_jobs} ジョブ")

    if total_jobs == 0:
        logger.error("対象データが見つかりません")
        sys.exit(1)

    # 出力ディレクトリ
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = RESULTS_DIR / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"出力先: {output_dir}")

    # Phase 1: 全最適化
    logger.info("\n" + "=" * 60)
    logger.info("  Phase 1: 全銘柄 x 全TF x 全期間の最適化")
    logger.info("=" * 60)
    t0 = time.time()

    all_results = run_all_optimizations(
        available=available,
        use_oos=use_oos,
        exit_profiles=exit_profiles,
        output_dir=output_dir,
        n_workers=args.workers,
        force=args.force,
        template_filter=template_filter,
    )

    phase1_time = time.time() - t0
    logger.info(f"\nPhase 1 完了 ({phase1_time:.0f}s)")

    # Phase 2: 自動ランキング
    logger.info("\n" + "=" * 60)
    logger.info("  Phase 2: 自動ランキング")
    logger.info("=" * 60)

    ranking = auto_rank(all_results)

    n_strategies = len(ranking["strategies"])
    n_recommended = len([
        s for s in ranking["strategies"]
        if s["oos_pass_rate"] >= RANKING_MIN_OOS_PASS_RATE
        and s["symbol_count"] >= RANKING_MIN_SYMBOLS
    ])
    logger.info(f"戦略グループ: {n_strategies}")
    logger.info(f"推奨戦略: {n_recommended}")

    # Phase 3: レポート生成
    logger.info("\n" + "=" * 60)
    logger.info("  Phase 3: レポート生成")
    logger.info("=" * 60)

    total_time = time.time() - t0
    report_path = generate_report(ranking, config, output_dir, total_time)

    logger.info(f"\n全処理完了 ({total_time:.0f}s)")
    logger.info(f"結果: {output_dir}")
    logger.info(f"レポート: {report_path}")


if __name__ == "__main__":
    main()
