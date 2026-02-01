"""
バッチ自動最適化スクリプト

30銘柄×2期間（2024/2025）の全データに対し、
14テンプレート×パラメータグリッドの自動最適化を実行。
レジーム別の「コンセンサス戦略」を自動抽出し、
クロスバリデーション（2024→2025）で過学習を検出する。

使い方:
    python scripts/batch_optimize.py
    python scripts/batch_optimize.py --symbols BTCUSDT,ETHUSDT
    python scripts/batch_optimize.py --periods 20250201-20260130
"""

import argparse
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

from data.binance_loader import BinanceCSVLoader
from data.base import OHLCVData
from analysis.trend import TrendDetector, TrendRegime
from optimizer.grid import GridSearchOptimizer
from optimizer.templates import BUILTIN_TEMPLATES
from optimizer.results import OptimizationResultSet, OptimizationEntry
from optimizer.scoring import ScoringWeights

# =====================================================================
# 設定（ここを変更して各種パラメータを調整）
# =====================================================================

EXEC_TF = "15m"
HTF = "1h"
TREND_METHOD = "ma_cross"  # "ma_cross", "adx", "combined"
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

# コンセンサス分析: 各レジームでTOP何位まで集計するか
CONSENSUS_TOP_N = 5

# クロスバリデーション: 各レジームでTOP何戦略を検証するか
CV_TOP_N = 3

# データディレクトリ
INPUTDATA_DIR = PROJECT_DIR / "inputdata"
RESULTS_DIR = PROJECT_DIR / "results" / "batch"

# 期間定義
PERIODS = [
    "20240201-20250131",  # 2024期間
    "20250201-20260130",  # 2025期間
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
    """inf/nanをJSON安全な値に変換"""
    if math.isinf(v) or math.isnan(v):
        return 9999.0 if v > 0 else -9999.0 if v < 0 else 0.0
    return v


# =====================================================================
# データロード
# =====================================================================

def scan_available_data(
    inputdata_dir: Path, exec_tf: str, htf: str, periods: List[str],
) -> Dict[str, List[str]]:
    """
    inputdata/ をスキャンし、利用可能な銘柄×期間の組み合わせを検出

    Returns:
        {period: [symbol, ...]} の辞書
    """
    available = {}
    for period in periods:
        symbols = set()
        # exec_tf ファイルの存在チェック
        for f in inputdata_dir.glob(f"*-{exec_tf}-{period}-merged.csv"):
            symbol = f.name.split(f"-{exec_tf}")[0]
            # htf ファイルも存在するか確認
            htf_file = inputdata_dir / f"{symbol}-{htf}-{period}-merged.csv"
            if htf_file.exists():
                symbols.add(symbol)
            else:
                logger.warning(f"{symbol}: HTFファイルなし ({htf_file.name})")
        available[period] = sorted(symbols)
    return available


def load_symbol_data(
    symbol: str, period: str, exec_tf: str, htf: str,
    inputdata_dir: Path,
) -> Dict[str, OHLCVData]:
    """1銘柄のexec_tf + htf データをロード"""
    loader = BinanceCSVLoader()
    tf_dict = {}

    exec_path = inputdata_dir / f"{symbol}-{exec_tf}-{period}-merged.csv"
    htf_path = inputdata_dir / f"{symbol}-{htf}-{period}-merged.csv"

    tf_dict[exec_tf] = loader.load(str(exec_path), symbol=symbol)
    tf_dict[htf] = loader.load(str(htf_path), symbol=symbol)

    return tf_dict


# =====================================================================
# 単一銘柄最適化（UIの _execute_single_optimization と同等ロジック）
# =====================================================================

def optimize_single(
    tf_dict: Dict[str, OHLCVData],
    exec_tf: str,
    htf: str,
    trend_method: str,
    ma_fast: int,
    ma_slow: int,
    adx_period: int,
    adx_trend_th: float,
    adx_range_th: float,
    target_regimes: List[str],
    initial_capital: float,
    commission_pct: float,
    slippage_pct: float,
    n_workers: int,
    top_n_results: int,
) -> OptimizationResultSet:
    """
    1銘柄分の全テンプレート×パラメータのグリッドサーチを実行

    _execute_single_optimization() のStreamlit非依存版
    """
    exec_ohlcv = tf_dict[exec_tf]
    exec_df = exec_ohlcv.df.copy()

    # トレンドラベル付与
    if htf and htf in tf_dict:
        htf_ohlcv = tf_dict[htf]
        htf_df = htf_ohlcv.df.copy()

        detector = TrendDetector()
        if trend_method == "ma_cross":
            htf_df = detector.detect_ma_cross(
                htf_df, fast_period=ma_fast, slow_period=ma_slow
            )
        elif trend_method == "adx":
            htf_df = detector.detect_adx(
                htf_df, adx_period=adx_period,
                trend_threshold=adx_trend_th,
                range_threshold=adx_range_th,
            )
        else:  # combined
            htf_df = detector.detect_combined(
                htf_df, ma_fast=ma_fast, ma_slow=ma_slow,
                adx_period=adx_period,
                adx_trend_threshold=adx_trend_th,
                adx_range_threshold=adx_range_th,
            )

        exec_df = TrendDetector.label_execution_tf(exec_df, htf_df)
    else:
        exec_df["trend_regime"] = TrendRegime.RANGE.value

    # 全テンプレートからconfig生成
    all_configs = []
    for tname, template in BUILTIN_TEMPLATES.items():
        configs = template.generate_configs()
        all_configs.extend(configs)

    # グリッドサーチ実行
    optimizer = GridSearchOptimizer(
        initial_capital=initial_capital,
        commission_pct=commission_pct,
        slippage_pct=slippage_pct,
        top_n_results=top_n_results,
    )

    result_set = optimizer.run(
        df=exec_df,
        configs=all_configs,
        target_regimes=target_regimes,
        n_workers=n_workers,
    )

    result_set.symbol = exec_ohlcv.symbol
    result_set.execution_tf = exec_tf
    result_set.htf = htf or ""

    return result_set


# =====================================================================
# 結果保存
# =====================================================================

def save_optimization_result(
    result_set: OptimizationResultSet,
    symbol: str,
    period: str,
    output_dir: Path,
) -> Path:
    """最適化結果をJSONで保存"""
    json_rows = []
    for e in result_set.ranked():
        json_rows.append({
            "template": e.template_name,
            "params": e.params,
            "regime": e.trend_regime,
            "score": round(_safe_float(e.composite_score), 4),
            "metrics": {
                "trades": e.metrics.total_trades,
                "win_rate": round(_safe_float(e.metrics.win_rate), 1),
                "profit_factor": round(_safe_float(e.metrics.profit_factor), 2),
                "total_pnl": round(_safe_float(e.metrics.total_profit_pct), 2),
                "max_dd": round(_safe_float(e.metrics.max_drawdown_pct), 2),
                "sharpe": round(_safe_float(e.metrics.sharpe_ratio), 2),
            },
            "config": e.config,
        })

    json_path = output_dir / f"{symbol}_{period}.json"
    data = {
        "symbol": symbol,
        "period": period,
        "execution_tf": result_set.execution_tf,
        "htf": result_set.htf,
        "total_combinations": result_set.total_combinations,
        "results": json_rows,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return json_path


# =====================================================================
# Phase 1: 全銘柄×全期間の最適化
# =====================================================================

def run_all_optimizations(
    available: Dict[str, List[str]],
    config: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Dict[str, OptimizationResultSet]]:
    """
    全銘柄×全期間の最適化を実行

    Returns:
        {period: {symbol: OptimizationResultSet}}
    """
    opt_dir = output_dir / "optimization"
    opt_dir.mkdir(parents=True, exist_ok=True)

    all_results: Dict[str, Dict[str, OptimizationResultSet]] = {}
    total_jobs = sum(len(syms) for syms in available.values())
    current_job = 0

    for period, symbols in available.items():
        all_results[period] = {}
        for symbol in symbols:
            current_job += 1
            logger.info(
                f"[{current_job}/{total_jobs}] {symbol} ({period})"
            )

            # 既存結果があればスキップ
            existing = opt_dir / f"{symbol}_{period}.json"
            if existing.exists():
                logger.info(f"  -> スキップ（既存結果あり）")
                result_set = OptimizationResultSet.from_json(str(existing))
                result_set.symbol = symbol
                result_set.execution_tf = config["exec_tf"]
                result_set.htf = config["htf"]
                all_results[period][symbol] = result_set
                continue

            try:
                t0 = time.time()
                tf_dict = load_symbol_data(
                    symbol, period,
                    config["exec_tf"], config["htf"],
                    config["inputdata_dir"],
                )
                exec_bars = tf_dict[config["exec_tf"]].bars
                htf_bars = tf_dict[config["htf"]].bars
                logger.info(
                    f"  データ: exec={exec_bars} bars, htf={htf_bars} bars"
                )

                result_set = optimize_single(
                    tf_dict=tf_dict,
                    exec_tf=config["exec_tf"],
                    htf=config["htf"],
                    trend_method=config["trend_method"],
                    ma_fast=config["ma_fast"],
                    ma_slow=config["ma_slow"],
                    adx_period=config["adx_period"],
                    adx_trend_th=config["adx_trend_th"],
                    adx_range_th=config["adx_range_th"],
                    target_regimes=config["target_regimes"],
                    initial_capital=config["initial_capital"],
                    commission_pct=config["commission_pct"],
                    slippage_pct=config["slippage_pct"],
                    n_workers=config["n_workers"],
                    top_n_results=config["top_n_results"],
                )

                elapsed = time.time() - t0
                logger.info(
                    f"  完了: {result_set.total_combinations} 組み合わせ "
                    f"({elapsed:.1f}s)"
                )

                save_optimization_result(result_set, symbol, period, opt_dir)
                all_results[period][symbol] = result_set

            except Exception as e:
                logger.error(f"  エラー: {e}")
                continue

    return all_results


# =====================================================================
# Phase 2: コンセンサス分析
# =====================================================================

def analyze_consensus(
    all_results: Dict[str, Dict[str, OptimizationResultSet]],
    top_n: int = CONSENSUS_TOP_N,
) -> Dict[str, Any]:
    """
    レジーム別コンセンサス戦略を分析

    各銘柄のレジーム別TOP-N戦略を集計し、
    複数銘柄で共通して上位に来る戦略を抽出する。
    """
    consensus = {}

    for regime in TARGET_REGIMES:
        # strategy_key = (template, params_tuple) → 出現情報リスト
        strategy_stats: Dict[Tuple, List[Dict]] = defaultdict(list)
        template_counts: Dict[str, int] = defaultdict(int)
        total_symbols = 0

        for period, symbol_results in all_results.items():
            for symbol, result_set in symbol_results.items():
                regime_set = result_set.filter_regime(regime)
                ranked = regime_set.ranked()
                if not ranked:
                    continue

                total_symbols += 1

                for i, entry in enumerate(ranked[:top_n]):
                    params_tuple = tuple(sorted(entry.params.items()))
                    key = (entry.template_name, params_tuple)

                    strategy_stats[key].append({
                        "symbol": symbol,
                        "period": period,
                        "rank": i + 1,
                        "score": _safe_float(entry.composite_score),
                        "pnl": _safe_float(entry.metrics.total_profit_pct),
                        "win_rate": _safe_float(entry.metrics.win_rate),
                        "profit_factor": _safe_float(entry.metrics.profit_factor),
                        "sharpe": _safe_float(entry.metrics.sharpe_ratio),
                        "trades": entry.metrics.total_trades,
                    })

                    template_counts[entry.template_name] += 1

        # 出現頻度でソート
        top_strategies = []
        for (template, params_tuple), appearances in sorted(
            strategy_stats.items(),
            key=lambda x: len(x[1]),
            reverse=True,
        )[:20]:
            params = dict(params_tuple)
            symbols_seen = list(set(a["symbol"] for a in appearances))
            pnls = [a["pnl"] for a in appearances]
            scores = [a["score"] for a in appearances]
            win_rates = [a["win_rate"] for a in appearances]
            pfs = [a["profit_factor"] for a in appearances]

            top_strategies.append({
                "template": template,
                "params": params,
                "symbol_count": len(symbols_seen),
                "appearance_count": len(appearances),
                "appearance_rate": round(
                    len(symbols_seen) / max(total_symbols, 1), 3
                ),
                "median_score": round(median(scores), 4) if scores else 0,
                "median_pnl": round(median(pnls), 2) if pnls else 0,
                "median_win_rate": round(
                    median(win_rates), 1
                ) if win_rates else 0,
                "median_profit_factor": round(
                    median(pfs), 2
                ) if pfs else 0,
                "symbols": symbols_seen,
            })

        # テンプレート頻度（正規化）
        total_appearances = sum(template_counts.values()) or 1
        template_freq = {
            k: round(v / total_appearances, 3)
            for k, v in sorted(
                template_counts.items(), key=lambda x: -x[1]
            )
        }

        consensus[regime] = {
            "total_symbols_periods": total_symbols,
            "top_strategies": top_strategies,
            "template_frequency": template_freq,
        }

    return consensus


# =====================================================================
# Phase 3: クロスバリデーション
# =====================================================================

def cross_validate(
    all_results: Dict[str, Dict[str, OptimizationResultSet]],
    config: Dict[str, Any],
    train_period: str = "20240201-20250131",
    test_period: str = "20250201-20260130",
    top_n: int = CV_TOP_N,
) -> Dict[str, Any]:
    """
    2024期間のレジーム別ベスト戦略を2025データで検証

    train_periodの各レジームTOP-N戦略を取得し、
    test_periodの全銘柄で同じ戦略を個別実行して性能変化を比較。
    """
    if train_period not in all_results or test_period not in all_results:
        logger.warning("クロスバリデーション: 両期間の結果が必要です")
        return {}

    train_results = all_results[train_period]
    test_results = all_results[test_period]
    cv_results = {}

    for regime in TARGET_REGIMES:
        # train期間で頻出するTOP戦略を特定
        strategy_scores: Dict[Tuple, List[Dict]] = defaultdict(list)

        for symbol, result_set in train_results.items():
            regime_set = result_set.filter_regime(regime)
            ranked = regime_set.ranked()
            if not ranked:
                continue
            best = ranked[0]
            params_tuple = tuple(sorted(best.params.items()))
            key = (best.template_name, params_tuple)
            strategy_scores[key].append({
                "symbol": symbol,
                "score": _safe_float(best.composite_score),
                "pnl": _safe_float(best.metrics.total_profit_pct),
                "win_rate": _safe_float(best.metrics.win_rate),
                "profit_factor": _safe_float(best.metrics.profit_factor),
                "sharpe": _safe_float(best.metrics.sharpe_ratio),
            })

        # 出現頻度上位N戦略
        top_strategies = sorted(
            strategy_scores.items(),
            key=lambda x: len(x[1]),
            reverse=True,
        )[:top_n]

        regime_cv = []
        for (template_name, params_tuple), train_appearances in top_strategies:
            params = dict(params_tuple)
            logger.info(
                f"  CV [{regime}] {template_name} "
                f"{params} ({len(train_appearances)} symbols)"
            )

            # test_periodの全銘柄で同じ戦略の結果を探す
            results_by_symbol = {}
            train_syms = {a["symbol"] for a in train_appearances}

            # train期間の結果
            for appearance in train_appearances:
                sym = appearance["symbol"]
                results_by_symbol[sym] = {
                    "train": {
                        "pnl": round(_safe_float(appearance["pnl"]), 2),
                        "win_rate": round(_safe_float(appearance["win_rate"]), 1),
                        "pf": round(_safe_float(appearance["profit_factor"]), 2),
                        "sharpe": round(_safe_float(appearance["sharpe"]), 2),
                    }
                }

            # test期間: 同じ戦略のエントリーを探す
            for sym, result_set in test_results.items():
                if sym not in results_by_symbol:
                    results_by_symbol[sym] = {}

                regime_set = result_set.filter_regime(regime)
                # 同じテンプレート+パラメータのエントリーを探す
                match = None
                for entry in regime_set.entries:
                    entry_params = tuple(sorted(entry.params.items()))
                    if (entry.template_name == template_name
                            and entry_params == params_tuple):
                        match = entry
                        break

                if match:
                    results_by_symbol[sym]["test"] = {
                        "pnl": round(_safe_float(match.metrics.total_profit_pct), 2),
                        "win_rate": round(_safe_float(match.metrics.win_rate), 1),
                        "pf": round(_safe_float(match.metrics.profit_factor), 2),
                        "sharpe": round(_safe_float(match.metrics.sharpe_ratio), 2),
                    }

            # サマリー計算（両期間にデータがある銘柄のみ）
            both_count = 0
            profitable_both = 0
            pnl_decays = []

            for sym, data in results_by_symbol.items():
                if "train" in data and "test" in data:
                    both_count += 1
                    if data["train"]["pnl"] > 0 and data["test"]["pnl"] > 0:
                        profitable_both += 1
                    if data["train"]["pnl"] != 0:
                        decay = (
                            (data["test"]["pnl"] - data["train"]["pnl"])
                            / abs(data["train"]["pnl"])
                        )
                        pnl_decays.append(decay)

            regime_cv.append({
                "template": template_name,
                "params": params,
                "train_period": train_period,
                "test_period": test_period,
                "train_symbol_count": len(train_appearances),
                "results_by_symbol": results_by_symbol,
                "summary": {
                    "profitable_in_both": profitable_both,
                    "total_symbols_with_both": both_count,
                    "consistency_rate": round(
                        profitable_both / max(both_count, 1), 3
                    ),
                    "avg_pnl_decay": round(
                        sum(pnl_decays) / max(len(pnl_decays), 1), 3
                    ) if pnl_decays else None,
                },
            })

        cv_results[regime] = regime_cv

    return cv_results


# =====================================================================
# レポート生成
# =====================================================================

def generate_report(
    all_results: Dict[str, Dict[str, OptimizationResultSet]],
    consensus: Dict[str, Any],
    cv_results: Dict[str, Any],
    config: Dict[str, Any],
    output_dir: Path,
):
    """最終レポートをJSON + テキストで出力"""

    # consensus.json
    consensus_path = output_dir / "consensus.json"
    # symbolsリストはJSONが大きくなるので省略版にする
    consensus_compact = {}
    for regime, data in consensus.items():
        compact_strategies = []
        for s in data["top_strategies"]:
            s_copy = dict(s)
            s_copy["symbols"] = s_copy["symbols"][:5]  # 先頭5のみ
            if len(s["symbols"]) > 5:
                s_copy["symbols_truncated"] = True
            compact_strategies.append(s_copy)
        consensus_compact[regime] = {
            "total_symbols_periods": data["total_symbols_periods"],
            "top_strategies": compact_strategies,
            "template_frequency": data["template_frequency"],
        }

    with open(consensus_path, "w", encoding="utf-8") as f:
        json.dump(consensus_compact, f, ensure_ascii=False, indent=2)
    logger.info(f"コンセンサス分析保存: {consensus_path}")

    # cross_validation.json
    cv_path = output_dir / "cross_validation.json"
    with open(cv_path, "w", encoding="utf-8") as f:
        json.dump(cv_results, f, ensure_ascii=False, indent=2)
    logger.info(f"クロスバリデーション保存: {cv_path}")

    # config.json
    config_path = output_dir / "config.json"
    config_save = {k: v for k, v in config.items() if k != "inputdata_dir"}
    config_save["inputdata_dir"] = str(config.get("inputdata_dir", ""))
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_save, f, ensure_ascii=False, indent=2)

    # report.txt（テキストレポート）
    report_path = output_dir / "report.txt"
    lines = []
    lines.append("=" * 70)
    lines.append("  バッチ自動最適化レポート")
    lines.append(f"  生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"設定: exec_tf={config['exec_tf']}, htf={config['htf']}, "
                 f"trend={config['trend_method']}, "
                 f"MA {config['ma_fast']}/{config['ma_slow']}")
    lines.append(f"対象期間: {', '.join(config['periods'])}")
    lines.append("")

    # 概要
    for period in config["periods"]:
        if period in all_results:
            n_symbols = len(all_results[period])
            total_combos = sum(
                rs.total_combinations
                for rs in all_results[period].values()
            )
            lines.append(
                f"期間 {period}: {n_symbols} 銘柄, "
                f"{total_combos} 組み合わせ"
            )
    lines.append("")

    # コンセンサス分析
    lines.append("-" * 70)
    lines.append("  コンセンサス戦略（レジーム別）")
    lines.append("-" * 70)

    for regime in TARGET_REGIMES:
        if regime not in consensus:
            continue
        data = consensus[regime]
        lines.append(f"\n  [{regime.upper()}]")
        lines.append(
            f"  テンプレート頻度: "
            f"{', '.join(f'{k}={v:.0%}' for k, v in list(data['template_frequency'].items())[:5])}"
        )

        for i, s in enumerate(data["top_strategies"][:5]):
            param_str = ", ".join(f"{k}={v}" for k, v in s["params"].items())
            lines.append(
                f"  #{i+1} {s['template']} ({param_str})"
            )
            lines.append(
                f"      出現: {s['symbol_count']}銘柄 "
                f"({s['appearance_rate']:.0%}), "
                f"PnL中央値={s['median_pnl']:.1f}%, "
                f"WR={s['median_win_rate']:.0f}%, "
                f"PF={s['median_profit_factor']:.2f}"
            )
    lines.append("")

    # クロスバリデーション
    lines.append("-" * 70)
    lines.append("  クロスバリデーション（2024 → 2025）")
    lines.append("-" * 70)

    for regime in TARGET_REGIMES:
        if regime not in cv_results:
            continue
        lines.append(f"\n  [{regime.upper()}]")
        for cv in cv_results[regime]:
            param_str = ", ".join(
                f"{k}={v}" for k, v in cv["params"].items()
            )
            summary = cv["summary"]
            lines.append(
                f"  {cv['template']} ({param_str})"
            )
            lines.append(
                f"      一貫性: {summary['profitable_in_both']}"
                f"/{summary['total_symbols_with_both']} 銘柄 "
                f"({summary['consistency_rate']:.0%})"
            )
            if summary["avg_pnl_decay"] is not None:
                lines.append(
                    f"      PnL変化率: {summary['avg_pnl_decay']:+.0%}"
                )
    lines.append("")
    lines.append("=" * 70)

    report_text = "\n".join(lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    logger.info(f"レポート保存: {report_path}")

    # コンソールにも出力
    print(report_text)


# =====================================================================
# メイン
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="バッチ自動最適化スクリプト"
    )
    parser.add_argument(
        "--symbols", type=str, default="",
        help="対象銘柄（カンマ区切り）。未指定で全銘柄",
    )
    parser.add_argument(
        "--periods", type=str, default="",
        help="対象期間（カンマ区切り）。未指定で全期間",
    )
    parser.add_argument(
        "--workers", type=int, default=N_WORKERS,
        help=f"並列ワーカー数（デフォルト: {N_WORKERS}）",
    )
    parser.add_argument(
        "--skip-cv", action="store_true",
        help="クロスバリデーションをスキップ",
    )
    args = parser.parse_args()

    # 設定辞書
    config = {
        "exec_tf": EXEC_TF,
        "htf": HTF,
        "trend_method": TREND_METHOD,
        "ma_fast": MA_FAST,
        "ma_slow": MA_SLOW,
        "adx_period": ADX_PERIOD,
        "adx_trend_th": ADX_TREND_TH,
        "adx_range_th": ADX_RANGE_TH,
        "target_regimes": TARGET_REGIMES,
        "initial_capital": INITIAL_CAPITAL,
        "commission_pct": COMMISSION_PCT,
        "slippage_pct": SLIPPAGE_PCT,
        "n_workers": args.workers,
        "top_n_results": TOP_N_RESULTS,
        "inputdata_dir": INPUTDATA_DIR,
        "periods": PERIODS,
    }

    # 期間フィルタ
    periods = PERIODS
    if args.periods:
        periods = [p.strip() for p in args.periods.split(",")]
        config["periods"] = periods

    # 利用可能データのスキャン
    logger.info("データスキャン中...")
    available = scan_available_data(
        INPUTDATA_DIR, EXEC_TF, HTF, periods
    )

    # 銘柄フィルタ
    if args.symbols:
        filter_syms = set(s.strip() for s in args.symbols.split(","))
        for period in available:
            available[period] = [
                s for s in available[period] if s in filter_syms
            ]

    total_jobs = sum(len(syms) for syms in available.values())
    for period, symbols in available.items():
        logger.info(f"  {period}: {len(symbols)} 銘柄")
    logger.info(f"合計: {total_jobs} ジョブ")

    if total_jobs == 0:
        logger.error("対象データが見つかりません")
        sys.exit(1)

    # 出力ディレクトリ
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = RESULTS_DIR / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"出力先: {output_dir}")

    # Phase 1: 全銘柄最適化
    logger.info("")
    logger.info("=" * 50)
    logger.info("  Phase 1: 全銘柄×全期間の最適化")
    logger.info("=" * 50)
    t0 = time.time()

    all_results = run_all_optimizations(available, config, output_dir)

    phase1_time = time.time() - t0
    logger.info(f"Phase 1 完了 ({phase1_time:.0f}s)")

    # Phase 2: コンセンサス分析
    logger.info("")
    logger.info("=" * 50)
    logger.info("  Phase 2: コンセンサス分析")
    logger.info("=" * 50)

    consensus = analyze_consensus(all_results, top_n=CONSENSUS_TOP_N)

    # Phase 3: クロスバリデーション
    cv_results = {}
    if not args.skip_cv and len(periods) >= 2:
        logger.info("")
        logger.info("=" * 50)
        logger.info("  Phase 3: クロスバリデーション")
        logger.info("=" * 50)

        cv_results = cross_validate(
            all_results, config,
            train_period=periods[0],
            test_period=periods[1],
            top_n=CV_TOP_N,
        )

    # レポート生成
    logger.info("")
    logger.info("=" * 50)
    logger.info("  レポート生成")
    logger.info("=" * 50)

    generate_report(all_results, consensus, cv_results, config, output_dir)

    total_time = time.time() - t0
    logger.info(f"\n全処理完了 ({total_time:.0f}s)")
    logger.info(f"結果: {output_dir}")


if __name__ == "__main__":
    main()
