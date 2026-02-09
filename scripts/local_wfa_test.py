"""
WFA（Walk-Forward Analysis）ローカル検証スクリプト

確定3戦略（rsi_bb_long_f35, rsi_bb_short_f67, rsi_bb_short_f65）の
堅牢性をWFAで検証する。

WFA方式: Anchored（5フォールド）
対象銘柄: 3期間連続PASS銘柄を優先（AAVE, BNB, SOL, SUI）
対象レジーム: 戦略ごとに方向一致のもの

使い方:
    python3 scripts/local_wfa_test.py
    python3 scripts/local_wfa_test.py --symbols AAVEUSDT,BNBUSDT
    python3 scripts/local_wfa_test.py --period 20240201-20250131
"""

import argparse
import copy
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.binance_loader import BinanceCSVLoader
from analysis.trend import TrendDetector
from optimizer.grid import GridSearchOptimizer
from optimizer.templates import BUILTIN_TEMPLATES
from optimizer.exit_profiles import get_profiles
from optimizer.walk_forward import WFAConfig, run_walk_forward_analysis

# --- 設定 ---
MA_FAST = 20
MA_SLOW = 50
INITIAL_CAPITAL = 10000.0
COMMISSION_PCT = 0.04
N_WORKERS = 1  # WFA内はシーケンシャル（フォールド依存あるため）

# 戦略ごとの対象レジーム（方向一致のみ）
STRATEGY_REGIMES = {
    "rsi_bb_long_f35": ["uptrend"],
    "rsi_bb_short_f67": ["range"],
    "rsi_bb_short_f65": ["range"],
}

# 3期間連続PASS銘柄
DEFAULT_SYMBOLS = ["AAVEUSDT", "BNBUSDT", "SOLUSDT", "SUIUSDT"]
DEFAULT_PERIODS = ["20230201-20240131", "20240201-20250131", "20250201-20260130"]

# WFE / CR 判定閾値
WFE_THRESHOLD = 0.5
CR_THRESHOLD = 0.6


def _safe_float(v):
    if math.isinf(v) or math.isnan(v):
        return 9999.0 if v > 0 else -9999.0 if v < 0 else 0.0
    return v


def load_data(symbol: str, period: str, inputdata_dir: Path):
    """1銘柄×1期間のデータを読み込み、レジーム付与済みDFを返す"""
    exec_tf = "15m"
    htf = "1h"
    super_htf = "4h"

    exec_path = inputdata_dir / f"{symbol}-{exec_tf}-{period}-merged.csv"
    htf_path = inputdata_dir / f"{symbol}-{htf}-{period}-merged.csv"
    super_htf_path = inputdata_dir / f"{symbol}-{super_htf}-{period}-merged.csv"

    if not exec_path.exists() or not htf_path.exists():
        return None

    loader = BinanceCSVLoader()
    exec_ohlcv = loader.load(str(exec_path), symbol=symbol)
    htf_ohlcv = loader.load(str(htf_path), symbol=symbol)

    exec_df = exec_ohlcv.df.copy()
    htf_df = htf_ohlcv.df.copy()
    detector = TrendDetector()

    if super_htf_path.exists():
        super_htf_ohlcv = loader.load(str(super_htf_path), symbol=symbol)
        super_htf_df = super_htf_ohlcv.df.copy()
        htf_df = detector.detect_dual_tf_ema(
            htf_df, super_htf_df, fast_period=MA_FAST, slow_period=MA_SLOW
        )
    else:
        htf_df = detector.detect_ma_cross(htf_df, fast_period=MA_FAST, slow_period=MA_SLOW)

    exec_df = TrendDetector.label_execution_tf(exec_df, htf_df)
    return exec_df


def run_wfa_for_strategy(
    template_name: str,
    target_regimes: list,
    exec_df,
    symbol: str,
    period: str,
    exit_filter: str = "",
) -> dict:
    """1戦略×1銘柄×1期間のWFA実行"""
    template = BUILTIN_TEMPLATES[template_name]
    exit_profiles = get_profiles("atr_compact")
    if exit_filter:
        exit_profiles = [p for p in exit_profiles if p["name"] == exit_filter]
    all_configs = template.generate_configs(exit_profiles=exit_profiles)

    optimizer = GridSearchOptimizer(
        initial_capital=INITIAL_CAPITAL,
        commission_pct=COMMISSION_PCT,
        slippage_pct=0.0,
        top_n_results=20,
    )

    wfa_config = WFAConfig(
        n_folds=5,
        min_is_pct=0.4,
        use_validation=True,
        val_pct_within_is=0.2,
        top_n_for_val=10,
        min_trades_for_val=10,
    )

    t0 = time.time()

    try:
        wfa_result = run_walk_forward_analysis(
            df=exec_df,
            all_configs=copy.deepcopy(all_configs),
            target_regimes=target_regimes,
            wfa_config=wfa_config,
            optimizer=optimizer,
            trend_column="trend_regime",
            n_workers=N_WORKERS,
        )
    except Exception as e:
        return {
            "template": template_name,
            "symbol": symbol,
            "period": period,
            "status": "error",
            "error": str(e),
        }

    elapsed = time.time() - t0

    # 結果整理
    result = {
        "template": template_name,
        "symbol": symbol,
        "period": period,
        "status": "done",
        "bars": len(exec_df),
        "n_folds": len(wfa_result.folds),
        "elapsed": round(elapsed, 1),
        "regimes": {},
    }

    for regime in target_regimes:
        wfe = wfa_result.wfe.get(regime)
        cr = wfa_result.consistency_ratio.get(regime)
        stitched_pnl = wfa_result.stitched_oos_pnl.get(regime)
        stability = wfa_result.strategy_stability.get(regime)

        # フォールド詳細
        fold_details = []
        for fold in wfa_result.folds:
            is_entry = fold.selected_strategy.get(regime)
            oos_entry = fold.oos_results.get(regime)
            fold_info = {
                "fold": fold.fold_index,
                "is_range": list(fold.is_range),
                "oos_range": list(fold.oos_range),
            }
            if is_entry:
                fold_info["is_template"] = is_entry.template_name
                fold_info["is_params"] = is_entry.params
                fold_info["is_pnl"] = round(_safe_float(is_entry.metrics.total_profit_pct), 2)
                fold_info["is_trades"] = is_entry.metrics.total_trades
            if oos_entry:
                fold_info["oos_pnl"] = round(_safe_float(oos_entry.metrics.total_profit_pct), 2)
                fold_info["oos_trades"] = oos_entry.metrics.total_trades
                fold_info["oos_win_rate"] = round(_safe_float(oos_entry.metrics.win_rate), 1)
            fold_details.append(fold_info)

        # 判定
        wfe_pass = wfe is not None and wfe > WFE_THRESHOLD
        cr_pass = cr is not None and cr >= CR_THRESHOLD
        robust = wfe_pass and cr_pass

        regime_result = {
            "wfe": round(_safe_float(wfe), 4) if wfe is not None else None,
            "consistency_ratio": round(_safe_float(cr), 4) if cr is not None else None,
            "stitched_oos_pnl": round(_safe_float(stitched_pnl), 2) if stitched_pnl is not None else None,
            "strategy_stability": round(_safe_float(stability), 4) if stability is not None else None,
            "wfe_pass": wfe_pass,
            "cr_pass": cr_pass,
            "robust": robust,
            "folds": fold_details,
        }
        result["regimes"][regime] = regime_result

    return result


def main():
    parser = argparse.ArgumentParser(description="WFA検証スクリプト")
    parser.add_argument("--symbols", type=str, default="")
    parser.add_argument("--period", type=str, default=",".join(DEFAULT_PERIODS))
    parser.add_argument("--templates", type=str, default="")
    parser.add_argument("--exit-filter", type=str, default="", help="Exit profile名でフィルタ (例: atr_tp20_sl20)")
    parser.add_argument("--regimes", type=str, default="", help="レジーム指定 (例: downtrend,uptrend)")
    args = parser.parse_args()

    inputdata_dir = Path(__file__).resolve().parent.parent / "inputdata"
    results_dir = Path(__file__).resolve().parent.parent / "results" / "wfa"
    results_dir.mkdir(parents=True, exist_ok=True)

    symbols = [s.strip() for s in args.symbols.split(",")] if args.symbols else DEFAULT_SYMBOLS
    periods = [p.strip() for p in args.period.split(",")]

    if args.templates:
        template_names = [t.strip() for t in args.templates.split(",")]
        if args.regimes:
            override_regimes = [r.strip() for r in args.regimes.split(",")]
            strategy_regimes = {t: override_regimes for t in template_names}
        else:
            strategy_regimes = {t: STRATEGY_REGIMES.get(t, ["uptrend", "range"]) for t in template_names}
    else:
        strategy_regimes = STRATEGY_REGIMES

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 70)
    print("  WFA（Walk-Forward Analysis）検証")
    print("=" * 70)
    print(f"銘柄: {symbols}")
    print(f"期間: {periods}")
    print(f"戦略: {list(strategy_regimes.keys())}")
    print(f"WFA: 5フォールド Anchored / IS内 Train/Val分割")
    print(f"判定基準: WFE > {WFE_THRESHOLD}, CR > {CR_THRESHOLD}")
    print(f"Run ID: {run_id}")
    print()

    all_results = []
    total_jobs = len(strategy_regimes) * len(symbols) * len(periods)
    done_count = 0

    for template_name, target_regimes in strategy_regimes.items():
        print(f"\n--- {template_name} (regimes: {target_regimes}) ---")

        for period in periods:
            for symbol in symbols:
                done_count += 1
                job_id = f"{template_name}/{symbol}/{period}"

                exec_df = load_data(symbol, period, inputdata_dir)
                if exec_df is None:
                    print(f"  [{done_count}/{total_jobs}] {job_id}: SKIP (データなし)")
                    continue

                print(f"  [{done_count}/{total_jobs}] {job_id}: {len(exec_df)} bars ... ", end="", flush=True)

                result = run_wfa_for_strategy(
                    template_name=template_name,
                    target_regimes=target_regimes,
                    exec_df=exec_df,
                    symbol=symbol,
                    period=period,
                    exit_filter=args.exit_filter,
                )
                all_results.append(result)

                if result["status"] == "done":
                    for regime, rdata in result["regimes"].items():
                        tag = "ROBUST" if rdata["robust"] else "WEAK"
                        wfe_str = f"WFE={rdata['wfe']}" if rdata["wfe"] is not None else "WFE=N/A"
                        cr_str = f"CR={rdata['consistency_ratio']}" if rdata["consistency_ratio"] is not None else "CR=N/A"
                        pnl_str = f"PnL={rdata['stitched_oos_pnl']}%" if rdata["stitched_oos_pnl"] is not None else "PnL=N/A"
                        print(f"{tag} [{regime}] {wfe_str}, {cr_str}, {pnl_str} ({result['elapsed']:.0f}s)")
                elif result["status"] == "error":
                    print(f"ERROR: {result['error']}")

    # --- 集計 ---
    print(f"\n{'=' * 70}")
    print("  WFA集計サマリー")
    print(f"{'=' * 70}\n")

    # 戦略×レジーム別集計
    summary = {}
    for r in all_results:
        if r["status"] != "done":
            continue
        tpl = r["template"]
        for regime, rdata in r["regimes"].items():
            key = f"{tpl}/{regime}"
            if key not in summary:
                summary[key] = {"robust": 0, "weak": 0, "wfe_list": [], "cr_list": [], "pnl_list": []}
            if rdata["robust"]:
                summary[key]["robust"] += 1
            else:
                summary[key]["weak"] += 1
            if rdata["wfe"] is not None:
                summary[key]["wfe_list"].append(rdata["wfe"])
            if rdata["consistency_ratio"] is not None:
                summary[key]["cr_list"].append(rdata["consistency_ratio"])
            if rdata["stitched_oos_pnl"] is not None:
                summary[key]["pnl_list"].append(rdata["stitched_oos_pnl"])

    for key, data in sorted(summary.items()):
        total = data["robust"] + data["weak"]
        robust_pct = 100 * data["robust"] / max(total, 1)
        avg_wfe = sum(data["wfe_list"]) / max(len(data["wfe_list"]), 1)
        avg_cr = sum(data["cr_list"]) / max(len(data["cr_list"]), 1)
        avg_pnl = sum(data["pnl_list"]) / max(len(data["pnl_list"]), 1)

        print(f"  {key}:")
        print(f"    ROBUST: {data['robust']}/{total} ({robust_pct:.0f}%)")
        print(f"    平均WFE: {avg_wfe:.3f}  平均CR: {avg_cr:.3f}  平均PnL: {avg_pnl:.1f}%")

    # JSON保存
    output = {
        "run_id": run_id,
        "strategies": list(strategy_regimes.keys()),
        "symbols": symbols,
        "periods": periods,
        "wfa_config": {
            "n_folds": 5,
            "min_is_pct": 0.4,
            "use_validation": True,
            "val_pct_within_is": 0.2,
        },
        "thresholds": {
            "wfe": WFE_THRESHOLD,
            "cr": CR_THRESHOLD,
        },
        "summary": {
            k: {
                "robust": v["robust"],
                "total": v["robust"] + v["weak"],
                "avg_wfe": round(sum(v["wfe_list"]) / max(len(v["wfe_list"]), 1), 4),
                "avg_cr": round(sum(v["cr_list"]) / max(len(v["cr_list"]), 1), 4),
                "avg_pnl": round(sum(v["pnl_list"]) / max(len(v["pnl_list"]), 1), 2),
            }
            for k, v in summary.items()
        },
        "results": all_results,
    }
    out_path = results_dir / f"wfa_{run_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n結果保存: {out_path}")


if __name__ == "__main__":
    main()
