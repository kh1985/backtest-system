"""
ローカル複合テンプレート横断テスト

rsi_bb_reversal_short と bb_volume_reversal_short を
ローカルの全銘柄×全期間で実行し、銘柄横断性を検証する。

使い方:
    python3 scripts/local_composite_test.py
    python3 scripts/local_composite_test.py --period 20240201-20250131
    python3 scripts/local_composite_test.py --symbols BTCUSDT,ETHUSDT
"""

import argparse
import copy
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.binance_loader import BinanceCSVLoader
from analysis.trend import TrendDetector
from optimizer.grid import GridSearchOptimizer
from optimizer.templates import BUILTIN_TEMPLATES
from optimizer.exit_profiles import get_profiles
from optimizer.validation import DataSplitConfig, run_validated_optimization


# --- 設定 ---
MA_FAST = 20
MA_SLOW = 50
INITIAL_CAPITAL = 10000.0
COMMISSION_PCT = 0.04
TOP_N_RESULTS = 20
TARGET_REGIMES = ["uptrend", "downtrend", "range"]
OOS_TRAIN_PCT = 0.6
OOS_VAL_PCT = 0.2
OOS_TOP_N_FOR_VAL = 10
N_WORKERS = 8
MIN_OOS_TRADES = 20

COMPOSITE_TEMPLATES = [
    "rsi_bb_reversal",
    "rsi_bb_reversal_short",
    "bb_volume_reversal",
    "bb_volume_reversal_short",
]


def _safe_float(v):
    if math.isinf(v) or math.isnan(v):
        return 9999.0 if v > 0 else -9999.0 if v < 0 else 0.0
    return v


def _entry_to_dict(e):
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


def run_one(symbol: str, period: str, inputdata_dir: Path, run_id: str) -> dict:
    """1銘柄×1期間の最適化"""
    exec_tf = "15m"
    htf = "1h"
    super_htf = "4h"
    job_id = f"{symbol}_{period}"

    exec_path = inputdata_dir / f"{symbol}-{exec_tf}-{period}-merged.csv"
    htf_path = inputdata_dir / f"{symbol}-{htf}-{period}-merged.csv"
    super_htf_path = inputdata_dir / f"{symbol}-{super_htf}-{period}-merged.csv"

    if not exec_path.exists() or not htf_path.exists():
        return {"job_id": job_id, "status": "skipped", "reason": "データなし"}

    t0 = time.time()

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
        regime_method = "Dual-TF EMA"
    else:
        htf_df = detector.detect_ma_cross(htf_df, fast_period=MA_FAST, slow_period=MA_SLOW)
        regime_method = "MA Cross fallback"

    exec_df = TrendDetector.label_execution_tf(exec_df, htf_df)

    # Config生成（複合テンプレートのみ）
    exit_profiles = get_profiles("atr_compact")
    all_configs = []
    for tname in COMPOSITE_TEMPLATES:
        template = BUILTIN_TEMPLATES[tname]
        configs = template.generate_configs(exit_profiles=exit_profiles)
        all_configs.extend(configs)

    # OOS最適化
    optimizer = GridSearchOptimizer(
        initial_capital=INITIAL_CAPITAL,
        commission_pct=COMMISSION_PCT,
        slippage_pct=0.0,
        top_n_results=TOP_N_RESULTS,
    )

    split_config = DataSplitConfig(
        train_pct=OOS_TRAIN_PCT,
        val_pct=OOS_VAL_PCT,
        top_n_for_val=OOS_TOP_N_FOR_VAL,
    )

    result = run_validated_optimization(
        df=exec_df,
        all_configs=copy.deepcopy(all_configs),
        target_regimes=TARGET_REGIMES,
        split_config=split_config,
        optimizer=optimizer,
        n_workers=N_WORKERS,
    )

    # 結果整理
    test_results = {}
    for regime, entry in result.test_results.items():
        d = _entry_to_dict(entry)
        pnl = d["metrics"]["total_pnl"]
        trades = d["metrics"]["trades"]
        oos_pass = pnl > 0 and trades >= MIN_OOS_TRADES
        d["oos_pass"] = oos_pass
        test_results[regime] = d

    elapsed = time.time() - t0
    return {
        "job_id": job_id,
        "status": "done",
        "symbol": symbol,
        "period": period,
        "regime_method": regime_method,
        "bars": len(exec_df),
        "configs": len(all_configs),
        "elapsed": round(elapsed, 1),
        "test_results": test_results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=str, default="")
    parser.add_argument("--period", type=str, default="20230201-20240131,20240201-20250131,20250201-20260130")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    inputdata_dir = Path(__file__).resolve().parent.parent / "inputdata"
    results_dir = Path(__file__).resolve().parent.parent / "results" / "local"
    results_dir.mkdir(parents=True, exist_ok=True)

    periods = [p.strip() for p in args.period.split(",")]

    # 銘柄リスト
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",")]
    else:
        # inputdataから自動検出
        sym_set = set()
        for p in periods:
            for f in inputdata_dir.glob(f"*-15m-{p}-merged.csv"):
                sym = f.name.split("-15m")[0]
                htf_f = inputdata_dir / f"{sym}-1h-{p}-merged.csv"
                if htf_f.exists():
                    sym_set.add(sym)
        symbols = sorted(sym_set)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print("  複合テンプレート ローカル横断テスト")
    print("=" * 60)
    print(f"銘柄: {len(symbols)} 銘柄")
    print(f"期間: {periods}")
    print(f"テンプレート: {COMPOSITE_TEMPLATES}")
    print(f"並列ワーカー: {args.workers}")
    print(f"Run ID: {run_id}")
    print()

    # ジョブ生成
    jobs = []
    for period in periods:
        for symbol in symbols:
            jobs.append((symbol, period))

    print(f"{len(jobs)} ジョブ実行中...\n")

    # 並列実行
    all_results = []
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_one, sym, per, inputdata_dir, run_id): (sym, per)
            for sym, per in jobs
        }
        done_count = 0
        for future in as_completed(futures):
            done_count += 1
            result = future.result()
            all_results.append(result)
            status = result.get("status", "unknown")
            if status == "done":
                # PASS/FAIL サマリー
                passes = sum(
                    1 for r in result["test_results"].values()
                    if r.get("oos_pass", False)
                )
                total = len(result["test_results"])
                print(f"  [{done_count}/{len(jobs)}] {result['job_id']}: "
                      f"{result['bars']} bars, {result['elapsed']:.0f}s, "
                      f"PASS {passes}/{total}")
            elif status == "skipped":
                print(f"  [{done_count}/{len(jobs)}] {result['job_id']}: SKIP")

    total_time = time.time() - t0

    # --- 集計 ---
    print(f"\n{'=' * 60}")
    print(f"  集計")
    print(f"{'=' * 60}\n")

    # テンプレート別PASS集計
    template_pass = {}
    regime_pass = {"uptrend": 0, "downtrend": 0, "range": 0}
    regime_total = {"uptrend": 0, "downtrend": 0, "range": 0}
    total_pass = 0
    total_entries = 0

    for r in all_results:
        if r["status"] != "done":
            continue
        for regime, entry in r["test_results"].items():
            total_entries += 1
            regime_total[regime] = regime_total.get(regime, 0) + 1
            tpl = entry["template"]
            if tpl not in template_pass:
                template_pass[tpl] = {"pass": 0, "total": 0, "pass_details": []}
            template_pass[tpl]["total"] += 1
            if entry.get("oos_pass", False):
                total_pass += 1
                regime_pass[regime] = regime_pass.get(regime, 0) + 1
                template_pass[tpl]["pass"] += 1
                template_pass[tpl]["pass_details"].append(
                    f"{r['symbol']}/{r['period'][:4]}/{regime} "
                    f"+{entry['metrics']['total_pnl']:.1f}% ({entry['metrics']['trades']}t)"
                )

    print(f"PASS率: {total_pass}/{total_entries} ({100*total_pass/max(total_entries,1):.1f}%)\n")

    print("レジーム別:")
    for reg in TARGET_REGIMES:
        p = regime_pass.get(reg, 0)
        t = regime_total.get(reg, 0)
        print(f"  {reg}: {p}/{t} ({100*p/max(t,1):.0f}%)")

    print(f"\nテンプレート別:")
    for tpl, data in sorted(template_pass.items(), key=lambda x: -x[1]["pass"]):
        print(f"  {tpl}: PASS {data['pass']}/{data['total']}")
        for detail in data["pass_details"][:10]:
            print(f"    {detail}")

    # JSON保存
    output = {
        "run_id": run_id,
        "templates": COMPOSITE_TEMPLATES,
        "periods": periods,
        "symbols": symbols,
        "total_pass": total_pass,
        "total_entries": total_entries,
        "template_pass": {k: {"pass": v["pass"], "total": v["total"]} for k, v in template_pass.items()},
        "results": [r for r in all_results if r["status"] == "done"],
    }
    out_path = results_dir / f"composite_test_{run_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n実行時間: {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"結果: {out_path}")


if __name__ == "__main__":
    main()
