"""
Modal上でWalk-Forward Analysisを実行するスクリプト

3期間のデータを結合して1つの3年分DataFrameにし、
Anchored WFA (5フォールド) で時間的頑健性を検証する。

使い方:
    # デフォルト: 10銘柄 × volume_spike_short × downtrend
    modal run scripts/modal_wfa.py

    # テンプレート・レジーム指定
    modal run scripts/modal_wfa.py --templates ma_crossover_short --regimes uptrend

    # フォールド数変更
    modal run scripts/modal_wfa.py --n-folds 8
"""

import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import modal

vol_data = modal.Volume.from_name("prism-data", create_if_missing=True)
vol_results = modal.Volume.from_name("prism-results", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "pandas>=2.0,<3.0",
        "numpy>=1.24,<2.0",
        "numba>=0.59",
        "pyyaml>=6.0",
    )
    .add_local_dir(".", "/app", copy=True, ignore=[
        "inputdata", "results", "sample_data", "cache",
        ".git", "__pycache__", "*.pyc", ".claude",
    ])
)

app = modal.App("prism-wfa", image=image)


@app.function(
    cpu=8,
    memory=16384,
    timeout=3600,
    volumes={
        "/data": vol_data,
        "/results": vol_results,
    },
)
def wfa_one(
    symbol: str,
    periods: List[str],
    exec_tf: str,
    htf: str,
    super_htf: str,
    template_filter: str,
    target_regimes: List[str],
    exit_profiles_mode: str,
    n_folds: int,
    run_id: str,
) -> Dict[str, Any]:
    """1銘柄のWFAを実行"""
    import sys
    sys.path.insert(0, "/app")

    import copy
    import math
    from pathlib import Path

    import pandas as pd

    from data.binance_loader import BinanceCSVLoader
    from analysis.trend import TrendDetector
    from optimizer.grid import GridSearchOptimizer
    from optimizer.templates import BUILTIN_TEMPLATES
    from optimizer.exit_profiles import get_profiles
    from optimizer.walk_forward import WFAConfig, run_walk_forward_analysis

    MA_FAST = 20
    MA_SLOW = 50
    INITIAL_CAPITAL = 10000.0
    COMMISSION_PCT = 0.04
    TOP_N_RESULTS = 20
    N_WORKERS = 8

    inputdata_dir = Path("/data")
    output_dir = Path(f"/results/{run_id}/wfa")
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    job_id = f"{symbol}_wfa"
    print(f"[START] {job_id}")

    # --- 3期間のデータを結合 ---
    loader = BinanceCSVLoader()
    detector = TrendDetector()

    exec_dfs = []
    for period in periods:
        exec_path = inputdata_dir / f"{symbol}-{exec_tf}-{period}-merged.csv"
        htf_path = inputdata_dir / f"{symbol}-{htf}-{period}-merged.csv"
        super_htf_path = inputdata_dir / f"{symbol}-{super_htf}-{period}-merged.csv"

        if not exec_path.exists() or not htf_path.exists():
            print(f"  {job_id}: {period} データなし、スキップ")
            continue

        exec_ohlcv = loader.load(str(exec_path), symbol=symbol)
        htf_ohlcv = loader.load(str(htf_path), symbol=symbol)
        exec_df = exec_ohlcv.df.copy()
        htf_df = htf_ohlcv.df.copy()

        # Dual-TF EMA レジーム検出
        if super_htf_path.exists():
            super_htf_ohlcv = loader.load(str(super_htf_path), symbol=symbol)
            super_htf_df = super_htf_ohlcv.df.copy()
            htf_df = detector.detect_dual_tf_ema(
                htf_df, super_htf_df, fast_period=MA_FAST, slow_period=MA_SLOW
            )
        else:
            htf_df = detector.detect_ma_cross(
                htf_df, fast_period=MA_FAST, slow_period=MA_SLOW
            )

        exec_df = TrendDetector.label_execution_tf(exec_df, htf_df)
        exec_dfs.append(exec_df)

    if not exec_dfs:
        return {"job_id": job_id, "status": "skipped", "reason": "データなし"}

    # 結合して重複排除・ソート
    combined_df = pd.concat(exec_dfs, ignore_index=False)
    combined_df = combined_df[~combined_df.index.duplicated(keep="first")]
    combined_df = combined_df.sort_index()
    combined_df = combined_df.reset_index(drop=True)

    total_bars = len(combined_df)
    print(f"  {job_id}: {len(exec_dfs)}期間結合 → {total_bars} bars")

    # --- Config生成 ---
    exit_profiles = None
    if exit_profiles_mode != "none":
        exit_profiles = get_profiles(exit_profiles_mode)

    filter_patterns = [p.strip().lower() for p in template_filter.split(",")]

    all_configs = []
    for tname, template in BUILTIN_TEMPLATES.items():
        if not any(p in tname.lower() for p in filter_patterns):
            continue
        configs = template.generate_configs(exit_profiles=exit_profiles)
        all_configs.extend(configs)

    print(f"  {job_id}: {len(all_configs)} configs, {n_folds} folds")

    # --- WFA実行 ---
    optimizer = GridSearchOptimizer(
        initial_capital=INITIAL_CAPITAL,
        commission_pct=COMMISSION_PCT,
        top_n_results=TOP_N_RESULTS,
    )

    wfa_config = WFAConfig(
        n_folds=n_folds,
        min_is_pct=0.4,
        use_validation=True,
        val_pct_within_is=0.2,
        top_n_for_val=10,
        min_trades_for_val=20,
    )

    wfa_result = run_walk_forward_analysis(
        df=combined_df,
        all_configs=all_configs,
        target_regimes=target_regimes,
        wfa_config=wfa_config,
        optimizer=optimizer,
        n_workers=N_WORKERS,
    )

    # --- 結果をJSON化 ---
    def _safe_float(v):
        if v is None:
            return 0.0
        if math.isinf(v) or math.isnan(v):
            return 9999.0 if v > 0 else -9999.0 if v < 0 else 0.0
        return round(v, 4)

    def _entry_to_dict(e):
        if e is None:
            return None
        return {
            "template": e.template_name,
            "params": e.params,
            "regime": e.trend_regime,
            "exit_profile": e.config.get("_exit_profile", "default"),
            "metrics": {
                "trades": e.metrics.total_trades,
                "win_rate": round(_safe_float(e.metrics.win_rate), 1),
                "profit_factor": round(_safe_float(e.metrics.profit_factor), 2),
                "total_pnl": round(_safe_float(e.metrics.total_profit_pct), 2),
                "max_dd": round(_safe_float(e.metrics.max_drawdown_pct), 2),
                "sharpe": round(_safe_float(e.metrics.sharpe_ratio), 2),
            },
        }

    folds_data = []
    for fold in wfa_result.folds:
        fold_data = {
            "fold_index": fold.fold_index,
            "is_range": list(fold.is_range),
            "oos_range": list(fold.oos_range),
            "selected": {
                regime: _entry_to_dict(entry)
                for regime, entry in fold.selected_strategy.items()
            },
            "oos_results": {
                regime: _entry_to_dict(entry)
                for regime, entry in fold.oos_results.items()
            },
        }
        folds_data.append(fold_data)

    data = {
        "symbol": symbol,
        "periods": periods,
        "execution_tf": exec_tf,
        "htf": htf,
        "super_htf": super_htf,
        "template_filter": template_filter,
        "target_regimes": target_regimes,
        "total_bars": total_bars,
        "n_folds": n_folds,
        "wfa": True,
        "folds": folds_data,
        "aggregate": {
            "wfe": {k: round(v, 4) for k, v in wfa_result.wfe.items()},
            "consistency_ratio": {k: round(v, 4) for k, v in wfa_result.consistency_ratio.items()},
            "stitched_oos_pnl": {k: round(v, 2) for k, v in wfa_result.stitched_oos_pnl.items()},
            "strategy_stability": {k: round(v, 4) for k, v in wfa_result.strategy_stability.items()},
        },
    }

    # JSON保存
    fname = f"{symbol}_wfa.json"
    json_path = output_dir / fname
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    vol_results.commit()

    elapsed = time.time() - t0
    print(f"[DONE] {job_id}: {total_bars} bars, {n_folds} folds, {elapsed:.0f}s")

    # サマリー出力
    for regime in target_regimes:
        cr = wfa_result.consistency_ratio.get(regime, 0)
        wfe = wfa_result.wfe.get(regime, 0)
        pnl = wfa_result.stitched_oos_pnl.get(regime, 0)
        stability = wfa_result.strategy_stability.get(regime, 0)
        print(f"  {regime}: CR={cr:.0%} WFE={wfe:.2f} OOS_PnL={pnl:+.1f}% Stability={stability:.0%}")

    return {
        "job_id": job_id,
        "status": "done",
        "total_bars": total_bars,
        "elapsed": round(elapsed, 1),
        "aggregate": data["aggregate"],
    }


@app.local_entrypoint()
def main(
    symbols: str = "BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,ADAUSDT,AVAXUSDT,DOGEUSDT,DOTUSDT,TRXUSDT",
    periods: str = "20230201-20240131,20240201-20250131,20250201-20260130",
    tf_combo: str = "15m:1h",
    super_htf: str = "4h",
    templates: str = "volume_spike_short",
    regimes: str = "downtrend",
    exit_profiles: str = "atr_compact",
    n_folds: int = 5,
):
    symbol_list = [s.strip() for s in symbols.split(",")]
    period_list = [p.strip() for p in periods.split(",")]
    regime_list = [r.strip() for r in regimes.split(",")]
    parts = tf_combo.strip().split(":")
    exec_tf, htf = parts[0], parts[1]

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_wfa"

    print("=" * 60)
    print("  Prism Walk-Forward Analysis (Modal)")
    print("=" * 60)
    print(f"銘柄: {symbol_list}")
    print(f"期間: {period_list} (結合して使用)")
    print(f"TF: {exec_tf}/{htf}, Super HTF: {super_htf}")
    print(f"テンプレート: {templates}")
    print(f"レジーム: {regime_list}")
    print(f"Exit: {exit_profiles}")
    print(f"フォールド数: {n_folds}")
    print(f"Run ID: {run_id}")
    print()

    # 全銘柄を並列投入
    handles = []
    for symbol in symbol_list:
        h = wfa_one.spawn(
            symbol=symbol,
            periods=period_list,
            exec_tf=exec_tf,
            htf=htf,
            super_htf=super_htf,
            template_filter=templates,
            target_regimes=regime_list,
            exit_profiles_mode=exit_profiles,
            n_folds=n_folds,
            run_id=run_id,
        )
        handles.append((symbol, h))

    # 結果回収
    t0 = time.time()
    results = []
    for symbol, h in handles:
        result = h.get()
        results.append(result)
        status = result.get("status", "unknown")
        if status == "done":
            print(f"  {symbol}: {result['elapsed']:.0f}s")
            agg = result.get("aggregate", {})
            for regime in regime_list:
                cr = agg.get("consistency_ratio", {}).get(regime, 0)
                wfe = agg.get("wfe", {}).get(regime, 0)
                pnl = agg.get("stitched_oos_pnl", {}).get(regime, 0)
                stability = agg.get("strategy_stability", {}).get(regime, 0)
                label = "PASS" if cr >= 0.6 else "FAIL"
                print(f"    {regime}: CR={cr:.0%} WFE={wfe:.2f} PnL={pnl:+.1f}% Stab={stability:.0%} → {label}")
        else:
            print(f"  {symbol}: {status}")

    total_time = time.time() - t0

    # サマリー
    print()
    print("=" * 60)
    print("  WFA サマリー")
    print("=" * 60)

    for regime in regime_list:
        print(f"\n  [{regime}]")
        print(f"  {'銘柄':<12} {'CR':>6} {'WFE':>8} {'OOS PnL':>10} {'Stability':>10} {'判定':>6}")
        print(f"  {'-'*54}")

        pass_count = 0
        for result in results:
            if result.get("status") != "done":
                continue
            symbol = result["job_id"].replace("_wfa", "")
            agg = result.get("aggregate", {})
            cr = agg.get("consistency_ratio", {}).get(regime, 0)
            wfe = agg.get("wfe", {}).get(regime, 0)
            pnl = agg.get("stitched_oos_pnl", {}).get(regime, 0)
            stability = agg.get("strategy_stability", {}).get(regime, 0)
            label = "PASS" if cr >= 0.6 else "FAIL"
            if cr >= 0.6:
                pass_count += 1
            print(f"  {symbol:<12} {cr:>5.0%} {wfe:>8.2f} {pnl:>+9.1f}% {stability:>9.0%} {label:>6}")

        total_done = sum(1 for r in results if r.get("status") == "done")
        print(f"\n  PASS: {pass_count}/{total_done}")

    print(f"\n  実行時間: {total_time:.0f}s ({total_time / 60:.1f}min)")
    print(f"  Run ID: {run_id}")
    print(f"  結果DL: modal run scripts/modal_download.py --run-id {run_id}")
    print("=" * 60)
