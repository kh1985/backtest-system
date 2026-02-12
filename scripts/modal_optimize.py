"""
Modal上でバッチ最適化を並列実行するスクリプト

使い方:
    # 5銘柄 × 15m:1h（デフォルト）
    modal run scripts/modal_optimize.py

    # 特定銘柄・TF
    modal run scripts/modal_optimize.py --symbols BTCUSDT,ETHUSDT --tf-combos 15m:1h,1h:4h

    # OOSなし（高速テスト）
    modal run scripts/modal_optimize.py --no-oos

    # exit profiles指定
    modal run scripts/modal_optimize.py --exit-profiles fixed

# 2026-02-11: テンプレートフィルターロジック修正（部分一致→完全一致）
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import modal

# --- Modal リソース定義 ---
vol_data = modal.Volume.from_name("prism-data", create_if_missing=True)
vol_results = modal.Volume.from_name("prism-results", create_if_missing=True)

# コンテナイメージ（依存パッケージ + ソースコード）
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
        "CLAUDE.md", "*.bak",
    ])
)

app = modal.App("prism-optimize", image=image)


# --- 1銘柄の最適化（これが銘柄数ぶん並列で走る） ---
@app.function(
    cpu=8,
    memory=16384,
    timeout=3600,
    volumes={
        "/data": vol_data,
        "/results": vol_results,
    },
)
def optimize_one(
    symbol: str,
    period: str,
    exec_tf: str,
    htf: str,
    use_oos: bool,
    exit_profiles_mode: str,
    template_filter: Optional[str],
    run_id: str,
    super_htf: str = "",
) -> Dict[str, Any]:
    """1銘柄×1TF×1期間の最適化を実行"""
    import sys
    sys.path.insert(0, "/app")

    import copy
    import math
    from pathlib import Path

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
    from optimizer.exit_profiles import get_profiles

    # --- 設定 ---
    MA_FAST = 20
    MA_SLOW = 50
    INITIAL_CAPITAL = 10000.0
    COMMISSION_PCT = 0.04
    SLIPPAGE_PCT = 0.0
    TOP_N_RESULTS = 20
    TARGET_REGIMES = ["uptrend", "downtrend", "range"]
    OOS_TRAIN_PCT = 0.6
    OOS_VAL_PCT = 0.2
    OOS_TOP_N_FOR_VAL = 10
    N_WORKERS = 8  # コンテナ内の並列ワーカー数

    inputdata_dir = Path("/data")
    output_dir = Path(f"/results/{run_id}/optimization")
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    job_id = f"{symbol}_{period}_{exec_tf}_{htf}"
    print(f"[START] {job_id}")

    # --- データ読み込み ---
    loader = BinanceCSVLoader()
    exec_path = inputdata_dir / f"{symbol}-{exec_tf}-{period}-merged.csv"
    htf_path = inputdata_dir / f"{symbol}-{htf}-{period}-merged.csv"

    if not exec_path.exists() or not htf_path.exists():
        msg = f"データなし: {exec_path.name} or {htf_path.name}"
        print(f"[SKIP] {job_id}: {msg}")
        return {"job_id": job_id, "status": "skipped", "reason": msg}

    tf_dict = {}
    tf_dict[exec_tf] = loader.load(str(exec_path), symbol=symbol)
    tf_dict[htf] = loader.load(str(htf_path), symbol=symbol)

    # --- トレンドラベル付与 ---
    exec_ohlcv = tf_dict[exec_tf]
    exec_df = exec_ohlcv.df.copy()
    htf_ohlcv = tf_dict[htf]
    htf_df = htf_ohlcv.df.copy()
    detector = TrendDetector()

    if super_htf:
        super_htf_path = inputdata_dir / f"{symbol}-{super_htf}-{period}-merged.csv"
        if super_htf_path.exists():
            super_htf_ohlcv = loader.load(str(super_htf_path), symbol=symbol)
            super_htf_df = super_htf_ohlcv.df.copy()
            htf_df = detector.detect_dual_tf_ema(
                htf_df, super_htf_df, fast_period=MA_FAST, slow_period=MA_SLOW
            )
            print(f"  {job_id}: Dual-TF EMA ({htf}+{super_htf})")
        else:
            htf_df = detector.detect_ma_cross(htf_df, fast_period=MA_FAST, slow_period=MA_SLOW)
            print(f"  {job_id}: {super_htf}データなし → MA Cross fallback")
    else:
        htf_df = detector.detect_ma_cross(htf_df, fast_period=MA_FAST, slow_period=MA_SLOW)

    exec_df = TrendDetector.label_execution_tf(exec_df, htf_df)

    bars = len(exec_df)
    print(f"  {job_id}: {bars} bars")

    # --- Config生成 ---
    exit_profiles = None
    if exit_profiles_mode not in ("none", "fixed"):
        exit_profiles = get_profiles(exit_profiles_mode)
    # "fixed" の場合は exit_profiles=None → テンプレート定義のexitを使用

    filter_patterns = None
    if template_filter:
        filter_patterns = [p.strip().lower() for p in template_filter.split(",")]

    all_configs = []
    for tname, template in BUILTIN_TEMPLATES.items():
        if filter_patterns:
            # 完全一致判定（修正版: 2026-02-11）
            if tname.lower() not in filter_patterns:
                continue
        configs = template.generate_configs(exit_profiles=exit_profiles)
        all_configs.extend(configs)

    print(f"  {job_id}: {len(all_configs)} configs")

    # --- 最適化実行 ---
    optimizer = GridSearchOptimizer(
        initial_capital=INITIAL_CAPITAL,
        commission_pct=COMMISSION_PCT,
        slippage_pct=SLIPPAGE_PCT,
        top_n_results=TOP_N_RESULTS,
    )

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

    if use_oos:
        split_config = DataSplitConfig(
            train_pct=OOS_TRAIN_PCT,
            val_pct=OOS_VAL_PCT,
            top_n_for_val=OOS_TOP_N_FOR_VAL,
        )
        configs = copy.deepcopy(all_configs)
        result = run_validated_optimization(
            df=exec_df,
            all_configs=configs,
            target_regimes=TARGET_REGIMES,
            split_config=split_config,
            optimizer=optimizer,
            n_workers=N_WORKERS,
        )
        # JSON保存
        data = {
            "symbol": symbol,
            "period": period,
            "execution_tf": exec_tf,
            "htf": htf,
            "oos": True,
            "train_results": [
                _entry_to_dict(e)
                for e in result.train_results.ranked()[:TOP_N_RESULTS]
            ],
            "test_results": {
                regime: _entry_to_dict(entry)
                for regime, entry in result.test_results.items()
            },
            "val_best": {
                regime: _entry_to_dict(entry)
                for regime, entry in result.val_best.items()
            },
            "warnings": result.overfitting_warnings,
            "total_combinations": result.train_results.total_combinations,
        }
    else:
        configs = copy.deepcopy(all_configs)
        result = optimizer.run(
            df=exec_df,
            configs=configs,
            target_regimes=TARGET_REGIMES,
            n_workers=N_WORKERS,
        )
        data = {
            "symbol": symbol,
            "period": period,
            "execution_tf": exec_tf,
            "htf": htf,
            "oos": False,
            "total_combinations": result.total_combinations,
            "results": [
                _entry_to_dict(e)
                for e in result.ranked()[:TOP_N_RESULTS]
            ],
        }

    # JSON保存
    fname = f"{symbol}_{period}_{exec_tf}_{htf}.json"
    json_path = output_dir / fname
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    vol_results.commit()

    elapsed = time.time() - t0
    combos = data.get("total_combinations", 0)
    print(f"[DONE] {job_id}: {combos} combos, {elapsed:.0f}s")

    return {
        "job_id": job_id,
        "status": "done",
        "combinations": combos,
        "elapsed": round(elapsed, 1),
        "json_file": fname,
    }


# --- ランキング＆レポート生成 ---
@app.function(
    cpu=2,
    memory=4096,
    timeout=600,
    volumes={"/results": vol_results},
)
def generate_ranking(run_id: str, use_oos: bool, config_info: Dict[str, Any], min_oos_trades: int = 20) -> str:
    """全結果JSONを読み込んでランキングとレポートを生成"""
    import json
    from collections import defaultdict
    from statistics import median
    from pathlib import Path

    opt_dir = Path(f"/results/{run_id}/optimization")
    output_dir = Path(f"/results/{run_id}")

    # --- 全JSONファイル読み込み ---
    json_files = sorted(opt_dir.glob("*.json"))
    if not json_files:
        return "結果JSONファイルがありません"

    print(f"結果ファイル: {len(json_files)} 件")

    # --- OOS結果を抽出 ---
    TARGET_REGIMES = ["uptrend", "downtrend", "range"]
    rows = []
    all_symbols = set()

    for jf in json_files:
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)

        symbol = data["symbol"]
        period = data["period"]
        exec_tf = data["execution_tf"]
        htf = data["htf"]
        is_oos = data.get("oos", False)
        all_symbols.add(symbol)

        if is_oos and "test_results" in data:
            for regime, entry in data["test_results"].items():
                rows.append({
                    "period": period,
                    "exec_tf": exec_tf,
                    "htf": htf,
                    "symbol": symbol,
                    "regime": regime,
                    "template": entry["template"],
                    "exit_profile": entry.get("exit_profile", "default"),
                    "oos_pnl": entry["metrics"]["total_pnl"],
                    "oos_trades": entry["metrics"]["trades"],
                    "oos_win_rate": entry["metrics"]["win_rate"],
                    "oos_sharpe": entry["metrics"]["sharpe"],
                    "oos_pf": entry["metrics"]["profit_factor"],
                    "oos_pass": (
                        entry["metrics"]["total_pnl"] > 0
                        and entry["metrics"]["trades"] >= min_oos_trades
                    ),
                    "data_source": "oos_test",
                })
            # OOS結果がないレジームのTrain結果
            for regime in TARGET_REGIMES:
                if regime in data["test_results"]:
                    continue
                train_results = data.get("train_results", [])
                regime_entries = [e for e in train_results if e["regime"] == regime]
                if regime_entries:
                    best = max(regime_entries, key=lambda e: e["score"])
                    rows.append({
                        "period": period,
                        "exec_tf": exec_tf,
                        "htf": htf,
                        "symbol": symbol,
                        "regime": regime,
                        "template": best["template"],
                        "exit_profile": best.get("exit_profile", "default"),
                        "oos_pnl": best["metrics"]["total_pnl"],
                        "oos_trades": best["metrics"]["trades"],
                        "oos_win_rate": best["metrics"]["win_rate"],
                        "oos_sharpe": best["metrics"]["sharpe"],
                        "oos_pf": best["metrics"]["profit_factor"],
                        "oos_pass": False,
                        "data_source": "train_only",
                    })
        elif not is_oos and "results" in data:
            for entry in data["results"][:3]:  # レジーム別ベスト
                rows.append({
                    "period": period,
                    "exec_tf": exec_tf,
                    "htf": htf,
                    "symbol": symbol,
                    "regime": entry["regime"],
                    "template": entry["template"],
                    "exit_profile": entry.get("exit_profile", "default"),
                    "oos_pnl": entry["metrics"]["total_pnl"],
                    "oos_trades": entry["metrics"]["trades"],
                    "oos_win_rate": entry["metrics"]["win_rate"],
                    "oos_sharpe": entry["metrics"]["sharpe"],
                    "oos_pf": entry["metrics"]["profit_factor"],
                    "oos_pass": False,
                    "data_source": "train_only",
                })

    if not rows:
        return "ランキング対象の結果がありません"

    # --- グルーピング＆スコア計算 ---
    PERIODS = list(set(r["period"] for r in rows))
    groups = defaultdict(list)
    for r in rows:
        key = (r["regime"], r["template"], r["exit_profile"], r["exec_tf"], r["htf"])
        groups[key].append(r)

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
        periods_seen = set(e["period"] for e in entries)
        period_consistency = len(periods_seen) / max(len(PERIODS), 1)
        symbol_coverage = n_symbols / max(len(all_symbols), 1)

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
            "entries": n_entries,
        })

    strategies.sort(key=lambda x: -x["score"])

    # --- ランキングJSON保存 ---
    ranking = {"strategies": strategies}
    ranking_path = output_dir / "ranking.json"
    with open(ranking_path, "w", encoding="utf-8") as f:
        json.dump(ranking, f, ensure_ascii=False, indent=2)

    # --- レポートMarkdown生成 ---
    lines = []
    lines.append("# バッチ自動最適化レポート (Modal)")
    lines.append("")
    lines.append(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"結果ファイル数: {len(json_files)}")
    lines.append(f"銘柄数: {len(all_symbols)}")
    lines.append(f"OOS: {'ON' if use_oos else 'OFF'}")
    lines.append("")

    # 推奨戦略
    RANKING_MIN_SYMBOLS = 3
    RANKING_MIN_OOS_PASS_RATE = 0.5
    recommended = [
        s for s in strategies
        if s["oos_pass_rate"] >= RANKING_MIN_OOS_PASS_RATE
        and s["symbol_count"] >= RANKING_MIN_SYMBOLS
    ]

    lines.append("## 推奨戦略ランキング")
    lines.append("")
    if recommended:
        lines.append("| Rank | TF | Regime | Template | Exit | OOS通過率 | 平均PnL | 銘柄数 | Score |")
        lines.append("|------|------|--------|----------|------|---------|---------|--------|-------|")
        for i, s in enumerate(recommended[:20]):
            lines.append(
                f"| {i+1} | {s['exec_tf']}/{s['htf']} | {s['regime']} | "
                f"{s['template']} | {s['exit_profile']} | "
                f"{s['oos_pass_rate']:.0%} | {s['avg_pnl']:+.1f}% | "
                f"{s['symbol_count']} | {s['score']:.3f} |"
            )
    else:
        lines.append("(推奨戦略なし)")
    lines.append("")

    # 全戦略Top10
    lines.append("## 全戦略 Top 10")
    lines.append("")
    lines.append("| Rank | TF | Regime | Template | Exit | OOS通過率 | 平均PnL | 銘柄数 | Score |")
    lines.append("|------|------|--------|----------|------|---------|---------|--------|-------|")
    for i, s in enumerate(strategies[:10]):
        lines.append(
            f"| {i+1} | {s['exec_tf']}/{s['htf']} | {s['regime']} | "
            f"{s['template']} | {s['exit_profile']} | "
            f"{s['oos_pass_rate']:.0%} | {s['avg_pnl']:+.1f}% | "
            f"{s['symbol_count']} | {s['score']:.3f} |"
        )
    lines.append("")

    report_text = "\n".join(lines)
    report_path = output_dir / "report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    vol_results.commit()

    summary = f"戦略グループ: {len(strategies)}, 推奨: {len(recommended)}"
    print(summary)
    return summary


# --- エントリーポイント（ローカルから呼び出す） ---
@app.local_entrypoint()
def main(
    symbols: str = "",
    period: str = "20250201-20260130",
    tf_combos: str = "15m:1h",
    exit_profiles: str = "all",
    templates: str = "",
    no_oos: bool = False,
    workers: int = 8,
    super_htf: str = "4h",
    min_oos_trades: int = 20,
):
    """ローカルから実行するエントリーポイント"""
    use_oos = not no_oos
    template_filter = templates if templates else None

    # TF組み合わせパース
    tf_list = []
    for tc in tf_combos.split(","):
        parts = tc.strip().split(":")
        if len(parts) == 2:
            tf_list.append((parts[0], parts[1]))
    if not tf_list:
        print("ERROR: 有効なTF組み合わせがありません")
        return

    # 期間リスト
    period_list = [p.strip() for p in period.split(",")]

    # 銘柄リスト
    if symbols:
        symbol_list = [s.strip() for s in symbols.split(",")]
    else:
        # Volume内のデータファイルをスキャンして銘柄を自動検出
        # 最初の期間でスキャン（全期間で共通の銘柄セットを想定）
        symbol_list = scan_volume_symbols.remote(period_list[0], tf_list)
        if not symbol_list:
            print("ERROR: Volume内にデータが見つかりません")
            return
        print(f"自動検出された銘柄: {symbol_list}")

    # 実行ID
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print("  Prism バッチ最適化 (Modal)")
    print("=" * 60)
    print(f"銘柄: {symbol_list}")
    print(f"期間: {period_list}")
    print(f"TF: {tf_list}")
    print(f"OOS: {'ON' if use_oos else 'OFF'}")
    print(f"Super HTF: {super_htf if super_htf else 'なし（MA Cross fallback）'}")
    print(f"Exit profiles: {exit_profiles}")
    print(f"Min OOS trades: {min_oos_trades}")
    print(f"Run ID: {run_id}")

    # --- 全ジョブを一括投入 ---
    jobs = []
    for p in period_list:
        for exec_tf, htf in tf_list:
            for symbol in symbol_list:
                jobs.append(
                    optimize_one.spawn(
                        symbol=symbol,
                        period=p,
                        exec_tf=exec_tf,
                        htf=htf,
                        use_oos=use_oos,
                        exit_profiles_mode=exit_profiles,
                        template_filter=template_filter,
                        run_id=run_id,
                        super_htf=super_htf,
                    )
                )

    print(f"\n{len(jobs)} ジョブ投入完了。並列実行中...\n")

    # --- 全ジョブの完了を待つ ---
    t0 = time.time()
    done = 0
    errors = 0
    for job in jobs:
        result = job.get()
        done += 1
        status = result.get("status", "unknown")
        if status == "done":
            elapsed = result.get("elapsed", 0)
            combos = result.get("combinations", 0)
            print(f"  [{done}/{len(jobs)}] {result['job_id']}: {combos} combos, {elapsed:.0f}s")
        elif status == "skipped":
            print(f"  [{done}/{len(jobs)}] {result['job_id']}: SKIP - {result.get('reason', '')}")
        else:
            errors += 1
            print(f"  [{done}/{len(jobs)}] {result['job_id']}: ERROR")

    total_time = time.time() - t0

    # --- ランキング＆レポート生成 ---
    print(f"\nランキング生成中...")
    config_info = {
        "tf_combos": tf_list,
        "periods": period_list,
        "use_oos": use_oos,
        "exit_mode": exit_profiles,
        "min_oos_trades": min_oos_trades,
    }
    summary = generate_ranking.remote(run_id, use_oos, config_info, min_oos_trades)
    print(f"  {summary}")

    # --- 完了 ---
    print(f"\n{'=' * 60}")
    print(f"  完了!")
    print(f"  ジョブ: {done} 完了, {errors} エラー")
    print(f"  実行時間: {total_time:.0f}s ({total_time / 60:.1f}min)")
    print(f"  Run ID: {run_id}")
    print(f"  結果ダウンロード: modal run scripts/modal_download.py --run-id {run_id}")
    print(f"{'=' * 60}")


# --- Volume内のデータファイルをスキャンする補助関数 ---
@app.function(
    cpu=1,
    memory=512,
    timeout=60,
    volumes={"/data": vol_data},
)
def scan_volume_symbols(period: str, tf_list: List[Tuple[str, str]]) -> List[str]:
    """Volume内のCSVファイルから利用可能な銘柄を検出"""
    from pathlib import Path

    data_dir = Path("/data")
    symbols = set()

    for exec_tf, htf in tf_list:
        for f in data_dir.glob(f"*-{exec_tf}-{period}-merged.csv"):
            symbol = f.name.split(f"-{exec_tf}")[0]
            htf_file = data_dir / f"{symbol}-{htf}-{period}-merged.csv"
            if htf_file.exists():
                symbols.add(symbol)

    return sorted(symbols)
