"""
マルチTF戻り売り検証（Modal上で実行）

15mでシグナル検知 + 1h ATRでTP/SL管理 vs 純粋1hシグナル を比較。
「15mの精度で入って、1hの幅で持つ」アプローチの効果測定。

使い方:
    PYTHONIOENCODING=utf-8 py -3 -m modal run scripts/modal_multitf_explore.py
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import modal

vol_data = modal.Volume.from_name("prism-data", create_if_missing=True)

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

app = modal.App("prism-multitf-explore", image=image)


@app.function(
    cpu=4,
    memory=8192,
    timeout=600,
    volumes={"/data": vol_data},
)
def explore_symbol(
    symbol: str,
    period: str = "20250201-20260130",
) -> Dict[str, Any]:
    """1銘柄: 15m検知+1h exit vs 純粋1h の比較"""
    import sys
    sys.path.insert(0, "/app")

    import numpy as np
    import pandas as pd

    from data.binance_loader import BinanceCSVLoader
    from analysis.trend import TrendDetector
    from indicators.registry import create_indicator

    loader = BinanceCSVLoader()
    data_dir = Path("/data")

    # --- データ読み込み（15m, 1h, 4h）---
    p15m = data_dir / f"{symbol}-15m-{period}-merged.csv"
    p1h = data_dir / f"{symbol}-1h-{period}-merged.csv"
    p4h = data_dir / f"{symbol}-4h-{period}-merged.csv"

    for p in [p15m, p1h, p4h]:
        if not p.exists():
            return {"symbol": symbol, "status": "skipped", "reason": f"{p.name} not found"}

    df_15m = loader.load(str(p15m), symbol=symbol).df.copy()
    df_1h = loader.load(str(p1h), symbol=symbol).df.copy()
    df_4h = loader.load(str(p4h), symbol=symbol).df.copy()

    # --- レジーム検出（4h MA Cross → 1h → 15m にラベル）---
    detector = TrendDetector()
    df_4h = detector.detect_ma_cross(df_4h, fast_period=20, slow_period=50)
    df_1h = TrendDetector.label_execution_tf(df_1h, df_4h)
    df_15m = TrendDetector.label_execution_tf(df_15m, df_4h)

    dt_mask_1h = df_1h["trend_regime"] == "downtrend"
    dt_mask_15m = df_15m["trend_regime"] == "downtrend"

    dt_bars_1h = dt_mask_1h.sum()
    if dt_bars_1h < 50:
        return {"symbol": symbol, "status": "skipped",
                "reason": f"DT bars too few: {dt_bars_1h}"}

    # --- インジケーター計算 ---
    # 15m: SMA20
    sma_15m = create_indicator("sma", period=20)
    df_15m = sma_15m.calculate(df_15m)

    # 1h: SMA20, ATR14
    sma_1h = create_indicator("sma", period=20)
    df_1h = sma_1h.calculate(df_1h)
    atr_1h = create_indicator("atr", period=14)
    df_1h = atr_1h.calculate(df_1h)

    # --- 1h ATR を 15m に forward-fill ---
    atr_labels = df_1h[["datetime", "atr_14"]].copy()
    atr_labels = atr_labels.rename(columns={"atr_14": "atr_1h"})
    atr_labels = atr_labels.sort_values("datetime")
    df_15m = df_15m.sort_values("datetime").reset_index(drop=True)
    merged = pd.merge_asof(
        df_15m[["datetime"]],
        atr_labels,
        on="datetime",
        direction="backward",
    )
    df_15m["atr_1h"] = merged["atr_1h"].values

    # --- 事前計算 ---
    # 15m arrays
    h_15m = df_15m["high"].values
    lo_15m = df_15m["low"].values
    cl_15m = df_15m["close"].values
    o_15m = df_15m["open"].values
    sma20_15m = df_15m["sma_20"].values
    atr_1h_on_15m = df_15m["atr_1h"].values
    dt_15m = dt_mask_15m.values

    # 1h arrays
    h_1h = df_1h["high"].values
    lo_1h = df_1h["low"].values
    cl_1h = df_1h["close"].values
    o_1h = df_1h["open"].values
    sma20_1h = df_1h["sma_20"].values
    atr_1h_arr = df_1h["atr_14"].values
    dt_1h = dt_mask_1h.values

    # --- シグナル生成 ---
    # Approach A: 15m検知（pullback reject on 15m）, 15mエントリー, 1h ATR exit
    sig_15m = (h_15m >= sma20_15m) & (cl_15m < sma20_15m) & dt_15m
    sig_15m = sig_15m & ~np.isnan(sma20_15m) & ~np.isnan(atr_1h_on_15m)

    # Approach B: 純粋1hシグナル, 1h ATR exit（ベースライン）
    sig_1h = (h_1h >= sma20_1h) & (cl_1h < sma20_1h) & dt_1h
    sig_1h = sig_1h & ~np.isnan(sma20_1h) & ~np.isnan(atr_1h_arr)

    # --- TP/SLシミュレーション ---
    def simulate_short(prices_h, prices_l, prices_c, entry_indices, entry_prices,
                       atr_vals, tp_mult, sl_mult, max_hold):
        """ショートTP/SLシミュレーション"""
        wins = 0
        losses = 0
        timeouts = 0
        pnl_list = []
        n = len(prices_c)
        last_exit = -1

        for k in range(len(entry_indices)):
            idx = entry_indices[k]
            if idx <= last_exit:
                continue
            ep = entry_prices[k]
            atr = atr_vals[k]
            if ep <= 0 or np.isnan(ep) or np.isnan(atr) or atr <= 0:
                continue

            tp_price = ep - atr * tp_mult
            sl_price = ep + atr * sl_mult

            resolved = False
            for j in range(idx + 1, min(idx + max_hold + 1, n)):
                if prices_l[j] <= tp_price:
                    wins += 1
                    pnl_list.append((ep - tp_price) / ep * 100)
                    last_exit = j
                    resolved = True
                    break
                if prices_h[j] >= sl_price:
                    losses += 1
                    pnl_list.append((ep - sl_price) / ep * 100)
                    last_exit = j
                    resolved = True
                    break

            if not resolved:
                end_idx = min(idx + max_hold, n - 1)
                exit_p = prices_c[end_idx]
                pnl_list.append((ep - exit_p) / ep * 100)
                timeouts += 1
                last_exit = end_idx

        trades = len(pnl_list)
        if trades == 0:
            return {"trades": 0, "wins": 0, "losses": 0, "timeouts": 0,
                    "pnl_total": 0.0, "win_rate": 0.0, "max_dd": 0.0,
                    "avg_pnl": 0.0}

        pnl_arr = np.array(pnl_list)
        cumsum = np.cumsum(pnl_arr)
        running_max = np.maximum.accumulate(cumsum)
        max_dd = (running_max - cumsum).max()

        return {
            "trades": trades,
            "wins": wins,
            "losses": losses,
            "timeouts": timeouts,
            "pnl_total": float(pnl_arr.sum()),
            "win_rate": float(wins / trades * 100),
            "max_dd": float(max_dd),
            "avg_pnl": float(pnl_arr.mean()),
        }

    # --- Exit設定 ---
    exit_configs = [
        ("atr_tp15_sl20", 1.5, 2.0),
        ("atr_tp20_sl20", 2.0, 2.0),
        ("atr_tp15_sl15", 1.5, 1.5),
        ("atr_tp20_sl15", 2.0, 1.5),
    ]

    results = {}

    for exit_name, tp_m, sl_m in exit_configs:
        # === Approach A: 15m検知 + 15mバーでTP/SL判定（1h ATR使用）===
        entries_a = np.where(sig_15m)[0]
        entry_prices_a = cl_15m[entries_a]
        atr_a = atr_1h_on_15m[entries_a]
        # max_hold: 1hで72本 = 15mで288本
        res_a = simulate_short(h_15m, lo_15m, cl_15m,
                               entries_a, entry_prices_a, atr_a,
                               tp_m, sl_m, max_hold=288)

        # === Approach B: 純粋1h + 1hバーでTP/SL判定 ===
        entries_b = np.where(sig_1h)[0]
        entry_prices_b = cl_1h[entries_b]
        atr_b = atr_1h_arr[entries_b]
        res_b = simulate_short(h_1h, lo_1h, cl_1h,
                               entries_b, entry_prices_b, atr_b,
                               tp_m, sl_m, max_hold=72)

        # === Approach C: 15m検知 + 1hバーでTP/SL判定（1h解像度exit）===
        # 15mシグナルを1hバーにマッピング → その1hバーでエントリー
        # 15mのdatetimeを1hの時間にフロア
        sig_15m_idx = np.where(sig_15m)[0]
        if len(sig_15m_idx) > 0:
            sig_15m_dt = df_15m["datetime"].iloc[sig_15m_idx].values
            # 1h datetimeに最も近いものにマッピング
            dt_1h_vals = df_1h["datetime"].values
            mapped_1h_idx = np.searchsorted(dt_1h_vals, sig_15m_dt, side="right") - 1
            mapped_1h_idx = np.clip(mapped_1h_idx, 0, len(df_1h) - 1)
            # ユニークな1hインデックスのみ（同じ1hバー内の複数15mシグナルは1回だけ）
            unique_1h_idx = np.unique(mapped_1h_idx)
            # DTフィルタ
            unique_1h_idx = unique_1h_idx[dt_1h[unique_1h_idx]]

            entry_prices_c = cl_1h[unique_1h_idx]
            atr_c = atr_1h_arr[unique_1h_idx]
            res_c = simulate_short(h_1h, lo_1h, cl_1h,
                                   unique_1h_idx, entry_prices_c, atr_c,
                                   tp_m, sl_m, max_hold=72)
        else:
            res_c = {"trades": 0, "wins": 0, "losses": 0, "timeouts": 0,
                     "pnl_total": 0.0, "win_rate": 0.0, "max_dd": 0.0,
                     "avg_pnl": 0.0}

        results[exit_name] = {
            "A_15m_detect_15m_exec": res_a,
            "B_pure_1h": res_b,
            "C_15m_detect_1h_exec": res_c,
        }

    return {
        "symbol": symbol,
        "status": "done",
        "dt_bars_1h": int(dt_bars_1h),
        "dt_bars_15m": int(dt_mask_15m.sum()),
        "total_signals_15m": int(sig_15m.sum()),
        "total_signals_1h": int(sig_1h.sum()),
        "results": results,
    }


@app.function(cpu=1, memory=512, timeout=60, volumes={"/data": vol_data})
def scan_symbols(period: str) -> List[str]:
    data_dir = Path("/data")
    symbols = set()
    for f in data_dir.glob(f"*-1h-{period}-merged.csv"):
        sym = f.name.split("-1h")[0]
        if (data_dir / f"{sym}-15m-{period}-merged.csv").exists() and \
           (data_dir / f"{sym}-4h-{period}-merged.csv").exists():
            symbols.add(sym)
    return sorted(symbols)


@app.local_entrypoint()
def main(
    symbols: str = "",
    period: str = "20250201-20260130",
):
    """マルチTF戻り売り比較（全銘柄並列）"""
    import json

    if symbols:
        symbol_list = [s.strip() for s in symbols.split(",")]
    else:
        symbol_list = scan_symbols.remote(period)
        if not symbol_list:
            print("ERROR: No data")
            return

    print("=" * 70)
    print("  Multi-TF Pullback Short Comparison (Modal)")
    print("=" * 70)
    print(f"Symbols: {len(symbol_list)}")
    print(f"Approaches:")
    print(f"  A: 15m signal detect + 15m bar TP/SL (1h ATR)")
    print(f"  B: Pure 1h signal + 1h bar TP/SL (baseline)")
    print(f"  C: 15m signal detect → mapped to 1h bar + 1h TP/SL")
    print()

    t0 = time.time()

    jobs = [explore_symbol.spawn(symbol=sym, period=period) for sym in symbol_list]

    all_results = []
    done = 0
    for job in jobs:
        result = job.get()
        done += 1
        sym = result["symbol"]
        if result["status"] == "done":
            sigs_15 = result["total_signals_15m"]
            sigs_1h = result["total_signals_1h"]
            print(f"  [{done}/{len(jobs)}] {sym}: 15m={sigs_15} sigs, 1h={sigs_1h} sigs")
            all_results.append(result)
        else:
            print(f"  [{done}/{len(jobs)}] {sym}: SKIP - {result.get('reason')}")

    elapsed = time.time() - t0
    print(f"\nTime: {elapsed:.0f}s")
    print(f"Valid: {len(all_results)}/{len(symbol_list)}")

    if not all_results:
        return

    # === 集約 ===
    print("\n" + "=" * 70)
    print("  Results Comparison")
    print("=" * 70)

    exit_names = ["atr_tp15_sl20", "atr_tp20_sl20", "atr_tp15_sl15", "atr_tp20_sl15"]
    approaches = ["A_15m_detect_15m_exec", "B_pure_1h", "C_15m_detect_1h_exec"]
    approach_labels = {
        "A_15m_detect_15m_exec": "A: 15m detect + 15m exec",
        "B_pure_1h": "B: Pure 1h (baseline)",
        "C_15m_detect_1h_exec": "C: 15m detect + 1h exec",
    }

    for exit_name in exit_names:
        print(f"\n--- {exit_name} ---")
        print(f"{'Approach':<30} {'Syms':>4} {'Trades':>7} {'WR%':>6} {'PnL%':>9} {'AvgPnL':>8} {'MaxDD':>7}")
        print("-" * 75)

        for appr in approaches:
            total_trades = 0
            total_wins = 0
            total_pnl = 0.0
            syms_with_trades = 0
            max_dd_list = []
            pnl_per_sym = []

            for res in all_results:
                r = res["results"][exit_name][appr]
                if r["trades"] > 0:
                    syms_with_trades += 1
                total_trades += r["trades"]
                total_wins += r["wins"]
                total_pnl += r["pnl_total"]
                max_dd_list.append(r["max_dd"])
                pnl_per_sym.append(r["pnl_total"])

            wr = total_wins / total_trades * 100 if total_trades > 0 else 0
            avg_pnl = total_pnl / syms_with_trades if syms_with_trades > 0 else 0
            max_dd = max(max_dd_list) if max_dd_list else 0
            label = approach_labels[appr]
            print(f"{label:<30} {syms_with_trades:>4} {total_trades:>7} {wr:>5.1f}% {total_pnl:>+8.1f}% {avg_pnl:>+7.1f}% {max_dd:>6.1f}%")

    # --- 銘柄別比較（メインexit: atr_tp15_sl20）---
    main_exit = "atr_tp15_sl20"
    print(f"\n\n--- Per-Symbol: {main_exit} ---")
    print(f"{'Symbol':<12} {'15m+15m':>10} {'Pure 1h':>10} {'15m+1h':>10} {'Best':>12}")
    print("-" * 58)

    for res in sorted(all_results, key=lambda x: x["symbol"]):
        sym = res["symbol"]
        pnl_a = res["results"][main_exit]["A_15m_detect_15m_exec"]["pnl_total"]
        pnl_b = res["results"][main_exit]["B_pure_1h"]["pnl_total"]
        pnl_c = res["results"][main_exit]["C_15m_detect_1h_exec"]["pnl_total"]

        best = "A(15m+15m)" if pnl_a >= pnl_b and pnl_a >= pnl_c else \
               "B(pure 1h)" if pnl_b >= pnl_a and pnl_b >= pnl_c else "C(15m+1h)"

        print(f"{sym:<12} {pnl_a:>+9.1f}% {pnl_b:>+9.1f}% {pnl_c:>+9.1f}% {best:>12}")

    # --- 勝率比較 ---
    print(f"\n--- Win Rate Comparison: {main_exit} ---")
    for appr in approaches:
        label = approach_labels[appr]
        a_wins = sum(1 for res in all_results
                     if res["results"][main_exit][appr]["pnl_total"] > 0)
        print(f"  {label}: {a_wins}/{len(all_results)} symbols profitable")

    # JSON保存
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {"period": period, "n_symbols": len(all_results)},
        "symbol_results": all_results,
    }
    out_path = Path("multitf_explore_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out_path}")
    print(f"\n{'=' * 70}")
    print(f"  Done! {elapsed:.0f}s")
    print(f"{'=' * 70}")
