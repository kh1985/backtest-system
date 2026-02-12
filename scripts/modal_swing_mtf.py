"""
15m+1h HH/LL MTFレジーム=トレード 概念検証（Modal）

3モード比較:
- 15m: 15mスイングのみ（ベースライン）
- 1h:  1hスイングのみ（15mバーで約定）
- mtf: 15m+1hスイングが一致した時のみトレード

使い方:
    modal run scripts/modal_swing_mtf.py
    modal run scripts/modal_swing_mtf.py --mode mtf          # MTFのみ
    modal run scripts/modal_swing_mtf.py --mode 15m,1h,mtf   # 全比較
    modal run scripts/modal_swing_mtf.py --msb-15m 3 --msb-1h 3
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import modal

vol_data = modal.Volume.from_name("prism-data", create_if_missing=True)
vol_results = modal.Volume.from_name("prism-results", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "pandas>=2.0,<3.0",
        "numpy>=1.24,<2.0",
        "pyyaml>=6.0",
    )
    .add_local_dir(".", "/app", copy=True, ignore=[
        "inputdata", "results", "sample_data", "cache",
        ".git", "__pycache__", "*.pyc", ".claude",
        "CLAUDE.md", "*.bak",
    ])
)

app = modal.App("prism-swing-mtf", image=image)


@app.function(
    cpu=2,
    memory=4096,
    timeout=600,
    volumes={"/data": vol_data, "/results": vol_results},
)
def test_swing_mtf(
    symbol: str,
    period: str,
    mode: str = "mtf",
    msb_15m: int = 3,
    msb_1h: int = 3,
    atr_filter: float = 0.5,
    commission_pct: float = 0.04,
) -> Dict[str, Any]:
    """1銘柄×1期間×1モードのswing MTFテスト"""
    import sys
    sys.path.insert(0, "/app")

    import numpy as np
    import pandas as pd
    from data.binance_loader import BinanceCSVLoader
    from indicators.structure import SwingStructure

    job_id = f"{mode}_{symbol}_{period}"
    t0 = time.time()
    print(f"[START] {job_id}")

    loader = BinanceCSVLoader()

    # --- 15mデータ読み込み ---
    path_15m = Path(f"/data/{symbol}-15m-{period}-merged.csv")
    if not path_15m.exists():
        print(f"[SKIP] {job_id}: 15mデータなし")
        return {"job_id": job_id, "status": "skipped", "reason": "no_15m"}

    df_15m = loader.load(str(path_15m), symbol=symbol).df.copy()

    # --- 1hデータ読み込み（1hモード or mtfモードで必要） ---
    df_1h = None
    if mode in ("1h", "mtf"):
        path_1h = Path(f"/data/{symbol}-1h-{period}-merged.csv")
        if not path_1h.exists():
            print(f"[SKIP] {job_id}: 1hデータなし")
            return {"job_id": job_id, "status": "skipped", "reason": "no_1h"}
        df_1h = loader.load(str(path_1h), symbol=symbol).df.copy()

    print(f"  {job_id}: 15m={len(df_15m)} bars" +
          (f", 1h={len(df_1h)} bars" if df_1h is not None else ""))

    # --- SwingStructure計算 ---
    swing_15m = SwingStructure(min_swing_bars=msb_15m, atr_filter=atr_filter)
    df_15m = swing_15m.calculate(df_15m)

    if df_1h is not None:
        swing_1h = SwingStructure(min_swing_bars=msb_1h, atr_filter=atr_filter)
        df_1h = swing_1h.calculate(df_1h)

    # --- 1hトレンドを15mにマージ（ルックアヘッド防止） ---
    # 1hバーは完了後に確定 → shift(1)で1h遅延
    # merge_asofで最新の確定済み1hトレンドを15mの各バーに割り当て
    if df_1h is not None:
        df_1h_signal = df_1h[["datetime", "swing_trend"]].copy()
        df_1h_signal = df_1h_signal.rename(columns={"swing_trend": "swing_trend_1h"})
        # shift(1): 1hバーが確定するのは次の1hバー開始時
        df_1h_signal["swing_trend_1h"] = df_1h_signal["swing_trend_1h"].shift(1)
        df_1h_signal = df_1h_signal.dropna(subset=["swing_trend_1h"])
        df_1h_signal["swing_trend_1h"] = df_1h_signal["swing_trend_1h"].astype(int)

        # merge_asof: 15mの各行に、それ以前で最も近い1hのトレンドを割り当て
        df_15m = df_15m.sort_values("datetime")
        df_1h_signal = df_1h_signal.sort_values("datetime")
        df_15m = pd.merge_asof(
            df_15m, df_1h_signal,
            on="datetime", direction="backward"
        )
        # 1hデータが始まる前のバーはNaN → 0（不明）で埋める
        df_15m["swing_trend_1h"] = df_15m["swing_trend_1h"].fillna(0).astype(int)
    else:
        df_15m["swing_trend_1h"] = 0

    # --- トレードシグナル生成（モード別） ---
    swing_15m_arr = df_15m["swing_trend"].values
    swing_1h_arr = df_15m["swing_trend_1h"].values
    opens = df_15m["open"].values
    closes = df_15m["close"].values
    n = len(df_15m)

    # 合成シグナル配列を作る
    if mode == "15m":
        # 15mのみ: swing_trend_15mがそのままシグナル
        signal = swing_15m_arr.copy()
    elif mode == "1h":
        # 1hのみ: 1hのswing_trendがシグナル（15mバーで約定）
        signal = swing_1h_arr.copy()
    elif mode == "mtf":
        # MTF: 15mと1hが一致した時だけシグナル、不一致=0（ノーポジ）
        signal = np.where(swing_15m_arr == swing_1h_arr, swing_15m_arr, 0)
    else:
        signal = swing_15m_arr.copy()

    # --- レジーム分布 ---
    total_bars = n
    regime_dist = {
        "uptrend": int(np.sum(signal == 1)),
        "downtrend": int(np.sum(signal == -1)),
        "neutral": int(np.sum(signal == 0)),
    }
    regime_pct = {k: round(v / total_bars * 100, 1) for k, v in regime_dist.items()}

    # --- レジーム=トレード シミュレーション ---
    results_by_side = {}

    for side, target_trend in [("short", -1), ("long", 1)]:
        trades = []
        in_position = False
        entry_price = 0.0
        entry_idx = 0

        for i in range(1, n):
            prev_signal = signal[i - 1]

            # エントリー: 前バーでtarget_trendに変化 → 今バーのopenで約定
            if not in_position and prev_signal == target_trend and (i < 2 or signal[i - 2] != target_trend):
                entry_price = opens[i]
                entry_idx = i
                in_position = True

            # 決済: 前バーでtarget_trend以外に変化 → 今バーのopenで約定
            elif in_position and prev_signal != target_trend:
                exit_price = opens[i]
                if side == "short":
                    pnl_pct = (entry_price - exit_price) / entry_price * 100
                else:
                    pnl_pct = (exit_price - entry_price) / entry_price * 100
                pnl_pct -= commission_pct * 2

                trades.append({
                    "entry_idx": entry_idx,
                    "exit_idx": i,
                    "pnl_pct": round(pnl_pct, 4),
                    "duration_bars": i - entry_idx,
                })
                in_position = False

        # 未決済ポジション → 最終バーで決済
        if in_position:
            exit_price = closes[-1]
            if side == "short":
                pnl_pct = (entry_price - exit_price) / entry_price * 100
            else:
                pnl_pct = (exit_price - entry_price) / entry_price * 100
            pnl_pct -= commission_pct * 2
            trades.append({
                "entry_idx": entry_idx,
                "exit_idx": n - 1,
                "pnl_pct": round(pnl_pct, 4),
                "duration_bars": n - 1 - entry_idx,
            })

        # --- 統計 ---
        n_trades = len(trades)
        if n_trades > 0:
            pnls = [t["pnl_pct"] for t in trades]
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p <= 0]
            durations = [t["duration_bars"] for t in trades]

            compound = 1.0
            for p in pnls:
                compound *= (1 + p / 100)
            compound_pnl = (compound - 1) * 100

            win_rate = len(wins) / n_trades * 100
            profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else 999.0

            # Max Drawdown
            equity = [1.0]
            for p in pnls:
                equity.append(equity[-1] * (1 + p / 100))
            equity = np.array(equity)
            peak = np.maximum.accumulate(equity)
            dd = (equity - peak) / peak * 100
            max_dd = float(np.min(dd))

            stats = {
                "trades": n_trades,
                "compound_pnl": round(compound_pnl, 2),
                "win_rate": round(win_rate, 1),
                "profit_factor": round(min(profit_factor, 999), 2),
                "max_drawdown": round(max_dd, 2),
                "avg_duration_bars": round(np.mean(durations), 1),
                "avg_duration_hours": round(np.mean(durations) * 0.25, 1),
            }
        else:
            stats = {"trades": 0, "compound_pnl": 0}

        results_by_side[side] = stats

    elapsed = time.time() - t0
    print(f"[DONE] {job_id}: short={results_by_side['short']['trades']}件, "
          f"long={results_by_side['long']['trades']}件, {elapsed:.1f}s")

    return {
        "job_id": job_id,
        "status": "done",
        "symbol": symbol,
        "period": period,
        "mode": mode,
        "total_bars": total_bars,
        "params": {
            "mode": mode,
            "msb_15m": msb_15m,
            "msb_1h": msb_1h,
            "atr_filter": atr_filter,
        },
        "regime_distribution": regime_dist,
        "regime_pct": regime_pct,
        "short": results_by_side["short"],
        "long": results_by_side["long"],
        "elapsed": round(elapsed, 1),
    }


@app.local_entrypoint()
def main(
    symbols: str = "",
    period: str = "20230201-20240131,20240201-20250131,20250201-20260130",
    mode: str = "15m,1h,mtf",
    msb_15m: int = 3,
    msb_1h: int = 3,
    atr_filter: float = 0.5,
):
    """15m+1h MTF HH/LL レジーム=トレード 概念検証"""

    period_list = [p.strip() for p in period.split(",")]
    mode_list = [m.strip() for m in mode.split(",")]

    if symbols:
        symbol_list = [s.strip() for s in symbols.split(",")]
    else:
        symbol_list = _scan_symbols.remote(period_list[0])
        if not symbol_list:
            print("ERROR: Volume内に15mデータが見つかりません")
            return

    print("=" * 60)
    print("  Prism: 15m+1h MTF HH/LL レジーム=トレード")
    print("=" * 60)
    print(f"銘柄: {len(symbol_list)}銘柄")
    print(f"期間: {period_list}")
    print(f"モード: {mode_list}")
    print(f"MSB(15m): {msb_15m}, MSB(1h): {msb_1h}")
    print(f"atr_filter: {atr_filter}")
    print()

    # --- 全ジョブ投入 ---
    jobs = []
    for m in mode_list:
        for p in period_list:
            for sym in symbol_list:
                jobs.append(
                    test_swing_mtf.spawn(
                        symbol=sym,
                        period=p,
                        mode=m,
                        msb_15m=msb_15m,
                        msb_1h=msb_1h,
                        atr_filter=atr_filter,
                    )
                )

    n_per_mode = len(period_list) * len(symbol_list)
    print(f"{len(jobs)} ジョブ投入（{len(mode_list)} モード x {n_per_mode} 銘柄期間）\n")

    # --- 結果収集 ---
    t0 = time.time()
    all_results = []
    for i, job in enumerate(jobs):
        result = job.get()
        all_results.append(result)
        if result["status"] == "done":
            s = result["short"]
            l = result["long"]
            print(f"  [{i+1}/{len(jobs)}] {result['job_id']}: "
                  f"S {s['trades']}件 {s.get('compound_pnl', 0):+.1f}% | "
                  f"L {l['trades']}件 {l.get('compound_pnl', 0):+.1f}%")
        else:
            print(f"  [{i+1}/{len(jobs)}] {result['job_id']}: SKIP ({result.get('reason', '')})")

    total_time = time.time() - t0

    # --- サマリー ---
    done_results = [r for r in all_results if r["status"] == "done"]
    if not done_results:
        print("\n結果なし")
        return

    def _avg(lst):
        return sum(lst) / len(lst) if lst else 0

    def _pos_rate(lst):
        return sum(1 for x in lst if x > 0) / len(lst) * 100 if lst else 0

    print(f"\n{'=' * 80}")
    print("  モード別比較サマリー")
    print(f"{'=' * 80}\n")

    print(f"{'Mode':<6} | {'--- SHORT ---':^38} | {'--- LONG ---':^38}")
    print(f"{'':>6} | {'n':>3} {'PnL':>8} {'PnL+':>5} {'件数':>5} {'WR':>5} {'PF':>5}"
          f" | {'n':>3} {'PnL':>8} {'PnL+':>5} {'件数':>5} {'WR':>5} {'PF':>5}")
    print("-" * 88)

    for m in mode_list:
        mr = [r for r in done_results if r["mode"] == m]
        if not mr:
            continue

        s_pnls = [r["short"].get("compound_pnl", 0) for r in mr if r["short"]["trades"] > 0]
        s_trades = [r["short"]["trades"] for r in mr if r["short"]["trades"] > 0]
        s_wrs = [r["short"].get("win_rate", 0) for r in mr if r["short"]["trades"] > 0]
        s_pfs = [r["short"].get("profit_factor", 0) for r in mr if r["short"]["trades"] > 0]

        l_pnls = [r["long"].get("compound_pnl", 0) for r in mr if r["long"]["trades"] > 0]
        l_trades = [r["long"]["trades"] for r in mr if r["long"]["trades"] > 0]
        l_wrs = [r["long"].get("win_rate", 0) for r in mr if r["long"]["trades"] > 0]
        l_pfs = [r["long"].get("profit_factor", 0) for r in mr if r["long"]["trades"] > 0]

        print(f"{m:<6} | {len(s_pnls):>3} {_avg(s_pnls):>+7.1f}% {_pos_rate(s_pnls):>4.0f}% {_avg(s_trades):>5.0f} {_avg(s_wrs):>4.1f}% {_avg(s_pfs):>5.2f}"
              f" | {len(l_pnls):>3} {_avg(l_pnls):>+7.1f}% {_pos_rate(l_pnls):>4.0f}% {_avg(l_trades):>5.0f} {_avg(l_wrs):>4.1f}% {_avg(l_pfs):>5.2f}")

    # --- モード×期間 詳細 ---
    print(f"\n{'=' * 80}")
    print("  モード×期間 詳細")
    print(f"{'=' * 80}\n")

    for m in mode_list:
        print(f"--- {m.upper()} ---")
        for p in period_list:
            mr = [r for r in done_results if r["mode"] == m and r["period"] == p]
            if not mr:
                continue
            s_pnls = [r["short"].get("compound_pnl", 0) for r in mr if r["short"]["trades"] > 0]
            s_trades = [r["short"]["trades"] for r in mr if r["short"]["trades"] > 0]
            l_pnls = [r["long"].get("compound_pnl", 0) for r in mr if r["long"]["trades"] > 0]
            l_trades = [r["long"]["trades"] for r in mr if r["long"]["trades"] > 0]
            print(f"  {p}: S {len(s_pnls)}件 avg={_avg(s_pnls):+.1f}% PnL+={_pos_rate(s_pnls):.0f}% avg_n={_avg(s_trades):.0f}"
                  f" | L {len(l_pnls)}件 avg={_avg(l_pnls):+.1f}% PnL+={_pos_rate(l_pnls):.0f}% avg_n={_avg(l_trades):.0f}")
        print()

    # --- ベスト個別（MTFのみ） ---
    mtf_results = [r for r in done_results if r["mode"] == "mtf"]
    if mtf_results:
        print(f"{'=' * 80}")
        print("  MTFモード ベスト個別 (Top 10)")
        print(f"{'=' * 80}\n")

        print("--- SHORT Top 10 ---")
        print(f"{'銘柄':<12} {'期間':<24} {'件数':>4} {'PnL':>8} {'WR':>6} {'PF':>6} {'DD':>7} {'h':>6}")
        print("-" * 78)
        short_all = [(r["symbol"], r["period"], r["short"])
                     for r in mtf_results if r["short"]["trades"] > 0]
        short_all.sort(key=lambda x: x[2].get("compound_pnl", 0), reverse=True)
        for sym, per, s in short_all[:10]:
            print(f"{sym:<12} {per:<24} {s['trades']:>4} {s['compound_pnl']:>+7.1f}% "
                  f"{s.get('win_rate', 0):>5.1f}% {s.get('profit_factor', 0):>5.1f} "
                  f"{s.get('max_drawdown', 0):>6.1f}% {s.get('avg_duration_hours', 0):>5.1f}")
        print()

        print("--- LONG Top 10 ---")
        print(f"{'銘柄':<12} {'期間':<24} {'件数':>4} {'PnL':>8} {'WR':>6} {'PF':>6} {'DD':>7} {'h':>6}")
        print("-" * 78)
        long_all = [(r["symbol"], r["period"], r["long"])
                    for r in mtf_results if r["long"]["trades"] > 0]
        long_all.sort(key=lambda x: x[2].get("compound_pnl", 0), reverse=True)
        for sym, per, l in long_all[:10]:
            print(f"{sym:<12} {per:<24} {l['trades']:>4} {l['compound_pnl']:>+7.1f}% "
                  f"{l.get('win_rate', 0):>5.1f}% {l.get('profit_factor', 0):>5.1f} "
                  f"{l.get('max_drawdown', 0):>6.1f}% {l.get('avg_duration_hours', 0):>5.1f}")

    # --- レジーム分布 ---
    print(f"\n{'=' * 80}")
    print("  レジーム分布（モード別平均）")
    print(f"{'=' * 80}\n")
    print(f"{'Mode':<6} {'UT%':>7} {'DT%':>7} {'Neutral%':>9}")
    print("-" * 32)
    for m in mode_list:
        mr = [r for r in done_results if r["mode"] == m]
        if not mr:
            continue
        avg_ut = _avg([r["regime_pct"]["uptrend"] for r in mr])
        avg_dt = _avg([r["regime_pct"]["downtrend"] for r in mr])
        avg_n = _avg([r["regime_pct"]["neutral"] for r in mr])
        print(f"{m:<6} {avg_ut:>6.1f}% {avg_dt:>6.1f}% {avg_n:>8.1f}%")

    print(f"\n実行時間: {total_time:.0f}s ({total_time/60:.1f}min)")

    # --- JSON保存 ---
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_swing_mtf")
    save_results.remote(run_id, all_results, {
        "modes": mode_list,
        "msb_15m": msb_15m,
        "msb_1h": msb_1h,
        "atr_filter": atr_filter,
        "periods": period_list,
        "symbols": symbol_list,
    })
    print(f"\n結果保存: {run_id}")


@app.function(cpu=1, memory=512, timeout=60, volumes={"/data": vol_data})
def _scan_symbols(period: str) -> List[str]:
    """Volume内の15mデータから銘柄を検出"""
    data_dir = Path("/data")
    symbols = set()
    for f in data_dir.glob(f"*-15m-{period}-merged.csv"):
        symbol = f.name.split("-15m")[0]
        symbols.add(symbol)
    return sorted(symbols)


@app.function(cpu=1, memory=512, timeout=60, volumes={"/results": vol_results})
def save_results(run_id: str, results: List[Dict], params: Dict):
    """結果をJSONで保存"""
    output_dir = Path(f"/results/{run_id}")
    output_dir.mkdir(parents=True, exist_ok=True)

    data = {
        "type": "swing_mtf_poc",
        "params": params,
        "results": results,
        "generated_at": datetime.now().isoformat(),
    }

    with open(output_dir / "swing_mtf_results.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    vol_results.commit()
    print(f"保存完了: {run_id}/swing_mtf_results.json")
