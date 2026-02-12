"""
15m HH/LL レジーム=トレード 概念検証（Modal）

SwingStructureを15mデータに適用し、
swing_trend=-1の期間をショート、swing_trend=1の期間をロングとして
PnL・トレード数・勝率を計算する。

使い方:
    modal run scripts/modal_swing_poc.py
    modal run scripts/modal_swing_poc.py --symbols BTCUSDT,ETHUSDT
    modal run scripts/modal_swing_poc.py --min-swing-bars 5
    modal run scripts/modal_swing_poc.py --min-swing-bars 3,5,7,10,15  # 複数値テスト
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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

app = modal.App("prism-swing-poc", image=image)


@app.function(
    cpu=2,
    memory=4096,
    timeout=600,
    volumes={"/data": vol_data, "/results": vol_results},
)
def test_swing_regime(
    symbol: str,
    period: str,
    min_swing_bars: int = 3,
    atr_filter: float = 0.5,
    commission_pct: float = 0.04,
) -> Dict[str, Any]:
    """1銘柄×1期間のswing regime=tradeテスト"""
    import sys
    sys.path.insert(0, "/app")

    import numpy as np
    import pandas as pd
    from data.binance_loader import BinanceCSVLoader
    from indicators.structure import SwingStructure

    job_id = f"{symbol}_{period}"
    t0 = time.time()
    print(f"[START] {job_id}")

    # --- データ読み込み（15m） ---
    data_path = Path(f"/data/{symbol}-15m-{period}-merged.csv")
    if not data_path.exists():
        print(f"[SKIP] {job_id}: データなし")
        return {"job_id": job_id, "status": "skipped"}

    loader = BinanceCSVLoader()
    ohlcv = loader.load(str(data_path), symbol=symbol)
    df = ohlcv.df.copy()

    print(f"  {job_id}: {len(df)} bars (15m)")

    # --- SwingStructure計算 ---
    swing = SwingStructure(min_swing_bars=min_swing_bars, atr_filter=atr_filter)
    df = swing.calculate(df)

    # --- レジーム分布 ---
    trend_counts = df["swing_trend"].value_counts()
    total_bars = len(df)
    regime_dist = {
        "uptrend": int(trend_counts.get(1, 0)),
        "downtrend": int(trend_counts.get(-1, 0)),
        "range": int(trend_counts.get(0, 0)),
    }
    regime_pct = {k: round(v / total_bars * 100, 1) for k, v in regime_dist.items()}

    # --- レジーム=トレード シミュレーション ---
    # ルール:
    # - swing_trend が -1 に変わった → 次のバーのopenでショートエントリー
    # - swing_trend が -1 以外に変わった → 次のバーのopenでショート決済
    # - swing_trend が 1 に変わった → 次のバーのopenでロングエントリー
    # - swing_trend が 1 以外に変わった → 次のバーのopenでロング決済
    # → 「next bar open」で約定（ルックアヘッド防止）

    swing_trend = df["swing_trend"].values
    opens = df["open"].values
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    swing_highs = df["swing_high"].values
    swing_lows = df["swing_low"].values
    datetimes = df["datetime"].values if "datetime" in df.columns else np.arange(len(df))

    results_by_side = {}

    for side, target_trend in [("short", -1), ("long", 1)]:
        trades = []
        in_position = False
        entry_price = 0.0
        entry_idx = 0

        for i in range(1, len(df)):
            prev_trend = swing_trend[i - 1]
            # エントリー: 前バーでtarget_trendに変化 → 今バーのopenで約定
            if not in_position and prev_trend == target_trend and (i < 2 or swing_trend[i - 2] != target_trend):
                entry_price = opens[i]
                entry_idx = i
                in_position = True

            # 決済: 前バーでtarget_trend以外に変化 → 今バーのopenで約定
            elif in_position and prev_trend != target_trend:
                exit_price = opens[i]
                # PnL計算
                if side == "short":
                    pnl_pct = (entry_price - exit_price) / entry_price * 100
                else:
                    pnl_pct = (exit_price - entry_price) / entry_price * 100

                # 手数料（往復）
                pnl_pct -= commission_pct * 2

                duration_bars = i - entry_idx
                trades.append({
                    "entry_idx": entry_idx,
                    "exit_idx": i,
                    "entry_price": round(entry_price, 4),
                    "exit_price": round(exit_price, 4),
                    "pnl_pct": round(pnl_pct, 4),
                    "duration_bars": duration_bars,
                })
                in_position = False

        # まだポジション持ってる場合は最終バーで決済
        if in_position:
            exit_price = closes[-1]
            if side == "short":
                pnl_pct = (entry_price - exit_price) / entry_price * 100
            else:
                pnl_pct = (exit_price - entry_price) / entry_price * 100
            pnl_pct -= commission_pct * 2
            trades.append({
                "entry_idx": entry_idx,
                "exit_idx": len(df) - 1,
                "entry_price": round(entry_price, 4),
                "exit_price": round(exit_price, 4),
                "pnl_pct": round(pnl_pct, 4),
                "duration_bars": len(df) - 1 - entry_idx,
            })

        # --- 統計計算 ---
        n_trades = len(trades)
        if n_trades > 0:
            pnls = [t["pnl_pct"] for t in trades]
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p <= 0]
            durations = [t["duration_bars"] for t in trades]

            total_pnl = sum(pnls)
            # 複利PnL
            compound = 1.0
            for p in pnls:
                compound *= (1 + p / 100)
            compound_pnl = (compound - 1) * 100

            win_rate = len(wins) / n_trades * 100
            avg_win = np.mean(wins) if wins else 0
            avg_loss = np.mean(losses) if losses else 0
            profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else 999.0
            avg_duration = np.mean(durations)
            max_duration = max(durations)
            min_duration = min(durations)

            # Max Drawdown（複利）
            equity = [1.0]
            for p in pnls:
                equity.append(equity[-1] * (1 + p / 100))
            equity = np.array(equity)
            peak = np.maximum.accumulate(equity)
            dd = (equity - peak) / peak * 100
            max_dd = float(np.min(dd))

            stats = {
                "trades": n_trades,
                "total_pnl": round(total_pnl, 2),
                "compound_pnl": round(compound_pnl, 2),
                "win_rate": round(win_rate, 1),
                "avg_win": round(avg_win, 2),
                "avg_loss": round(avg_loss, 2),
                "profit_factor": round(min(profit_factor, 999), 2),
                "max_drawdown": round(max_dd, 2),
                "avg_duration_bars": round(avg_duration, 1),
                "max_duration_bars": max_duration,
                "min_duration_bars": min_duration,
                "avg_duration_hours": round(avg_duration * 0.25, 1),  # 15m bars → hours
            }
        else:
            stats = {"trades": 0}

        results_by_side[side] = stats

    elapsed = time.time() - t0
    print(f"[DONE] {job_id}: short={results_by_side['short']['trades']}件, "
          f"long={results_by_side['long']['trades']}件, {elapsed:.1f}s")

    return {
        "job_id": job_id,
        "status": "done",
        "symbol": symbol,
        "period": period,
        "total_bars": total_bars,
        "params": {
            "min_swing_bars": min_swing_bars,
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
    min_swing_bars: str = "3",
    atr_filter: float = 0.5,
):
    """15m HH/LL レジーム=トレード 概念検証"""

    period_list = [p.strip() for p in period.split(",")]
    msb_list = [int(x.strip()) for x in min_swing_bars.split(",")]

    # 銘柄自動検出 or 指定
    if symbols:
        symbol_list = [s.strip() for s in symbols.split(",")]
    else:
        symbol_list = _scan_symbols.remote(period_list[0])
        if not symbol_list:
            print("ERROR: Volume内に15mデータが見つかりません")
            return

    print("=" * 60)
    print("  Prism: 15m HH/LL レジーム=トレード 概念検証")
    print("=" * 60)
    print(f"銘柄: {symbol_list} ({len(symbol_list)}銘柄)")
    print(f"期間: {period_list}")
    print(f"min_swing_bars: {msb_list}")
    print(f"atr_filter: {atr_filter}")
    print()

    # --- 全ジョブ投入（銘柄×期間×min_swing_bars） ---
    jobs = []
    job_meta = []  # (symbol, period, msb) の対応
    for msb in msb_list:
        for p in period_list:
            for sym in symbol_list:
                jobs.append(
                    test_swing_regime.spawn(
                        symbol=sym,
                        period=p,
                        min_swing_bars=msb,
                        atr_filter=atr_filter,
                    )
                )
                job_meta.append((sym, p, msb))

    print(f"{len(jobs)} ジョブ投入完了（{len(msb_list)} MSB値 × {len(period_list)} 期間 × {len(symbol_list)} 銘柄）\n")

    # --- 結果収集 ---
    t0 = time.time()
    all_results = []
    for i, job in enumerate(jobs):
        result = job.get()
        all_results.append(result)
        if result["status"] == "done":
            s = result["short"]
            l = result["long"]
            msb = result["params"]["min_swing_bars"]
            print(f"  [{i+1}/{len(jobs)}] msb={msb} {result['job_id']}: "
                  f"Short {s['trades']}件 PnL={s.get('compound_pnl', 0):+.1f}% | "
                  f"Long {l['trades']}件 PnL={l.get('compound_pnl', 0):+.1f}%")
        else:
            print(f"  [{i+1}/{len(jobs)}] {result['job_id']}: SKIP")

    total_time = time.time() - t0

    # --- サマリー（min_swing_bars別） ---
    done_results = [r for r in all_results if r["status"] == "done"]
    if not done_results:
        print("\n結果なし")
        return

    print(f"\n{'=' * 80}")
    print("  min_swing_bars 比較サマリー")
    print(f"{'=' * 80}\n")

    # MSB別集計テーブル
    print(f"{'MSB':>4} | {'--- SHORT ---':^42} | {'--- LONG ---':^42}")
    print(f"{'':>4} | {'テスト':>4} {'平均PnL':>8} {'PnL+率':>7} {'平均件数':>7} {'平均WR':>6} {'平均PF':>6}"
          f" | {'テスト':>4} {'平均PnL':>8} {'PnL+率':>7} {'平均件数':>7} {'平均WR':>6} {'平均PF':>6}")
    print("-" * 100)

    for msb in msb_list:
        msb_results = [r for r in done_results if r["params"]["min_swing_bars"] == msb]
        if not msb_results:
            continue

        # Short
        s_pnls = [r["short"].get("compound_pnl", 0) for r in msb_results if r["short"]["trades"] > 0]
        s_trades = [r["short"]["trades"] for r in msb_results if r["short"]["trades"] > 0]
        s_wrs = [r["short"].get("win_rate", 0) for r in msb_results if r["short"]["trades"] > 0]
        s_pfs = [r["short"].get("profit_factor", 0) for r in msb_results if r["short"]["trades"] > 0]
        # Long
        l_pnls = [r["long"].get("compound_pnl", 0) for r in msb_results if r["long"]["trades"] > 0]
        l_trades = [r["long"]["trades"] for r in msb_results if r["long"]["trades"] > 0]
        l_wrs = [r["long"].get("win_rate", 0) for r in msb_results if r["long"]["trades"] > 0]
        l_pfs = [r["long"].get("profit_factor", 0) for r in msb_results if r["long"]["trades"] > 0]

        def _avg(lst):
            return sum(lst) / len(lst) if lst else 0

        def _pos_rate(lst):
            return sum(1 for x in lst if x > 0) / len(lst) * 100 if lst else 0

        print(f"{msb:>4} | {len(s_pnls):>4} {_avg(s_pnls):>+7.1f}% {_pos_rate(s_pnls):>6.0f}% {_avg(s_trades):>7.0f} {_avg(s_wrs):>5.1f}% {_avg(s_pfs):>5.2f}"
              f" | {len(l_pnls):>4} {_avg(l_pnls):>+7.1f}% {_pos_rate(l_pnls):>6.0f}% {_avg(l_trades):>7.0f} {_avg(l_wrs):>5.1f}% {_avg(l_pfs):>5.2f}")

    # --- 全MSBの中でベストなケース ---
    print(f"\n{'=' * 80}")
    print("  ベスト個別結果 (PnL Top 10)")
    print(f"{'=' * 80}\n")

    # Short Top 10
    print("--- SHORT Top 10 ---")
    print(f"{'MSB':>4} {'銘柄':<12} {'期間':<24} {'件数':>4} {'PnL':>8} {'勝率':>6} {'PF':>6} {'MaxDD':>7} {'平均h':>6}")
    print("-" * 85)
    short_all = [(r["params"]["min_swing_bars"], r["symbol"], r["period"], r["short"])
                 for r in done_results if r["short"]["trades"] > 0]
    short_all.sort(key=lambda x: x[3].get("compound_pnl", 0), reverse=True)
    for msb, sym, per, s in short_all[:10]:
        print(f"{msb:>4} {sym:<12} {per:<24} {s['trades']:>4} {s['compound_pnl']:>+7.1f}% "
              f"{s['win_rate']:>5.1f}% {s['profit_factor']:>5.1f} {s['max_drawdown']:>6.1f}% "
              f"{s['avg_duration_hours']:>5.1f}")

    print()

    # Long Top 10
    print("--- LONG Top 10 ---")
    print(f"{'MSB':>4} {'銘柄':<12} {'期間':<24} {'件数':>4} {'PnL':>8} {'勝率':>6} {'PF':>6} {'MaxDD':>7} {'平均h':>6}")
    print("-" * 85)
    long_all = [(r["params"]["min_swing_bars"], r["symbol"], r["period"], r["long"])
                for r in done_results if r["long"]["trades"] > 0]
    long_all.sort(key=lambda x: x[3].get("compound_pnl", 0), reverse=True)
    for msb, sym, per, l in long_all[:10]:
        print(f"{msb:>4} {sym:<12} {per:<24} {l['trades']:>4} {l['compound_pnl']:>+7.1f}% "
              f"{l['win_rate']:>5.1f}% {l['profit_factor']:>5.1f} {l['max_drawdown']:>6.1f}% "
              f"{l['avg_duration_hours']:>5.1f}")

    # レジーム分布（MSB別）
    print(f"\n{'=' * 80}")
    print("  レジーム分布（MSB別平均）")
    print(f"{'=' * 80}\n")
    print(f"{'MSB':>4} {'UT%':>7} {'DT%':>7} {'Range%':>7}")
    print("-" * 30)
    for msb in msb_list:
        msb_results = [r for r in done_results if r["params"]["min_swing_bars"] == msb]
        if not msb_results:
            continue
        avg_ut = sum(r["regime_pct"]["uptrend"] for r in msb_results) / len(msb_results)
        avg_dt = sum(r["regime_pct"]["downtrend"] for r in msb_results) / len(msb_results)
        avg_rg = sum(r["regime_pct"]["range"] for r in msb_results) / len(msb_results)
        print(f"{msb:>4} {avg_ut:>6.1f}% {avg_dt:>6.1f}% {avg_rg:>6.1f}%")

    print(f"\n実行時間: {total_time:.0f}s ({total_time/60:.1f}min)")

    # --- JSON保存 ---
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_swing_poc")
    save_results.remote(run_id, all_results, {
        "min_swing_bars": msb_list,
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
        "type": "swing_regime_poc",
        "params": params,
        "results": results,
        "generated_at": datetime.now().isoformat(),
    }

    with open(output_dir / "swing_poc_results.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    vol_results.commit()
    print(f"保存完了: {run_id}/swing_poc_results.json")
