"""
戻り売り（Pullback Short）パターン探索（Modal上で実行）

DTレジームで価格がEMA/SMAまで戻り、拒否されるパターンを検証。
1h:4h 構成（MA Cross fallback）。

使い方:
    PYTHONIOENCODING=utf-8 py -3 -m modal run scripts/modal_pullback_explore.py
    PYTHONIOENCODING=utf-8 py -3 -m modal run scripts/modal_pullback_explore.py --symbols BTCUSDT,ETHUSDT
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

app = modal.App("prism-pullback-explore", image=image)


# ====== パターン定義 ======
PATTERNS = [
    # --- Base: EMA touch rejection (high >= EMA, close < EMA) ---
    ("pb_ema13", "pullback_reject", {"ma_col": "ema_13", "bearish": False}),
    ("pb_ema21", "pullback_reject", {"ma_col": "ema_21", "bearish": False}),
    ("pb_sma20", "pullback_reject", {"ma_col": "sma_20", "bearish": False}),
    ("pb_sma50", "pullback_reject", {"ma_col": "sma_50", "bearish": False}),

    # --- With bearish candle confirmation ---
    ("pb_ema13_bear", "pullback_reject", {"ma_col": "ema_13", "bearish": True}),
    ("pb_ema21_bear", "pullback_reject", {"ma_col": "ema_21", "bearish": True}),
    ("pb_sma20_bear", "pullback_reject", {"ma_col": "sma_20", "bearish": True}),
    ("pb_sma50_bear", "pullback_reject", {"ma_col": "sma_50", "bearish": True}),

    # --- Cross below (prev_close > MA, close < MA) ---
    ("pb_ema13_xbelow", "pullback_cross_below", {"ma_col": "ema_13", "bearish": False}),
    ("pb_ema21_xbelow", "pullback_cross_below", {"ma_col": "ema_21", "bearish": False}),
    ("pb_ema13_xbelow_bear", "pullback_cross_below", {"ma_col": "ema_13", "bearish": True}),

    # --- With engulfing confirmation ---
    ("pb_ema13_engulf", "pullback_engulf", {"ma_col": "ema_13"}),
    ("pb_ema21_engulf", "pullback_engulf", {"ma_col": "ema_21"}),

    # --- With lower high confirmation ---
    ("pb_ema13_lohi", "pullback_lower_high", {"ma_col": "ema_13"}),
    ("pb_ema21_lohi", "pullback_lower_high", {"ma_col": "ema_21"}),

    # --- With volume filter (weak pullback = low volume) ---
    ("pb_ema13_lowvol", "pullback_low_volume", {"ma_col": "ema_13"}),
    ("pb_ema21_lowvol", "pullback_low_volume", {"ma_col": "ema_21"}),

    # --- Combo: bearish + lower high ---
    ("pb_ema13_bear_lohi", "pullback_bear_lohi", {"ma_col": "ema_13"}),
    ("pb_ema21_bear_lohi", "pullback_bear_lohi", {"ma_col": "ema_21"}),

    # --- 比較用: 前回有効だったパターン ---
    ("ref_ema_crossdown", "ema_crossdown", {"fast": 5, "slow": 13}),
    ("ref_engulf_no_bb", "engulfing_no_bb", {}),
]

# Exit設定（TP%/SL%）
EXIT_SETTINGS = [
    ("tp15_sl15", 1.5, 1.5),
    ("tp15_sl20", 1.5, 2.0),
    ("tp20_sl15", 2.0, 1.5),
    ("tp20_sl20", 2.0, 2.0),
    ("tp10_sl15", 1.0, 1.5),
]


@app.function(
    cpu=4,
    memory=8192,
    timeout=600,
    volumes={"/data": vol_data},
)
def explore_symbol(
    symbol: str,
    period: str = "20250201-20260130",
    exec_tf: str = "1h",
    htf: str = "4h",
) -> Dict[str, Any]:
    """1銘柄のダウントレンド区間で戻り売りパターン探索"""
    import sys
    sys.path.insert(0, "/app")

    import numpy as np
    import pandas as pd

    from data.binance_loader import BinanceCSVLoader
    from analysis.trend import TrendDetector
    from indicators.registry import create_indicator

    loader = BinanceCSVLoader()
    inputdata_dir = Path("/data")

    # --- データ読み込み ---
    exec_path = inputdata_dir / f"{symbol}-{exec_tf}-{period}-merged.csv"
    htf_path = inputdata_dir / f"{symbol}-{htf}-{period}-merged.csv"

    for p in [exec_path, htf_path]:
        if not p.exists():
            return {"symbol": symbol, "status": "skipped", "reason": f"{p.name} not found"}

    exec_ohlcv = loader.load(str(exec_path), symbol=symbol)
    htf_ohlcv = loader.load(str(htf_path), symbol=symbol)

    exec_df = exec_ohlcv.df.copy()
    htf_df = htf_ohlcv.df.copy()

    # --- レジーム検出（MA Cross on 4h → label to 1h）---
    detector = TrendDetector()
    htf_df = detector.detect_ma_cross(htf_df, fast_period=20, slow_period=50)
    exec_df = TrendDetector.label_execution_tf(exec_df, htf_df)

    total_bars = len(exec_df)
    dt_mask = exec_df["trend_regime"] == "downtrend"
    dt_bars = dt_mask.sum()

    if dt_bars < 50:
        return {
            "symbol": symbol,
            "status": "skipped",
            "reason": f"DT bars too few: {dt_bars}",
            "total_bars": total_bars,
            "dt_bars": int(dt_bars),
        }

    # --- インジケーター計算 ---
    df = exec_df.copy()

    # ATR
    atr_ind = create_indicator("atr", period=14)
    df = atr_ind.calculate(df)

    # EMA
    for ep in [5, 13, 21]:
        ema_ind = create_indicator("ema", period=ep)
        df = ema_ind.calculate(df)

    # SMA
    for sp in [20, 50]:
        sma_ind = create_indicator("sma", period=sp)
        df = sma_ind.calculate(df)

    # Volume MA
    df["vol_ma20"] = df["volume"].rolling(20).mean()

    # --- 事前計算 ---
    o = df["open"].values
    h = df["high"].values
    lo = df["low"].values
    cl = df["close"].values
    vol = df["volume"].values
    atr = df["atr_14"].values

    is_bearish = cl < o

    prev_o = np.roll(o, 1); prev_o[0] = np.nan
    prev_cl = np.roll(cl, 1); prev_cl[0] = np.nan
    prev_h = np.roll(h, 1); prev_h[0] = np.nan
    prev_is_bull = prev_cl > prev_o

    vol_ma = df["vol_ma20"].values

    # --- シグナル生成関数 ---
    def pullback_reject(params):
        """EMA/SMA touch rejection: high >= MA AND close < MA"""
        ma_col = params["ma_col"]
        need_bearish = params.get("bearish", False)
        if ma_col not in df.columns:
            return np.zeros(len(df), dtype=bool)
        ma = df[ma_col].values
        sig = (h >= ma) & (cl < ma)
        if need_bearish:
            sig = sig & is_bearish
        return np.where(np.isnan(ma), False, sig)

    def pullback_cross_below(params):
        """Cross below: prev_close > MA AND close < MA"""
        ma_col = params["ma_col"]
        need_bearish = params.get("bearish", False)
        if ma_col not in df.columns:
            return np.zeros(len(df), dtype=bool)
        ma = df[ma_col].values
        sig = (prev_cl > ma) & (cl < ma)
        if need_bearish:
            sig = sig & is_bearish
        return np.where(np.isnan(ma) | np.isnan(prev_cl), False, sig)

    def pullback_engulf(params):
        """Pullback rejection + bearish engulfing"""
        ma_col = params["ma_col"]
        if ma_col not in df.columns:
            return np.zeros(len(df), dtype=bool)
        ma = df[ma_col].values
        # EMA rejection
        reject = (h >= ma) & (cl < ma) & is_bearish
        # Engulfing: prev is bull, current close <= prev open
        engulf = prev_is_bull & (cl <= prev_o)
        sig = reject & engulf
        return np.where(np.isnan(ma) | np.isnan(prev_cl), False, sig)

    def pullback_lower_high(params):
        """Pullback rejection + lower high (current high < prev high)"""
        ma_col = params["ma_col"]
        if ma_col not in df.columns:
            return np.zeros(len(df), dtype=bool)
        ma = df[ma_col].values
        reject = (h >= ma) & (cl < ma) & is_bearish
        lower_h = h < prev_h
        sig = reject & lower_h
        return np.where(np.isnan(ma) | np.isnan(prev_h), False, sig)

    def pullback_low_volume(params):
        """Pullback rejection + low volume (vol < vol_ma = weak pullback)"""
        ma_col = params["ma_col"]
        if ma_col not in df.columns:
            return np.zeros(len(df), dtype=bool)
        ma = df[ma_col].values
        reject = (h >= ma) & (cl < ma) & is_bearish
        low_vol = vol < vol_ma
        sig = reject & low_vol
        return np.where(np.isnan(ma) | np.isnan(vol_ma), False, sig)

    def pullback_bear_lohi(params):
        """Pullback rejection + bearish + lower high (combo)"""
        ma_col = params["ma_col"]
        if ma_col not in df.columns:
            return np.zeros(len(df), dtype=bool)
        ma = df[ma_col].values
        reject = (h >= ma) & (cl < ma) & is_bearish
        lower_h = h < prev_h
        sig = reject & lower_h
        return np.where(np.isnan(ma) | np.isnan(prev_h), False, sig)

    def ema_crossdown(params):
        """比較用: EMAクロスダウン"""
        fast_col = f"ema_{params['fast']}"
        slow_col = f"ema_{params['slow']}"
        if fast_col not in df.columns or slow_col not in df.columns:
            return np.zeros(len(df), dtype=bool)
        fast = df[fast_col].values
        slow = df[slow_col].values
        fast_prev = np.roll(fast, 1); fast_prev[0] = np.nan
        slow_prev = np.roll(slow, 1); slow_prev[0] = np.nan
        sig = (fast_prev >= slow_prev) & (fast < slow)
        return np.where(np.isnan(fast) | np.isnan(slow), False, sig)

    def engulfing_no_bb(params):
        """比較用: 包み陰線（BBなし）"""
        body = np.abs(cl - o)
        engulf = is_bearish & prev_is_bull & (cl <= prev_o)
        body_filter = body >= atr * 0.3
        sig = engulf & body_filter
        return np.where(np.isnan(atr) | np.isnan(prev_cl), False, sig)

    signal_funcs = {
        "pullback_reject": pullback_reject,
        "pullback_cross_below": pullback_cross_below,
        "pullback_engulf": pullback_engulf,
        "pullback_lower_high": pullback_lower_high,
        "pullback_low_volume": pullback_low_volume,
        "pullback_bear_lohi": pullback_bear_lohi,
        "ema_crossdown": ema_crossdown,
        "engulfing_no_bb": engulfing_no_bb,
    }

    # --- TP/SLシミュレーション（ショート） ---
    def simulate_trades(entry_mask, tp_pct, sl_pct, max_hold=72):
        entries = np.where(entry_mask)[0]
        if len(entries) == 0:
            return {"trades": 0, "wins": 0, "losses": 0, "timeouts": 0,
                    "pnl_total": 0.0, "pnl_avg": 0.0, "win_rate": 0.0,
                    "max_dd": 0.0}

        tp_mult = tp_pct / 100.0
        sl_mult = sl_pct / 100.0

        wins = 0
        losses = 0
        timeouts = 0
        pnl_list = []
        n = len(df)
        last_exit = -1

        for idx in entries:
            if idx <= last_exit:
                continue
            entry_price = cl[idx]
            if entry_price <= 0 or np.isnan(entry_price):
                continue

            tp_price = entry_price * (1 - tp_mult)
            sl_price = entry_price * (1 + sl_mult)

            resolved = False
            for j in range(idx + 1, min(idx + max_hold + 1, n)):
                if lo[j] <= tp_price:
                    wins += 1
                    pnl_list.append(tp_pct)
                    last_exit = j
                    resolved = True
                    break
                if h[j] >= sl_price:
                    losses += 1
                    pnl_list.append(-sl_pct)
                    last_exit = j
                    resolved = True
                    break

            if not resolved:
                end_idx = min(idx + max_hold, n - 1)
                exit_price = cl[end_idx]
                pnl = (entry_price - exit_price) / entry_price * 100
                timeouts += 1
                pnl_list.append(pnl)
                last_exit = end_idx

        trades = len(pnl_list)
        if trades == 0:
            return {"trades": 0, "wins": 0, "losses": 0, "timeouts": 0,
                    "pnl_total": 0.0, "pnl_avg": 0.0, "win_rate": 0.0,
                    "max_dd": 0.0}

        pnl_arr = np.array(pnl_list)
        cumsum = np.cumsum(pnl_arr)
        running_max = np.maximum.accumulate(cumsum)
        drawdowns = running_max - cumsum
        max_dd = drawdowns.max() if len(drawdowns) > 0 else 0.0

        return {
            "trades": trades,
            "wins": wins,
            "losses": losses,
            "timeouts": timeouts,
            "pnl_total": float(pnl_arr.sum()),
            "pnl_avg": float(pnl_arr.mean()),
            "win_rate": float(wins / trades * 100) if trades > 0 else 0.0,
            "max_dd": float(max_dd),
        }

    # --- 全パターン × Exit のテスト ---
    dt_indices = np.where(dt_mask.values)[0]
    results = []

    for pname, func_name, params in PATTERNS:
        func = signal_funcs.get(func_name)
        if func is None:
            continue

        raw_signal = func(params)

        # ダウントレンド区間のみ
        dt_signal = np.zeros(len(df), dtype=bool)
        dt_signal[dt_indices] = raw_signal[dt_indices]

        total_signals = dt_signal.sum()
        if total_signals == 0:
            for exit_name, tp, sl in EXIT_SETTINGS:
                results.append({
                    "pattern": pname, "exit": exit_name,
                    "signals": 0, "trades": 0, "wins": 0, "losses": 0,
                    "timeouts": 0, "pnl_total": 0.0, "pnl_avg": 0.0,
                    "win_rate": 0.0, "max_dd": 0.0,
                })
            continue

        for exit_name, tp, sl in EXIT_SETTINGS:
            stats = simulate_trades(dt_signal, tp, sl)
            results.append({
                "pattern": pname, "exit": exit_name,
                "signals": int(total_signals), **stats,
            })

    return {
        "symbol": symbol,
        "status": "done",
        "total_bars": total_bars,
        "dt_bars": int(dt_bars),
        "dt_pct": round(dt_bars / total_bars * 100, 1),
        "results": results,
    }


@app.function(
    cpu=1, memory=512, timeout=60,
    volumes={"/data": vol_data},
)
def scan_symbols(period: str, exec_tf: str, htf: str) -> List[str]:
    """Volume内の利用可能銘柄をスキャン"""
    data_dir = Path("/data")
    symbols = set()
    for f in data_dir.glob(f"*-{exec_tf}-{period}-merged.csv"):
        sym = f.name.split(f"-{exec_tf}")[0]
        htf_file = data_dir / f"{sym}-{htf}-{period}-merged.csv"
        if htf_file.exists():
            symbols.add(sym)
    return sorted(symbols)


@app.local_entrypoint()
def main(
    symbols: str = "",
    period: str = "20250201-20260130",
    exec_tf: str = "1h",
    htf: str = "4h",
):
    """戻り売りパターン探索（全銘柄並列）"""
    import json

    if symbols:
        symbol_list = [s.strip() for s in symbols.split(",")]
    else:
        symbol_list = scan_symbols.remote(period, exec_tf, htf)
        if not symbol_list:
            print("ERROR: No data found")
            return

    print("=" * 70)
    print("  Prism Pullback Short Explorer (Modal)")
    print("=" * 70)
    print(f"Symbols: {len(symbol_list)} {symbol_list}")
    print(f"TF: {exec_tf}:{htf} (MA Cross regime)")
    print(f"Patterns: {len(PATTERNS)}")
    print(f"Exit settings: {len(EXIT_SETTINGS)}")
    print(f"Tests per symbol: {len(PATTERNS) * len(EXIT_SETTINGS)}")
    print()

    t0 = time.time()

    # 全銘柄並列実行
    jobs = []
    for sym in symbol_list:
        jobs.append(explore_symbol.spawn(
            symbol=sym, period=period,
            exec_tf=exec_tf, htf=htf,
        ))

    all_results = []
    done = 0
    for job in jobs:
        result = job.get()
        done += 1
        sym = result["symbol"]
        status = result["status"]
        if status == "done":
            dt_pct = result["dt_pct"]
            n_res = len(result["results"])
            print(f"  [{done}/{len(jobs)}] {sym}: DT={dt_pct}%, {n_res} tests")
            all_results.append(result)
        else:
            reason = result.get("reason", "unknown")
            print(f"  [{done}/{len(jobs)}] {sym}: SKIP - {reason}")

    elapsed = time.time() - t0
    print(f"\nTime: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"Valid symbols: {len(all_results)}/{len(symbol_list)}")

    if not all_results:
        print("No valid data")
        return

    # === 集約分析 ===
    print("\n" + "=" * 70)
    print("  Pullback Short Results")
    print("=" * 70)

    from collections import defaultdict
    agg = defaultdict(lambda: {
        "symbols": 0, "trades_total": 0, "wins_total": 0,
        "pnl_sum": 0.0, "pnl_list": [], "dd_list": [],
        "signal_list": [],
    })

    for res in all_results:
        for r in res["results"]:
            key = (r["pattern"], r["exit"])
            a = agg[key]
            if r["trades"] > 0:
                a["symbols"] += 1
            a["trades_total"] += r["trades"]
            a["wins_total"] += r["wins"]
            a["pnl_sum"] += r["pnl_total"]
            a["pnl_list"].append(r["pnl_total"])
            a["dd_list"].append(r["max_dd"])
            a["signal_list"].append(r["signals"])

    # --- ベストExit (tp15_sl20) でのパターン比較 ---
    default_exit = "tp15_sl20"
    print(f"\n--- Default Exit ({default_exit}) Pattern Comparison ---")
    print(f"{'Pattern':<25} {'Syms':>4} {'Trades':>6} {'WR%':>6} {'PnL%':>8} {'AvgPnL':>8} {'MaxDD':>7}")
    print("-" * 70)

    pattern_scores = []
    for pname, _, _ in PATTERNS:
        key = (pname, default_exit)
        a = agg[key]
        trades = a["trades_total"]
        wins = a["wins_total"]
        wr = wins / trades * 100 if trades > 0 else 0
        pnl = a["pnl_sum"]
        avg_pnl = pnl / a["symbols"] if a["symbols"] > 0 else 0
        max_dd = max(a["dd_list"]) if a["dd_list"] else 0
        print(f"{pname:<25} {a['symbols']:>4} {trades:>6} {wr:>5.1f}% {pnl:>+7.1f}% {avg_pnl:>+7.1f}% {max_dd:>6.1f}%")
        pattern_scores.append((pname, trades, wr, pnl, avg_pnl, a["symbols"]))

    # --- 上位パターンの Exit 比較 ---
    print("\n\n--- Top Patterns: Exit Comparison ---")
    by_pnl = sorted(pattern_scores, key=lambda x: x[3], reverse=True)

    top_patterns = []
    seen = set()
    for item in by_pnl[:8]:
        if item[0] not in seen:
            top_patterns.append(item[0])
            seen.add(item[0])

    for pname in top_patterns:
        print(f"\n  {pname}:")
        print(f"  {'Exit':<12} {'Syms':>4} {'Trades':>6} {'WR%':>6} {'PnL%':>8} {'AvgPnL':>8} {'MaxDD':>7}")
        for exit_name, _, _ in EXIT_SETTINGS:
            key = (pname, exit_name)
            a = agg[key]
            trades = a["trades_total"]
            wins = a["wins_total"]
            wr = wins / trades * 100 if trades > 0 else 0
            pnl = a["pnl_sum"]
            avg_pnl = pnl / a["symbols"] if a["symbols"] > 0 else 0
            max_dd = max(a["dd_list"]) if a["dd_list"] else 0
            print(f"  {exit_name:<12} {a['symbols']:>4} {trades:>6} {wr:>5.1f}% {pnl:>+7.1f}% {avg_pnl:>+7.1f}% {max_dd:>6.1f}%")

    # --- 銘柄別ベスト ---
    print(f"\n\n--- Best Pattern per Symbol ({default_exit}) ---")
    print(f"{'Symbol':<12} {'Pattern':<25} {'Trades':>6} {'WR%':>6} {'PnL%':>8}")
    print("-" * 60)
    for res in sorted(all_results, key=lambda x: x["symbol"]):
        sym = res["symbol"]
        best = None
        best_pnl = -999
        for r in res["results"]:
            if r["exit"] == default_exit and r["pnl_total"] > best_pnl and r["trades"] >= 3:
                best_pnl = r["pnl_total"]
                best = r
        if best:
            print(f"{sym:<12} {best['pattern']:<25} {best['trades']:>6} {best['win_rate']:>5.1f}% {best['pnl_total']:>+7.1f}%")
        else:
            print(f"{sym:<12} {'(no trades)':>25}")

    # --- PnLプラスの銘柄×パターン全列挙 ---
    print(f"\n\n--- All Profitable Symbol x Pattern ({default_exit}) ---")
    print(f"{'Symbol':<12} {'Pattern':<25} {'Trades':>6} {'WR%':>6} {'PnL%':>8}")
    print("-" * 60)
    profitable = []
    for res in sorted(all_results, key=lambda x: x["symbol"]):
        sym = res["symbol"]
        for r in res["results"]:
            if r["exit"] == default_exit and r["pnl_total"] > 0 and r["trades"] >= 3:
                profitable.append((sym, r))
    profitable.sort(key=lambda x: x[1]["pnl_total"], reverse=True)
    for sym, r in profitable[:30]:
        print(f"{sym:<12} {r['pattern']:<25} {r['trades']:>6} {r['win_rate']:>5.1f}% {r['pnl_total']:>+7.1f}%")

    # JSON保存
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "exec_tf": exec_tf,
            "htf": htf,
            "regime": "MA Cross",
            "period": period,
            "n_symbols": len(all_results),
            "patterns": len(PATTERNS),
            "exits": len(EXIT_SETTINGS),
        },
        "symbol_results": all_results,
    }

    out_path = Path("pullback_explore_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out_path}")

    print(f"\n{'=' * 70}")
    print(f"  Done! {elapsed:.0f}s")
    print(f"{'=' * 70}")
