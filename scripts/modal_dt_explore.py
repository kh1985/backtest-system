"""
ダウントレンド向けパターン探索（Modal上で実行）

多数のエントリーシグナルバリエーション × Exit設定を
全銘柄のダウントレンド区間でテストし、有望パターンを特定する。

使い方:
    set PYTHONIOENCODING=utf-8 && py -3 -m modal run scripts/modal_dt_explore.py
    set PYTHONIOENCODING=utf-8 && py -3 -m modal run scripts/modal_dt_explore.py --symbols BTCUSDT,ETHUSDT
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

app = modal.App("prism-dt-explore", image=image)


# ====== パターン定義 ======
# 各パターンは (名前, シグナル生成関数名, パラメータ) のタプル
PATTERNS = [
    # --- Group A: Engulfing系 ---
    ("eng_strict_bb_upper", "engulfing_signal", {"strict": True, "bb_filter": "upper"}),
    ("eng_strict_bb_mid", "engulfing_signal", {"strict": True, "bb_filter": "middle"}),
    ("eng_strict_no_bb", "engulfing_signal", {"strict": True, "bb_filter": None}),
    ("eng_relaxed_bb_upper", "engulfing_signal", {"strict": False, "bb_filter": "upper"}),
    ("eng_relaxed_bb_mid", "engulfing_signal", {"strict": False, "bb_filter": "middle"}),
    ("eng_relaxed_no_bb", "engulfing_signal", {"strict": False, "bb_filter": None}),
    ("eng_relaxed_no_bb_body03", "engulfing_signal", {"strict": False, "bb_filter": None, "body_atr": 0.3}),
    ("eng_relaxed_bb_mid_body03", "engulfing_signal", {"strict": False, "bb_filter": "middle", "body_atr": 0.3}),

    # --- Group B: ReversalHigh系 ---
    ("rh_strict", "reversal_high_signal", {"lookback": 3, "wick_ratio": 2.0, "atr_mult": 1.5}),
    ("rh_medium", "reversal_high_signal", {"lookback": 3, "wick_ratio": 1.5, "atr_mult": 1.0}),
    ("rh_relaxed", "reversal_high_signal", {"lookback": 4, "wick_ratio": 1.0, "atr_mult": 0.5}),
    ("rh_lb2_medium", "reversal_high_signal", {"lookback": 2, "wick_ratio": 1.5, "atr_mult": 1.0}),

    # --- Group C: WickFill系 ---
    ("wf_base_ema13", "wick_fill_signal", {"lookback": 3, "wick_ratio": 1.5, "ema_period": 13}),
    ("wf_base_ema5", "wick_fill_signal", {"lookback": 3, "wick_ratio": 1.5, "ema_period": 5}),
    ("wf_lb5_ema13", "wick_fill_signal", {"lookback": 5, "wick_ratio": 1.5, "ema_period": 13}),
    ("wf_rsi_filter", "wick_fill_rsi_signal", {"lookback": 3, "wick_ratio": 1.5, "ema_period": 13, "rsi_min": 55}),
    ("wf_rsi50_filter", "wick_fill_rsi_signal", {"lookback": 3, "wick_ratio": 1.5, "ema_period": 13, "rsi_min": 50}),

    # --- Group D: シンプルパターン ---
    ("shooting_star", "shooting_star_signal", {"wick_mult": 2.0}),
    ("shooting_star_relaxed", "shooting_star_signal", {"wick_mult": 1.5}),
    ("ema_cross_down", "ema_crossdown_signal", {"fast": 5, "slow": 13}),
    ("bb_upper_reject", "bb_upper_reject_signal", {"bb_period": 20}),
    ("bearish_marubozu", "marubozu_signal", {"body_atr": 1.0, "wick_pct": 0.1}),
]

# Exit設定（TP%/SL%）
EXIT_SETTINGS = [
    ("tp15_sl15", 1.5, 1.5),
    ("tp20_sl15", 2.0, 1.5),
    ("tp15_sl20", 1.5, 2.0),
    ("tp20_sl20", 2.0, 2.0),
    ("tp30_sl20", 3.0, 2.0),
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
    exec_tf: str = "15m",
    htf: str = "1h",
    super_htf: str = "4h",
) -> Dict[str, Any]:
    """1銘柄のダウントレンド区間でパターン探索"""
    import sys
    sys.path.insert(0, "/app")

    import numpy as np
    import pandas as pd
    from pathlib import Path

    from data.binance_loader import BinanceCSVLoader
    from analysis.trend import TrendDetector
    from indicators.registry import create_indicator

    loader = BinanceCSVLoader()
    inputdata_dir = Path("/data")

    # --- データ読み込み ---
    exec_path = inputdata_dir / f"{symbol}-{exec_tf}-{period}-merged.csv"
    htf_path = inputdata_dir / f"{symbol}-{htf}-{period}-merged.csv"
    super_htf_path = inputdata_dir / f"{symbol}-{super_htf}-{period}-merged.csv"

    for p in [exec_path, htf_path, super_htf_path]:
        if not p.exists():
            return {"symbol": symbol, "status": "skipped", "reason": f"{p.name} not found"}

    exec_ohlcv = loader.load(str(exec_path), symbol=symbol)
    htf_ohlcv = loader.load(str(htf_path), symbol=symbol)
    super_htf_ohlcv = loader.load(str(super_htf_path), symbol=symbol)

    exec_df = exec_ohlcv.df.copy()
    htf_df = htf_ohlcv.df.copy()
    super_htf_df = super_htf_ohlcv.df.copy()

    # --- レジーム検出 ---
    detector = TrendDetector()
    htf_df = detector.detect_dual_tf_ema(htf_df, super_htf_df, fast_period=20, slow_period=50)
    exec_df = TrendDetector.label_execution_tf(exec_df, htf_df)

    total_bars = len(exec_df)
    dt_mask = exec_df["trend_regime"] == "downtrend"
    dt_bars = dt_mask.sum()

    if dt_bars < 100:
        return {
            "symbol": symbol,
            "status": "skipped",
            "reason": f"DT bars too few: {dt_bars}",
            "total_bars": total_bars,
            "dt_bars": int(dt_bars),
        }

    # --- インジケーター計算（全体に対して） ---
    df = exec_df.copy()

    # ATR
    atr_ind = create_indicator("atr", period=14)
    df = atr_ind.calculate(df)

    # BB
    for bp in [15, 20]:
        bb_ind = create_indicator("bollinger", period=bp, std_dev=2.0)
        df = bb_ind.calculate(df)

    # EMA
    for ep in [5, 13]:
        ema_ind = create_indicator("ema", period=ep)
        df = ema_ind.calculate(df)

    # RSI
    rsi_ind = create_indicator("rsi", period=14)
    df = rsi_ind.calculate(df)

    # Rally
    from indicators.candle_pattern import RallyCalc, WickSpikeCalc
    for lb in [2, 3, 4]:
        rc = RallyCalc(lookback=lb)
        df = rc.calculate(df)

    # WickSpike
    for lb in [3, 5]:
        for wr in [1.5, 2.0]:
            ws = WickSpikeCalc(lookback=lb, wick_ratio=wr)
            df = ws.calculate(df)

    # --- 事前計算カラム ---
    o = df["open"].values
    h = df["high"].values
    lo = df["low"].values
    cl = df["close"].values
    atr = df["atr_14"].values

    body_top = np.maximum(o, cl)
    body_bot = np.minimum(o, cl)
    body = body_top - body_bot
    upper_wick = h - body_top
    lower_wick = body_bot - lo
    is_bearish = cl < o

    # --- シグナル生成関数 ---
    def engulfing_signal(params):
        strict = params.get("strict", True)
        bb_filter = params.get("bb_filter", None)
        body_atr = params.get("body_atr", 0.5)

        po = df["open"].shift(1)
        pc = df["close"].shift(1)
        is_prev_bull = pc > po

        if strict:
            # open >= prev_close AND close <= prev_open
            engulf = is_bearish & is_prev_bull & (df["open"] >= pc) & (df["close"] <= po)
        else:
            # relaxed: close <= prev_open のみ
            engulf = is_bearish & is_prev_bull & (df["close"] <= po)

        # body filter
        body_s = (df["open"] - df["close"]).abs()
        engulf = engulf & (body_s >= df["atr_14"] * body_atr)

        if bb_filter == "upper":
            engulf = engulf & (df["high"] >= df["bb_upper_20"])
        elif bb_filter == "middle":
            engulf = engulf & (df["close"].shift(1) > df["bb_middle_20"])

        return engulf.fillna(False).values

    def reversal_high_signal(params):
        lookback = params["lookback"]
        wick_ratio = params["wick_ratio"]
        atr_mult = params["atr_mult"]

        rally_col = f"_rally_{lookback}"
        if rally_col not in df.columns:
            return np.zeros(len(df), dtype=bool)

        rally = df[rally_col].values
        denom = body + lower_wick
        with np.errstate(divide="ignore", invalid="ignore"):
            wr = np.where(denom > 0, upper_wick / denom, 0)

        sig = is_bearish & (rally >= atr * atr_mult) & (wr >= wick_ratio)
        return np.where(np.isnan(rally) | np.isnan(atr), False, sig)

    def wick_fill_signal(params):
        lookback = params["lookback"]
        wick_ratio = params["wick_ratio"]
        ema_period = params["ema_period"]

        spike_body_col = f"_spike_body_top_{lookback}"
        ema_col = f"ema_{ema_period}"

        if spike_body_col not in df.columns or ema_col not in df.columns:
            return np.zeros(len(df), dtype=bool)

        spike_body = df[spike_body_col].values
        ema = df[ema_col].values
        ema_prev = np.roll(ema, 1)
        ema_prev[0] = np.nan

        has_spike = ~np.isnan(spike_body)
        reach = h >= spike_body
        ema_down = ema < ema_prev

        sig = is_bearish & has_spike & reach & ema_down
        return np.where(np.isnan(spike_body) | np.isnan(ema), False, sig)

    def wick_fill_rsi_signal(params):
        base_sig = wick_fill_signal(params)
        rsi_min = params.get("rsi_min", 55)
        rsi_vals = df["rsi_14"].values
        rsi_ok = rsi_vals >= rsi_min
        return base_sig & np.where(np.isnan(rsi_vals), False, rsi_ok)

    def shooting_star_signal(params):
        wick_mult = params["wick_mult"]
        # upper_wick > body * wick_mult, bearish, small lower_wick
        sig = is_bearish & (upper_wick > body * wick_mult) & (lower_wick < body * 0.5)
        return np.where(np.isnan(atr), False, sig)

    def ema_crossdown_signal(params):
        fast_p = params["fast"]
        slow_p = params["slow"]
        fast_col = f"ema_{fast_p}"
        slow_col = f"ema_{slow_p}"
        if fast_col not in df.columns or slow_col not in df.columns:
            return np.zeros(len(df), dtype=bool)
        fast = df[fast_col].values
        slow = df[slow_col].values
        fast_prev = np.roll(fast, 1)
        fast_prev[0] = np.nan
        slow_prev = np.roll(slow, 1)
        slow_prev[0] = np.nan
        # クロスダウン: 前回fast>=slow → 今回fast<slow
        sig = (fast_prev >= slow_prev) & (fast < slow)
        return np.where(np.isnan(fast) | np.isnan(slow), False, sig)

    def bb_upper_reject_signal(params):
        bp = params["bb_period"]
        upper_col = f"bb_upper_{bp}"
        if upper_col not in df.columns:
            return np.zeros(len(df), dtype=bool)
        bb_upper = df[upper_col].values
        # high >= BB upper + bearish
        sig = is_bearish & (h >= bb_upper)
        return np.where(np.isnan(bb_upper), False, sig)

    def marubozu_signal(params):
        body_atr_mult = params["body_atr"]
        wick_pct = params["wick_pct"]
        # 大陰線: body >= ATR * mult, upper+lower wick < body * wick_pct
        total_wick = upper_wick + lower_wick
        sig = is_bearish & (body >= atr * body_atr_mult) & (total_wick < body * wick_pct)
        return np.where(np.isnan(atr), False, sig)

    signal_funcs = {
        "engulfing_signal": engulfing_signal,
        "reversal_high_signal": reversal_high_signal,
        "wick_fill_signal": wick_fill_signal,
        "wick_fill_rsi_signal": wick_fill_rsi_signal,
        "shooting_star_signal": shooting_star_signal,
        "ema_crossdown_signal": ema_crossdown_signal,
        "bb_upper_reject_signal": bb_upper_reject_signal,
        "marubozu_signal": marubozu_signal,
    }

    # --- TP/SLシミュレーション（ベクトル化） ---
    def simulate_trades(entry_mask, tp_pct, sl_pct, max_hold=96):
        """エントリーシグナルに対してTP/SLの勝敗をベクトル的に計算（ショート）"""
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
        equity = 0.0
        peak = 0.0
        max_dd = 0.0

        n = len(df)
        last_exit = -1

        for idx in entries:
            if idx <= last_exit:
                continue  # 前のトレードがまだ終わっていない
            entry_price = cl[idx]
            if entry_price <= 0 or np.isnan(entry_price):
                continue

            tp_price = entry_price * (1 - tp_mult)
            sl_price = entry_price * (1 + sl_mult)

            resolved = False
            for j in range(idx + 1, min(idx + max_hold + 1, n)):
                # ショート: low <= tp_price → Win
                if lo[j] <= tp_price:
                    wins += 1
                    pnl_list.append(tp_pct)
                    last_exit = j
                    resolved = True
                    break
                # ショート: high >= sl_price → Loss
                if h[j] >= sl_price:
                    losses += 1
                    pnl_list.append(-sl_pct)
                    last_exit = j
                    resolved = True
                    break

            if not resolved:
                # タイムアウト: 最終バーの終値で決済
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

        # シグナル生成
        raw_signal = func(params)

        # ダウントレンド区間のみに絞る
        dt_signal = np.zeros(len(df), dtype=bool)
        dt_signal[dt_indices] = raw_signal[dt_indices]

        total_signals = dt_signal.sum()
        if total_signals == 0:
            for exit_name, tp, sl in EXIT_SETTINGS:
                results.append({
                    "pattern": pname,
                    "exit": exit_name,
                    "signals": 0,
                    "trades": 0,
                    "wins": 0,
                    "losses": 0,
                    "timeouts": 0,
                    "pnl_total": 0.0,
                    "pnl_avg": 0.0,
                    "win_rate": 0.0,
                    "max_dd": 0.0,
                })
            continue

        for exit_name, tp, sl in EXIT_SETTINGS:
            stats = simulate_trades(dt_signal, tp, sl)
            results.append({
                "pattern": pname,
                "exit": exit_name,
                "signals": int(total_signals),
                **stats,
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
    cpu=1,
    memory=512,
    timeout=60,
    volumes={"/data": vol_data},
)
def scan_symbols(period: str, exec_tf: str, htf: str, super_htf: str) -> List[str]:
    """Volume内の利用可能銘柄をスキャン"""
    from pathlib import Path
    data_dir = Path("/data")
    symbols = set()
    for f in data_dir.glob(f"*-{exec_tf}-{period}-merged.csv"):
        sym = f.name.split(f"-{exec_tf}")[0]
        htf_file = data_dir / f"{sym}-{htf}-{period}-merged.csv"
        super_file = data_dir / f"{sym}-{super_htf}-{period}-merged.csv"
        if htf_file.exists() and super_file.exists():
            symbols.add(sym)
    return sorted(symbols)


@app.local_entrypoint()
def main(
    symbols: str = "",
    period: str = "20250201-20260130",
    exec_tf: str = "15m",
    htf: str = "1h",
    super_htf: str = "4h",
):
    """ダウントレンドパターン探索（全銘柄並列）"""
    import json

    # 銘柄リスト
    if symbols:
        symbol_list = [s.strip() for s in symbols.split(",")]
    else:
        symbol_list = scan_symbols.remote(period, exec_tf, htf, super_htf)
        if not symbol_list:
            print("ERROR: データなし")
            return

    print("=" * 70)
    print("  Prism DT Pattern Explorer (Modal)")
    print("=" * 70)
    print(f"銘柄: {len(symbol_list)}個 {symbol_list}")
    print(f"TF: {exec_tf}:{htf} (super: {super_htf})")
    print(f"パターン: {len(PATTERNS)}種")
    print(f"Exit設定: {len(EXIT_SETTINGS)}種")
    print(f"テスト総数: {len(PATTERNS) * len(EXIT_SETTINGS)} per symbol")
    print()

    t0 = time.time()

    # 全銘柄並列実行
    jobs = []
    for sym in symbol_list:
        jobs.append(explore_symbol.spawn(
            symbol=sym, period=period,
            exec_tf=exec_tf, htf=htf, super_htf=super_htf,
        ))

    # 結果収集
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
    print(f"\n実行時間: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"有効銘柄: {len(all_results)}/{len(symbol_list)}")

    if not all_results:
        print("有効データなし")
        return

    # === 集約分析 ===
    print("\n" + "=" * 70)
    print("  集約結果: パターン × Exit")
    print("=" * 70)

    # パターン×Exit別に集約
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

    # サマリーテーブル（tp20_sl15をデフォルトで表示）
    print("\n--- デフォルトExit (tp20_sl15) でのパターン比較 ---")
    print(f"{'Pattern':<30} {'Syms':>4} {'Trades':>6} {'WR%':>6} {'PnL%':>8} {'AvgPnL':>8} {'MaxDD':>7}")
    print("-" * 75)

    default_exit = "tp20_sl15"
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
        total_sigs = sum(a["signal_list"])
        print(f"{pname:<30} {a['symbols']:>4} {trades:>6} {wr:>5.1f}% {pnl:>+7.1f}% {avg_pnl:>+7.1f}% {max_dd:>6.1f}%")
        pattern_scores.append((pname, trades, wr, pnl, avg_pnl, a["symbols"]))

    # ベストパターンを各Exitで詳細表示
    print("\n\n--- 上位パターンの Exit 比較 ---")
    # PnL上位5 + WR上位5 をユニークに取得
    by_pnl = sorted(pattern_scores, key=lambda x: x[3], reverse=True)
    by_wr = sorted(pattern_scores, key=lambda x: x[2] if x[1] >= 10 else 0, reverse=True)
    top_patterns = []
    seen = set()
    for item in by_pnl[:5] + by_wr[:5]:
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

    # 銘柄別ベスト
    print("\n\n--- 銘柄別ベストパターン (tp20_sl15) ---")
    print(f"{'Symbol':<12} {'Pattern':<30} {'Trades':>6} {'WR%':>6} {'PnL%':>8}")
    print("-" * 65)
    for res in sorted(all_results, key=lambda x: x["symbol"]):
        sym = res["symbol"]
        best = None
        best_pnl = -999
        for r in res["results"]:
            if r["exit"] == default_exit and r["pnl_total"] > best_pnl and r["trades"] >= 3:
                best_pnl = r["pnl_total"]
                best = r
        if best:
            wr = best["win_rate"]
            print(f"{sym:<12} {best['pattern']:<30} {best['trades']:>6} {wr:>5.1f}% {best['pnl_total']:>+7.1f}%")
        else:
            print(f"{sym:<12} {'(no trades)':>30}")

    # JSON保存
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "exec_tf": exec_tf,
            "htf": htf,
            "super_htf": super_htf,
            "period": period,
            "n_symbols": len(all_results),
            "patterns": len(PATTERNS),
            "exits": len(EXIT_SETTINGS),
        },
        "symbol_results": all_results,
    }

    out_path = Path("dt_explore_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n結果保存: {out_path}")

    print(f"\n{'=' * 70}")
    print(f"  完了! {elapsed:.0f}s")
    print(f"{'=' * 70}")
