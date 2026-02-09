"""
Numba JIT 最適化バックテストループ

エントリーシグナルは pandas ベクトル演算で一括計算し、
ポジション管理ループ（TP/SL/トレーリング/タイムアウト）のみを
Numba @njit でコンパイルして高速化する。

Numba 未インストール時は通常の Python ループにフォールバック。
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    def njit(func=None, cache=False):
        """Numba未インストール時のダミーデコレータ"""
        if func is not None:
            return func
        def wrapper(f):
            return f
        return wrapper


@njit(cache=True)
def _compute_tp_sl(
    entry_price: float,
    is_long: bool,
    tp_pct: float,
    sl_pct: float,
    use_atr_exit: bool,
    atr_val: float,
    atr_tp_mult: float,
    atr_sl_mult: float,
) -> Tuple[float, float]:
    """
    TP/SL価格を計算するヘルパー。

    ATRモード・固定%モード・SLなし（sl=0）を統一的に処理。
    """
    if use_atr_exit:
        # ATRベースTP/SL
        if is_long:
            tp_price = (entry_price + atr_val * atr_tp_mult) if atr_tp_mult > 0.0 else 1e18
            sl_price = (entry_price - atr_val * atr_sl_mult) if atr_sl_mult > 0.0 else -1.0
        else:
            tp_price = (entry_price - atr_val * atr_tp_mult) if atr_tp_mult > 0.0 else -1.0
            sl_price = (entry_price + atr_val * atr_sl_mult) if atr_sl_mult > 0.0 else 1e18
    else:
        # 固定%モード（sl_pct=0 → SLなし対応）
        if is_long:
            tp_price = entry_price * (1.0 + tp_pct / 100.0) if tp_pct > 0.0 else 1e18
            sl_price = entry_price * (1.0 - sl_pct / 100.0) if sl_pct > 0.0 else -1.0
        else:
            tp_price = entry_price * (1.0 - tp_pct / 100.0) if tp_pct > 0.0 else -1.0
            sl_price = entry_price * (1.0 + sl_pct / 100.0) if sl_pct > 0.0 else 1e18
    return tp_price, sl_price


@njit(cache=True)
def _backtest_loop(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    entry_signals: np.ndarray,
    regime_mask: np.ndarray,
    is_long: bool,
    tp_pct: float,
    sl_pct: float,
    trailing_pct: float,
    timeout_bars: int,
    commission_pct: float,
    slippage_pct: float,
    initial_capital: float,
    atr: np.ndarray,
    use_atr_exit: bool,
    atr_tp_mult: float,
    atr_sl_mult: float,
    bb_upper: np.ndarray,
    bb_lower: np.ndarray,
    use_bb_exit: bool,
    vwap_upper: np.ndarray,
    vwap_lower: np.ndarray,
    use_vwap_exit: bool,
    use_atr_trailing: bool,
    atr_trailing_mult: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Numba JIT コンパイルされたバックテストループ。

    Args:
        high, low, close: OHLC numpy配列
        entry_signals: エントリーシグナル（bool配列、事前にベクトル化済み）
        regime_mask: レジームフィルタ（bool配列、全True = フィルタなし）
        is_long: True=ロング、False=ショート
        tp_pct, sl_pct: TP/SLパーセンテージ（固定%モード）
        trailing_pct: トレーリングストップ%（0.0=無効、固定%モード）
        timeout_bars: タイムアウトバー数（0=無効）
        commission_pct: 手数料%
        slippage_pct: スリッページ%
        initial_capital: 初期資金
        atr: ATR配列（ATRベースexit時に使用。未使用時は長さ0の配列を渡す）
        use_atr_exit: ATRベースのTP/SL計算を使用
        atr_tp_mult: ATR TP倍率（0.0=TP無効）
        atr_sl_mult: ATR SL倍率（0.0=SL無効）
        bb_upper: BB上限バンド配列（BB exit時に使用。未使用時は長さ0の配列を渡す）
        bb_lower: BB下限バンド配列（BB exit時に使用。未使用時は長さ0の配列を渡す）
        use_bb_exit: True=BB帯で動的TP（ロング→上限、ショート→下限で決済）
        vwap_upper: VWAP上限バンド配列（VWAP exit時に使用。未使用時は長さ0の配列を渡す）
        vwap_lower: VWAP下限バンド配列（VWAP exit時に使用。未使用時は長さ0の配列を渡す）
        use_vwap_exit: True=VWAPバンドで動的TP（ロング→上限、ショート→下限で決済）
        use_atr_trailing: True=ATRベースのトレーリングストップ幅を使用
        atr_trailing_mult: ATRトレーリング倍率（use_atr_trailing=True時に使用）

    Returns:
        (profit_pcts, durations, equity_curve)
        - profit_pcts: 各トレードの損益% (float64)
        - durations: 各トレードの保有バー数 (int64)
        - equity_curve: エクイティカーブ (float64)
    """
    n = len(high)
    max_trades = n
    profit_pcts = np.empty(max_trades, dtype=np.float64)
    durations = np.empty(max_trades, dtype=np.int64)
    trade_count = 0

    # エクイティ追跡
    equity = initial_capital
    equity_curve = np.empty(max_trades + 1, dtype=np.float64)
    equity_curve[0] = equity

    # ポジション状態
    in_position = False
    entry_price = 0.0
    entry_idx = 0
    tp_price = 0.0
    sl_price = 0.0
    highest_price = 0.0
    lowest_price = 1e18
    trailing_dist = 0.0  # ATRベーストレーリング用の絶対値幅

    for i in range(1, n):
        # === 決済判定 ===
        if in_position:
            h = high[i]
            l = low[i]
            c = close[i]

            # BB動的exit: 毎バーTP価格をBB帯で更新
            if use_bb_exit and len(bb_upper) > i:
                if is_long:
                    bb_val = bb_upper[i]
                    if bb_val > 0.0 and bb_val < 1e17:
                        tp_price = bb_val
                else:
                    bb_val = bb_lower[i]
                    if bb_val > 0.0 and bb_val < 1e17:
                        tp_price = bb_val

            # VWAP動的exit: 毎バーTP価格をVWAPバンドで更新
            if use_vwap_exit and len(vwap_upper) > i:
                if is_long:
                    vwap_val = vwap_upper[i]
                    if vwap_val > 0.0 and vwap_val < 1e17:
                        tp_price = vwap_val
                else:
                    vwap_val = vwap_lower[i]
                    if vwap_val > 0.0 and vwap_val < 1e17:
                        tp_price = vwap_val

            # トレーリングストップ更新（固定%またはATRベース）
            if trailing_pct > 0.0 or trailing_dist > 0.0:
                if is_long:
                    if h > highest_price:
                        highest_price = h
                    # ATRベースの場合は絶対値、固定%の場合は%で計算
                    if trailing_dist > 0.0:
                        new_sl = highest_price - trailing_dist
                    else:
                        new_sl = highest_price * (1.0 - trailing_pct / 100.0)
                    if new_sl > sl_price:
                        sl_price = new_sl
                else:
                    if l < lowest_price:
                        lowest_price = l
                    if trailing_dist > 0.0:
                        new_sl = lowest_price + trailing_dist
                    else:
                        new_sl = lowest_price * (1.0 + trailing_pct / 100.0)
                    if new_sl < sl_price:
                        sl_price = new_sl

            duration = i - entry_idx
            exited = False
            exit_price = 0.0

            # タイムアウト
            if timeout_bars > 0 and duration >= timeout_bars:
                exit_price = c
                exited = True

            # TP / SL 判定（同一バー両ヒット時はSL優先＝保守的）
            if not exited:
                if is_long:
                    tp_hit = h >= tp_price
                    sl_hit = l <= sl_price
                    if tp_hit and sl_hit:
                        exit_price = sl_price
                        exited = True
                    elif tp_hit:
                        exit_price = tp_price
                        exited = True
                    elif sl_hit:
                        exit_price = sl_price
                        exited = True
                else:
                    tp_hit = l <= tp_price
                    sl_hit = h >= sl_price
                    if tp_hit and sl_hit:
                        exit_price = sl_price
                        exited = True
                    elif tp_hit:
                        exit_price = tp_price
                        exited = True
                    elif sl_hit:
                        exit_price = sl_price
                        exited = True

            if exited:
                if is_long:
                    pnl_pct = (exit_price - entry_price) / entry_price * 100.0
                else:
                    pnl_pct = (entry_price - exit_price) / entry_price * 100.0

                pnl_pct -= commission_pct * 2.0

                profit_pcts[trade_count] = pnl_pct
                durations[trade_count] = duration
                trade_count += 1

                equity += equity * (pnl_pct / 100.0)
                equity_curve[trade_count] = equity

                in_position = False

        # === エントリー判定 ===
        if not in_position and entry_signals[i] and regime_mask[i]:
            entry_price = close[i]

            if slippage_pct > 0.0:
                if is_long:
                    entry_price *= (1.0 + slippage_pct / 100.0)
                else:
                    entry_price *= (1.0 - slippage_pct / 100.0)

            entry_idx = i

            # ATR値取得（ATRベースexit/トレーリング時）
            atr_val = atr[i] if (use_atr_exit or use_atr_trailing) and len(atr) > i else 0.0

            tp_price, sl_price = _compute_tp_sl(
                entry_price, is_long,
                tp_pct, sl_pct,
                use_atr_exit, atr_val, atr_tp_mult, atr_sl_mult,
            )

            # ATRベーストレーリング幅の計算
            trailing_dist = atr_val * atr_trailing_mult if use_atr_trailing else 0.0

            # BB exit モード: 初期TPをBB帯に設定（次バーから動的更新される）
            if use_bb_exit and len(bb_upper) > i:
                if is_long:
                    bb_val = bb_upper[i]
                    if bb_val > 0.0 and bb_val < 1e17:
                        tp_price = bb_val
                else:
                    bb_val = bb_lower[i]
                    if bb_val > 0.0 and bb_val < 1e17:
                        tp_price = bb_val

            # VWAP exit モード: 初期TPをVWAPバンドに設定（次バーから動的更新される）
            if use_vwap_exit and len(vwap_upper) > i:
                if is_long:
                    vwap_val = vwap_upper[i]
                    if vwap_val > 0.0 and vwap_val < 1e17:
                        tp_price = vwap_val
                else:
                    vwap_val = vwap_lower[i]
                    if vwap_val > 0.0 and vwap_val < 1e17:
                        tp_price = vwap_val

            highest_price = entry_price
            lowest_price = entry_price
            in_position = True

    # 未決済ポジション強制決済
    if in_position:
        c = close[n - 1]
        duration = (n - 1) - entry_idx
        if is_long:
            pnl_pct = (c - entry_price) / entry_price * 100.0
        else:
            pnl_pct = (entry_price - c) / entry_price * 100.0
        pnl_pct -= commission_pct * 2.0

        profit_pcts[trade_count] = pnl_pct
        durations[trade_count] = duration
        trade_count += 1
        equity += equity * (pnl_pct / 100.0)
        equity_curve[trade_count] = equity

    return (
        profit_pcts[:trade_count],
        durations[:trade_count],
        equity_curve[:trade_count + 1],
    )


@njit(cache=True)
def _backtest_loop_regime_switching(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    entry_signals_0: np.ndarray,
    entry_signals_1: np.ndarray,
    entry_signals_2: np.ndarray,
    regime_array: np.ndarray,
    is_long_0: bool, is_long_1: bool, is_long_2: bool,
    tp_pct_0: float, tp_pct_1: float, tp_pct_2: float,
    sl_pct_0: float, sl_pct_1: float, sl_pct_2: float,
    trailing_pct_0: float, trailing_pct_1: float, trailing_pct_2: float,
    timeout_bars_0: int, timeout_bars_1: int, timeout_bars_2: int,
    commission_pct: float,
    slippage_pct: float,
    initial_capital: float,
    atr: np.ndarray,
    use_atr_0: bool, use_atr_1: bool, use_atr_2: bool,
    atr_tp_mult_0: float, atr_tp_mult_1: float, atr_tp_mult_2: float,
    atr_sl_mult_0: float, atr_sl_mult_1: float, atr_sl_mult_2: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    レジーム切替バックテストループ。

    3レジーム（0=uptrend, 1=downtrend, 2=range）ごとに
    異なるシグナル・パラメータを使用。ポジション保持中はエントリー時のパラメータ維持。

    Returns:
        (profit_pcts, durations, equity_curve, trade_regimes)
    """
    n = len(high)
    max_trades = n
    profit_pcts = np.empty(max_trades, dtype=np.float64)
    durations = np.empty(max_trades, dtype=np.int64)
    trade_regimes = np.empty(max_trades, dtype=np.int64)
    trade_count = 0

    equity = initial_capital
    equity_curve = np.empty(max_trades + 1, dtype=np.float64)
    equity_curve[0] = equity

    in_position = False
    entry_price = 0.0
    entry_idx = 0
    tp_price = 0.0
    sl_price = 0.0
    highest_price = 0.0
    lowest_price = 1e18
    pos_is_long = True
    pos_trailing_pct = 0.0
    pos_timeout_bars = 0

    for i in range(1, n):
        # === 決済判定 ===
        if in_position:
            h = high[i]
            l = low[i]
            c = close[i]

            if pos_trailing_pct > 0.0:
                if pos_is_long:
                    if h > highest_price:
                        highest_price = h
                    new_sl = highest_price * (1.0 - pos_trailing_pct / 100.0)
                    if new_sl > sl_price:
                        sl_price = new_sl
                else:
                    if l < lowest_price:
                        lowest_price = l
                    new_sl = lowest_price * (1.0 + pos_trailing_pct / 100.0)
                    if new_sl < sl_price:
                        sl_price = new_sl

            duration = i - entry_idx
            exited = False
            exit_price = 0.0

            if pos_timeout_bars > 0 and duration >= pos_timeout_bars:
                exit_price = c
                exited = True

            # TP / SL 判定（同一バー両ヒット時はSL優先＝保守的）
            if not exited:
                if pos_is_long:
                    tp_hit = h >= tp_price
                    sl_hit = l <= sl_price
                    if tp_hit and sl_hit:
                        exit_price = sl_price
                        exited = True
                    elif tp_hit:
                        exit_price = tp_price
                        exited = True
                    elif sl_hit:
                        exit_price = sl_price
                        exited = True
                else:
                    tp_hit = l <= tp_price
                    sl_hit = h >= sl_price
                    if tp_hit and sl_hit:
                        exit_price = sl_price
                        exited = True
                    elif tp_hit:
                        exit_price = tp_price
                        exited = True
                    elif sl_hit:
                        exit_price = sl_price
                        exited = True

            if exited:
                if pos_is_long:
                    pnl_pct = (exit_price - entry_price) / entry_price * 100.0
                else:
                    pnl_pct = (entry_price - exit_price) / entry_price * 100.0
                pnl_pct -= commission_pct * 2.0

                profit_pcts[trade_count] = pnl_pct
                durations[trade_count] = duration
                trade_count += 1
                equity += equity * (pnl_pct / 100.0)
                equity_curve[trade_count] = equity
                in_position = False

        # === エントリー判定（レジーム切替） ===
        if not in_position:
            r = regime_array[i]
            if r == 0:
                signal = entry_signals_0[i]
                cur_is_long = is_long_0
                cur_tp = tp_pct_0
                cur_sl = sl_pct_0
                cur_trail = trailing_pct_0
                cur_tout = timeout_bars_0
                cur_use_atr = use_atr_0
                cur_atr_tp = atr_tp_mult_0
                cur_atr_sl = atr_sl_mult_0
            elif r == 1:
                signal = entry_signals_1[i]
                cur_is_long = is_long_1
                cur_tp = tp_pct_1
                cur_sl = sl_pct_1
                cur_trail = trailing_pct_1
                cur_tout = timeout_bars_1
                cur_use_atr = use_atr_1
                cur_atr_tp = atr_tp_mult_1
                cur_atr_sl = atr_sl_mult_1
            else:
                signal = entry_signals_2[i]
                cur_is_long = is_long_2
                cur_tp = tp_pct_2
                cur_sl = sl_pct_2
                cur_trail = trailing_pct_2
                cur_tout = timeout_bars_2
                cur_use_atr = use_atr_2
                cur_atr_tp = atr_tp_mult_2
                cur_atr_sl = atr_sl_mult_2

            if signal:
                entry_price = close[i]
                if slippage_pct > 0.0:
                    if cur_is_long:
                        entry_price *= (1.0 + slippage_pct / 100.0)
                    else:
                        entry_price *= (1.0 - slippage_pct / 100.0)

                entry_idx = i
                pos_is_long = cur_is_long
                pos_trailing_pct = cur_trail
                pos_timeout_bars = cur_tout

                atr_val = atr[i] if cur_use_atr and len(atr) > i else 0.0

                tp_price, sl_price = _compute_tp_sl(
                    entry_price, cur_is_long,
                    cur_tp, cur_sl,
                    cur_use_atr, atr_val, cur_atr_tp, cur_atr_sl,
                )

                highest_price = entry_price
                lowest_price = entry_price
                in_position = True
                trade_regimes[trade_count] = r

    # 未決済ポジション強制決済
    if in_position:
        c = close[n - 1]
        duration = (n - 1) - entry_idx
        if pos_is_long:
            pnl_pct = (c - entry_price) / entry_price * 100.0
        else:
            pnl_pct = (entry_price - c) / entry_price * 100.0
        pnl_pct -= commission_pct * 2.0

        profit_pcts[trade_count] = pnl_pct
        durations[trade_count] = duration
        trade_count += 1
        equity += equity * (pnl_pct / 100.0)
        equity_curve[trade_count] = equity

    return (
        profit_pcts[:trade_count],
        durations[:trade_count],
        equity_curve[:trade_count + 1],
        trade_regimes[:trade_count],
    )


def compute_atr_numpy(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """
    numpy配列からATRを計算。

    Numbaループに渡す前にATR配列を事前計算するためのヘルパー。
    インジケーターモジュールのATRクラスと同等の計算をnumpy純粋実装で行う。
    """
    n = len(high)
    if n == 0:
        return np.empty(0, dtype=np.float64)

    # True Range
    tr = np.empty(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, max(hc, lc))

    # EWM (Exponential Weighted Mean)
    alpha = 2.0 / (period + 1)
    atr = np.empty(n, dtype=np.float64)
    atr[0] = tr[0]
    for i in range(1, n):
        atr[i] = alpha * tr[i] + (1.0 - alpha) * atr[i - 1]

    return atr


def vectorize_entry_signals(
    df: pd.DataFrame,
    entry_conditions: List[Dict[str, Any]],
    entry_logic: str = "and",
) -> np.ndarray:
    """
    エントリー条件を pandas ベクトル演算で一括計算し bool 配列を返す。

    各条件を DataFrame 全体に対して一度に評価する。
    Condition.evaluate(row, prev_row) の row-by-row 呼び出しを排除。
    """
    n = len(df)
    if not entry_conditions:
        return np.zeros(n, dtype=np.bool_)

    signals = []

    for c in entry_conditions:
        ctype = c["type"]

        if ctype == "threshold":
            col = c["column"]
            op = c["operator"]
            val = c["value"]
            if col not in df.columns:
                signals.append(np.zeros(n, dtype=np.bool_))
                continue
            series = df[col]
            if op == ">":
                sig = series > val
            elif op == "<":
                sig = series < val
            elif op == ">=":
                sig = series >= val
            elif op == "<=":
                sig = series <= val
            elif op == "==":
                sig = series == val
            else:
                signals.append(np.zeros(n, dtype=np.bool_))
                continue
            signals.append(sig.fillna(False).values.astype(np.bool_))

        elif ctype == "crossover":
            fast_col = c["fast"]
            slow_col = c["slow"]
            direction = c.get("direction", "above")
            if fast_col not in df.columns or slow_col not in df.columns:
                signals.append(np.zeros(n, dtype=np.bool_))
                continue
            prev_fast = df[fast_col].shift(1)
            prev_slow = df[slow_col].shift(1)
            cur_fast = df[fast_col]
            cur_slow = df[slow_col]
            if direction == "above":
                sig = (prev_fast <= prev_slow) & (cur_fast > cur_slow)
            else:
                sig = (prev_fast >= prev_slow) & (cur_fast < cur_slow)
            signals.append(sig.fillna(False).values.astype(np.bool_))

        elif ctype == "candle":
            pattern = c["pattern"]
            if pattern == "bullish":
                sig = df["close"] > df["open"]
            else:
                sig = df["close"] < df["open"]
            signals.append(sig.values.astype(np.bool_))

        elif ctype == "column_compare":
            col_a = c["column_a"]
            op = c["operator"]
            col_b = c["column_b"]
            if col_a not in df.columns or col_b not in df.columns:
                signals.append(np.zeros(n, dtype=np.bool_))
                continue
            a = df[col_a]
            b = df[col_b]
            if op == ">":
                sig = a > b
            elif op == "<":
                sig = a < b
            elif op == ">=":
                sig = a >= b
            elif op == "<=":
                sig = a <= b
            elif op == "==":
                sig = a == b
            else:
                signals.append(np.zeros(n, dtype=np.bool_))
                continue
            signals.append(sig.fillna(False).values.astype(np.bool_))

        else:
            signals.append(np.zeros(n, dtype=np.bool_))

    if not signals:
        return np.zeros(n, dtype=np.bool_)

    result = signals[0]
    for s in signals[1:]:
        if entry_logic == "and":
            result = result & s
        else:
            result = result | s

    return result
