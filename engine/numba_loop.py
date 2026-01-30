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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Numba JIT コンパイルされたバックテストループ。

    Args:
        high, low, close: OHLC numpy配列
        entry_signals: エントリーシグナル（bool配列、事前にベクトル化済み）
        regime_mask: レジームフィルタ（bool配列、全True = フィルタなし）
        is_long: True=ロング、False=ショート
        tp_pct, sl_pct: TP/SLパーセンテージ
        trailing_pct: トレーリングストップ%（0.0=無効）
        timeout_bars: タイムアウトバー数（0=無効）
        commission_pct: 手数料%
        slippage_pct: スリッページ%
        initial_capital: 初期資金

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

    for i in range(1, n):
        # === 決済判定 ===
        if in_position:
            h = high[i]
            l = low[i]
            c = close[i]

            # トレーリングストップ更新
            if trailing_pct > 0.0:
                if is_long:
                    if h > highest_price:
                        highest_price = h
                    new_sl = highest_price * (1.0 - trailing_pct / 100.0)
                    if new_sl > sl_price:
                        sl_price = new_sl
                else:
                    if l < lowest_price:
                        lowest_price = l
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

            # TP / SL 判定
            if not exited:
                if is_long:
                    if h >= tp_price:
                        exit_price = tp_price
                        exited = True
                    elif l <= sl_price:
                        exit_price = sl_price
                        exited = True
                else:
                    if l <= tp_price:
                        exit_price = tp_price
                        exited = True
                    elif h >= sl_price:
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

            if is_long:
                tp_price = entry_price * (1.0 + tp_pct / 100.0)
                sl_price = entry_price * (1.0 - sl_pct / 100.0)
            else:
                tp_price = entry_price * (1.0 - tp_pct / 100.0)
                sl_price = entry_price * (1.0 + sl_pct / 100.0)

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
