"""
ATR ベース exit 機能のテスト

_compute_tp_sl(), compute_atr_numpy(), _backtest_loop() の
ATR モード動作を検証。
"""

import numpy as np
import pytest

from engine.numba_loop import _compute_tp_sl, compute_atr_numpy, _backtest_loop


# ---------------------------------------------------------------------------
# _compute_tp_sl
# ---------------------------------------------------------------------------

class TestComputeTpSl:
    """TP/SL 計算ヘルパーのテスト"""

    def test_fixed_long(self):
        """固定%モード・ロング: TP=2%, SL=1%"""
        tp, sl = _compute_tp_sl(
            entry_price=100.0, is_long=True,
            tp_pct=2.0, sl_pct=1.0,
            use_atr_exit=False, atr_val=0.0,
            atr_tp_mult=0.0, atr_sl_mult=0.0,
        )
        assert tp == pytest.approx(102.0)
        assert sl == pytest.approx(99.0)

    def test_fixed_short(self):
        """固定%モード・ショート: TP=2%, SL=1%"""
        tp, sl = _compute_tp_sl(
            entry_price=100.0, is_long=False,
            tp_pct=2.0, sl_pct=1.0,
            use_atr_exit=False, atr_val=0.0,
            atr_tp_mult=0.0, atr_sl_mult=0.0,
        )
        assert tp == pytest.approx(98.0)
        assert sl == pytest.approx(101.0)

    def test_fixed_no_sl_long(self):
        """固定%モード・SLなし: sl_pct=0 → SL発動しない"""
        tp, sl = _compute_tp_sl(
            entry_price=100.0, is_long=True,
            tp_pct=2.0, sl_pct=0.0,
            use_atr_exit=False, atr_val=0.0,
            atr_tp_mult=0.0, atr_sl_mult=0.0,
        )
        assert tp == pytest.approx(102.0)
        assert sl == -1.0  # 到達不可能

    def test_fixed_no_sl_short(self):
        """固定%モード・SLなし・ショート"""
        tp, sl = _compute_tp_sl(
            entry_price=100.0, is_long=False,
            tp_pct=2.0, sl_pct=0.0,
            use_atr_exit=False, atr_val=0.0,
            atr_tp_mult=0.0, atr_sl_mult=0.0,
        )
        assert tp == pytest.approx(98.0)
        assert sl == 1e18  # 到達不可能

    def test_fixed_no_tp_long(self):
        """固定%モード・TPなし: tp_pct=0"""
        tp, sl = _compute_tp_sl(
            entry_price=100.0, is_long=True,
            tp_pct=0.0, sl_pct=1.0,
            use_atr_exit=False, atr_val=0.0,
            atr_tp_mult=0.0, atr_sl_mult=0.0,
        )
        assert tp == 1e18  # 到達不可能
        assert sl == pytest.approx(99.0)

    def test_atr_long(self):
        """ATRモード・ロング: ATR=5, TP×3, SL×1.5"""
        tp, sl = _compute_tp_sl(
            entry_price=100.0, is_long=True,
            tp_pct=0.0, sl_pct=0.0,
            use_atr_exit=True, atr_val=5.0,
            atr_tp_mult=3.0, atr_sl_mult=1.5,
        )
        assert tp == pytest.approx(115.0)  # 100 + 5*3
        assert sl == pytest.approx(92.5)   # 100 - 5*1.5

    def test_atr_short(self):
        """ATRモード・ショート: ATR=5, TP×3, SL×1.5"""
        tp, sl = _compute_tp_sl(
            entry_price=100.0, is_long=False,
            tp_pct=0.0, sl_pct=0.0,
            use_atr_exit=True, atr_val=5.0,
            atr_tp_mult=3.0, atr_sl_mult=1.5,
        )
        assert tp == pytest.approx(85.0)   # 100 - 5*3
        assert sl == pytest.approx(107.5)  # 100 + 5*1.5

    def test_atr_no_sl(self):
        """ATRモード・SLなし: atr_sl_mult=0"""
        tp, sl = _compute_tp_sl(
            entry_price=100.0, is_long=True,
            tp_pct=0.0, sl_pct=0.0,
            use_atr_exit=True, atr_val=5.0,
            atr_tp_mult=3.0, atr_sl_mult=0.0,
        )
        assert tp == pytest.approx(115.0)
        assert sl == -1.0  # 到達不可能

    def test_atr_no_tp(self):
        """ATRモード・TPなし: atr_tp_mult=0（トレーリングのみ想定）"""
        tp, sl = _compute_tp_sl(
            entry_price=100.0, is_long=True,
            tp_pct=0.0, sl_pct=0.0,
            use_atr_exit=True, atr_val=5.0,
            atr_tp_mult=0.0, atr_sl_mult=1.5,
        )
        assert tp == 1e18  # 到達不可能
        assert sl == pytest.approx(92.5)

    def test_atr_zero_val(self):
        """ATR値=0の場合（データ不足等）→ TP/SL = entry_price"""
        tp, sl = _compute_tp_sl(
            entry_price=100.0, is_long=True,
            tp_pct=0.0, sl_pct=0.0,
            use_atr_exit=True, atr_val=0.0,
            atr_tp_mult=3.0, atr_sl_mult=1.5,
        )
        assert tp == pytest.approx(100.0)  # 100 + 0*3
        assert sl == pytest.approx(100.0)  # 100 - 0*1.5


# ---------------------------------------------------------------------------
# compute_atr_numpy
# ---------------------------------------------------------------------------

class TestComputeAtrNumpy:
    """ATR 計算のテスト"""

    def test_basic(self):
        """基本的なATR計算"""
        n = 100
        high = np.full(n, 105.0)
        low = np.full(n, 95.0)
        close = np.full(n, 100.0)
        atr = compute_atr_numpy(high, low, close, period=14)
        assert len(atr) == n
        # 一定のH-L=10なのでATRは10に収束
        assert atr[-1] == pytest.approx(10.0, abs=0.5)

    def test_empty(self):
        """空配列 → 空のATR"""
        atr = compute_atr_numpy(
            np.empty(0), np.empty(0), np.empty(0), period=14,
        )
        assert len(atr) == 0

    def test_single_bar(self):
        """1本のみ → ATR = H-L"""
        atr = compute_atr_numpy(
            np.array([110.0]), np.array([90.0]), np.array([100.0]),
            period=14,
        )
        assert len(atr) == 1
        assert atr[0] == pytest.approx(20.0)

    def test_varying_volatility(self):
        """ボラが変わるとATRも追従"""
        n = 200
        high = np.empty(n)
        low = np.empty(n)
        close = np.empty(n)
        # 前半: ボラ小
        for i in range(100):
            high[i] = 101.0
            low[i] = 99.0
            close[i] = 100.0
        # 後半: ボラ大
        for i in range(100, 200):
            high[i] = 110.0
            low[i] = 90.0
            close[i] = 100.0
        atr = compute_atr_numpy(high, low, close, period=14)
        # 前半終了時点ではATR ≈ 2
        assert atr[99] < 5.0
        # 後半終了時点ではATR ≈ 20 に近づく
        assert atr[199] > 15.0


# ---------------------------------------------------------------------------
# _backtest_loop ATR モード統合テスト
# ---------------------------------------------------------------------------

class TestBacktestLoopATR:
    """_backtest_loop の ATR モード動作テスト"""

    def _make_price_data(self, n=500):
        """一定のトレンドがある価格データを生成"""
        np.random.seed(42)
        close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n)) * 0.5
        low = close - np.abs(np.random.randn(n)) * 0.5
        return high, low, close

    def test_backward_compat_no_atr(self):
        """use_atr_exit=False → 従来の固定%モードと同じ動作"""
        high, low, close = self._make_price_data()
        n = len(high)
        signals = np.zeros(n, dtype=np.bool_)
        signals[10::50] = True  # 50本ごとにエントリー
        mask = np.ones(n, dtype=np.bool_)

        profit_pcts, durations, equity = _backtest_loop(
            high, low, close, signals, mask,
            is_long=True, tp_pct=2.0, sl_pct=1.0,
            trailing_pct=0.0, timeout_bars=0,
            commission_pct=0.04, slippage_pct=0.0,
            initial_capital=10000.0,
            atr=np.empty(0, dtype=np.float64),
            use_atr_exit=False,
            atr_tp_mult=0.0, atr_sl_mult=0.0,
            bb_upper=np.empty(0, dtype=np.float64),
            bb_lower=np.empty(0, dtype=np.float64),
            use_bb_exit=False,
            vwap_upper=np.empty(0, dtype=np.float64),
            vwap_lower=np.empty(0, dtype=np.float64),
            use_vwap_exit=False,
        )
        assert len(profit_pcts) > 0
        assert len(equity) == len(profit_pcts) + 1

    def test_atr_mode_produces_trades(self):
        """ATRモードでトレードが発生する"""
        high, low, close = self._make_price_data()
        n = len(high)
        atr = compute_atr_numpy(high, low, close, period=14)
        signals = np.zeros(n, dtype=np.bool_)
        signals[20::50] = True
        mask = np.ones(n, dtype=np.bool_)

        profit_pcts, durations, equity = _backtest_loop(
            high, low, close, signals, mask,
            is_long=True, tp_pct=0.0, sl_pct=0.0,
            trailing_pct=0.0, timeout_bars=0,
            commission_pct=0.04, slippage_pct=0.0,
            initial_capital=10000.0,
            atr=atr,
            use_atr_exit=True,
            atr_tp_mult=3.0, atr_sl_mult=1.5,
            bb_upper=np.empty(0, dtype=np.float64),
            bb_lower=np.empty(0, dtype=np.float64),
            use_bb_exit=False,
            vwap_upper=np.empty(0, dtype=np.float64),
            vwap_lower=np.empty(0, dtype=np.float64),
            use_vwap_exit=False,
        )
        assert len(profit_pcts) > 0

    def test_no_sl_with_timeout(self):
        """SLなし + タイムアウト: 全トレードがタイムアウトで決済"""
        n = 500
        close = np.full(n, 100.0)
        high = np.full(n, 100.5)
        low = np.full(n, 99.5)

        signals = np.zeros(n, dtype=np.bool_)
        signals[10] = True
        mask = np.ones(n, dtype=np.bool_)

        profit_pcts, durations, equity = _backtest_loop(
            high, low, close, signals, mask,
            is_long=True, tp_pct=0.0, sl_pct=0.0,
            trailing_pct=0.0, timeout_bars=30,
            commission_pct=0.04, slippage_pct=0.0,
            initial_capital=10000.0,
            atr=np.empty(0, dtype=np.float64),
            use_atr_exit=False,
            atr_tp_mult=0.0, atr_sl_mult=0.0,
            bb_upper=np.empty(0, dtype=np.float64),
            bb_lower=np.empty(0, dtype=np.float64),
            use_bb_exit=False,
            vwap_upper=np.empty(0, dtype=np.float64),
            vwap_lower=np.empty(0, dtype=np.float64),
            use_vwap_exit=False,
        )
        assert len(profit_pcts) == 1
        assert durations[0] == 30  # タイムアウト
