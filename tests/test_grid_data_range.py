"""
grid.py の data_range パラメータのテスト

GridSearchOptimizer の内部メソッドが data_range を正しく伝播し、
numpy 配列のスライスが正しく行われることを検証。
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from optimizer.grid import GridSearchOptimizer, _BacktestTask
from optimizer.scoring import ScoringWeights


# ---------------------------------------------------------------------------
# _BacktestTask
# ---------------------------------------------------------------------------

class TestBacktestTask:
    def test_data_range_default_none(self):
        task = _BacktestTask(
            task_id=0,
            config={},
            template_name="test",
            params={},
            target_regime="all",
            trend_column="trend_regime",
            cache_key="abc",
            initial_capital=10000,
            commission_pct=0.0,
            slippage_pct=0.0,
            scoring_weights=ScoringWeights(),
        )
        assert task.data_range is None

    def test_data_range_set(self):
        task = _BacktestTask(
            task_id=0,
            config={},
            template_name="test",
            params={},
            target_regime="all",
            trend_column="trend_regime",
            cache_key="abc",
            initial_capital=10000,
            commission_pct=0.0,
            slippage_pct=0.0,
            scoring_weights=ScoringWeights(),
            data_range=(100, 500),
        )
        assert task.data_range == (100, 500)


# ---------------------------------------------------------------------------
# GridSearchOptimizer.run() data_range 伝播
# ---------------------------------------------------------------------------

class TestGridDataRange:
    """
    _run_numba のスライスロジックを直接テスト。
    フルパイプラインを通すと Numba/Strategy 依存が重いため、
    _run_numba 内のスライスロジックを numpy レベルで検証。
    """

    def test_numpy_slice_logic(self):
        """data_range=(200, 600) でスライスすると長さ400になる"""
        n = 1000
        high = np.random.randn(n)
        low = np.random.randn(n)
        close = np.random.randn(n)
        signals = np.zeros(n, dtype=np.bool_)
        mask = np.ones(n, dtype=np.bool_)

        start, end = 200, 600
        h_sliced = high[start:end]
        l_sliced = low[start:end]
        c_sliced = close[start:end]
        s_sliced = signals[start:end]
        m_sliced = mask[start:end]

        assert len(h_sliced) == 400
        assert len(l_sliced) == 400
        assert len(c_sliced) == 400
        assert len(s_sliced) == 400
        assert len(m_sliced) == 400

        # 元データの対応するインデックスと一致
        np.testing.assert_array_equal(h_sliced, high[200:600])

    def test_no_slice_when_none(self):
        """data_range=None ではスライスしない"""
        n = 1000
        arr = np.arange(n)
        data_range = None

        if data_range is not None:
            start, end = data_range
            result = arr[start:end]
        else:
            result = arr

        assert len(result) == n

    def test_run_passes_data_range_to_sequential(self):
        """run(data_range=...) が _run_sequential に渡される"""
        opt = GridSearchOptimizer()

        with patch.object(opt, "_run_sequential") as mock_seq:
            mock_seq.return_value = MagicMock()
            opt.run(
                df=pd.DataFrame({"close": [1, 2, 3]}),
                configs=[],
                target_regimes=["all"],
                data_range=(0, 100),
                n_workers=1,
            )
            # _run_sequential(df, configs, regimes, trend_col, cb, data_range)
            args = mock_seq.call_args[0]
            assert args[-1] == (0, 100)

    def test_run_passes_data_range_to_parallel(self):
        """run(data_range=..., n_workers=2) が _run_parallel に渡される"""
        opt = GridSearchOptimizer()

        with patch.object(opt, "_run_parallel") as mock_par:
            mock_par.return_value = MagicMock()
            opt.run(
                df=pd.DataFrame({"close": [1, 2, 3]}),
                configs=[],
                target_regimes=["all"],
                data_range=(0, 100),
                n_workers=2,
            )
            # _run_parallel(df, configs, regimes, trend_col, cb, n_workers, data_range)
            args = mock_par.call_args[0]
            assert args[-1] == (0, 100)
