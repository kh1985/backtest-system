"""
metrics.calculator の時間軸依存テスト

- bars_per_year に応じて Sharpe が変化する
- 短系列/空トレードで NaN/inf を返さない
"""

import numpy as np

from metrics.calculator import calculate_metrics_from_arrays


class TestMetricsTimeframe:
    def test_sharpe_depends_on_bars_per_year(self):
        profit_pcts = np.array([1.0, -0.5, 0.8, -0.2, 0.4], dtype=np.float64)
        durations = np.array([3, 2, 4, 1, 2], dtype=np.int64)
        equity_curve = np.array(
            [10000.0, 10100.0, 10049.5, 10129.896, 10109.636, 10150.075],
            dtype=np.float64,
        )

        m_15m = calculate_metrics_from_arrays(
            profit_pcts, durations, equity_curve, bars_per_year=365 * 24 * 4
        )
        m_1h = calculate_metrics_from_arrays(
            profit_pcts, durations, equity_curve, bars_per_year=365 * 24
        )

        assert m_15m.sharpe_ratio > m_1h.sharpe_ratio

    def test_short_series_is_safe(self):
        profit_pcts = np.array([0.5], dtype=np.float64)
        durations = np.array([1], dtype=np.int64)
        equity_curve = np.array([10000.0, 10050.0], dtype=np.float64)

        m = calculate_metrics_from_arrays(
            profit_pcts, durations, equity_curve, bars_per_year=365 * 24 * 4
        )
        assert np.isfinite(m.sharpe_ratio)

    def test_empty_trades_is_safe(self):
        m = calculate_metrics_from_arrays(
            np.array([], dtype=np.float64),
            np.array([], dtype=np.int64),
            np.array([10000.0], dtype=np.float64),
            bars_per_year=365 * 24 * 4,
        )
        assert m.total_trades == 0
        assert m.sharpe_ratio == 0.0
