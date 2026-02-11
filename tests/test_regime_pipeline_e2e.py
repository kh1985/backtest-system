"""
レジーム最適化パイプラインのE2Eテスト（モックなし）
"""

import numpy as np
import pandas as pd

from analysis.trend import TrendDetector, TrendRegime
from optimizer.grid import GridSearchOptimizer
from optimizer.validation import DataSplitConfig, run_validated_optimization
from optimizer.walk_forward import WFAConfig, run_walk_forward_analysis


def _simple_config():
    return [{
        "_template_name": "always_long",
        "_params": {},
        "name": "always_long",
        "side": "long",
        "indicators": [],
        "entry_conditions": [
            {"type": "threshold", "column": "close", "operator": ">", "value": 0.0},
        ],
        "entry_logic": "and",
        "exit": {"take_profit_pct": 1.0, "stop_loss_pct": 1.0},
    }]


def _make_exec_df(n=360):
    rng = np.random.default_rng(42)
    dt = pd.date_range("2024-01-01", periods=n, freq="15min")
    close = 100 + np.cumsum(rng.normal(0.03, 0.3, size=n))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + 0.4
    low = np.minimum(open_, close) - 0.4
    volume = np.full(n, 1000.0)
    return pd.DataFrame({
        "datetime": dt,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


class TestRegimePipelineE2E:
    def test_pipeline_validation_and_wfa_runs(self):
        exec_df = _make_exec_df(360)
        htf_df = (
            exec_df.set_index("datetime")
            .resample("1h")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
            .dropna()
            .reset_index()
        )

        detector = TrendDetector()
        htf_labeled = detector.detect_ma_cross(htf_df, fast_period=5, slow_period=10, apply_shift=True)
        exec_labeled = TrendDetector.label_execution_tf(exec_df, htf_labeled)

        optimizer = GridSearchOptimizer(
            entry_on_next_open=True,
            bars_per_year=365 * 24 * 4,
        )
        configs = _simple_config()

        validated = run_validated_optimization(
            df=exec_labeled,
            all_configs=configs,
            target_regimes=["uptrend", "downtrend", "range"],
            split_config=DataSplitConfig(top_n_for_val=1, min_trades_for_val=0),
            optimizer=optimizer,
            n_workers=1,
        )
        assert validated.train_results.total_combinations > 0
        assert validated.train_end > 0
        assert validated.val_end > validated.train_end

        wfa = run_walk_forward_analysis(
            df=exec_labeled,
            all_configs=configs,
            target_regimes=["uptrend", "downtrend", "range"],
            wfa_config=WFAConfig(n_folds=3, min_is_pct=0.4, use_validation=False),
            optimizer=optimizer,
            n_workers=1,
        )
        assert len(wfa.folds) == 3
        assert "uptrend" in wfa.valid_fold_count

    def test_leakage_guard_shift_changes_labels(self):
        detector = TrendDetector()
        htf = pd.DataFrame({
            "datetime": pd.date_range("2024-01-01", periods=20, freq="1h"),
            "open": np.linspace(100, 110, 20),
            "high": np.linspace(101, 111, 20),
            "low": np.linspace(99, 109, 20),
            "close": np.r_[np.linspace(100, 105, 10), np.linspace(120, 90, 10)],
            "volume": np.full(20, 1000.0),
        })
        exec_df = pd.DataFrame({
            "datetime": pd.date_range("2024-01-01", periods=80, freq="15min"),
            "open": np.full(80, 100.0),
            "high": np.full(80, 101.0),
            "low": np.full(80, 99.0),
            "close": np.full(80, 100.0),
            "volume": np.full(80, 1000.0),
        })

        shifted = detector.detect_ma_cross(htf, fast_period=3, slow_period=5, apply_shift=True)
        not_shifted = detector.detect_ma_cross(htf, fast_period=3, slow_period=5, apply_shift=False)

        exec_shifted = TrendDetector.label_execution_tf(exec_df, shifted)
        exec_not_shifted = TrendDetector.label_execution_tf(exec_df, not_shifted)

        assert not exec_shifted["trend_regime"].equals(exec_not_shifted["trend_regime"])
        assert shifted["trend_regime"].iloc[0] == TrendRegime.RANGE.value

    def test_next_open_is_more_conservative_than_same_close(self):
        n = 60
        df = pd.DataFrame({
            "datetime": pd.date_range("2024-01-01", periods=n, freq="15min"),
            "open": np.where(np.arange(n) % 2 == 0, 100.0, 103.0),
            "high": np.full(n, 101.5),
            "low": np.full(n, 99.5),
            "close": np.full(n, 100.0),
            "volume": np.full(n, 1000.0),
            "trend_regime": np.full(n, TrendRegime.RANGE.value),
        })
        configs = _simple_config()

        opt_same_close = GridSearchOptimizer(entry_on_next_open=False, bars_per_year=365 * 24 * 4)
        res_same_close = opt_same_close.run(df, configs, target_regimes=["all"], n_workers=1)

        opt_next_open = GridSearchOptimizer(entry_on_next_open=True, bars_per_year=365 * 24 * 4)
        res_next_open = opt_next_open.run(df, configs, target_regimes=["all"], n_workers=1)

        assert res_same_close.best is not None
        assert res_next_open.best is not None
        assert res_next_open.best.metrics.total_profit_pct < res_same_close.best.metrics.total_profit_pct
