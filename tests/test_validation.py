"""
validation.py のユニットテスト

- DataSplitConfig のインデックス計算
- ValidatedResultSet のデータ構造
- run_validated_optimization の統合テスト（モック使用）
"""

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from unittest.mock import MagicMock, patch, call

import pytest

from metrics.calculator import BacktestMetrics
from optimizer.results import OptimizationEntry, OptimizationResultSet
from optimizer.validation import (
    DataSplitConfig,
    ValidatedResultSet,
    _rebuild_configs_from_entries,
    run_validated_optimization,
)


# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------

def _make_metrics(**overrides) -> BacktestMetrics:
    defaults = dict(
        total_trades=50,
        winning_trades=30,
        losing_trades=20,
        win_rate=60.0,
        total_profit_pct=15.0,
        avg_profit_pct=1.5,
        avg_loss_pct=-1.0,
        profit_factor=1.5,
        max_drawdown_pct=10.0,
        sharpe_ratio=1.2,
        avg_duration_bars=5.0,
        best_trade_pct=5.0,
        worst_trade_pct=-3.0,
        equity_curve=[10000.0],
        cumulative_returns=[],
        drawdown_series=[],
    )
    defaults.update(overrides)
    return BacktestMetrics(**defaults)


def _make_entry(
    template="test_tmpl",
    regime="uptrend",
    score=0.5,
    params=None,
    config=None,
    **metric_overrides,
) -> OptimizationEntry:
    return OptimizationEntry(
        template_name=template,
        params=params or {"sma_period": 20},
        trend_regime=regime,
        config=config or {"indicators": [], "entry_conditions": []},
        metrics=_make_metrics(**metric_overrides),
        composite_score=score,
    )


# ---------------------------------------------------------------------------
# DataSplitConfig
# ---------------------------------------------------------------------------

class TestDataSplitConfig:
    def test_default_values(self):
        cfg = DataSplitConfig()
        assert cfg.train_pct == 0.6
        assert cfg.val_pct == 0.2
        assert cfg.test_pct == 0.2
        assert cfg.min_trades_for_val == 30

    def test_test_pct_computed(self):
        cfg = DataSplitConfig(train_pct=0.7, val_pct=0.15)
        assert cfg.test_pct == pytest.approx(0.15, abs=0.01)

    def test_compute_indices(self):
        cfg = DataSplitConfig(train_pct=0.6, val_pct=0.2)
        train_end, val_end = cfg.compute_indices(1000)
        assert train_end == 600
        assert val_end == 800

    def test_compute_indices_small_data(self):
        cfg = DataSplitConfig()
        train_end, val_end = cfg.compute_indices(100)
        assert train_end == 60
        assert val_end == 80


# ---------------------------------------------------------------------------
# _rebuild_configs_from_entries
# ---------------------------------------------------------------------------

class TestRebuildConfigs:
    def test_basic_rebuild(self):
        entry = _make_entry(
            template="sma_cross",
            params={"fast": 10, "slow": 50},
            config={"indicators": [{"type": "sma"}], "side": "long"},
        )
        configs = _rebuild_configs_from_entries([entry])
        assert len(configs) == 1
        c = configs[0]
        assert c["_template_name"] == "sma_cross"
        assert c["_params"] == {"fast": 10, "slow": 50}
        assert c["indicators"] == [{"type": "sma"}]

    def test_deep_copy(self):
        """元の entry.config を変更しても rebuilt config に影響しない"""
        entry = _make_entry(config={"indicators": [{"type": "sma"}]})
        configs = _rebuild_configs_from_entries([entry])
        configs[0]["indicators"].append({"type": "rsi"})
        assert len(entry.config["indicators"]) == 1  # 元は変わらない

    def test_multiple_entries(self):
        entries = [
            _make_entry(template=f"tmpl_{i}", score=0.5 + i * 0.1)
            for i in range(5)
        ]
        configs = _rebuild_configs_from_entries(entries)
        assert len(configs) == 5
        names = [c["_template_name"] for c in configs]
        assert names == [f"tmpl_{i}" for i in range(5)]


# ---------------------------------------------------------------------------
# ValidatedResultSet
# ---------------------------------------------------------------------------

class TestValidatedResultSet:
    def test_default_values(self):
        vrs = ValidatedResultSet(
            train_results=OptimizationResultSet(),
        )
        assert vrs.val_best == {}
        assert vrs.test_results == {}
        assert vrs.overfitting_warnings == []

    def test_stores_split_info(self):
        cfg = DataSplitConfig(train_pct=0.7, val_pct=0.15)
        vrs = ValidatedResultSet(
            train_results=OptimizationResultSet(),
            split_config=cfg,
            train_end=700,
            val_end=850,
            total_bars=1000,
        )
        assert vrs.train_end == 700
        assert vrs.val_end == 850
        assert vrs.total_bars == 1000


# ---------------------------------------------------------------------------
# run_validated_optimization（モックベース統合テスト）
# ---------------------------------------------------------------------------

class TestRunValidatedOptimization:
    """
    GridSearchOptimizer.run() をモックし、
    3フェーズ（Train/Val/Test）が正しく呼ばれることを検証。
    """

    def _setup_mock_optimizer(self, regime="uptrend"):
        """
        optimizer.run() をモック。
        Train → 3エントリー返す
        Val → 1エントリー返す
        Test → 1エントリー返す
        """
        mock_optimizer = MagicMock()

        # Train 結果
        train_entries = [
            _make_entry(template=f"tmpl_{i}", regime=regime, score=0.8 - i * 0.1)
            for i in range(3)
        ]
        train_result = OptimizationResultSet(entries=train_entries)

        # Val 結果
        val_entry = _make_entry(
            template="tmpl_0", regime=regime, score=0.7,
        )
        val_result = OptimizationResultSet(entries=[val_entry])

        # Test 結果
        test_entry = _make_entry(
            template="tmpl_0", regime=regime, score=0.6,
            total_profit_pct=8.0,  # Train(15.0) より低い → 劣化
        )
        test_result = OptimizationResultSet(entries=[test_entry])

        mock_optimizer.run.side_effect = [train_result, val_result, test_result]
        return mock_optimizer

    def test_three_phases_called(self):
        """optimizer.run() が3回呼ばれる（Train, Val, Test）"""
        import pandas as pd

        mock_opt = self._setup_mock_optimizer()
        df = pd.DataFrame({"close": range(100)})
        configs = [
            {"_template_name": "tmpl_0", "_params": {"a": 1}, "indicators": []},
        ]

        result = run_validated_optimization(
            df=df,
            all_configs=configs,
            target_regimes=["uptrend"],
            split_config=DataSplitConfig(),
            optimizer=mock_opt,
        )

        assert mock_opt.run.call_count == 3

    def test_data_range_passed_correctly(self):
        """各フェーズに正しい data_range が渡される"""
        import pandas as pd

        mock_opt = self._setup_mock_optimizer()
        df = pd.DataFrame({"close": range(1000)})
        configs = [
            {"_template_name": "tmpl_0", "_params": {"a": 1}, "indicators": []},
        ]
        split = DataSplitConfig(train_pct=0.6, val_pct=0.2)

        result = run_validated_optimization(
            df=df,
            all_configs=configs,
            target_regimes=["uptrend"],
            split_config=split,
            optimizer=mock_opt,
        )

        calls = mock_opt.run.call_args_list
        # Train: (0, 600)
        assert calls[0].kwargs.get("data_range") == (0, 600)
        # Val: (600, 800)
        assert calls[1].kwargs.get("data_range") == (600, 800)
        # Test: (800, 1000)
        assert calls[2].kwargs.get("data_range") == (800, 1000)

    def test_result_structure(self):
        """返り値の ValidatedResultSet が正しい構造"""
        import pandas as pd

        mock_opt = self._setup_mock_optimizer()
        df = pd.DataFrame({"close": range(1000)})
        configs = [
            {"_template_name": "tmpl_0", "_params": {"a": 1}, "indicators": []},
        ]

        result = run_validated_optimization(
            df=df,
            all_configs=configs,
            target_regimes=["uptrend"],
            split_config=DataSplitConfig(),
            optimizer=mock_opt,
        )

        assert isinstance(result, ValidatedResultSet)
        assert result.train_results.total_combinations == 3
        assert "uptrend" in result.val_best
        assert "uptrend" in result.test_results
        assert result.train_end == 600
        assert result.val_end == 800
        assert result.total_bars == 1000

    def test_overfitting_warnings_generated(self):
        """Train → Test の劣化で警告が生成される"""
        import pandas as pd

        mock_opt = MagicMock()

        # Train: PF=3.0（高い）, trades=10（少ない）
        train_entry = _make_entry(
            regime="uptrend", score=0.9,
            profit_factor=3.0, total_trades=10, total_profit_pct=50.0,
        )
        train_result = OptimizationResultSet(entries=[train_entry])

        # Val
        val_entry = _make_entry(regime="uptrend", score=0.7)
        val_result = OptimizationResultSet(entries=[val_entry])

        # Test: PnL大幅劣化
        test_entry = _make_entry(
            regime="uptrend", score=0.3,
            total_profit_pct=5.0,  # 50 → 5: 90%劣化
        )
        test_result = OptimizationResultSet(entries=[test_entry])

        mock_opt.run.side_effect = [train_result, val_result, test_result]

        df = pd.DataFrame({"close": range(100)})
        configs = [{"_template_name": "t", "_params": {}, "indicators": []}]

        result = run_validated_optimization(
            df=df,
            all_configs=configs,
            target_regimes=["uptrend"],
            split_config=DataSplitConfig(),
            optimizer=mock_opt,
        )

        # PF>2.0, trades<30, OOS劣化 の3つの警告
        assert len(result.overfitting_warnings) >= 2

    def test_validation_tie_break_prefers_lower_warning_entry(self):
        """同点時は過学習警告が少ない候補を優先する"""
        import pandas as pd

        mock_opt = MagicMock()

        # Train: 同点スコア
        entry_warn = _make_entry(
            template="warn", score=0.8, regime="uptrend",
            profit_factor=3.0, total_trades=10, total_profit_pct=40.0,
        )
        entry_clean = _make_entry(
            template="clean", score=0.8, regime="uptrend",
            profit_factor=1.2, total_trades=120, total_profit_pct=12.0,
        )
        train_result = OptimizationResultSet(entries=[entry_warn, entry_clean])

        # Val: 渡されたconfigのtemplateをそのままベストとして返す
        val_result = OptimizationResultSet(entries=[entry_clean])
        test_result = OptimizationResultSet(entries=[entry_clean])
        mock_opt.run.side_effect = [train_result, val_result, test_result]

        df = pd.DataFrame({"close": range(200)})
        configs = [{"_template_name": "t", "_params": {}, "indicators": []}]

        result = run_validated_optimization(
            df=df,
            all_configs=configs,
            target_regimes=["uptrend"],
            split_config=DataSplitConfig(top_n_for_val=1),
            optimizer=mock_opt,
        )

        assert result.val_best["uptrend"].template_name == "clean"

    def test_progress_callback(self):
        """進捗コールバックが呼ばれる"""
        import pandas as pd

        mock_opt = self._setup_mock_optimizer()
        df = pd.DataFrame({"close": range(100)})
        configs = [{"_template_name": "t", "_params": {}, "indicators": []}]

        callback_calls = []

        def cb(current, total, desc):
            callback_calls.append((current, total, desc))

        run_validated_optimization(
            df=df,
            all_configs=configs,
            target_regimes=["uptrend"],
            split_config=DataSplitConfig(),
            optimizer=mock_opt,
            progress_callback=cb,
        )

        # 最後に「完了」が呼ばれる
        assert callback_calls[-1][2] == "完了"
        # current == total で完了
        assert callback_calls[-1][0] == callback_calls[-1][1]

    def test_empty_regime_results(self):
        """Train で該当レジームの結果が0件の場合"""
        import pandas as pd

        mock_opt = MagicMock()
        # Train: entries は空
        train_result = OptimizationResultSet(entries=[])
        mock_opt.run.side_effect = [train_result]

        df = pd.DataFrame({"close": range(100)})
        configs = [{"_template_name": "t", "_params": {}, "indicators": []}]

        result = run_validated_optimization(
            df=df,
            all_configs=configs,
            target_regimes=["uptrend"],
            split_config=DataSplitConfig(),
            optimizer=mock_opt,
        )

        # Val/Test は実行されない
        assert mock_opt.run.call_count == 1
        assert result.val_best == {}
        assert result.test_results == {}

    def test_min_trades_filter_excludes_low_trade_entries(self):
        """min_trades_for_val がTrainのトレード数でフィルタする"""
        import pandas as pd

        mock_opt = MagicMock()

        # Train: 2エントリー（1つは5trades、1つは50trades）
        low_trade_entry = _make_entry(
            template="low_trade", regime="uptrend", score=0.9,
            total_trades=5,  # min_trades=30 未満
        )
        high_trade_entry = _make_entry(
            template="high_trade", regime="uptrend", score=0.6,
            total_trades=50,  # min_trades=30 以上
        )
        train_result = OptimizationResultSet(
            entries=[low_trade_entry, high_trade_entry]
        )

        # Val: high_trade のみ通されるべき
        val_entry = _make_entry(
            template="high_trade", regime="uptrend", score=0.5,
        )
        val_result = OptimizationResultSet(entries=[val_entry])

        # Test
        test_entry = _make_entry(
            template="high_trade", regime="uptrend", score=0.4,
        )
        test_result = OptimizationResultSet(entries=[test_entry])

        mock_opt.run.side_effect = [train_result, val_result, test_result]

        df = pd.DataFrame({"close": range(100)})
        configs = [{"_template_name": "t", "_params": {}, "indicators": []}]

        split = DataSplitConfig(min_trades_for_val=30)
        result = run_validated_optimization(
            df=df,
            all_configs=configs,
            target_regimes=["uptrend"],
            split_config=split,
            optimizer=mock_opt,
        )

        # Val フェーズで渡される configs は high_trade のみ（1件）
        val_call = mock_opt.run.call_args_list[1]
        val_configs = val_call.kwargs.get("configs", val_call[1].get("configs") if len(val_call) > 1 else None)
        if val_configs is None:
            val_configs = val_call[0][1]  # positional arg
        assert len(val_configs) == 1
        assert val_configs[0]["_template_name"] == "high_trade"

    def test_min_trades_filter_fallback_when_no_entries_pass(self):
        """全エントリーが min_trades 未満の場合、フィルタなしで続行"""
        import pandas as pd

        mock_opt = MagicMock()

        # Train: 全て5trades（min_trades=30未満）
        entries = [
            _make_entry(template=f"tmpl_{i}", regime="uptrend", score=0.8 - i * 0.1,
                        total_trades=5)
            for i in range(3)
        ]
        train_result = OptimizationResultSet(entries=entries)

        # Val/Test
        val_entry = _make_entry(regime="uptrend", score=0.5, total_trades=5)
        val_result = OptimizationResultSet(entries=[val_entry])
        test_entry = _make_entry(regime="uptrend", score=0.4, total_trades=5)
        test_result = OptimizationResultSet(entries=[test_entry])

        mock_opt.run.side_effect = [train_result, val_result, test_result]

        df = pd.DataFrame({"close": range(100)})
        configs = [{"_template_name": "t", "_params": {}, "indicators": []}]

        split = DataSplitConfig(min_trades_for_val=30)
        result = run_validated_optimization(
            df=df,
            all_configs=configs,
            target_regimes=["uptrend"],
            split_config=split,
            optimizer=mock_opt,
        )

        # フォールバック: 全3件がValに渡される
        assert mock_opt.run.call_count == 3
        val_call = mock_opt.run.call_args_list[1]
        val_configs = val_call.kwargs.get("configs", val_call[1].get("configs") if len(val_call) > 1 else None)
        if val_configs is None:
            val_configs = val_call[0][1]
        assert len(val_configs) == 3

    def test_min_trades_zero_disables_filter(self):
        """min_trades_for_val=0 でフィルタ無効"""
        import pandas as pd

        mock_opt = MagicMock()

        # Train: 1エントリー（2trades）
        entry = _make_entry(
            template="low", regime="uptrend", score=0.9,
            total_trades=2,
        )
        train_result = OptimizationResultSet(entries=[entry])

        val_entry = _make_entry(regime="uptrend", score=0.5)
        val_result = OptimizationResultSet(entries=[val_entry])
        test_entry = _make_entry(regime="uptrend", score=0.4)
        test_result = OptimizationResultSet(entries=[test_entry])

        mock_opt.run.side_effect = [train_result, val_result, test_result]

        df = pd.DataFrame({"close": range(100)})
        configs = [{"_template_name": "t", "_params": {}, "indicators": []}]

        split = DataSplitConfig(min_trades_for_val=0)
        result = run_validated_optimization(
            df=df,
            all_configs=configs,
            target_regimes=["uptrend"],
            split_config=split,
            optimizer=mock_opt,
        )

        # フィルタなし: 2tradesエントリーもValに通される
        assert mock_opt.run.call_count == 3
