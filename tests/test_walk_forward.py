"""
walk_forward.py のユニットテスト

- WFAConfig のフォールド計算
- run_walk_forward_analysis の統合テスト（モック使用）
- 集約メトリクス計算
"""

import copy
from unittest.mock import MagicMock

import pytest

from metrics.calculator import BacktestMetrics
from optimizer.results import OptimizationEntry, OptimizationResultSet
from optimizer.walk_forward import (
    WFAConfig,
    WFAFoldResult,
    WFAResultSet,
    _compute_aggregate_metrics,
    run_walk_forward_analysis,
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
# WFAConfig テスト
# ---------------------------------------------------------------------------

class TestWFAConfig:
    def test_default_values(self):
        cfg = WFAConfig()
        assert cfg.n_folds == 5
        assert cfg.min_is_pct == 0.4
        assert cfg.use_validation is True
        assert cfg.val_pct_within_is == 0.2
        assert cfg.top_n_for_val == 10
        assert cfg.min_trades_for_val == 30

    def test_compute_fold_ranges_basic(self):
        """1000本/5フォールド: 各OOS区間が120本"""
        cfg = WFAConfig(n_folds=5, min_is_pct=0.4)
        folds = cfg.compute_fold_ranges(1000)

        assert len(folds) == 5

        # Fold 0: IS=[0, 400], OOS=[400, 520]
        assert folds[0] == ((0, 400), (400, 520))
        # Fold 1: IS=[0, 520], OOS=[520, 640]
        assert folds[1] == ((0, 520), (520, 640))
        # Fold 4: IS=[0, 880], OOS=[880, 1000]
        assert folds[4] == ((0, 880), (880, 1000))

    def test_fold_ranges_no_overlap(self):
        """OOS区間が重複しないこと"""
        cfg = WFAConfig(n_folds=5, min_is_pct=0.4)
        folds = cfg.compute_fold_ranges(1000)

        for i in range(len(folds) - 1):
            _, oos_current = folds[i]
            _, oos_next = folds[i + 1]
            # 現在の OOS end == 次の OOS start
            assert oos_current[1] == oos_next[0]

    def test_fold_ranges_cover_remaining(self):
        """全OOS区間がmin_is以降のデータをカバー"""
        cfg = WFAConfig(n_folds=5, min_is_pct=0.4)
        folds = cfg.compute_fold_ranges(1000)

        # 最初のOOS開始 = min_is_bars
        first_oos_start = folds[0][1][0]
        assert first_oos_start == 400

        # 最後のOOS終了 = 1000
        last_oos_end = folds[-1][1][1]
        assert last_oos_end == 1000

    def test_fold_is_always_starts_at_zero(self):
        """Anchored WFA: ISは常に0から開始"""
        cfg = WFAConfig(n_folds=5, min_is_pct=0.4)
        folds = cfg.compute_fold_ranges(1000)

        for is_range, _ in folds:
            assert is_range[0] == 0

    def test_fold_is_grows(self):
        """Anchored WFA: IS範囲が段階的に拡大"""
        cfg = WFAConfig(n_folds=5, min_is_pct=0.4)
        folds = cfg.compute_fold_ranges(1000)

        is_ends = [is_range[1] for is_range, _ in folds]
        for i in range(len(is_ends) - 1):
            assert is_ends[i] < is_ends[i + 1]

    def test_too_few_bars_raises(self):
        """データ不足でValueError"""
        cfg = WFAConfig(n_folds=10, min_is_pct=0.9)
        with pytest.raises(ValueError, match="データ不足"):
            cfg.compute_fold_ranges(10)

    def test_different_config(self):
        """min_is_pct=0.3, n_folds=7 でも正しく計算"""
        cfg = WFAConfig(n_folds=7, min_is_pct=0.3)
        folds = cfg.compute_fold_ranges(1000)

        assert len(folds) == 7
        # min_is = 300, remaining = 700, step = 100
        assert folds[0] == ((0, 300), (300, 400))
        assert folds[6] == ((0, 900), (900, 1000))


# ---------------------------------------------------------------------------
# run_walk_forward_analysis 統合テスト
# ---------------------------------------------------------------------------

class TestRunWalkForwardAnalysis:
    def _setup_mock_optimizer(self, n_folds, n_regimes, use_validation):
        """モック optimizer を構築"""
        mock_opt = MagicMock()

        # 各フォールドで:
        # use_validation=True: Train + Val + OOS (per regime) = 1 + n_regimes + n_regimes
        # use_validation=False: IS + OOS (per regime) = 1 + n_regimes
        # + 最終フォールドの IS結果取得用に追加1回

        results = []
        for fold_idx in range(n_folds):
            if use_validation:
                # Train結果
                train_entries = [
                    _make_entry(
                        template="tmpl_a", regime=f"regime_{r}",
                        score=0.8, total_profit_pct=10.0 + fold_idx,
                    )
                    for r in range(n_regimes)
                ]
                results.append(OptimizationResultSet(entries=train_entries))

                # Val結果 (per regime)
                for r in range(n_regimes):
                    val_entry = _make_entry(
                        template="tmpl_a", regime=f"regime_{r}",
                        score=0.7, total_profit_pct=8.0 + fold_idx,
                    )
                    results.append(OptimizationResultSet(entries=[val_entry]))
            else:
                # IS結果
                is_entries = [
                    _make_entry(
                        template="tmpl_a", regime=f"regime_{r}",
                        score=0.8, total_profit_pct=10.0 + fold_idx,
                    )
                    for r in range(n_regimes)
                ]
                results.append(OptimizationResultSet(entries=is_entries))

            # OOS結果 (per regime)
            for r in range(n_regimes):
                oos_entry = _make_entry(
                    template="tmpl_a", regime=f"regime_{r}",
                    score=0.5, total_profit_pct=5.0 + fold_idx,
                )
                results.append(OptimizationResultSet(entries=[oos_entry]))

        # 最終フォールドの IS結果取得用
        final_is_entries = [
            _make_entry(
                template="tmpl_a", regime=f"regime_{r}",
                score=0.8, total_profit_pct=10.0,
            )
            for r in range(n_regimes)
        ]
        results.append(OptimizationResultSet(entries=final_is_entries))

        mock_opt.run.side_effect = results
        return mock_opt

    def test_all_folds_executed(self):
        """全フォールドが実行される"""
        import pandas as pd

        n_folds = 3
        mock_opt = self._setup_mock_optimizer(n_folds, 1, use_validation=False)

        df = pd.DataFrame({"close": range(100)})
        configs = [{"_template_name": "t", "_params": {}, "indicators": []}]
        cfg = WFAConfig(n_folds=n_folds, min_is_pct=0.4, use_validation=False)

        result = run_walk_forward_analysis(
            df=df,
            all_configs=configs,
            target_regimes=["regime_0"],
            wfa_config=cfg,
            optimizer=mock_opt,
        )

        assert len(result.folds) == n_folds

    def test_data_ranges_correct(self):
        """各フォールドのdata_rangeがoptimizer.runに正しく渡される"""
        import pandas as pd

        n_folds = 3
        mock_opt = self._setup_mock_optimizer(n_folds, 1, use_validation=False)

        df = pd.DataFrame({"close": range(1000)})
        configs = [{"_template_name": "t", "_params": {}, "indicators": []}]
        cfg = WFAConfig(n_folds=n_folds, min_is_pct=0.4, use_validation=False)

        result = run_walk_forward_analysis(
            df=df,
            all_configs=configs,
            target_regimes=["regime_0"],
            wfa_config=cfg,
            optimizer=mock_opt,
        )

        # フォールド範囲を確認
        assert result.folds[0].is_range == (0, 400)
        assert result.folds[0].oos_range == (400, 600)
        assert result.folds[1].is_range == (0, 600)
        assert result.folds[1].oos_range == (600, 800)
        assert result.folds[2].is_range == (0, 800)
        assert result.folds[2].oos_range == (800, 1000)

    def test_aggregate_metrics_computed(self):
        """WFE, CR, stitched PnL が計算される"""
        import pandas as pd

        n_folds = 3
        mock_opt = self._setup_mock_optimizer(n_folds, 1, use_validation=False)

        df = pd.DataFrame({"close": range(1000)})
        configs = [{"_template_name": "t", "_params": {}, "indicators": []}]
        cfg = WFAConfig(n_folds=n_folds, min_is_pct=0.4, use_validation=False)

        result = run_walk_forward_analysis(
            df=df,
            all_configs=configs,
            target_regimes=["regime_0"],
            wfa_config=cfg,
            optimizer=mock_opt,
        )

        assert "regime_0" in result.wfe
        assert "regime_0" in result.consistency_ratio
        assert "regime_0" in result.stitched_oos_pnl

    def test_with_validation(self):
        """use_validation=True で Train/Val 分割が機能する"""
        import pandas as pd

        n_folds = 2
        mock_opt = self._setup_mock_optimizer(n_folds, 1, use_validation=True)

        df = pd.DataFrame({"close": range(1000)})
        configs = [{"_template_name": "t", "_params": {}, "indicators": []}]
        cfg = WFAConfig(
            n_folds=n_folds, min_is_pct=0.4,
            use_validation=True, val_pct_within_is=0.2,
        )

        result = run_walk_forward_analysis(
            df=df,
            all_configs=configs,
            target_regimes=["regime_0"],
            wfa_config=cfg,
            optimizer=mock_opt,
        )

        assert len(result.folds) == n_folds
        # 各フォールドに戦略が選択されている
        for fold in result.folds:
            assert "regime_0" in fold.selected_strategy
            assert "regime_0" in fold.oos_results

    def test_progress_callback(self):
        """進捗コールバックが呼ばれ最終呼び出しが「WFA完了」"""
        import pandas as pd

        n_folds = 2
        mock_opt = self._setup_mock_optimizer(n_folds, 1, use_validation=False)
        cb = MagicMock()

        df = pd.DataFrame({"close": range(1000)})
        configs = [{"_template_name": "t", "_params": {}, "indicators": []}]
        cfg = WFAConfig(n_folds=n_folds, min_is_pct=0.4, use_validation=False)

        run_walk_forward_analysis(
            df=df,
            all_configs=configs,
            target_regimes=["regime_0"],
            wfa_config=cfg,
            optimizer=mock_opt,
            progress_callback=cb,
        )

        # 最終呼び出し
        last_call = cb.call_args_list[-1]
        assert last_call[0][2] == "WFA完了"

    def test_final_train_results_stored(self):
        """最終フォールドのIS結果がfinal_train_resultsに格納される"""
        import pandas as pd

        n_folds = 2
        mock_opt = self._setup_mock_optimizer(n_folds, 1, use_validation=False)

        df = pd.DataFrame({"close": range(1000)})
        configs = [{"_template_name": "t", "_params": {}, "indicators": []}]
        cfg = WFAConfig(n_folds=n_folds, min_is_pct=0.4, use_validation=False)

        result = run_walk_forward_analysis(
            df=df,
            all_configs=configs,
            target_regimes=["regime_0"],
            wfa_config=cfg,
            optimizer=mock_opt,
        )

        assert result.final_train_results is not None


# ---------------------------------------------------------------------------
# 集約メトリクス計算テスト
# ---------------------------------------------------------------------------

class TestComputeAggregateMetrics:
    def test_wfe_calculation(self):
        """IS PnL=10, OOS PnL=5 → WFE=0.5"""
        result_set = WFAResultSet(config=WFAConfig())

        for i in range(3):
            is_entry = _make_entry(
                regime="uptrend", total_profit_pct=10.0,
            )
            oos_entry = _make_entry(
                regime="uptrend", total_profit_pct=5.0,
            )
            result_set.folds.append(WFAFoldResult(
                fold_index=i,
                is_range=(0, 100),
                oos_range=(100, 200),
                selected_strategy={"uptrend": is_entry},
                oos_results={"uptrend": oos_entry},
            ))

        _compute_aggregate_metrics(result_set, ["uptrend"])
        assert result_set.wfe["uptrend"] == pytest.approx(0.5, abs=0.01)

    def test_consistency_ratio(self):
        """3フォールド中2つOOS正 → CR=2/3"""
        result_set = WFAResultSet(config=WFAConfig())

        pnls = [5.0, -3.0, 2.0]
        for i, pnl in enumerate(pnls):
            is_entry = _make_entry(regime="uptrend", total_profit_pct=10.0)
            oos_entry = _make_entry(regime="uptrend", total_profit_pct=pnl)
            result_set.folds.append(WFAFoldResult(
                fold_index=i,
                is_range=(0, 100),
                oos_range=(100, 200),
                selected_strategy={"uptrend": is_entry},
                oos_results={"uptrend": oos_entry},
            ))

        _compute_aggregate_metrics(result_set, ["uptrend"])
        assert result_set.consistency_ratio["uptrend"] == pytest.approx(2.0 / 3.0, abs=0.01)

    def test_stitched_oos_pnl(self):
        """OOS PnL合計 = 5 + (-3) + 2 = 4"""
        result_set = WFAResultSet(config=WFAConfig())

        pnls = [5.0, -3.0, 2.0]
        for i, pnl in enumerate(pnls):
            is_entry = _make_entry(regime="uptrend", total_profit_pct=10.0)
            oos_entry = _make_entry(regime="uptrend", total_profit_pct=pnl)
            result_set.folds.append(WFAFoldResult(
                fold_index=i,
                is_range=(0, 100),
                oos_range=(100, 200),
                selected_strategy={"uptrend": is_entry},
                oos_results={"uptrend": oos_entry},
            ))

        _compute_aggregate_metrics(result_set, ["uptrend"])
        assert result_set.stitched_oos_pnl["uptrend"] == pytest.approx(4.0, abs=0.01)

    def test_strategy_stability_all_same(self):
        """全フォールド同一戦略 → stability=1.0"""
        result_set = WFAResultSet(config=WFAConfig())

        for i in range(5):
            is_entry = _make_entry(
                template="tmpl_a", regime="uptrend",
                params={"sma": 20}, total_profit_pct=10.0,
            )
            oos_entry = _make_entry(regime="uptrend", total_profit_pct=5.0)
            result_set.folds.append(WFAFoldResult(
                fold_index=i,
                is_range=(0, 100),
                oos_range=(100, 200),
                selected_strategy={"uptrend": is_entry},
                oos_results={"uptrend": oos_entry},
            ))

        _compute_aggregate_metrics(result_set, ["uptrend"])
        assert result_set.strategy_stability["uptrend"] == 1.0

    def test_strategy_stability_mixed(self):
        """5フォールド中3つが同一戦略 → stability=3/5=0.6"""
        result_set = WFAResultSet(config=WFAConfig())

        templates = ["tmpl_a", "tmpl_a", "tmpl_a", "tmpl_b", "tmpl_c"]
        for i, tmpl in enumerate(templates):
            is_entry = _make_entry(
                template=tmpl, regime="uptrend",
                params={"sma": 20}, total_profit_pct=10.0,
            )
            oos_entry = _make_entry(regime="uptrend", total_profit_pct=5.0)
            result_set.folds.append(WFAFoldResult(
                fold_index=i,
                is_range=(0, 100),
                oos_range=(100, 200),
                selected_strategy={"uptrend": is_entry},
                oos_results={"uptrend": oos_entry},
            ))

        _compute_aggregate_metrics(result_set, ["uptrend"])
        assert result_set.strategy_stability["uptrend"] == pytest.approx(0.6, abs=0.01)
