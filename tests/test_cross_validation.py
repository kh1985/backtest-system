"""
cross_validation.py のユニットテスト

- CrossValidationVerdict の判定ロジック
- run_cross_validation の統合テスト
"""

from metrics.calculator import BacktestMetrics
from optimizer.results import OptimizationEntry, OptimizationResultSet
from optimizer.cross_validation import (
    CrossValidationVerdict,
    StrategyProfile,
    run_cross_validation,
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
    **metric_overrides,
) -> OptimizationEntry:
    return OptimizationEntry(
        template_name=template,
        params={"sma_period": 20},
        trend_regime=regime,
        config={"indicators": [], "entry_conditions": []},
        metrics=_make_metrics(**metric_overrides),
        composite_score=score,
    )


def _make_result_set(symbol, entries) -> OptimizationResultSet:
    rs = OptimizationResultSet(entries=entries)
    rs.symbol = symbol
    return rs


# ---------------------------------------------------------------------------
# テスト
# ---------------------------------------------------------------------------

class TestCrossValidation:
    def test_all_pass_verdict(self):
        """全銘柄PnL正 → ALL_PASS"""
        results = [
            _make_result_set("BTCUSDT", [
                _make_entry("ma_cross", "uptrend", total_profit_pct=10.0),
            ]),
            _make_result_set("ETHUSDT", [
                _make_entry("ma_cross", "uptrend", total_profit_pct=5.0),
            ]),
            _make_result_set("SOLUSDT", [
                _make_entry("ma_cross", "uptrend", total_profit_pct=3.0),
            ]),
        ]

        cv = run_cross_validation(results, ["uptrend"])
        regime_result = cv.regime_results["uptrend"]

        assert len(regime_result.strategy_profiles) == 1
        profile = regime_result.strategy_profiles[0]
        assert profile.template_name == "ma_cross"
        assert profile.verdict == CrossValidationVerdict.ALL_PASS
        assert profile.pass_rate == 1.0
        assert len(profile.symbols_passed) == 3

    def test_majority_verdict(self):
        """3/5銘柄PnL正 → MAJORITY"""
        results = [
            _make_result_set("BTC", [_make_entry("tmpl_a", "uptrend", total_profit_pct=10.0)]),
            _make_result_set("ETH", [_make_entry("tmpl_a", "uptrend", total_profit_pct=5.0)]),
            _make_result_set("SOL", [_make_entry("tmpl_a", "uptrend", total_profit_pct=2.0)]),
            _make_result_set("DOT", [_make_entry("tmpl_a", "uptrend", total_profit_pct=-3.0)]),
            _make_result_set("ADA", [_make_entry("tmpl_a", "uptrend", total_profit_pct=-1.0)]),
        ]

        cv = run_cross_validation(results, ["uptrend"])
        profile = cv.regime_results["uptrend"].strategy_profiles[0]

        assert profile.verdict == CrossValidationVerdict.MAJORITY
        assert len(profile.symbols_passed) == 3
        assert len(profile.symbols_failed) == 2

    def test_fail_verdict(self):
        """全銘柄PnL負 → FAIL"""
        results = [
            _make_result_set("BTC", [_make_entry("tmpl_a", "uptrend", total_profit_pct=-5.0)]),
            _make_result_set("ETH", [_make_entry("tmpl_a", "uptrend", total_profit_pct=-3.0)]),
        ]

        cv = run_cross_validation(results, ["uptrend"])
        profile = cv.regime_results["uptrend"].strategy_profiles[0]

        assert profile.verdict == CrossValidationVerdict.FAIL
        assert len(profile.symbols_passed) == 0

    def test_minority_verdict(self):
        """1/3銘柄PnL正 → MINORITY"""
        results = [
            _make_result_set("BTC", [_make_entry("tmpl_a", "uptrend", total_profit_pct=5.0)]),
            _make_result_set("ETH", [_make_entry("tmpl_a", "uptrend", total_profit_pct=-3.0)]),
            _make_result_set("SOL", [_make_entry("tmpl_a", "uptrend", total_profit_pct=-1.0)]),
        ]

        cv = run_cross_validation(results, ["uptrend"])
        profile = cv.regime_results["uptrend"].strategy_profiles[0]

        assert profile.verdict == CrossValidationVerdict.MINORITY
        assert len(profile.symbols_passed) == 1

    def test_dominant_template(self):
        """最頻テンプレートが正しく検出される"""
        results = [
            _make_result_set("BTC", [_make_entry("ma_cross", "uptrend", total_profit_pct=10.0, score=0.9)]),
            _make_result_set("ETH", [_make_entry("ma_cross", "uptrend", total_profit_pct=5.0, score=0.8)]),
            _make_result_set("SOL", [_make_entry("rsi_rev", "uptrend", total_profit_pct=8.0, score=0.85)]),
        ]

        cv = run_cross_validation(results, ["uptrend"])
        regime_result = cv.regime_results["uptrend"]

        assert regime_result.dominant_template == "ma_cross"
        assert regime_result.dominant_template_ratio == 2.0 / 3.0

    def test_empty_results(self):
        """空の結果セットでエラーにならない"""
        cv = run_cross_validation([], ["uptrend"])

        assert cv.n_symbols == 0
        assert "uptrend" in cv.regime_results
        assert len(cv.regime_results["uptrend"].strategy_profiles) == 0

    def test_multiple_regimes(self):
        """複数レジームでそれぞれ独立に分析される"""
        results = [
            _make_result_set("BTC", [
                _make_entry("tmpl_a", "uptrend", total_profit_pct=10.0),
                _make_entry("tmpl_b", "downtrend", total_profit_pct=-5.0),
            ]),
            _make_result_set("ETH", [
                _make_entry("tmpl_a", "uptrend", total_profit_pct=5.0),
                _make_entry("tmpl_b", "downtrend", total_profit_pct=3.0),
            ]),
        ]

        cv = run_cross_validation(results, ["uptrend", "downtrend"])

        # uptrend: 全銘柄PnL正
        up_profile = cv.regime_results["uptrend"].strategy_profiles[0]
        assert up_profile.verdict == CrossValidationVerdict.ALL_PASS

        # downtrend: 1/2 = 50% → MAJORITY（BTC=-5 fail, ETH=+3 pass）
        down_profile = cv.regime_results["downtrend"].strategy_profiles[0]
        assert down_profile.verdict == CrossValidationVerdict.MAJORITY

    def test_avg_metrics(self):
        """平均メトリクスが正しく計算される"""
        results = [
            _make_result_set("BTC", [
                _make_entry("tmpl_a", "uptrend",
                            total_profit_pct=10.0, sharpe_ratio=2.0, win_rate=60.0),
            ]),
            _make_result_set("ETH", [
                _make_entry("tmpl_a", "uptrend",
                            total_profit_pct=20.0, sharpe_ratio=3.0, win_rate=70.0),
            ]),
        ]

        cv = run_cross_validation(results, ["uptrend"])
        profile = cv.regime_results["uptrend"].strategy_profiles[0]

        assert profile.avg_pnl == 15.0
        assert profile.avg_sharpe == 2.5
        assert profile.avg_win_rate == 65.0

    def test_symbols_list(self):
        """シンボルリストが正しく記録される"""
        results = [
            _make_result_set("BTC", [_make_entry("tmpl_a", "uptrend")]),
            _make_result_set("ETH", [_make_entry("tmpl_a", "uptrend")]),
        ]

        cv = run_cross_validation(results, ["uptrend"])

        assert cv.symbols == ["BTC", "ETH"]
        assert cv.n_symbols == 2
