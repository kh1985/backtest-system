"""
scoring.py のユニットテスト

- calculate_composite_score の正規化・重み付けロジック
- detect_overfitting_warnings の各警告条件
"""

import pytest

from optimizer.scoring import (
    ScoringWeights,
    calculate_composite_score,
    detect_overfitting_warnings,
)
from metrics.calculator import BacktestMetrics


# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------

def _make_metrics(**overrides) -> BacktestMetrics:
    """テスト用 BacktestMetrics を生成"""
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


# ---------------------------------------------------------------------------
# ScoringWeights
# ---------------------------------------------------------------------------

class TestScoringWeights:
    def test_default_weights_sum_to_one(self):
        w = ScoringWeights()
        assert w.validate()

    def test_custom_weights_valid(self):
        w = ScoringWeights(
            profit_factor=0.3,
            win_rate=0.2,
            max_drawdown=0.2,
            sharpe_ratio=0.2,
            total_return=0.1,
        )
        assert w.validate()

    def test_invalid_weights_rejected(self):
        w = ScoringWeights(
            profit_factor=0.5,
            win_rate=0.5,
            max_drawdown=0.5,
            sharpe_ratio=0.5,
            total_return=0.5,
        )
        assert not w.validate()


# ---------------------------------------------------------------------------
# calculate_composite_score
# ---------------------------------------------------------------------------

class TestCompositeScore:
    def test_score_range_0_to_1(self):
        """任意の入力でスコアは 0~1 に収まる"""
        score = calculate_composite_score(
            profit_factor=1.5,
            win_rate=60.0,
            max_drawdown_pct=10.0,
            sharpe_ratio=1.2,
            total_return_pct=15.0,
        )
        assert 0.0 <= score <= 1.0

    def test_perfect_score(self):
        """全指標が最高値ならスコア ≈ 1.0"""
        score = calculate_composite_score(
            profit_factor=10.0,
            win_rate=100.0,
            max_drawdown_pct=0.0,
            sharpe_ratio=5.0,
            total_return_pct=100.0,
        )
        assert score == pytest.approx(1.0, abs=0.01)

    def test_worst_score(self):
        """全指標が最低値ならスコア ≈ 0.0"""
        score = calculate_composite_score(
            profit_factor=0.0,
            win_rate=0.0,
            max_drawdown_pct=100.0,
            sharpe_ratio=-2.0,
            total_return_pct=-50.0,
        )
        assert score == pytest.approx(0.0, abs=0.01)

    def test_pf_clipping(self):
        """PF > 10 はクリップされる"""
        s1 = calculate_composite_score(
            profit_factor=10.0, win_rate=50, max_drawdown_pct=10,
            sharpe_ratio=1, total_return_pct=0,
        )
        s2 = calculate_composite_score(
            profit_factor=100.0, win_rate=50, max_drawdown_pct=10,
            sharpe_ratio=1, total_return_pct=0,
        )
        assert s1 == pytest.approx(s2)

    def test_custom_weights_affect_score(self):
        """重みを変えるとスコアが変わる"""
        base_args = dict(
            profit_factor=5.0,
            win_rate=80.0,
            max_drawdown_pct=5.0,
            sharpe_ratio=2.0,
            total_return_pct=30.0,
        )
        s_default = calculate_composite_score(**base_args)
        s_pf_heavy = calculate_composite_score(
            **base_args,
            weights=ScoringWeights(
                profit_factor=0.6,
                win_rate=0.1,
                max_drawdown=0.1,
                sharpe_ratio=0.1,
                total_return=0.1,
            ),
        )
        # PF が高いので PF 重視の方がスコアが高い
        assert s_pf_heavy != s_default

    def test_zero_trades_forces_zero_score(self):
        """トレード0件ならスコアは常に0"""
        score = calculate_composite_score(
            profit_factor=10.0,
            win_rate=100.0,
            max_drawdown_pct=0.0,
            sharpe_ratio=5.0,
            total_return_pct=100.0,
            total_trades=0,
        )
        assert score == 0.0


# ---------------------------------------------------------------------------
# detect_overfitting_warnings
# ---------------------------------------------------------------------------

class TestOverfittingWarnings:
    def test_no_warnings_for_normal_metrics(self):
        """正常なメトリクスでは警告なし"""
        m = _make_metrics(
            profit_factor=1.5,
            sharpe_ratio=1.2,
            total_trades=50,
        )
        warnings = detect_overfitting_warnings(m)
        assert warnings == []

    def test_high_pf_warning(self):
        """PF > 2.0 で警告"""
        m = _make_metrics(profit_factor=2.5)
        warnings = detect_overfitting_warnings(m)
        assert any("PF=" in w and "過学習" in w for w in warnings)

    def test_high_sharpe_warning(self):
        """Sharpe > 3.0 で警告"""
        m = _make_metrics(sharpe_ratio=3.5)
        warnings = detect_overfitting_warnings(m)
        assert any("Sharpe=" in w and "非現実的" in w for w in warnings)

    def test_low_trades_warning(self):
        """トレード数 < 30 で警告"""
        m = _make_metrics(total_trades=15)
        warnings = detect_overfitting_warnings(m)
        assert any("統計的に不十分" in w for w in warnings)

    def test_oos_decay_warning(self):
        """OOS PnL 劣化 > 50% で警告"""
        train = _make_metrics(total_profit_pct=20.0)
        test = _make_metrics(total_profit_pct=5.0)  # 75% 劣化
        warnings = detect_overfitting_warnings(train, test)
        assert any("劣化" in w for w in warnings)

    def test_no_oos_decay_when_test_is_good(self):
        """OOS 劣化が軽微なら警告なし"""
        train = _make_metrics(total_profit_pct=20.0)
        test = _make_metrics(total_profit_pct=15.0)  # 25% 劣化
        warnings = detect_overfitting_warnings(train, test)
        assert not any("劣化" in w for w in warnings)

    def test_no_oos_warning_when_train_negative(self):
        """Train PnL がマイナスの場合は OOS 劣化警告を出さない"""
        train = _make_metrics(total_profit_pct=-5.0)
        test = _make_metrics(total_profit_pct=-10.0)
        warnings = detect_overfitting_warnings(train, test)
        assert not any("劣化" in w for w in warnings)

    def test_multiple_warnings(self):
        """複数条件に該当すれば複数警告"""
        m = _make_metrics(
            profit_factor=3.0,
            sharpe_ratio=4.0,
            total_trades=10,
        )
        warnings = detect_overfitting_warnings(m)
        assert len(warnings) == 3
