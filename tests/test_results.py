"""
results.py のユニットテスト

- OptimizationEntry の warnings フィールド
- OptimizationResultSet のフィルタリング・ランキング・DataFrame変換
- from_json による復元
"""

import json
import os
import tempfile

import pytest

from metrics.calculator import BacktestMetrics
from optimizer.results import OptimizationEntry, OptimizationResultSet


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
    warnings=None,
    **metric_overrides,
) -> OptimizationEntry:
    return OptimizationEntry(
        template_name=template,
        params={"sma_period": 20, "rsi_period": 14},
        trend_regime=regime,
        config={"indicators": []},
        metrics=_make_metrics(**metric_overrides),
        composite_score=score,
        warnings=warnings or [],
    )


# ---------------------------------------------------------------------------
# OptimizationEntry
# ---------------------------------------------------------------------------

class TestOptimizationEntry:
    def test_default_warnings_empty(self):
        entry = _make_entry()
        assert entry.warnings == []

    def test_warnings_field(self):
        entry = _make_entry(warnings=["PF=3.0 — 過学習の疑い"])
        assert len(entry.warnings) == 1
        assert "過学習" in entry.warnings[0]

    def test_param_str(self):
        entry = _make_entry()
        assert "sma_period=20" in entry.param_str
        assert "rsi_period=14" in entry.param_str


# ---------------------------------------------------------------------------
# OptimizationResultSet
# ---------------------------------------------------------------------------

class TestOptimizationResultSet:
    def test_add_and_count(self):
        rs = OptimizationResultSet()
        rs.add(_make_entry(score=0.7))
        rs.add(_make_entry(score=0.5))
        assert rs.total_combinations == 2

    def test_ranked_descending(self):
        rs = OptimizationResultSet()
        rs.add(_make_entry(score=0.3))
        rs.add(_make_entry(score=0.9))
        rs.add(_make_entry(score=0.6))
        ranked = rs.ranked()
        scores = [e.composite_score for e in ranked]
        assert scores == [0.9, 0.6, 0.3]

    def test_best(self):
        rs = OptimizationResultSet()
        rs.add(_make_entry(score=0.3))
        rs.add(_make_entry(score=0.9))
        assert rs.best.composite_score == 0.9

    def test_best_empty(self):
        rs = OptimizationResultSet()
        assert rs.best is None

    def test_filter_regime(self):
        rs = OptimizationResultSet()
        rs.add(_make_entry(regime="uptrend", score=0.7))
        rs.add(_make_entry(regime="downtrend", score=0.5))
        rs.add(_make_entry(regime="uptrend", score=0.3))
        filtered = rs.filter_regime("uptrend")
        assert filtered.total_combinations == 2
        assert all(e.trend_regime == "uptrend" for e in filtered.entries)

    def test_filter_template(self):
        rs = OptimizationResultSet()
        rs.add(_make_entry(template="sma_cross", score=0.7))
        rs.add(_make_entry(template="rsi_bounce", score=0.5))
        filtered = rs.filter_template("sma_cross")
        assert filtered.total_combinations == 1

    def test_to_dataframe_columns(self):
        rs = OptimizationResultSet()
        rs.add(_make_entry(score=0.7, warnings=["警告A"]))
        rs.add(_make_entry(score=0.5))
        df = rs.to_dataframe()
        assert "warnings" in df.columns
        assert df.iloc[0]["warnings"] == "警告A"
        assert df.iloc[1]["warnings"] == ""

    def test_to_dataframe_sorted_by_score(self):
        rs = OptimizationResultSet()
        rs.add(_make_entry(score=0.3))
        rs.add(_make_entry(score=0.9))
        rs.add(_make_entry(score=0.6))
        df = rs.to_dataframe()
        assert list(df["score"]) == [0.9, 0.6, 0.3]

    def test_rescore(self):
        """rescore で全エントリーのスコアが再計算される"""
        rs = OptimizationResultSet()
        e = _make_entry(score=0.5)
        rs.add(e)
        old_score = e.composite_score
        rs.rescore()
        # デフォルト重みで再計算されるのでスコアが変わる（元が手動設定の0.5）
        assert e.composite_score != old_score

    def test_metadata_preserved_in_filter(self):
        """フィルタリング後もメタデータが維持される"""
        rs = OptimizationResultSet(
            symbol="BTCUSDT",
            execution_tf="1h",
            htf="4h",
            data_source="trimmed",
            data_period_start="2025-01-01",
            data_period_end="2025-01-31",
        )
        rs.add(_make_entry(regime="uptrend"))
        filtered = rs.filter_regime("uptrend")
        assert filtered.symbol == "BTCUSDT"
        assert filtered.data_source == "trimmed"


# ---------------------------------------------------------------------------
# from_json
# ---------------------------------------------------------------------------

class TestFromJson:
    def test_round_trip(self):
        """JSON保存 → from_json で復元"""
        data = {
            "symbol": "ETHUSDT",
            "execution_tf": "1h",
            "htf": "4h",
            "data_source": "original",
            "data_period": {"start": "2025-01-01", "end": "2025-06-01"},
            "results": [
                {
                    "template": "sma_cross",
                    "params": {"sma_fast": 10, "sma_slow": 50},
                    "regime": "uptrend",
                    "score": 0.75,
                    "config": {"indicators": []},
                    "metrics": {
                        "trades": 40,
                        "win_rate": 55.0,
                        "profit_factor": 1.8,
                        "total_pnl": 20.5,
                        "max_dd": 8.0,
                        "sharpe": 1.5,
                    },
                },
            ],
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(data, f)
            tmp_path = f.name

        try:
            rs = OptimizationResultSet.from_json(tmp_path)
            assert rs.symbol == "ETHUSDT"
            assert rs.total_combinations == 1
            entry = rs.entries[0]
            assert entry.template_name == "sma_cross"
            assert entry.composite_score == 0.75
            assert entry.metrics.total_trades == 40
        finally:
            os.unlink(tmp_path)
