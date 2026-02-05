"""遺伝的アルゴリズム（GA）最適化のテスト"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from optimizer.genetic import (
    Individual,
    GenerationRecord,
    GAResult,
    GAConfig,
    GeneticOptimizer,
)
from optimizer.templates import BUILTIN_TEMPLATES


# ---------------------------------------------------------------------------
# Individual テスト
# ---------------------------------------------------------------------------

class TestIndividual:
    def test_create_individual(self):
        """個体を作成できる"""
        ind = Individual(
            template_name="rsi_reversal",
            params={"rsi_period": 14, "rsi_threshold": 30},
        )
        assert ind.template_name == "rsi_reversal"
        assert ind.params["rsi_period"] == 14
        assert ind.score == 0.0
        assert ind.status == "pending"

    def test_to_config(self):
        """個体からconfig辞書を生成できる"""
        ind = Individual(
            template_name="rsi_reversal",
            params={"rsi_period": 14, "rsi_threshold": 30},
        )
        config = ind.to_config()

        assert config["_template_name"] == "rsi_reversal"
        assert config["_params"]["rsi_period"] == 14
        assert config["side"] == "long"

    def test_to_config_with_exit_override(self):
        """exit条件を上書きできる"""
        ind = Individual(
            template_name="rsi_reversal",
            params={"rsi_period": 14, "rsi_threshold": 30},
        )
        exit_config = {"take_profit_pct": 3.0, "stop_loss_pct": 1.5}
        config = ind.to_config(exit_config)

        assert config["exit"]["take_profit_pct"] == 3.0
        assert config["exit"]["stop_loss_pct"] == 1.5

    def test_to_dict(self):
        """JSON出力用の辞書を生成できる"""
        ind = Individual(
            template_name="macd_signal",
            params={"macd_fast": 12, "macd_slow": 26},
            score=0.15,
            pnl=15.0,
            status="survived",
        )
        d = ind.to_dict()

        assert d["template"] == "macd_signal"
        assert d["params"]["macd_fast"] == 12
        assert d["score"] == 0.15
        assert d["status"] == "survived"


# ---------------------------------------------------------------------------
# GenerationRecord テスト
# ---------------------------------------------------------------------------

class TestGenerationRecord:
    def test_to_dict(self):
        """世代記録を辞書に変換できる"""
        record = GenerationRecord(
            generation=1,
            individuals=[{"template": "rsi_reversal", "score": 0.1}],
            best_score=0.15,
            avg_score=0.08,
        )
        d = record.to_dict()

        assert d["generation"] == 1
        assert d["best_score"] == 0.15
        assert len(d["individuals"]) == 1


# ---------------------------------------------------------------------------
# GAResult テスト
# ---------------------------------------------------------------------------

class TestGAResult:
    def test_to_dict(self):
        """GA結果を辞書に変換できる"""
        result = GAResult(
            symbol="BTCUSDT",
            regime="range",
            total_evaluations=200,
            execution_time_sec=120.5,
        )
        result.final_winner = {"template": "macd_signal", "score": 0.18}

        d = result.to_dict()

        assert d["symbol"] == "BTCUSDT"
        assert d["regime"] == "range"
        assert d["total_evaluations"] == 200
        assert d["final_winner"]["template"] == "macd_signal"


# ---------------------------------------------------------------------------
# GAConfig テスト
# ---------------------------------------------------------------------------

class TestGAConfig:
    def test_default_values(self):
        """デフォルト値が設定されている"""
        config = GAConfig()

        assert config.population_size == 20
        assert config.max_generations == 15
        assert config.elite_ratio == 0.25
        assert config.mutation_rate == 0.1

    def test_custom_values(self):
        """カスタム値を設定できる"""
        config = GAConfig(
            population_size=50,
            max_generations=30,
            elite_ratio=0.3,
        )

        assert config.population_size == 50
        assert config.max_generations == 30


# ---------------------------------------------------------------------------
# GeneticOptimizer テスト
# ---------------------------------------------------------------------------

class TestGeneticOptimizer:
    def test_create_initial_population(self):
        """初期集団を生成できる"""
        optimizer = GeneticOptimizer(
            templates=["rsi_reversal", "macd_signal"],
            ga_config=GAConfig(population_size=10),
        )
        population = optimizer._create_initial_population()

        assert len(population) == 10
        for ind in population:
            assert ind.template_name in ["rsi_reversal", "macd_signal"]
            assert ind.generation == 1

    def test_random_params(self):
        """ランダムパラメータを生成できる"""
        optimizer = GeneticOptimizer()
        template = BUILTIN_TEMPLATES["rsi_reversal"]
        params = optimizer._random_params(template)

        assert "rsi_period" in params
        assert "rsi_threshold" in params

    def test_tournament_select(self):
        """トーナメント選択が動作する"""
        optimizer = GeneticOptimizer(
            ga_config=GAConfig(tournament_size=2),
        )

        # スコア付きの個体リスト
        population = [
            Individual("rsi_reversal", {}, score=0.1),
            Individual("macd_signal", {}, score=0.5),
            Individual("bb_bounce", {}, score=0.3),
        ]

        # 複数回選択してもエラーにならない
        for _ in range(10):
            selected = optimizer._tournament_select(population)
            assert selected in population

    def test_crossover_same_template(self):
        """同一テンプレートの交叉"""
        optimizer = GeneticOptimizer()

        parent1 = Individual(
            "rsi_reversal",
            {"rsi_period": 14, "rsi_threshold": 30},
        )
        parent2 = Individual(
            "rsi_reversal",
            {"rsi_period": 20, "rsi_threshold": 25},
        )

        child = optimizer._crossover(parent1, parent2, generation=2)

        assert child.template_name == "rsi_reversal"
        assert child.generation == 2
        assert "rsi_period" in child.params

    def test_crossover_different_template(self):
        """異なるテンプレートの交叉"""
        optimizer = GeneticOptimizer()

        parent1 = Individual("rsi_reversal", {"rsi_period": 14})
        parent2 = Individual("macd_signal", {"macd_fast": 12})

        child = optimizer._crossover(parent1, parent2, generation=2)

        assert child.template_name in ["rsi_reversal", "macd_signal"]

    def test_mutate(self):
        """突然変異が動作する"""
        optimizer = GeneticOptimizer()

        ind = Individual(
            "rsi_reversal",
            {"rsi_period": 14, "rsi_threshold": 30},
        )

        # 複数回突然変異してもエラーにならない
        for _ in range(10):
            optimizer._mutate(ind)

        # パラメータがテンプレートの範囲内に収まっている
        template = BUILTIN_TEMPLATES["rsi_reversal"]
        for pr in template.param_ranges:
            assert ind.params[pr.name] in pr.values()

    def test_check_convergence_not_converged(self):
        """収束していない場合"""
        optimizer = GeneticOptimizer(
            ga_config=GAConfig(
                convergence_generations=3,
                convergence_threshold=0.001,
            ),
        )

        # スコアが改善している
        best_scores = [0.1, 0.12, 0.15, 0.18]
        assert not optimizer._check_convergence(best_scores)

    def test_check_convergence_converged(self):
        """収束した場合"""
        optimizer = GeneticOptimizer(
            ga_config=GAConfig(
                convergence_generations=3,
                convergence_threshold=0.001,
            ),
        )

        # スコアが変わらない
        best_scores = [0.18, 0.18, 0.1801, 0.1802]
        assert optimizer._check_convergence(best_scores)

    def test_create_next_generation(self):
        """次世代を生成できる"""
        optimizer = GeneticOptimizer(
            templates=["rsi_reversal", "macd_signal"],
            ga_config=GAConfig(
                population_size=10,
                elite_ratio=0.2,
            ),
        )

        # 評価済みの集団
        population = []
        for i in range(10):
            ind = Individual(
                "rsi_reversal",
                {"rsi_period": 14, "rsi_threshold": 30},
                score=i * 0.1,
            )
            population.append(ind)

        next_gen = optimizer._create_next_generation(population, current_gen=1)

        assert len(next_gen) == 10
        for ind in next_gen:
            assert ind.generation == 2

        # 上位個体が淘汰されていない
        survivors = [ind for ind in population if ind.status == "survived"]
        assert len(survivors) == 2  # 20% of 10

    def test_individual_config_applies_params(self):
        """パラメータがconfig内で正しく置換される"""
        ind = Individual(
            template_name="ma_crossover",
            params={"fast_period": 15, "slow_period": 60},
        )
        config = ind.to_config()

        # インジケーター設定でperiodが置換されている
        indicators = config["indicators"]
        periods = [i.get("period") for i in indicators]
        assert 15 in periods
        assert 60 in periods
