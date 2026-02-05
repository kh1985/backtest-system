"""
遺伝的アルゴリズム（GA）最適化エンジン

テンプレート × パラメータ を遺伝的アルゴリズムで効率的に探索。
グリッドサーチと比較して探索回数を大幅に削減（1/3〜1/5）。

主要コンポーネント:
- Individual: 個体（テンプレート + パラメータ）
- GeneticOptimizer: GA実行エンジン
- GAResult: 全世代の履歴を含む結果
"""

import copy
import json
import logging
import random
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .templates import BUILTIN_TEMPLATES, StrategyTemplate, ParameterRange
from .scoring import ScoringWeights, calculate_composite_score
from .results import OptimizationEntry
from .grid import GridSearchOptimizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# データ構造
# ---------------------------------------------------------------------------

@dataclass
class Individual:
    """GA個体: テンプレート + パラメータの組み合わせ"""
    template_name: str
    params: Dict[str, Any]

    # 評価結果（バックテスト後に設定）
    score: float = 0.0
    pnl: float = 0.0
    sharpe: float = 0.0
    trades: int = 0
    win_rate: float = 0.0
    max_dd: float = 0.0

    # 状態
    status: str = "pending"  # pending / survived / eliminated
    elimination_reason: str = ""
    generation: int = 0

    def to_config(self, exit_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """バックテスト用のconfig辞書を生成"""
        template = BUILTIN_TEMPLATES.get(self.template_name)
        if template is None:
            raise ValueError(f"Unknown template: {self.template_name}")

        # テンプレートのconfig_templateをコピーしてパラメータを適用
        config = copy.deepcopy(template.config_template)
        config = self._apply_params(config, self.params)
        config["_template_name"] = self.template_name
        config["_params"] = dict(self.params)

        # exit条件を上書き（固定の場合）
        if exit_config:
            config["exit"] = exit_config

        return config

    def _apply_params(self, obj: Any, params: Dict[str, Any]) -> Any:
        """テンプレート内のプレースホルダーを置換"""
        if isinstance(obj, dict):
            return {k: self._apply_params(v, params) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._apply_params(item, params) for item in obj]
        elif isinstance(obj, str):
            if obj.startswith("{") and obj.endswith("}"):
                key = obj[1:-1]
                if key in params:
                    return params[key]
            result = obj
            for key, val in params.items():
                result = result.replace("{" + key + "}", str(val))
            return result
        return obj

    def to_dict(self) -> Dict[str, Any]:
        """JSON出力用の辞書"""
        return {
            "template": self.template_name,
            "params": self.params,
            "score": self.score,
            "pnl": self.pnl,
            "sharpe": self.sharpe,
            "trades": self.trades,
            "win_rate": self.win_rate,
            "max_dd": self.max_dd,
            "status": self.status,
            "elimination_reason": self.elimination_reason,
            "generation": self.generation,
        }


@dataclass
class GenerationRecord:
    """1世代の記録"""
    generation: int
    individuals: List[Dict[str, Any]]
    best_score: float
    avg_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generation": self.generation,
            "individuals": self.individuals,
            "best_score": self.best_score,
            "avg_score": self.avg_score,
        }


@dataclass
class GAResult:
    """GA実行結果（全履歴含む）"""
    symbol: str
    regime: str
    generations: List[GenerationRecord] = field(default_factory=list)
    final_winner: Optional[Dict[str, Any]] = None
    total_evaluations: int = 0
    execution_time_sec: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "regime": self.regime,
            "generations": [g.to_dict() for g in self.generations],
            "final_winner": self.final_winner,
            "total_evaluations": self.total_evaluations,
            "execution_time_sec": self.execution_time_sec,
        }

    def save(self, output_dir: str = "results") -> str:
        """結果をJSONファイルに保存"""
        path = Path(output_dir)
        path.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ga_{self.symbol}_{self.regime}_{timestamp}.json"
        filepath = path / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

        logger.info(f"GA結果を保存: {filepath}")
        return str(filepath)


# ---------------------------------------------------------------------------
# GAオプティマイザー
# ---------------------------------------------------------------------------

@dataclass
class GAConfig:
    """GA設定"""
    population_size: int = 20
    max_generations: int = 15
    elite_ratio: float = 0.25  # 上位25%が生き残り
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    tournament_size: int = 3
    convergence_threshold: float = 0.001  # 改善がこれ以下で収束
    convergence_generations: int = 3  # 連続何世代で収束判定


class GeneticOptimizer:
    """遺伝的アルゴリズムによる戦略最適化"""

    def __init__(
        self,
        templates: Optional[List[str]] = None,
        exit_config: Optional[Dict[str, Any]] = None,
        ga_config: Optional[GAConfig] = None,
        scoring_weights: Optional[ScoringWeights] = None,
        initial_capital: float = 10000.0,
        commission_pct: float = 0.04,
        slippage_pct: float = 0.01,
    ):
        """
        Args:
            templates: 使用するテンプレート名のリスト（Noneで全テンプレート）
            exit_config: 固定のexit条件（Noneでテンプレートのデフォルト使用）
            ga_config: GA設定
            scoring_weights: スコアリング重み
        """
        self.templates = templates or list(BUILTIN_TEMPLATES.keys())
        self.exit_config = exit_config
        self.ga_config = ga_config or GAConfig()
        self.scoring_weights = scoring_weights or ScoringWeights()
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct

        # 内部で使うGridSearchOptimizer（評価用）
        self._grid_optimizer = GridSearchOptimizer(
            initial_capital=initial_capital,
            commission_pct=commission_pct,
            slippage_pct=slippage_pct,
            scoring_weights=scoring_weights,
            min_trades=5,
        )

    def run(
        self,
        df: pd.DataFrame,
        target_regime: str,
        trend_column: str = "trend_label",
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        symbol: str = "UNKNOWN",
    ) -> GAResult:
        """
        GA最適化を実行

        Args:
            df: OHLCVデータ（トレンドラベル付き）
            target_regime: 対象レジーム（uptrend/range/downtrend）
            trend_column: トレンドラベルのカラム名
            progress_callback: 進捗コールバック(current, total, description)
            symbol: 銘柄名（結果保存用）

        Returns:
            GAResult: 全世代の履歴を含む結果
        """
        import time
        start_time = time.time()

        result = GAResult(symbol=symbol, regime=target_regime)

        # 初期集団を生成
        population = self._create_initial_population()

        total_evals = 0
        best_scores: List[float] = []

        for gen in range(self.ga_config.max_generations):
            gen_num = gen + 1

            if progress_callback:
                progress_callback(
                    gen_num,
                    self.ga_config.max_generations,
                    f"世代 {gen_num}: 評価中..."
                )

            # 評価
            self._evaluate_population(
                population, df, target_regime, trend_column
            )
            total_evals += len(population)

            # 世代記録
            scored = sorted(population, key=lambda x: x.score, reverse=True)
            best_score = scored[0].score
            avg_score = sum(ind.score for ind in population) / len(population)

            gen_record = GenerationRecord(
                generation=gen_num,
                individuals=[ind.to_dict() for ind in population],
                best_score=best_score,
                avg_score=avg_score,
            )
            result.generations.append(gen_record)
            best_scores.append(best_score)

            logger.info(
                f"世代 {gen_num}: best={best_score:.4f}, avg={avg_score:.4f}"
            )

            # 収束判定
            if self._check_convergence(best_scores):
                logger.info(f"収束検出（世代 {gen_num}）")
                break

            # 最終世代でなければ次世代を生成
            if gen_num < self.ga_config.max_generations:
                population = self._create_next_generation(population, gen_num)

        # 最終勝者
        final_best = max(population, key=lambda x: x.score)
        result.final_winner = final_best.to_dict()
        result.total_evaluations = total_evals
        result.execution_time_sec = time.time() - start_time

        logger.info(
            f"GA完了: {total_evals}回評価, {result.execution_time_sec:.1f}秒, "
            f"勝者={final_best.template_name}"
        )

        return result

    def _create_initial_population(self) -> List[Individual]:
        """初期集団をランダム生成"""
        population = []

        for _ in range(self.ga_config.population_size):
            # ランダムにテンプレートを選択
            template_name = random.choice(self.templates)
            template = BUILTIN_TEMPLATES[template_name]

            # ランダムにパラメータを選択
            params = self._random_params(template)

            individual = Individual(
                template_name=template_name,
                params=params,
                generation=1,
            )
            population.append(individual)

        return population

    def _random_params(self, template: StrategyTemplate) -> Dict[str, Any]:
        """テンプレートのパラメータ範囲からランダムに選択"""
        params = {}
        for pr in template.param_ranges:
            values = pr.values()
            if values:
                params[pr.name] = random.choice(values)
        return params

    def _evaluate_population(
        self,
        population: List[Individual],
        df: pd.DataFrame,
        target_regime: str,
        trend_column: str,
    ):
        """集団全体を評価（バックテスト実行）"""
        for ind in population:
            try:
                config = ind.to_config(self.exit_config)

                # GridSearchOptimizerの_run_singleを使って評価
                entry = self._grid_optimizer._run_single(
                    df=df,
                    config=config,
                    template_name=ind.template_name,
                    params=ind.params,
                    target_regime=target_regime,
                    trend_column=trend_column,
                )

                # 結果を個体に反映
                ind.score = entry.composite_score
                ind.pnl = entry.metrics.total_pnl
                ind.sharpe = entry.metrics.sharpe_ratio
                ind.trades = entry.metrics.trades
                ind.win_rate = entry.metrics.win_rate
                ind.max_dd = entry.metrics.max_drawdown
                ind.status = "evaluated"

            except Exception as e:
                logger.debug(f"評価失敗: {ind.template_name} - {e}")
                ind.score = -999.0
                ind.status = "error"

    def _check_convergence(self, best_scores: List[float]) -> bool:
        """収束判定"""
        n = self.ga_config.convergence_generations
        if len(best_scores) < n:
            return False

        recent = best_scores[-n:]
        improvement = max(recent) - min(recent)
        return improvement < self.ga_config.convergence_threshold

    def _create_next_generation(
        self,
        population: List[Individual],
        current_gen: int,
    ) -> List[Individual]:
        """次世代を生成"""
        # スコアでソート
        sorted_pop = sorted(population, key=lambda x: x.score, reverse=True)

        # エリート選択（上位N%をそのまま次世代へ）
        elite_count = max(1, int(len(population) * self.ga_config.elite_ratio))
        elites = sorted_pop[:elite_count]

        # エリートに生存マーク
        for ind in elites:
            ind.status = "survived"

        # 残りに淘汰マーク
        for ind in sorted_pop[elite_count:]:
            ind.status = "eliminated"
            ind.elimination_reason = f"bottom {100 - int(self.ga_config.elite_ratio * 100)}%"

        # 新世代を構築
        next_gen = []

        # エリートをコピー
        for elite in elites:
            new_ind = Individual(
                template_name=elite.template_name,
                params=dict(elite.params),
                generation=current_gen + 1,
            )
            next_gen.append(new_ind)

        # 残りは交叉・突然変異で生成
        while len(next_gen) < self.ga_config.population_size:
            if random.random() < self.ga_config.crossover_rate:
                # 交叉
                parent1 = self._tournament_select(sorted_pop)
                parent2 = self._tournament_select(sorted_pop)
                child = self._crossover(parent1, parent2, current_gen + 1)
            else:
                # 突然変異のみ
                parent = self._tournament_select(sorted_pop)
                child = Individual(
                    template_name=parent.template_name,
                    params=dict(parent.params),
                    generation=current_gen + 1,
                )

            # 突然変異
            if random.random() < self.ga_config.mutation_rate:
                self._mutate(child)

            next_gen.append(child)

        return next_gen

    def _tournament_select(self, population: List[Individual]) -> Individual:
        """トーナメント選択"""
        tournament = random.sample(
            population,
            min(self.ga_config.tournament_size, len(population))
        )
        return max(tournament, key=lambda x: x.score)

    def _crossover(
        self,
        parent1: Individual,
        parent2: Individual,
        generation: int,
    ) -> Individual:
        """交叉: 2つの親から子を生成"""
        # 同じテンプレートなら、パラメータを混合
        if parent1.template_name == parent2.template_name:
            new_params = {}
            all_keys = set(parent1.params.keys()) | set(parent2.params.keys())
            for key in all_keys:
                if random.random() < 0.5:
                    new_params[key] = parent1.params.get(key, parent2.params.get(key))
                else:
                    new_params[key] = parent2.params.get(key, parent1.params.get(key))

            return Individual(
                template_name=parent1.template_name,
                params=new_params,
                generation=generation,
            )

        # 異なるテンプレートなら、どちらかをランダムに選択
        chosen = random.choice([parent1, parent2])
        return Individual(
            template_name=chosen.template_name,
            params=dict(chosen.params),
            generation=generation,
        )

    def _mutate(self, individual: Individual):
        """突然変異: パラメータをランダムに変更"""
        template = BUILTIN_TEMPLATES.get(individual.template_name)
        if template is None:
            return

        # パラメータの一部をランダムに変更
        for pr in template.param_ranges:
            if random.random() < 0.3:  # 各パラメータ30%の確率で変異
                values = pr.values()
                if values:
                    individual.params[pr.name] = random.choice(values)

        # 低確率でテンプレート自体を変更
        if random.random() < 0.1:
            new_template_name = random.choice(self.templates)
            new_template = BUILTIN_TEMPLATES[new_template_name]
            individual.template_name = new_template_name
            individual.params = self._random_params(new_template)
