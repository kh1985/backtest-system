"""
適応型探索の設定クラス

Numeraiのエージェント設計思想を参考にした効率的な探索設定:
- Scout→Scale: 少量データで探索 → 良いものだけフルデータで検証
- プラトー検出: 改善が止まったら自動終了
- ラウンド制: ベスト周辺を絞り込んで反復改善
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from .results import OptimizationResultSet


@dataclass
class ScoutConfig:
    """Scout（探索）フェーズの設定

    データの一部を使って高速に全configを試し、有望なものを絞り込む。
    """
    sample_ratio: float = 0.2
    """データの何%を使うか（デフォルト: 20%）"""

    sample_method: str = "head"
    """サンプリング方法: "head"（先頭から）, "tail"（末尾から）, "random"（ランダム）"""

    random_seed: int = 42
    """ランダムサンプリング時のシード"""

    min_samples: int = 10000
    """最低サンプル数（これ以下にはならない）"""

    top_n_to_scale: int = 20
    """Scaleフェーズに持ち越す上位config数"""


@dataclass
class PlateauConfig:
    """プラトー（頭打ち）検出の設定

    連続して改善が見られない場合に探索を早期終了する。
    """
    min_improvement: float = 0.001
    """この値より小さい改善は「改善なし」とみなす"""

    consecutive_rounds: int = 2
    """連続何ラウンド改善なしでプラトー判定するか"""

    use_relative: bool = False
    """True: 相対改善率で判定, False: 絶対値で判定"""

    relative_threshold: float = 0.01
    """相対改善率の閾値（1%未満で改善なしとみなす）"""


@dataclass
class RoundConfig:
    """ラウンド制の設定

    全組み合わせを1回で回すのではなく、複数ラウンドに分けて
    良いものを絞り込んでいく。
    """
    max_rounds: int = 5
    """最大ラウンド数"""

    configs_per_round: int = 50
    """1ラウンドで試すconfig数"""

    top_n_survivors: int = 10
    """次ラウンドに持ち越す上位config数"""

    exploration_ratio: float = 0.2
    """各ラウンドで新規configを試す割合（0.2 = 20%は未テストを試す）"""

    refine_params: bool = True
    """ベストconfigの周辺パラメータを自動生成するか"""

    refine_range: float = 0.5
    """パラメータ探索範囲の縮小率（0.5 = 範囲を半分に）"""


@dataclass
class AdaptiveSearchConfig:
    """適応型探索の全体設定"""

    scout: ScoutConfig = field(default_factory=ScoutConfig)
    """Scout（探索）フェーズの設定"""

    plateau: PlateauConfig = field(default_factory=PlateauConfig)
    """プラトー検出の設定"""

    round: RoundConfig = field(default_factory=RoundConfig)
    """ラウンド制の設定"""

    enable_scout: bool = True
    """Scoutフェーズを有効にするか"""

    enable_plateau: bool = True
    """プラトー検出を有効にするか"""

    enable_rounds: bool = True
    """ラウンド制を有効にするか（Falseなら1ラウンドで終了）"""

    verbose: bool = True
    """詳細ログを出力するか"""


@dataclass
class RoundResult:
    """1ラウンドの実行結果"""

    round_number: int
    """ラウンド番号（0始まり）"""

    result_set: OptimizationResultSet
    """このラウンドの最適化結果"""

    best_score: float
    """このラウンドの最良スコア"""

    improvement: float
    """前ラウンドからの改善（スコア差分）"""

    elapsed_time: float
    """実行時間（秒）"""

    configs_tested: int
    """テストしたconfig数"""

    is_plateau: bool = False
    """プラトーに達したか"""

    @property
    def best_entry(self):
        """最良のエントリを取得"""
        return self.result_set.best if self.result_set else None


@dataclass
class AdaptiveSearchResult:
    """適応型探索の最終結果"""

    final_result: OptimizationResultSet
    """最終的な最適化結果"""

    round_history: List[RoundResult]
    """各ラウンドの履歴"""

    scout_result: Optional[OptimizationResultSet] = None
    """Scoutフェーズの結果（有効な場合）"""

    total_configs_tested: int = 0
    """テストした総config数"""

    total_elapsed_time: float = 0.0
    """総実行時間（秒）"""

    early_stopped: bool = False
    """プラトーで早期終了したか"""

    early_stop_round: Optional[int] = None
    """早期終了したラウンド番号"""

    @property
    def best(self):
        """最良のエントリを取得"""
        return self.final_result.best if self.final_result else None

    @property
    def n_rounds_completed(self) -> int:
        """完了したラウンド数"""
        return len(self.round_history)

    def summary(self) -> Dict[str, Any]:
        """結果サマリを辞書で返す"""
        best = self.best
        return {
            "best_score": best.composite_score if best else None,
            "best_template": best.template_name if best else None,
            "best_regime": best.trend_regime if best else None,
            "total_configs_tested": self.total_configs_tested,
            "total_elapsed_time": self.total_elapsed_time,
            "n_rounds": self.n_rounds_completed,
            "early_stopped": self.early_stopped,
            "early_stop_round": self.early_stop_round,
        }
