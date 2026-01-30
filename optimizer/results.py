"""
最適化結果データ構造

グリッドサーチの結果を格納・ランキング・フィルタリングする。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import pandas as pd

from engine.backtest import BacktestResult
from metrics.calculator import BacktestMetrics


@dataclass
class OptimizationEntry:
    """1つのパラメータ組み合わせの結果"""
    template_name: str
    params: Dict[str, Any]
    trend_regime: str  # "uptrend", "downtrend", "range", "all"
    config: Dict[str, Any]
    metrics: BacktestMetrics
    composite_score: float
    backtest_result: Optional[BacktestResult] = None  # 上位N件のみ保持

    @property
    def param_str(self) -> str:
        return ", ".join(f"{k}={v}" for k, v in self.params.items())


@dataclass
class OptimizationResultSet:
    """最適化結果セット"""
    entries: List[OptimizationEntry] = field(default_factory=list)
    symbol: str = ""
    execution_tf: str = ""
    htf: str = ""

    def add(self, entry: OptimizationEntry):
        self.entries.append(entry)

    def ranked(self, ascending: bool = False) -> List[OptimizationEntry]:
        """スコア順にソート"""
        return sorted(
            self.entries,
            key=lambda e: e.composite_score,
            reverse=not ascending,
        )

    def filter_regime(self, regime: str) -> "OptimizationResultSet":
        """レジームでフィルタリング"""
        filtered = [e for e in self.entries if e.trend_regime == regime]
        result = OptimizationResultSet(
            entries=filtered,
            symbol=self.symbol,
            execution_tf=self.execution_tf,
            htf=self.htf,
        )
        return result

    def filter_template(self, template_name: str) -> "OptimizationResultSet":
        """テンプレート名でフィルタリング"""
        filtered = [e for e in self.entries if e.template_name == template_name]
        result = OptimizationResultSet(
            entries=filtered,
            symbol=self.symbol,
            execution_tf=self.execution_tf,
            htf=self.htf,
        )
        return result

    def to_dataframe(self) -> pd.DataFrame:
        """結果をDataFrameに変換"""
        rows = []
        for e in self.entries:
            row = {
                "template": e.template_name,
                "params": e.param_str,
                "regime": e.trend_regime,
                "score": round(e.composite_score, 4),
                "trades": e.metrics.total_trades,
                "win_rate": round(e.metrics.win_rate, 1),
                "profit_factor": round(e.metrics.profit_factor, 2),
                "total_pnl": round(e.metrics.total_profit_pct, 2),
                "max_dd": round(e.metrics.max_drawdown_pct, 2),
                "sharpe": round(e.metrics.sharpe_ratio, 2),
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("score", ascending=False).reset_index(drop=True)
        return df

    @property
    def best(self) -> Optional[OptimizationEntry]:
        """最高スコアのエントリー"""
        if not self.entries:
            return None
        return max(self.entries, key=lambda e: e.composite_score)

    @property
    def total_combinations(self) -> int:
        return len(self.entries)
