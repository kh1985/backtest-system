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
    data_source: str = "original"  # "original" or "trimmed"
    data_period_start: str = ""  # 切り出し開始日時 (例: "2025-01-15")
    data_period_end: str = ""  # 切り出し終了日時 (例: "2025-01-25")

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
            data_source=self.data_source,
            data_period_start=self.data_period_start,
            data_period_end=self.data_period_end,
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
            data_source=self.data_source,
            data_period_start=self.data_period_start,
            data_period_end=self.data_period_end,
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

    @classmethod
    def from_json(cls, json_path: str) -> "OptimizationResultSet":
        """保存済みJSONから結果セットを復元"""
        import json

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        entries = []
        for row in data.get("results", []):
            m = row["metrics"]
            total_trades = m["trades"]
            win_rate = m["win_rate"]
            winning = round(total_trades * win_rate / 100)
            losing = total_trades - winning

            metrics = BacktestMetrics(
                total_trades=total_trades,
                winning_trades=winning,
                losing_trades=losing,
                win_rate=win_rate,
                total_profit_pct=m["total_pnl"],
                avg_profit_pct=0.0,
                avg_loss_pct=0.0,
                profit_factor=m["profit_factor"],
                max_drawdown_pct=m["max_dd"],
                sharpe_ratio=m["sharpe"],
                avg_duration_bars=0.0,
                best_trade_pct=0.0,
                worst_trade_pct=0.0,
                equity_curve=[],
                cumulative_returns=[],
                drawdown_series=[],
            )

            entry = OptimizationEntry(
                template_name=row["template"],
                params=row["params"],
                trend_regime=row["regime"],
                config=row.get("config", {}),
                metrics=metrics,
                composite_score=row["score"],
                backtest_result=None,
            )
            entries.append(entry)

        data_period = data.get("data_period", {})
        return cls(
            entries=entries,
            symbol=data.get("symbol", ""),
            execution_tf=data.get("execution_tf", ""),
            htf=data.get("htf", ""),
            data_source=data.get("data_source", "original"),
            data_period_start=data_period.get("start", ""),
            data_period_end=data_period.get("end", ""),
        )
