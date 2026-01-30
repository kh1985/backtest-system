"""
グリッドサーチエンジン

テンプレート × パラメータ × トレンドレジーム の全組み合わせを
自動バックテストし、複合スコアでランキングする。
"""

import hashlib
import json
from typing import Callable, Dict, List, Optional, Any

import pandas as pd

from strategy.builder import ConfigStrategy
from engine.backtest import BacktestEngine, BacktestResult
from engine.position import Position, Trade
from engine.portfolio import Portfolio
from metrics.calculator import calculate_metrics
from analysis.trend import TrendRegime
from .scoring import ScoringWeights, calculate_composite_score
from .results import OptimizationEntry, OptimizationResultSet


class IndicatorCache:
    """
    インジケーター計算結果のキャッシュ

    同じインジケーター設定は1回だけ計算し、結果を再利用。
    キーはインジケーター設定のMD5ハッシュ。
    """

    def __init__(self):
        self._cache: Dict[str, pd.DataFrame] = {}

    def get_key(self, indicator_configs: List[Dict]) -> str:
        """インジケーター設定からキャッシュキーを生成"""
        serialized = json.dumps(indicator_configs, sort_keys=True)
        return hashlib.md5(serialized.encode()).hexdigest()

    def get(self, key: str) -> Optional[pd.DataFrame]:
        return self._cache.get(key)

    def put(self, key: str, df: pd.DataFrame):
        self._cache[key] = df

    def clear(self):
        self._cache.clear()


class GridSearchOptimizer:
    """
    グリッドサーチ最適化エンジン

    各(テンプレート × パラメータ組み合わせ × トレンドレジーム)に対して:
    1. テンプレートからconfig生成
    2. ConfigStrategy構築
    3. インジケーター計算（キャッシュ済みならスキップ）
    4. トレンドフィルタ付きbar-by-barループ
    5. メトリクス算出 → 複合スコア計算
    6. OptimizationEntryに記録
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        commission_pct: float = 0.04,
        slippage_pct: float = 0.0,
        scoring_weights: ScoringWeights = None,
        top_n_results: int = 20,
    ):
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.scoring_weights = scoring_weights or ScoringWeights()
        self.top_n_results = top_n_results
        self._indicator_cache = IndicatorCache()

    def run(
        self,
        df: pd.DataFrame,
        configs: List[Dict[str, Any]],
        target_regimes: List[str],
        trend_column: str = "trend_regime",
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> OptimizationResultSet:
        """
        グリッドサーチを実行

        Args:
            df: 実行TFのDataFrame（trend_regimeカラム付き）
            configs: ConfigStrategy用configのリスト
                     各configに "_template_name" と "_params" を含む
            target_regimes: 対象レジーム ["uptrend", "downtrend", "range"]
            trend_column: トレンドレジームカラム名
            progress_callback: 進捗コールバック(current, total, description)

        Returns:
            OptimizationResultSet
        """
        result_set = OptimizationResultSet()

        total = len(configs) * len(target_regimes)
        current = 0

        for config in configs:
            template_name = config.pop("_template_name", config.get("name", "unknown"))
            params = config.pop("_params", {})

            for regime in target_regimes:
                current += 1
                desc = f"{template_name} ({regime})"

                if progress_callback:
                    progress_callback(current, total, desc)

                try:
                    entry = self._run_single(
                        df=df,
                        config=config,
                        template_name=template_name,
                        params=params,
                        target_regime=regime,
                        trend_column=trend_column,
                    )
                    result_set.add(entry)
                except Exception:
                    # 個別のエラーはスキップ
                    pass

        # BacktestResult は上位N件のみ保持（メモリ節約）
        self._trim_results(result_set)

        return result_set

    def _run_single(
        self,
        df: pd.DataFrame,
        config: Dict[str, Any],
        template_name: str,
        params: Dict[str, Any],
        target_regime: str,
        trend_column: str,
    ) -> OptimizationEntry:
        """1つの組み合わせでバックテストを実行"""
        strategy = ConfigStrategy(config)

        # インジケーター計算（キャッシュ利用）
        indicator_configs = config.get("indicators", [])
        cache_key = self._indicator_cache.get_key(indicator_configs)
        cached_df = self._indicator_cache.get(cache_key)

        if cached_df is not None:
            work_df = cached_df.copy()
            # 条件だけ構築
            strategy._entry_condition = strategy._build_condition(
                config.get("entry_conditions", [])
            )
        else:
            work_df = strategy.setup(df.copy())
            self._indicator_cache.put(cache_key, work_df)

        # トレンドフィルタ付きbar-by-barループ
        if target_regime == "all":
            result = self._run_backtest(work_df, strategy)
        else:
            result = self._run_with_trend_filter(
                work_df, strategy, target_regime, trend_column
            )

        # メトリクス算出
        metrics = calculate_metrics(
            result.trades, result.portfolio.equity_curve
        )

        # 複合スコア
        score = calculate_composite_score(
            profit_factor=metrics.profit_factor,
            win_rate=metrics.win_rate,
            max_drawdown_pct=metrics.max_drawdown_pct,
            sharpe_ratio=metrics.sharpe_ratio,
            weights=self.scoring_weights,
        )

        return OptimizationEntry(
            template_name=template_name,
            params=params,
            trend_regime=target_regime,
            config=config,
            metrics=metrics,
            composite_score=score,
            backtest_result=result,
        )

    def _run_backtest(
        self,
        df: pd.DataFrame,
        strategy: ConfigStrategy,
    ) -> BacktestResult:
        """通常のbar-by-barバックテスト（レジームフィルタなし）"""
        trades: list = []
        portfolio = Portfolio(self.initial_capital)
        position = None

        for i in range(1, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i - 1]

            if position is not None:
                trade = position.check_exit(row, i)
                if trade:
                    if self.commission_pct > 0:
                        trade.profit_pct -= self.commission_pct * 2
                    portfolio.close_trade(trade)
                    trades.append(trade)
                    position = None

            if position is None:
                signal = strategy.check_entry(row, prev_row)
                if signal:
                    position = self._open_position(row, i, signal, strategy)

        if position is not None:
            trade = self._force_close(position, df.iloc[-1], len(df) - 1)
            if self.commission_pct > 0:
                trade.profit_pct -= self.commission_pct * 2
            portfolio.close_trade(trade)
            trades.append(trade)

        return BacktestResult(
            trades=trades,
            portfolio=portfolio,
            strategy_name=strategy.name,
            df=df,
        )

    def _run_with_trend_filter(
        self,
        df: pd.DataFrame,
        strategy: ConfigStrategy,
        target_regime: str,
        trend_column: str,
    ) -> BacktestResult:
        """
        トレンドフィルタ付きbar-by-barバックテスト

        エントリーは対象レジームでのみ許可。
        決済はレジームに関係なく実行。
        """
        trades: list = []
        portfolio = Portfolio(self.initial_capital)
        position = None

        for i in range(1, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i - 1]

            # 決済判定（レジーム不問）
            if position is not None:
                trade = position.check_exit(row, i)
                if trade:
                    if self.commission_pct > 0:
                        trade.profit_pct -= self.commission_pct * 2
                    portfolio.close_trade(trade)
                    trades.append(trade)
                    position = None

            # エントリー判定（対象レジームのみ）
            if position is None:
                current_regime = row.get(trend_column, "")
                if current_regime == target_regime:
                    signal = strategy.check_entry(row, prev_row)
                    if signal:
                        position = self._open_position(row, i, signal, strategy)

        if position is not None:
            trade = self._force_close(position, df.iloc[-1], len(df) - 1)
            if self.commission_pct > 0:
                trade.profit_pct -= self.commission_pct * 2
            portfolio.close_trade(trade)
            trades.append(trade)

        return BacktestResult(
            trades=trades,
            portfolio=portfolio,
            strategy_name=strategy.name,
            df=df,
        )

    def _open_position(self, row, index, signal, strategy) -> Position:
        entry_price = row["close"]
        if self.slippage_pct > 0:
            if signal.side.value == "long":
                entry_price *= 1 + self.slippage_pct / 100
            else:
                entry_price *= 1 - self.slippage_pct / 100

        exit_rule = strategy.exit_rule
        if signal.side.value == "long":
            tp_price = entry_price * (1 + exit_rule.take_profit_pct / 100)
            sl_price = entry_price * (1 - exit_rule.stop_loss_pct / 100)
        else:
            tp_price = entry_price * (1 - exit_rule.take_profit_pct / 100)
            sl_price = entry_price * (1 + exit_rule.stop_loss_pct / 100)

        return Position(
            entry_price=entry_price,
            entry_time=row.get("datetime", pd.Timestamp.now()),
            entry_index=index,
            side=signal.side.value,
            tp_price=tp_price,
            sl_price=sl_price,
            trailing_stop_pct=exit_rule.trailing_stop_pct,
            timeout_bars=exit_rule.timeout_bars,
            reason=signal.reason,
        )

    def _force_close(self, position, row, index) -> Trade:
        close_price = row["close"]
        duration = index - position.entry_index
        if position.side == "long":
            profit_pct = (close_price - position.entry_price) / position.entry_price * 100
        else:
            profit_pct = (position.entry_price - close_price) / position.entry_price * 100

        return Trade(
            entry_time=position.entry_time,
            exit_time=row.get("datetime", pd.Timestamp.now()),
            side=position.side,
            entry_price=position.entry_price,
            exit_price=close_price,
            profit_pct=profit_pct,
            duration_bars=duration,
            exit_type="FORCED",
            reason=position.reason,
        )

    def _trim_results(self, result_set: OptimizationResultSet):
        """上位N件以外のBacktestResultをクリア（メモリ節約）"""
        ranked = result_set.ranked()
        for i, entry in enumerate(ranked):
            if i >= self.top_n_results:
                entry.backtest_result = None
