"""
オプティマイザ用チャートコンポーネント

散布図（PF vs 勝率）、エクイティカーブオーバーレイ等。
"""

from typing import List, Optional

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from optimizer.results import OptimizationResultSet, OptimizationEntry


# レジーム別カラー
REGIME_COLORS = {
    "uptrend": "#26a69a",
    "downtrend": "#ef5350",
    "range": "#42a5f5",
    "all": "#ab47bc",
}


def create_scatter_chart(
    result_set: OptimizationResultSet,
    x_metric: str = "profit_factor",
    y_metric: str = "win_rate",
) -> go.Figure:
    """
    散布図を作成（X軸: PF, Y軸: 勝率、色: レジーム、サイズ: スコア）

    Args:
        result_set: 最適化結果セット
        x_metric: X軸メトリクス名
        y_metric: Y軸メトリクス名
    """
    df = result_set.to_dataframe()
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data", showarrow=False)
        return fig

    fig = px.scatter(
        df,
        x=x_metric,
        y=y_metric,
        color="regime",
        size="score",
        hover_data=["template", "params", "trades", "total_pnl", "max_dd"],
        color_discrete_map=REGIME_COLORS,
        title=f"{y_metric} vs {x_metric}",
    )

    fig.update_layout(
        template="plotly_dark",
        height=500,
        xaxis_title=x_metric.replace("_", " ").title(),
        yaxis_title=y_metric.replace("_", " ").title(),
    )

    return fig


def create_equity_overlay(
    entries: List[OptimizationEntry],
    max_entries: int = 10,
) -> go.Figure:
    """
    上位N件のエクイティカーブをオーバーレイ表示

    Args:
        entries: OptimizationEntryリスト（スコア順）
        max_entries: 表示する最大数
    """
    fig = go.Figure()

    for i, entry in enumerate(entries[:max_entries]):
        if entry.backtest_result is None:
            continue

        equity = entry.backtest_result.portfolio.equity_curve
        label = f"#{i+1} {entry.template_name} ({entry.trend_regime})"

        fig.add_trace(go.Scatter(
            y=equity,
            mode="lines",
            name=label,
            line=dict(width=1.5),
            hovertemplate=(
                f"{label}<br>"
                f"Score: {entry.composite_score:.4f}<br>"
                f"Equity: %{{y:.2f}}<br>"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        template="plotly_dark",
        title="Equity Curve Overlay (Top N)",
        height=400,
        xaxis_title="Bar",
        yaxis_title="Equity",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            font=dict(size=10),
        ),
    )

    return fig


def create_regime_distribution_chart(
    result_set: OptimizationResultSet,
) -> go.Figure:
    """レジーム別のスコア分布をボックスプロットで表示"""
    df = result_set.to_dataframe()
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data", showarrow=False)
        return fig

    fig = px.box(
        df,
        x="regime",
        y="score",
        color="regime",
        color_discrete_map=REGIME_COLORS,
        title="Score Distribution by Regime",
    )

    fig.update_layout(
        template="plotly_dark",
        height=350,
        showlegend=False,
    )

    return fig
