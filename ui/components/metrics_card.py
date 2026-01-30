"""
メトリクスカードコンポーネント

バックテスト結果のサマリーをカード形式で表示する。
"""

import streamlit as st

from metrics.calculator import BacktestMetrics


def render_metrics_cards(metrics: BacktestMetrics) -> None:
    """メトリクスをカード形式で表示"""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Total Trades",
            value=metrics.total_trades,
        )
    with col2:
        st.metric(
            label="Win Rate",
            value=f"{metrics.win_rate:.1f}%",
        )
    with col3:
        st.metric(
            label="Profit Factor",
            value=f"{metrics.profit_factor:.2f}",
        )
    with col4:
        st.metric(
            label="Max Drawdown",
            value=f"{metrics.max_drawdown_pct:.2f}%",
            delta=f"-{metrics.max_drawdown_pct:.2f}%",
            delta_color="inverse",
        )

    col5, col6, col7, col8 = st.columns(4)

    with col5:
        st.metric(
            label="Sharpe Ratio",
            value=f"{metrics.sharpe_ratio:.2f}",
        )
    with col6:
        color = "normal" if metrics.total_profit_pct >= 0 else "inverse"
        st.metric(
            label="Total Profit",
            value=f"{metrics.total_profit_pct:+.2f}%",
            delta=f"{metrics.total_profit_pct:+.2f}%",
            delta_color=color,
        )
    with col7:
        st.metric(
            label="Avg Win",
            value=f"{metrics.avg_profit_pct:+.2f}%",
        )
    with col8:
        st.metric(
            label="Avg Loss",
            value=f"{metrics.avg_loss_pct:.2f}%",
        )
