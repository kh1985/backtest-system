"""
トレード分析ページ

個別トレードの詳細分析、損益分布、統計チャート
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

from ui.components.chart import create_candlestick_chart


def render_trade_analysis_page():
    """トレード分析ページを描画"""
    st.header("🔍 Trade Analysis")
    st.caption("個別トレードの詳細分析・損益統計")

    if "backtest_result" not in st.session_state or st.session_state.backtest_result is None:
        st.warning("No backtest results. Run a backtest first.")
        return

    result = st.session_state.backtest_result
    metrics = st.session_state.backtest_metrics
    trades = result.trades

    if not trades:
        st.info("No trades to analyze.")
        return

    # 個別トレード分析
    st.subheader("Individual Trade")
    _render_individual_trade(result)

    st.divider()

    # 損益分布
    st.subheader("P/L Distribution")
    _render_pl_distribution(trades)

    st.divider()

    # 勝ち/負け分析
    st.subheader("Win/Loss Analysis")
    _render_win_loss_analysis(trades)

    st.divider()

    # Exit Type分布
    st.subheader("Exit Type Distribution")
    _render_exit_distribution(trades)


def _render_individual_trade(result):
    """個別トレードの詳細表示"""
    trades = result.trades
    df = result.df

    # トレード選択
    trade_options = [
        f"#{i+1} - {t.exit_type} {t.profit_pct:+.2f}% "
        f"({str(t.entry_time)[:19]})"
        for i, t in enumerate(trades)
    ]
    selected_idx = st.selectbox(
        "Select Trade",
        range(len(trades)),
        format_func=lambda x: trade_options[x],
    )

    trade = trades[selected_idx]

    # トレード詳細
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Side", trade.side.upper())
        st.metric("Entry Price", f"{trade.entry_price:.6f}")
    with col2:
        st.metric("Exit Type", trade.exit_type)
        st.metric("Exit Price", f"{trade.exit_price:.6f}")
    with col3:
        st.metric("P/L", f"{trade.profit_pct:+.2f}%")
        st.metric("Duration", f"{trade.duration_bars} bars")
    with col4:
        st.metric("Entry", str(trade.entry_time)[:19])
        st.metric("Exit", str(trade.exit_time)[:19])

    st.caption(f"Reason: {trade.reason}")

    # 該当範囲のチャート表示
    if "datetime" in df.columns:
        # エントリー前後のデータを抽出
        mask = (
            (df["datetime"] >= trade.entry_time - pd.Timedelta(minutes=30))
            & (df["datetime"] <= trade.exit_time + pd.Timedelta(minutes=30))
        )
        trade_df = df[mask].copy()

        if not trade_df.empty:
            fig = create_candlestick_chart(
                trade_df,
                title=f"Trade #{selected_idx + 1}",
                trades=[trade],
                height=500,
            )

            # TP/SLライン
            entry = trade.entry_price
            exit_rule = None
            if hasattr(st.session_state, "strategy_config"):
                exit_config = st.session_state.strategy_config.get("exit", {})
                tp_pct = exit_config.get("take_profit_pct", 1.0)
                sl_pct = exit_config.get("stop_loss_pct", 0.5)

                tp_price = entry * (1 + tp_pct / 100)
                sl_price = entry * (1 - sl_pct / 100)

                fig.add_hline(
                    y=tp_price,
                    line_dash="dash",
                    line_color="#26a69a",
                    annotation_text=f"TP ({tp_pct}%)",
                    row=1,
                    col=1,
                )
                fig.add_hline(
                    y=sl_price,
                    line_dash="dash",
                    line_color="#ef5350",
                    annotation_text=f"SL ({sl_pct}%)",
                    row=1,
                    col=1,
                )

            st.plotly_chart(fig, use_container_width=True)


def _render_pl_distribution(trades):
    """損益分布ヒストグラム"""
    profits = [t.profit_pct for t in trades]

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=profits,
            nbinsx=30,
            marker_color=[
                "#26a69a" if p > 0 else "#ef5350" for p in profits
            ],
            name="P/L Distribution",
        )
    )
    fig.update_layout(
        title="P/L Distribution",
        xaxis_title="P/L (%)",
        yaxis_title="Count",
        template="plotly_dark",
        height=300,
    )
    st.plotly_chart(fig, use_container_width=True)

    # 累積リターン
    cum_returns = np.cumsum(profits)
    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(
            y=cum_returns,
            mode="lines",
            name="Cumulative P/L",
            line=dict(color="#2196f3", width=2),
        )
    )
    fig2.update_layout(
        title="Cumulative P/L (%)",
        xaxis_title="Trade #",
        yaxis_title="Cumulative P/L (%)",
        template="plotly_dark",
        height=300,
    )
    st.plotly_chart(fig2, use_container_width=True)


def _render_win_loss_analysis(trades):
    """勝ち/負け統計"""
    wins = [t for t in trades if t.profit_pct > 0]
    losses = [t for t in trades if t.profit_pct <= 0]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Winning Trades**")
        if wins:
            st.metric("Count", len(wins))
            st.metric("Avg P/L", f"{np.mean([t.profit_pct for t in wins]):+.2f}%")
            st.metric("Best", f"{max(t.profit_pct for t in wins):+.2f}%")
            st.metric("Avg Duration", f"{np.mean([t.duration_bars for t in wins]):.0f} bars")
        else:
            st.info("No winning trades")

    with col2:
        st.markdown("**Losing Trades**")
        if losses:
            st.metric("Count", len(losses))
            st.metric("Avg P/L", f"{np.mean([t.profit_pct for t in losses]):.2f}%")
            st.metric("Worst", f"{min(t.profit_pct for t in losses):.2f}%")
            st.metric("Avg Duration", f"{np.mean([t.duration_bars for t in losses]):.0f} bars")
        else:
            st.info("No losing trades")


def _render_exit_distribution(trades):
    """Exit Type分布の円グラフ"""
    exit_types = {}
    for t in trades:
        exit_types[t.exit_type] = exit_types.get(t.exit_type, 0) + 1

    fig = go.Figure(
        data=[
            go.Pie(
                labels=list(exit_types.keys()),
                values=list(exit_types.values()),
                hole=0.4,
                marker=dict(
                    colors=["#26a69a", "#ef5350", "#ff9800", "#2196f3"]
                ),
            )
        ]
    )
    fig.update_layout(
        title="Exit Type Distribution",
        template="plotly_dark",
        height=300,
    )
    st.plotly_chart(fig, use_container_width=True)
