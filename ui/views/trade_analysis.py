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
    st.header("🔍 トレード分析")
    st.caption("個別トレードの詳細分析・損益統計")

    if "backtest_result" not in st.session_state or st.session_state.backtest_result is None:
        st.info(
            "▶️ **バックテスト結果がありません**\n\n"
            "トレード分析にはバックテスト結果が必要です。\n\n"
            "サイドバーの **▶️ バックテスト** ページで戦略を実行してください。"
        )
        return

    result = st.session_state.backtest_result
    metrics = st.session_state.backtest_metrics
    trades = result.trades

    if not trades:
        st.info("分析するトレードがありません。")
        return

    # 個別トレード分析
    st.subheader("個別トレード")
    _render_individual_trade(result)

    st.divider()

    # 損益分布
    st.subheader("損益分布")
    _render_pl_distribution(trades)

    st.divider()

    # 勝ち/負け分析
    st.subheader("勝敗分析")
    _render_win_loss_analysis(trades)

    st.divider()

    # 決済タイプ分布
    st.subheader("決済タイプ分布")
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
        "トレード選択",
        range(len(trades)),
        format_func=lambda x: trade_options[x],
    )

    trade = trades[selected_idx]

    # トレード詳細
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("売買方向", trade.side.upper())
        st.metric("エントリー価格", f"{trade.entry_price:.6f}")
    with col2:
        st.metric("決済タイプ", trade.exit_type)
        st.metric("決済価格", f"{trade.exit_price:.6f}")
    with col3:
        st.metric("損益", f"{trade.profit_pct:+.2f}%")
        st.metric("保有期間", f"{trade.duration_bars} 本")
    with col4:
        st.metric("エントリー", str(trade.entry_time)[:19])
        st.metric("決済", str(trade.exit_time)[:19])

    st.caption(f"理由: {trade.reason}")

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
        title="損益分布",
        xaxis_title="損益 (%)",
        yaxis_title="回数",
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
        title="累計損益 (%)",
        xaxis_title="トレード #",
        yaxis_title="累計損益 (%)",
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
        st.markdown("**勝ちトレード**")
        if wins:
            st.metric("回数", len(wins))
            st.metric("平均損益", f"{np.mean([t.profit_pct for t in wins]):+.2f}%")
            st.metric("最大利益", f"{max(t.profit_pct for t in wins):+.2f}%")
            st.metric("平均保有期間", f"{np.mean([t.duration_bars for t in wins]):.0f} 本")
        else:
            st.info("勝ちトレードなし")

    with col2:
        st.markdown("**負けトレード**")
        if losses:
            st.metric("回数", len(losses))
            st.metric("平均損益", f"{np.mean([t.profit_pct for t in losses]):.2f}%")
            st.metric("最大損失", f"{min(t.profit_pct for t in losses):.2f}%")
            st.metric("平均保有期間", f"{np.mean([t.duration_bars for t in losses]):.0f} 本")
        else:
            st.info("負けトレードなし")


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
        title="決済タイプ分布",
        template="plotly_dark",
        height=300,
    )
    st.plotly_chart(fig, use_container_width=True)
