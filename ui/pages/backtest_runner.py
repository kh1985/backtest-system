"""
バックテスト実行ページ

戦略の選択 → バックテスト実行 → 結果表示（メトリクス・チャート・トレード一覧）
"""

import streamlit as st

from engine.backtest import BacktestEngine
from metrics.calculator import calculate_metrics
from strategy.builder import ConfigStrategy
from ui.components.chart import (
    create_candlestick_chart,
    create_drawdown_chart,
    create_equity_curve,
)
from ui.components.metrics_card import render_metrics_cards
from ui.components.trade_table import render_trade_table


def render_backtest_runner_page():
    """バックテスト実行ページを描画"""
    st.header("Backtest Runner")

    # データと戦略の確認
    has_data = "ohlcv_data" in st.session_state and st.session_state.ohlcv_data is not None
    has_strategy = "strategy_config" in st.session_state and st.session_state.strategy_config.get("name")

    if not has_data:
        st.warning("Data not loaded. Go to 'Data' page first.")
        return

    if not has_strategy:
        st.warning("Strategy not configured. Go to 'Strategy' page first.")
        return

    ohlcv = st.session_state.ohlcv_data
    config = st.session_state.strategy_config

    # バックテスト設定
    st.subheader("Settings")
    col1, col2, col3 = st.columns(3)
    with col1:
        strategy_name = st.text_input(
            "Strategy",
            value=config.get("name", ""),
            disabled=True,
        )
    with col2:
        initial_capital = st.number_input(
            "Initial Capital (USDT)",
            value=10000.0,
            min_value=100.0,
            step=1000.0,
        )
    with col3:
        commission = st.number_input(
            "Commission (%)",
            value=0.04,
            min_value=0.0,
            step=0.01,
            format="%.4f",
        )

    # 実行ボタン
    if st.button("Run Backtest", type="primary", use_container_width=True):
        _run_backtest(ohlcv, config, initial_capital, commission)

    # 結果表示
    if "backtest_result" in st.session_state and st.session_state.backtest_result is not None:
        _render_results()


def _run_backtest(ohlcv, config, initial_capital, commission):
    """バックテスト実行"""
    with st.spinner("Running backtest..."):
        try:
            strategy = ConfigStrategy(config)
            engine = BacktestEngine(
                strategy=strategy,
                initial_capital=initial_capital,
                commission_pct=commission,
            )

            result = engine.run(ohlcv.df.copy())
            metrics = calculate_metrics(
                result.trades, result.portfolio.equity_curve
            )

            st.session_state.backtest_result = result
            st.session_state.backtest_metrics = metrics
            st.success(
                f"Backtest complete! {len(result.trades)} trades executed."
            )
            st.rerun()

        except Exception as e:
            st.error(f"Backtest error: {e}")
            import traceback
            st.code(traceback.format_exc())


def _render_results():
    """バックテスト結果を描画"""
    result = st.session_state.backtest_result
    metrics = st.session_state.backtest_metrics

    st.divider()
    st.subheader(f"Results: {result.strategy_name}")

    # メトリクスカード
    render_metrics_cards(metrics)

    st.divider()

    # エクイティカーブ
    if metrics.equity_curve:
        fig_equity = create_equity_curve(metrics.equity_curve)
        st.plotly_chart(fig_equity, use_container_width=True)

    # ドローダウン
    if metrics.drawdown_series:
        fig_dd = create_drawdown_chart(metrics.drawdown_series)
        st.plotly_chart(fig_dd, use_container_width=True)

    st.divider()

    # ローソク足チャート（トレードマーカー付き）
    st.subheader("Chart with Trades")

    max_bars = st.slider(
        "Display bars",
        min_value=50,
        max_value=min(2000, len(result.df)),
        value=min(500, len(result.df)),
        key="backtest_chart_bars",
    )

    display_df = result.df.tail(max_bars)

    # オーバーレイの自動検出
    overlays = _detect_overlays(result.df)

    fig = create_candlestick_chart(
        display_df,
        title=f"{result.strategy_name}",
        overlays=overlays,
        trades=result.trades,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # トレード一覧
    st.subheader("Trade List")
    render_trade_table(result.trades)


def _detect_overlays(df):
    """DataFrameからオーバーレイ対象のカラムを自動検出"""
    overlays = {}

    # 移動平均系
    ma_cols = [c for c in df.columns if c.startswith(("sma_", "ema_"))]
    if ma_cols:
        overlays["MA"] = ma_cols

    # ボリンジャーバンド
    bb_cols = [c for c in df.columns if c.startswith("bb_") and c != "bb_width"]
    if bb_cols:
        overlays["BB"] = bb_cols

    # VWAP
    if "vwap" in df.columns:
        overlays["VWAP"] = ["vwap"]

    return overlays if overlays else None
