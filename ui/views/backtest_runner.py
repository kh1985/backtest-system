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
    st.header("▶️ バックテスト")
    st.caption("戦略のバックテスト実行・結果表示")

    # データと戦略の確認
    datasets = st.session_state.get("datasets", {})
    has_data = len(datasets) > 0
    has_strategy = "strategy_config" in st.session_state and st.session_state.strategy_config.get("name")

    if not has_data:
        st.info(
            "📂 **データが読み込まれていません**\n\n"
            "バックテストにはOHLCVデータが必要です。\n\n"
            "サイドバーの **📂 データ** ページでCSVファイルを読み込んでください。"
        )
        return

    if not has_strategy:
        st.info(
            "🧩 **戦略が設定されていません**\n\n"
            "バックテストには戦略の設定が必要です。\n\n"
            "サイドバーの **🧩 戦略** ページでインジケーター・条件・決済ルールを設定してください。"
        )
        return

    config = st.session_state.strategy_config

    # --- データセット選択 ---
    st.subheader("📦 データセット")
    symbols = list(datasets.keys())
    col_sym, col_tf = st.columns(2)
    with col_sym:
        selected_symbol = st.selectbox(
            "シンボル", options=symbols, index=0, key="bt_symbol"
        )
    with col_tf:
        tf_dict = datasets[selected_symbol]
        tf_options = list(tf_dict.keys())
        selected_tf = st.selectbox(
            "タイムフレーム", options=tf_options, index=0, key="bt_tf"
        )

    ohlcv = tf_dict[selected_tf]
    st.caption(f"📊 {selected_symbol} {selected_tf} — {ohlcv.bars:,} bars")

    # --- バックテスト設定 ---
    st.subheader("⚙️ 設定")
    col1, col2, col3 = st.columns(3)
    with col1:
        strategy_name = st.text_input(
            "戦略",
            value=config.get("name", ""),
            disabled=True,
        )
    with col2:
        initial_capital = st.number_input(
            "初期資金 (USDT)",
            value=10000.0,
            min_value=100.0,
            step=1000.0,
            help="バックテスト開始時の資金",
        )
    with col3:
        commission = st.number_input(
            "手数料 (%)",
            value=0.04,
            min_value=0.0,
            step=0.01,
            format="%.4f",
            help="1トレードあたりの取引手数料率",
        )

    # 実行ボタン
    if st.button("バックテスト実行", type="primary", use_container_width=True):
        _run_backtest(ohlcv, config, initial_capital, commission)

    # 結果表示
    if "backtest_result" in st.session_state and st.session_state.backtest_result is not None:
        _render_results()


def _run_backtest(ohlcv, config, initial_capital, commission):
    """バックテスト実行"""
    with st.spinner("バックテスト実行中..."):
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
                f"バックテスト完了！ {len(result.trades)} 件のトレードを実行"
            )
            st.rerun()

        except Exception as e:
            st.error(f"バックテストエラー: {e}")
            import traceback
            st.code(traceback.format_exc())


def _render_results():
    """バックテスト結果を描画"""
    result = st.session_state.backtest_result
    metrics = st.session_state.backtest_metrics

    st.divider()
    st.subheader(f"結果: {result.strategy_name}")

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
    st.subheader("トレードチャート")

    max_bars = st.slider(
        "表示本数",
        min_value=50,
        max_value=min(2000, len(result.df)),
        value=min(500, len(result.df)),
        key="backtest_chart_bars",
        help="チャートに表示するローソク足の本数",
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
    st.subheader("トレード一覧")
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
