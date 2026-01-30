"""
Backtest System - Streamlit メインアプリ

起動方法: streamlit run ui/app.py
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import streamlit as st

from ui.components.styles import inject_custom_css
from ui.views.data_loader import render_data_loader_page
from ui.views.strategy_builder import render_strategy_builder_page
from ui.views.backtest_runner import render_backtest_runner_page
from ui.views.trade_analysis import render_trade_analysis_page
from ui.views.optimizer_page import render_optimizer_page


# ページ定義 (key, icon, label)
PAGES = [
    ("Data", "📂", "Data"),
    ("Strategy", "🧩", "Strategy"),
    ("Backtest", "▶️", "Backtest"),
    ("Analysis", "🔍", "Analysis"),
    ("Optimizer", "⚡", "Optimizer"),
]


def main():
    st.set_page_config(
        page_title="Backtest System v2",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # カスタムCSS注入
    inject_custom_css()

    # セッション初期化
    if "datasets" not in st.session_state:
        st.session_state.datasets = {}  # {symbol: {tf: OHLCVData}}
    # 後方互換
    if "ohlcv_data" not in st.session_state:
        st.session_state.ohlcv_data = None
    if "ohlcv_dict" not in st.session_state:
        st.session_state.ohlcv_dict = {}
    if "strategy" not in st.session_state:
        st.session_state.strategy = None
    if "backtest_result" not in st.session_state:
        st.session_state.backtest_result = None
    if "backtest_metrics" not in st.session_state:
        st.session_state.backtest_metrics = None

    # ステータス判定
    has_data = len(st.session_state.datasets) > 0
    has_strategy = st.session_state.strategy is not None
    has_result = st.session_state.backtest_result is not None
    has_optimization = (
        "optimization_result" in st.session_state
        and st.session_state.optimization_result is not None
    )

    # サイドバー
    with st.sidebar:
        st.markdown(
            '<h2 style="margin-bottom:0;color:#e6edf3;">📊 Backtest System</h2>',
            unsafe_allow_html=True,
        )
        st.caption("Strategy Optimizer & Analyzer")
        st.divider()

        page = st.radio(
            "Navigation",
            options=[p[0] for p in PAGES],
            format_func=lambda x: next(
                f"{icon} {label}" for key, icon, label in PAGES if key == x
            ),
            index=0,
        )

        st.divider()

        # ステップインジケーター
        steps = [
            ("Data", has_data),
            ("Strategy", has_strategy),
            ("Backtest", has_result),
            ("Optimizer", has_optimization),
        ]

        for step_name, done in steps:
            check = "✓" if done else "○"
            color = "#3fb950" if done else "#484f58"
            label_class = "" if done else "dimmed"
            st.markdown(
                f'<div class="step-indicator">'
                f'<span style="color:{color};font-size:0.9rem;">{check}</span>'
                f'<span class="step-label {label_class}">{step_name}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.divider()

        # データセット一覧
        if has_data:
            st.caption("📦 Datasets")
            for sym, tf_dict in st.session_state.datasets.items():
                tf_labels = ", ".join(
                    f"{tf}({ohlcv.bars:,})" for tf, ohlcv in tf_dict.items()
                )
                st.markdown(
                    f'<span class="status-badge status-ready">'
                    f'{sym}: {tf_labels}</span>',
                    unsafe_allow_html=True,
                )

        if has_strategy:
            name = getattr(st.session_state.strategy, "name", "")
            st.markdown(
                f'<span class="status-badge status-active">Strategy: {name}</span>',
                unsafe_allow_html=True,
            )
        if has_result:
            n_trades = len(st.session_state.backtest_result.trades)
            st.markdown(
                f'<span class="status-badge status-ready">Result: {n_trades} trades</span>',
                unsafe_allow_html=True,
            )

    # ページ描画
    if page == "Data":
        render_data_loader_page()
    elif page == "Strategy":
        render_strategy_builder_page()
    elif page == "Backtest":
        render_backtest_runner_page()
    elif page == "Analysis":
        render_trade_analysis_page()
    elif page == "Optimizer":
        render_optimizer_page()


if __name__ == "__main__":
    main()
