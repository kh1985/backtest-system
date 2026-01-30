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

from ui.pages.data_loader import render_data_loader_page
from ui.pages.strategy_builder import render_strategy_builder_page
from ui.pages.backtest_runner import render_backtest_runner_page
from ui.pages.trade_analysis import render_trade_analysis_page
from ui.pages.optimizer_page import render_optimizer_page


def main():
    st.set_page_config(
        page_title="Backtest System",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # セッション初期化
    if "ohlcv_data" not in st.session_state:
        st.session_state.ohlcv_data = None
    if "strategy" not in st.session_state:
        st.session_state.strategy = None
    if "backtest_result" not in st.session_state:
        st.session_state.backtest_result = None
    if "backtest_metrics" not in st.session_state:
        st.session_state.backtest_metrics = None

    # サイドバー
    with st.sidebar:
        st.title("Backtest System")
        st.divider()

        page = st.radio(
            "Navigation",
            options=["Data", "Strategy", "Backtest", "Analysis", "Optimizer"],
            index=0,
        )

        st.divider()

        # ステータス表示
        st.caption("Status")
        data_status = "Loaded" if st.session_state.ohlcv_data else "Not loaded"
        strategy_status = (
            st.session_state.strategy_config.get("name", "Not set")
            if "strategy_config" in st.session_state
            else "Not set"
        )
        result_status = (
            f"{len(st.session_state.backtest_result.trades)} trades"
            if st.session_state.backtest_result
            else "Not run"
        )

        st.text(f"Data: {data_status}")
        st.text(f"Strategy: {strategy_status}")
        st.text(f"Result: {result_status}")

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
