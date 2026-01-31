"""
Prism - Streamlit メインアプリ

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


# ページ定義 (key, icon, label_ja)
PAGES = [
    ("Data", "📂", "データ"),
    ("Strategy", "🧩", "戦略"),
    ("Backtest", "▶️", "バックテスト"),
    ("Analysis", "🔍", "分析"),
    ("Optimizer", "⚡", "最適化"),
]


def main():
    st.set_page_config(
        page_title="Prism",
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
    if "trimmed_datasets" not in st.session_state:
        st.session_state.trimmed_datasets = []  # [{id, symbol, label, start_dt, end_dt, data: {tf: OHLCVData}}]

    # ステータス判定
    has_data = len(st.session_state.datasets) > 0
    has_strategy = st.session_state.strategy is not None
    has_result = st.session_state.backtest_result is not None
    has_optimization = (
        "optimization_result" in st.session_state
        and st.session_state.optimization_result is not None
    )

    step_status = {
        "Data": has_data,
        "Strategy": has_strategy,
        "Backtest": has_result,
        "Analysis": has_result,  # 分析はバックテスト結果があれば可
        "Optimizer": has_optimization,
    }

    # サイドバー
    with st.sidebar:
        st.markdown(
            '<h2 style="margin-bottom:0;color:#e6edf3;">🔷 Prism</h2>',
            unsafe_allow_html=True,
        )
        st.caption("戦略バックテスト＆最適化ツール")
        st.divider()

        # ナビゲーション（進捗チェック統合）
        def _nav_label(key):
            icon = next(i for k, i, _ in PAGES if k == key)
            label = next(l for k, _, l in PAGES if k == key)
            done = step_status.get(key, False)
            check = " ✓" if done else ""
            return f"{icon} {label}{check}"

        page = st.radio(
            "ナビゲーション",
            options=[p[0] for p in PAGES],
            format_func=_nav_label,
            label_visibility="collapsed",
            key="nav_page",
        )

        st.divider()

        # --- 常駐ワークフローガイド ---
        _render_sidebar_guide(step_status)

        st.divider()

        # データセット一覧 / ステータス
        if has_data:
            st.caption("📦 読み込み済みデータ")
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
                f'<span class="status-badge status-active">戦略: {name}</span>',
                unsafe_allow_html=True,
            )
        if has_result:
            n_trades = len(st.session_state.backtest_result.trades)
            st.markdown(
                f'<span class="status-badge status-ready">結果: {n_trades}件</span>',
                unsafe_allow_html=True,
            )

    # ページ描画
    if page == "Data":
        if not has_data:
            _render_welcome()
        render_data_loader_page()
    elif page == "Strategy":
        render_strategy_builder_page()
    elif page == "Backtest":
        render_backtest_runner_page()
    elif page == "Analysis":
        render_trade_analysis_page()
    elif page == "Optimizer":
        render_optimizer_page()


def _render_sidebar_guide(step_status):
    """サイドバーに常駐するワークフローガイド"""

    def _step_html(label, done):
        if done:
            return f'<span class="guide-step done">{label} ✓</span>'
        return f'<span class="guide-step">{label}</span>'

    data_done = step_status.get("Data", False)
    strategy_done = step_status.get("Strategy", False)
    backtest_done = step_status.get("Backtest", False)
    analysis_done = step_status.get("Analysis", False)
    optimizer_done = step_status.get("Optimizer", False)

    st.markdown(f"""
    <div class="sidebar-guide">
        <div class="guide-title">使い方ガイド</div>
        <div class="guide-route">
            <div class="guide-route-label">ルートA: 手動テスト</div>
            <div class="guide-flow">
                {_step_html("データ", data_done)}
                <span class="guide-arrow">→</span>
                {_step_html("戦略", strategy_done)}
                <span class="guide-arrow">→</span>
                {_step_html("バックテスト", backtest_done)}
                <span class="guide-arrow">→</span>
                {_step_html("分析", analysis_done)}
            </div>
        </div>
        <div class="guide-route">
            <div class="guide-route-label">ルートB: 自動最適化</div>
            <div class="guide-flow">
                {_step_html("データ", data_done)}
                <span class="guide-arrow">→</span>
                {_step_html("最適化", optimizer_done)}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def _render_welcome():
    """初回訪問時のウェルカムガイド（メインエリア）"""
    st.markdown("""
    <div class="welcome-section">
        <h2 class="welcome-title">Prism へようこそ</h2>
        <p class="welcome-desc">
            トレード戦略のバックテスト・分析・最適化を行うツールです。<br>
            まず、下のフォームからOHLCVデータを読み込んでください。
        </p>
        <div class="welcome-routes">
            <div class="welcome-route">
                <div class="welcome-route-title">ルートA: 手動テスト</div>
                <div class="welcome-route-desc">
                    データ → 戦略設定 → バックテスト → 分析
                </div>
                <div class="welcome-route-note">自分で戦略を組んで検証</div>
            </div>
            <div class="welcome-route">
                <div class="welcome-route-title">ルートB: 自動最適化</div>
                <div class="welcome-route-desc">
                    データ → 最適化
                </div>
                <div class="welcome-route-note">テンプレートから最適戦略を自動探索</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()


if __name__ == "__main__":
    main()
