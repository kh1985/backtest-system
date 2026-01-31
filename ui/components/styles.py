"""
カスタムCSS / テーマスタイリング

Streamlit のデフォルトスタイルをオーバーライドして
プロフェッショナルな見た目に仕上げる。
"""

import streamlit as st


def inject_custom_css():
    """カスタムCSSを注入"""
    st.markdown("""
    <style>
    /* === 全体 === */
    .stApp {
        background-color: #0e1117;
    }

    /* === サイドバー === */
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #21262d;
    }

    section[data-testid="stSidebar"] .stRadio > label {
        font-size: 0.85rem;
    }

    /* === メトリックカード === */
    div[data-testid="stMetric"] {
        background-color: #161b22;
        border: 1px solid #21262d;
        border-radius: 8px;
        padding: 12px 16px;
    }

    div[data-testid="stMetric"] label {
        color: #8b949e !important;
        font-size: 0.75rem !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-size: 1.3rem !important;
        font-weight: 600;
    }

    /* === セクションヘッダー === */
    .section-header {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 4px;
        padding-bottom: 8px;
        border-bottom: 2px solid #21262d;
    }

    .section-header .icon {
        font-size: 1.2rem;
    }

    .section-header .title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #e6edf3;
    }

    .section-header .subtitle {
        font-size: 0.8rem;
        color: #8b949e;
        margin-left: auto;
    }

    /* === ステータスバッジ === */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 500;
    }

    .status-ready {
        background-color: #0d1117;
        border: 1px solid #238636;
        color: #3fb950;
    }

    .status-pending {
        background-color: #0d1117;
        border: 1px solid #30363d;
        color: #8b949e;
    }

    .status-active {
        background-color: #0d1117;
        border: 1px solid #1f6feb;
        color: #58a6ff;
    }

    /* === ベスト戦略カード === */
    .best-strategy-card {
        background: linear-gradient(135deg, #161b22 0%, #1c2333 100%);
        border: 1px solid #f0883e;
        border-radius: 12px;
        padding: 20px;
        margin: 12px 0;
    }

    .best-strategy-card .label {
        color: #f0883e;
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 4px;
    }

    .best-strategy-card .score {
        font-size: 2rem;
        font-weight: 700;
        color: #f0883e;
        margin-bottom: 8px;
    }

    .best-strategy-card .detail {
        font-size: 0.85rem;
        color: #c9d1d9;
        line-height: 1.6;
    }

    /* === ステップインジケーター === */
    .step-indicator {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 6px 0;
        font-size: 0.8rem;
    }

    .step-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        flex-shrink: 0;
    }

    .step-dot.done {
        background-color: #3fb950;
    }

    .step-dot.current {
        background-color: #58a6ff;
        box-shadow: 0 0 6px rgba(88, 166, 255, 0.4);
    }

    .step-dot.todo {
        background-color: #30363d;
    }

    .step-label {
        color: #e6edf3;
    }

    .step-label.dimmed {
        color: #484f58;
    }

    /* === テンプレートカテゴリ === */
    .template-category {
        background-color: #161b22;
        border: 1px solid #21262d;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 8px;
    }

    .template-category-header {
        display: flex;
        align-items: center;
        gap: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        margin-bottom: 8px;
    }

    .template-tag {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.7rem;
        font-weight: 600;
    }

    .tag-long {
        background-color: rgba(38, 166, 154, 0.15);
        color: #26a69a;
        border: 1px solid rgba(38, 166, 154, 0.3);
    }

    .tag-short {
        background-color: rgba(239, 83, 80, 0.15);
        color: #ef5350;
        border: 1px solid rgba(239, 83, 80, 0.3);
    }

    /* === プログレスバー === */
    .stProgress > div > div {
        background-color: #238636;
    }

    /* === データフレームスタイル === */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }

    /* === タブ === */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background-color: #161b22;
        border-radius: 8px 8px 0 0;
        padding: 0 4px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 6px 6px 0 0;
        padding: 8px 20px;
        font-weight: 500;
    }

    /* === ディバイダー === */
    hr {
        border-color: #21262d !important;
        margin: 16px 0 !important;
    }

    /* === エクスパンダー === */
    .streamlit-expanderHeader {
        background-color: #161b22;
        border-radius: 6px;
        font-size: 0.85rem;
    }

    /* === レジーム別ベストカード === */
    .regime-best-card {
        background-color: #161b22;
        border: 1px solid #21262d;
        border-radius: 10px;
        padding: 16px;
    }

    .regime-best-card.viable {
        border-color: #238636;
    }

    .regime-best-card.not-viable {
        border-color: #da3633;
        opacity: 0.7;
    }

    .regime-best-card .regime-title {
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 8px;
    }

    .regime-best-card .template-name {
        font-size: 0.85rem;
        color: #58a6ff;
        font-weight: 500;
        margin-bottom: 4px;
    }

    .regime-best-card .param-text {
        font-size: 0.75rem;
        color: #8b949e;
        margin-bottom: 10px;
    }

    .regime-best-card .metric-row {
        display: flex;
        justify-content: space-between;
        font-size: 0.8rem;
        padding: 3px 0;
        border-bottom: 1px solid #21262d;
    }

    .regime-best-card .metric-row:last-child {
        border-bottom: none;
    }

    .regime-best-card .metric-label {
        color: #8b949e;
    }

    .regime-best-card .metric-value {
        font-weight: 600;
        color: #e6edf3;
    }

    .regime-best-card .metric-value.positive {
        color: #3fb950;
    }

    .regime-best-card .metric-value.negative {
        color: #f85149;
    }

    .regime-best-card .verdict {
        margin-top: 10px;
        text-align: center;
        font-size: 0.85rem;
        font-weight: 600;
        padding: 6px;
        border-radius: 6px;
    }

    .regime-best-card .verdict.pass {
        background-color: rgba(35, 134, 54, 0.15);
        color: #3fb950;
    }

    .regime-best-card .verdict.fail {
        background-color: rgba(218, 54, 51, 0.15);
        color: #f85149;
    }

    /* === サイドバー ワークフローガイド === */
    .sidebar-guide {
        padding: 4px 0;
    }

    .guide-title {
        font-size: 0.75rem;
        font-weight: 600;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 10px;
    }

    .guide-route {
        margin-bottom: 10px;
    }

    .guide-route-label {
        font-size: 0.7rem;
        color: #58a6ff;
        font-weight: 600;
        margin-bottom: 4px;
    }

    .guide-flow {
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 2px;
    }

    .guide-step {
        font-size: 0.7rem;
        color: #484f58;
        padding: 2px 6px;
        border: 1px solid #30363d;
        border-radius: 4px;
        background-color: #0d1117;
        white-space: nowrap;
    }

    .guide-step.done {
        color: #3fb950;
        border-color: #238636;
        background-color: rgba(35, 134, 54, 0.1);
    }

    .guide-arrow {
        color: #30363d;
        font-size: 0.7rem;
        padding: 0 1px;
    }

    /* === ウェルカムセクション（メインエリア） === */
    .welcome-section {
        background: linear-gradient(135deg, #161b22 0%, #1c2333 100%);
        border: 1px solid #21262d;
        border-radius: 12px;
        padding: 32px;
        text-align: center;
        margin-bottom: 8px;
    }

    .welcome-title {
        color: #e6edf3;
        font-size: 1.6rem;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .welcome-desc {
        color: #8b949e;
        font-size: 0.9rem;
        line-height: 1.6;
        margin-bottom: 24px;
    }

    .welcome-routes {
        display: flex;
        gap: 16px;
        justify-content: center;
        flex-wrap: wrap;
    }

    .welcome-route {
        background-color: #0d1117;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 20px 24px;
        min-width: 240px;
        text-align: center;
    }

    .welcome-route-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #58a6ff;
        margin-bottom: 8px;
    }

    .welcome-route-desc {
        font-size: 0.9rem;
        font-weight: 600;
        color: #e6edf3;
        margin-bottom: 6px;
    }

    .welcome-route-note {
        font-size: 0.75rem;
        color: #8b949e;
    }

    /* === サマリーカード（Optimizer結果上部） === */
    .summary-row {
        display: flex;
        gap: 12px;
        margin-bottom: 16px;
    }

    .summary-card {
        flex: 1;
        background-color: #161b22;
        border: 1px solid #21262d;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }

    .summary-card .value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #e6edf3;
    }

    .summary-card .label {
        font-size: 0.7rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 4px;
    }
    </style>
    """, unsafe_allow_html=True)


def section_header(icon: str, title: str, subtitle: str = ""):
    """セクションヘッダーを描画"""
    sub_html = f'<span class="subtitle">{subtitle}</span>' if subtitle else ""
    st.markdown(
        f'<div class="section-header">'
        f'<span class="icon">{icon}</span>'
        f'<span class="title">{title}</span>'
        f'{sub_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


def status_badge(text: str, status: str = "pending"):
    """ステータスバッジを返す (HTML文字列)"""
    return f'<span class="status-badge status-{status}">{text}</span>'


def best_strategy_card(score: float, template: str, regime: str, params: str):
    """ベスト戦略カードを描画"""
    st.markdown(
        f"""
        <div class="best-strategy-card">
            <div class="label">Best Strategy</div>
            <div class="score">{score:.4f}</div>
            <div class="detail">
                Template: <strong>{template}</strong><br>
                Regime: <strong>{regime}</strong><br>
                Params: {params}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def template_tag(side: str) -> str:
    """Long/Shortタグ (HTML)"""
    cls = "tag-long" if side == "long" else "tag-short"
    label = "LONG" if side == "long" else "SHORT"
    return f'<span class="template-tag {cls}">{label}</span>'
