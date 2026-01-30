"""
Optimizerページ

戦略テンプレート×パラメータのグリッドサーチ自動最適化。
設定タブと結果タブで構成。
"""

import streamlit as st
import yaml

from data.base import Timeframe
from analysis.trend import TrendDetector, TrendRegime
from optimizer.templates import BUILTIN_TEMPLATES, ParameterRange
from optimizer.scoring import ScoringWeights
from optimizer.grid import GridSearchOptimizer


REGIME_OPTIONS = {
    "uptrend": "Uptrend",
    "downtrend": "Downtrend",
    "range": "Range",
}


def render_optimizer_page():
    """Optimizerページを描画"""
    st.header("Strategy Optimizer")

    if "ohlcv_dict" not in st.session_state or not st.session_state.ohlcv_dict:
        st.warning("まず Data Loader でデータを読み込んでください。")
        return

    if "optimization_result" not in st.session_state:
        st.session_state.optimization_result = None

    tab_config, tab_results = st.tabs(["Configuration", "Results"])

    with tab_config:
        _render_config_tab()

    with tab_results:
        _render_results_tab()


def _render_config_tab():
    """設定タブ"""

    # --- トレンド検出設定 ---
    st.subheader("1. Trend Detection")

    loaded_tfs = list(st.session_state.ohlcv_dict.keys())

    col1, col2 = st.columns(2)
    with col1:
        exec_tf = st.selectbox(
            "Execution Timeframe (実行TF)",
            options=loaded_tfs,
            index=0,
            key="opt_exec_tf",
            help="バックテストを実行するタイムフレーム",
        )
    with col2:
        htf_options = [tf for tf in loaded_tfs if tf != exec_tf]
        if htf_options:
            htf = st.selectbox(
                "Higher Timeframe (上位TF)",
                options=htf_options,
                index=0,
                key="opt_htf",
                help="トレンド判定に使用する上位タイムフレーム",
            )
        else:
            htf = None
            st.info("トレンド検出には2つ以上のTFデータが必要です")

    col3, col4 = st.columns(2)
    with col3:
        trend_method = st.selectbox(
            "Detection Method",
            options=["ma_cross", "adx", "combined"],
            format_func=lambda x: {
                "ma_cross": "MA Cross",
                "adx": "ADX",
                "combined": "MA Cross + ADX (Combined)",
            }[x],
            key="opt_trend_method",
        )
    with col4:
        target_regimes = st.multiselect(
            "Target Regimes",
            options=list(REGIME_OPTIONS.keys()),
            default=list(REGIME_OPTIONS.keys()),
            format_func=lambda x: REGIME_OPTIONS[x],
            key="opt_regimes",
        )

    # トレンド検出パラメータ
    with st.expander("Trend Detection Parameters", expanded=False):
        tcol1, tcol2, tcol3 = st.columns(3)
        with tcol1:
            ma_fast = st.number_input("MA Fast Period", value=20, min_value=5, key="opt_ma_fast")
            ma_slow = st.number_input("MA Slow Period", value=50, min_value=10, key="opt_ma_slow")
        with tcol2:
            adx_period = st.number_input("ADX Period", value=14, min_value=5, key="opt_adx_period")
            adx_trend_th = st.number_input("ADX Trend Threshold", value=25.0, key="opt_adx_trend_th")
        with tcol3:
            adx_range_th = st.number_input("ADX Range Threshold", value=20.0, key="opt_adx_range_th")

    st.divider()

    # --- テンプレート選択 ---
    st.subheader("2. Strategy Templates")

    selected_templates = st.multiselect(
        "Select Templates",
        options=list(BUILTIN_TEMPLATES.keys()),
        default=list(BUILTIN_TEMPLATES.keys()),
        format_func=lambda x: f"{x} - {BUILTIN_TEMPLATES[x].description}",
        key="opt_templates",
    )

    # 各テンプレートのパラメータ範囲
    custom_ranges = {}
    total_combinations = 0

    for tname in selected_templates:
        template = BUILTIN_TEMPLATES[tname]
        with st.expander(f"{tname} Parameters", expanded=False):
            tpl_ranges = {}
            for pr in template.param_ranges:
                pcol1, pcol2, pcol3 = st.columns(3)
                with pcol1:
                    min_val = st.number_input(
                        f"{pr.name} min",
                        value=int(pr.min_val) if pr.param_type == "int" else pr.min_val,
                        key=f"opt_{tname}_{pr.name}_min",
                    )
                with pcol2:
                    max_val = st.number_input(
                        f"{pr.name} max",
                        value=int(pr.max_val) if pr.param_type == "int" else pr.max_val,
                        key=f"opt_{tname}_{pr.name}_max",
                    )
                with pcol3:
                    step = st.number_input(
                        f"{pr.name} step",
                        value=int(pr.step) if pr.param_type == "int" else pr.step,
                        min_value=1 if pr.param_type == "int" else 0.01,
                        key=f"opt_{tname}_{pr.name}_step",
                    )
                tpl_ranges[pr.name] = ParameterRange(
                    pr.name, float(min_val), float(max_val), float(step), pr.param_type
                )

            custom_ranges[tname] = tpl_ranges
            count = template.combination_count(tpl_ranges)
            total_combinations += count
            st.caption(f"Combinations: {count}")

    st.divider()

    # --- スコア重み ---
    st.subheader("3. Scoring Weights")

    wcol1, wcol2, wcol3, wcol4 = st.columns(4)
    with wcol1:
        w_pf = st.slider("Profit Factor", 0.0, 1.0, 0.3, 0.05, key="opt_w_pf")
    with wcol2:
        w_wr = st.slider("Win Rate", 0.0, 1.0, 0.3, 0.05, key="opt_w_wr")
    with wcol3:
        w_dd = st.slider("Max DD (inv)", 0.0, 1.0, 0.2, 0.05, key="opt_w_dd")
    with wcol4:
        w_sh = st.slider("Sharpe", 0.0, 1.0, 0.2, 0.05, key="opt_w_sh")

    weight_sum = w_pf + w_wr + w_dd + w_sh
    if abs(weight_sum - 1.0) > 0.01:
        st.warning(f"Weights sum = {weight_sum:.2f} (should be 1.0)")

    st.divider()

    # --- バックテスト設定 ---
    st.subheader("4. Backtest Settings")

    bcol1, bcol2, bcol3 = st.columns(3)
    with bcol1:
        initial_capital = st.number_input(
            "Initial Capital", value=10000.0, min_value=100.0, key="opt_capital"
        )
    with bcol2:
        commission = st.number_input(
            "Commission (%)", value=0.04, min_value=0.0, step=0.01, key="opt_commission"
        )
    with bcol3:
        slippage = st.number_input(
            "Slippage (%)", value=0.0, min_value=0.0, step=0.01, key="opt_slippage"
        )

    # --- 実行 ---
    total_runs = total_combinations * len(target_regimes)
    st.info(
        f"Total combinations: {total_combinations} templates × "
        f"{len(target_regimes)} regimes = **{total_runs}** runs"
    )

    if st.button("Run Optimization", type="primary", use_container_width=True):
        if not selected_templates:
            st.error("テンプレートを1つ以上選択してください")
            return
        if not target_regimes:
            st.error("レジームを1つ以上選択してください")
            return

        _run_optimization(
            exec_tf=exec_tf,
            htf=htf,
            trend_method=trend_method,
            target_regimes=target_regimes,
            selected_templates=selected_templates,
            custom_ranges=custom_ranges,
            scoring_weights=ScoringWeights(w_pf, w_wr, w_dd, w_sh),
            initial_capital=initial_capital,
            commission=commission,
            slippage=slippage,
            ma_fast=int(ma_fast),
            ma_slow=int(ma_slow),
            adx_period=int(adx_period),
            adx_trend_th=float(adx_trend_th),
            adx_range_th=float(adx_range_th),
        )


def _run_optimization(
    exec_tf, htf, trend_method, target_regimes,
    selected_templates, custom_ranges, scoring_weights,
    initial_capital, commission, slippage,
    ma_fast, ma_slow, adx_period, adx_trend_th, adx_range_th,
):
    """最適化を実行"""
    exec_ohlcv = st.session_state.ohlcv_dict[exec_tf]
    exec_df = exec_ohlcv.df.copy()

    # トレンドラベル付与
    if htf and htf in st.session_state.ohlcv_dict:
        htf_ohlcv = st.session_state.ohlcv_dict[htf]
        htf_df = htf_ohlcv.df.copy()

        detector = TrendDetector()

        if trend_method == "ma_cross":
            htf_df = detector.detect_ma_cross(
                htf_df, fast_period=ma_fast, slow_period=ma_slow
            )
        elif trend_method == "adx":
            htf_df = detector.detect_adx(
                htf_df, adx_period=adx_period,
                trend_threshold=adx_trend_th,
                range_threshold=adx_range_th,
            )
        else:  # combined
            htf_df = detector.detect_combined(
                htf_df, ma_fast=ma_fast, ma_slow=ma_slow,
                adx_period=adx_period,
                adx_trend_threshold=adx_trend_th,
                adx_range_threshold=adx_range_th,
            )

        exec_df = TrendDetector.label_execution_tf(exec_df, htf_df)
    else:
        exec_df["trend_regime"] = TrendRegime.RANGE.value

    # config生成（テンプレートが_template_nameと_paramsを自動付加）
    all_configs = []
    for tname in selected_templates:
        template = BUILTIN_TEMPLATES[tname]
        tpl_ranges = custom_ranges.get(tname, {})
        configs = template.generate_configs(tpl_ranges)
        all_configs.extend(configs)

    # グリッドサーチ実行
    optimizer = GridSearchOptimizer(
        initial_capital=initial_capital,
        commission_pct=commission,
        slippage_pct=slippage,
        scoring_weights=scoring_weights,
    )

    progress_bar = st.progress(0, text="Starting optimization...")

    def on_progress(current, total, desc):
        progress_bar.progress(
            current / total,
            text=f"({current}/{total}) {desc}",
        )

    result_set = optimizer.run(
        df=exec_df,
        configs=all_configs,
        target_regimes=target_regimes,
        progress_callback=on_progress,
    )

    result_set.symbol = exec_ohlcv.symbol
    result_set.execution_tf = exec_tf
    result_set.htf = htf or ""

    st.session_state.optimization_result = result_set
    progress_bar.progress(1.0, text="Done!")
    st.success(
        f"Optimization complete: {result_set.total_combinations} results"
    )
    st.rerun()


def _render_results_tab():
    """結果タブ"""
    if st.session_state.optimization_result is None:
        st.info("まず Configuration タブで最適化を実行してください。")
        return

    from ui.components.optimizer_charts import (
        create_scatter_chart,
        create_equity_overlay,
        create_regime_distribution_chart,
    )

    result_set = st.session_state.optimization_result

    st.subheader(
        f"Results: {result_set.symbol} | "
        f"Exec: {result_set.execution_tf} | HTF: {result_set.htf}"
    )

    # フィルター
    fcol1, fcol2 = st.columns(2)
    with fcol1:
        filter_regime = st.selectbox(
            "Filter by Regime",
            options=["all"] + list(REGIME_OPTIONS.keys()),
            format_func=lambda x: "All Regimes" if x == "all" else REGIME_OPTIONS.get(x, x),
            key="result_filter_regime",
        )
    with fcol2:
        templates_in_results = list(
            set(e.template_name for e in result_set.entries)
        )
        filter_template = st.selectbox(
            "Filter by Template",
            options=["all"] + templates_in_results,
            key="result_filter_template",
        )

    # フィルタリング
    filtered = result_set
    if filter_regime != "all":
        filtered = filtered.filter_regime(filter_regime)
    if filter_template != "all":
        filtered = filtered.filter_template(filter_template)

    if not filtered.entries:
        st.warning("条件に一致する結果がありません。")
        return

    # ランキングテーブル
    st.subheader("Ranking")
    ranking_df = filtered.to_dataframe()
    st.dataframe(
        ranking_df,
        use_container_width=True,
        hide_index=False,
    )

    # チャート
    st.subheader("Charts")

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        scatter_fig = create_scatter_chart(filtered)
        st.plotly_chart(scatter_fig, use_container_width=True)
    with chart_col2:
        dist_fig = create_regime_distribution_chart(filtered)
        st.plotly_chart(dist_fig, use_container_width=True)

    # エクイティカーブオーバーレイ
    ranked_entries = filtered.ranked()
    entries_with_result = [
        e for e in ranked_entries if e.backtest_result is not None
    ]
    if entries_with_result:
        st.subheader("Equity Curve Overlay")
        top_n = st.slider(
            "Top N to display", 1, min(20, len(entries_with_result)),
            min(5, len(entries_with_result)), key="equity_top_n"
        )
        equity_fig = create_equity_overlay(entries_with_result, max_entries=top_n)
        st.plotly_chart(equity_fig, use_container_width=True)

    # ベスト戦略のYAMLエクスポート
    st.subheader("Export Best Strategy")
    best = filtered.best
    if best:
        st.metric("Best Score", f"{best.composite_score:.4f}")
        st.caption(
            f"Template: {best.template_name} | "
            f"Regime: {best.trend_regime} | "
            f"Params: {best.param_str}"
        )

        yaml_str = yaml.dump(best.config, default_flow_style=False, allow_unicode=True)
        st.code(yaml_str, language="yaml")
        st.download_button(
            "Download YAML",
            data=yaml_str,
            file_name=f"best_strategy_{best.template_name}.yaml",
            mime="text/yaml",
        )
