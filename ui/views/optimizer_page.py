"""
Optimizerページ

戦略テンプレート×パラメータのグリッドサーチ自動最適化。
設定 → 結果の2ビュー構成。最適化完了後は自動で結果表示に切り替え。
"""

import os
import time
import streamlit as st
import yaml
import pandas as pd

from data.base import Timeframe
from analysis.trend import TrendDetector, TrendRegime
from optimizer.templates import BUILTIN_TEMPLATES, ParameterRange
from optimizer.scoring import ScoringWeights
from optimizer.grid import GridSearchOptimizer
from ui.components.styles import section_header, best_strategy_card, template_tag


REGIME_OPTIONS = {
    "uptrend": "Uptrend",
    "downtrend": "Downtrend",
    "range": "Range",
}

REGIME_ICONS = {
    "uptrend": "📈",
    "downtrend": "📉",
    "range": "↔️",
}


def render_optimizer_page():
    """Optimizerページを描画"""
    st.header("⚡ Strategy Optimizer")

    if not st.session_state.get("datasets"):
        st.warning("まず Data Loader でデータを読み込んでください。")
        return

    if "optimization_result" not in st.session_state:
        st.session_state.optimization_result = None

    # 最適化完了後は自動で結果ビューを表示
    if "optimizer_view" not in st.session_state:
        st.session_state.optimizer_view = "config"

    has_results = st.session_state.optimization_result is not None

    # ビュー切り替え
    col_nav1, col_nav2, col_spacer = st.columns([1, 1, 4])
    with col_nav1:
        if st.button(
            "⚙️ Configuration",
            type="primary" if st.session_state.optimizer_view == "config" else "secondary",
            use_container_width=True,
        ):
            st.session_state.optimizer_view = "config"
            st.rerun()
    with col_nav2:
        btn_label = f"📊 Results ({st.session_state.optimization_result.total_combinations})" if has_results else "📊 Results"
        if st.button(
            btn_label,
            type="primary" if st.session_state.optimizer_view == "results" else "secondary",
            disabled=not has_results,
            use_container_width=True,
        ):
            st.session_state.optimizer_view = "results"
            st.rerun()

    st.divider()

    if st.session_state.optimizer_view == "config":
        _render_config_view()
    else:
        _render_results_view()


def _render_config_view():
    """設定ビュー"""

    datasets = st.session_state.datasets

    # --- 0. データセット選択 ---
    section_header("📦", "Dataset", "最適化に使用するデータ")

    symbols = list(datasets.keys())
    selected_symbol = st.selectbox(
        "Symbol",
        options=symbols,
        index=0,
        key="opt_symbol",
    )

    # 選択したシンボルのTF一覧
    active_tf_dict = datasets[selected_symbol]
    loaded_tfs = list(active_tf_dict.keys())

    # 選択シンボルの情報表示
    tf_info = ", ".join(
        f"{tf}({active_tf_dict[tf].bars:,})" for tf in loaded_tfs
    )
    st.caption(f"**{selected_symbol}**: {tf_info}")

    # ohlcv_dict を選択シンボルで同期
    st.session_state.ohlcv_dict = active_tf_dict

    st.divider()

    # --- 1. トレンド検出 ---
    section_header("📐", "Trend Detection", "トレンド判定の設定")

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
            format_func=lambda x: f"{REGIME_ICONS.get(x, '')} {REGIME_OPTIONS[x]}",
            key="opt_regimes",
        )

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

    # --- 2. テンプレート選択（Long/Short分類） ---
    section_header("🧩", "Strategy Templates", "テスト対象のテンプレート")

    # Long/Shortに分類
    long_templates = {k: v for k, v in BUILTIN_TEMPLATES.items()
                      if v.config_template.get("side", "long") == "long"}
    short_templates = {k: v for k, v in BUILTIN_TEMPLATES.items()
                       if v.config_template.get("side", "long") == "short"}

    col_long, col_short = st.columns(2)

    with col_long:
        st.markdown(
            f'{template_tag("long")} **Long Templates** ({len(long_templates)})',
            unsafe_allow_html=True,
        )
        selected_long = st.multiselect(
            "Long",
            options=list(long_templates.keys()),
            default=list(long_templates.keys()),
            format_func=lambda x: f"{x}",
            key="opt_long_templates",
            label_visibility="collapsed",
        )

    with col_short:
        st.markdown(
            f'{template_tag("short")} **Short Templates** ({len(short_templates)})',
            unsafe_allow_html=True,
        )
        selected_short = st.multiselect(
            "Short",
            options=list(short_templates.keys()),
            default=list(short_templates.keys()),
            format_func=lambda x: f"{x}",
            key="opt_short_templates",
            label_visibility="collapsed",
        )

    selected_templates = selected_long + selected_short

    # パラメータ範囲設定
    custom_ranges = {}
    total_combinations = 0

    if selected_templates:
        st.caption(f"選択中: {len(selected_templates)} templates")

        for tname in selected_templates:
            template = BUILTIN_TEMPLATES[tname]
            side = template.config_template.get("side", "long")
            tag = template_tag(side)

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
                st.caption(f"Combinations: **{count}**")

    st.divider()

    # --- 3. スコア重み ---
    section_header("🎯", "Scoring Weights", "複合スコアの重み配分")

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
        st.warning(f"⚠️ Weights sum = {weight_sum:.2f} (should be 1.0)")
    else:
        st.caption(f"✓ Weights sum = {weight_sum:.2f}")

    st.divider()

    # --- 4. バックテスト設定 ---
    section_header("⚙️", "Backtest Settings", "実行パラメータ")

    bcol1, bcol2, bcol3, bcol4 = st.columns(4)
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
    with bcol4:
        max_workers = os.cpu_count() or 4
        n_workers = st.number_input(
            "Workers (並列数)",
            value=max_workers,
            min_value=1,
            max_value=max_workers,
            step=1,
            key="opt_n_workers",
            help=f"CPU: {max_workers}コア",
        )

    st.divider()

    # --- 実行 ---
    total_runs = total_combinations * len(target_regimes)

    # サマリーカード
    scol1, scol2, scol3, scol4 = st.columns(4)
    with scol1:
        st.metric("Templates", f"{len(selected_templates)}")
    with scol2:
        st.metric("Regimes", f"{len(target_regimes)}")
    with scol3:
        st.metric("Combinations", f"{total_combinations:,}")
    with scol4:
        st.metric("Total Runs", f"{total_runs:,}")

    if st.button("🚀 Run Optimization", type="primary", use_container_width=True):
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
            n_workers=int(n_workers),
        )


def _run_optimization(
    exec_tf, htf, trend_method, target_regimes,
    selected_templates, custom_ranges, scoring_weights,
    initial_capital, commission, slippage,
    ma_fast, ma_slow, adx_period, adx_trend_th, adx_range_th,
    n_workers=1,
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

    # config生成
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
        elapsed = time.time() - start_time
        speed = current / elapsed if elapsed > 0 else 0
        progress_bar.progress(
            current / total,
            text=f"⚡ {current}/{total} ({speed:.0f} runs/s) [{elapsed:.1f}s]",
        )

    start_time = time.time()

    result_set = optimizer.run(
        df=exec_df,
        configs=all_configs,
        target_regimes=target_regimes,
        progress_callback=on_progress,
        n_workers=n_workers,
    )

    elapsed = time.time() - start_time

    result_set.symbol = exec_ohlcv.symbol
    result_set.execution_tf = exec_tf
    result_set.htf = htf or ""

    st.session_state.optimization_result = result_set
    progress_bar.progress(1.0, text=f"✅ Done! [{elapsed:.1f}s]")
    st.success(
        f"**{result_set.total_combinations}** results in **{elapsed:.1f}s** "
        f"(Workers: {n_workers})"
    )

    # 自動で結果ビューに切り替え
    st.session_state.optimizer_view = "results"
    st.rerun()


def _render_results_view():
    """結果ビュー"""
    if st.session_state.optimization_result is None:
        st.info("まず Configuration で最適化を実行してください。")
        return

    from ui.components.optimizer_charts import (
        create_scatter_chart,
        create_equity_overlay,
        create_regime_distribution_chart,
    )

    result_set = st.session_state.optimization_result

    # ヘッダー情報
    st.markdown(
        f"**{result_set.symbol}** | "
        f"Exec: `{result_set.execution_tf}` | "
        f"HTF: `{result_set.htf}` | "
        f"Total: **{result_set.total_combinations}** runs"
    )

    # --- ベスト戦略カード ---
    best = result_set.best
    if best:
        best_strategy_card(
            score=best.composite_score,
            template=best.template_name,
            regime=best.trend_regime,
            params=best.param_str,
        )

    st.divider()

    # --- フィルター ---
    section_header("🔎", "Filter & Ranking")

    fcol1, fcol2, fcol3 = st.columns([1, 1, 2])
    with fcol1:
        filter_regime = st.selectbox(
            "Regime",
            options=["all"] + list(REGIME_OPTIONS.keys()),
            format_func=lambda x: (
                "All Regimes" if x == "all"
                else f"{REGIME_ICONS.get(x, '')} {REGIME_OPTIONS.get(x, x)}"
            ),
            key="result_filter_regime",
        )
    with fcol2:
        templates_in_results = sorted(set(e.template_name for e in result_set.entries))
        filter_template = st.selectbox(
            "Template",
            options=["all"] + templates_in_results,
            key="result_filter_template",
        )
    with fcol3:
        min_trades = st.slider(
            "Min Trades",
            min_value=0,
            max_value=50,
            value=0,
            key="result_min_trades",
            help="最低トレード数でフィルタ",
        )

    # フィルタリング
    filtered = result_set
    if filter_regime != "all":
        filtered = filtered.filter_regime(filter_regime)
    if filter_template != "all":
        filtered = filtered.filter_template(filter_template)

    # min trades フィルタ
    if min_trades > 0:
        from optimizer.results import OptimizationResultSet
        filtered_entries = [e for e in filtered.entries if e.metrics.total_trades >= min_trades]
        filtered = OptimizationResultSet(
            entries=filtered_entries,
            symbol=filtered.symbol,
            execution_tf=filtered.execution_tf,
            htf=filtered.htf,
        )

    if not filtered.entries:
        st.warning("条件に一致する結果がありません。")
        return

    st.caption(f"Showing {len(filtered.entries)} results")

    # --- ランキングテーブル（スタイル付き） ---
    ranking_df = filtered.to_dataframe()

    # カラム設定でスタイリング
    column_config = {
        "score": st.column_config.ProgressColumn(
            "Score",
            min_value=0,
            max_value=1,
            format="%.4f",
        ),
        "win_rate": st.column_config.NumberColumn(
            "Win Rate %",
            format="%.1f%%",
        ),
        "profit_factor": st.column_config.NumberColumn(
            "PF",
            format="%.2f",
        ),
        "total_pnl": st.column_config.NumberColumn(
            "Total P/L %",
            format="%.2f%%",
        ),
        "max_dd": st.column_config.NumberColumn(
            "Max DD %",
            format="%.2f%%",
        ),
        "sharpe": st.column_config.NumberColumn(
            "Sharpe",
            format="%.2f",
        ),
        "trades": st.column_config.NumberColumn(
            "Trades",
            format="%d",
        ),
    }

    st.dataframe(
        ranking_df,
        use_container_width=True,
        hide_index=False,
        column_config=column_config,
        height=400,
    )

    st.divider()

    # --- チャート ---
    section_header("📊", "Charts")

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        scatter_fig = create_scatter_chart(filtered)
        st.plotly_chart(scatter_fig, use_container_width=True)
    with chart_col2:
        dist_fig = create_regime_distribution_chart(filtered)
        st.plotly_chart(dist_fig, use_container_width=True)

    # --- エクイティカーブオーバーレイ ---
    ranked_entries = filtered.ranked()
    entries_with_result = [
        e for e in ranked_entries if e.backtest_result is not None
    ]
    if entries_with_result:
        st.divider()
        section_header("📈", "Equity Curve Overlay", f"Top {min(len(entries_with_result), 10)}")

        top_n = st.slider(
            "Top N to display", 1, min(20, len(entries_with_result)),
            min(5, len(entries_with_result)), key="equity_top_n"
        )
        equity_fig = create_equity_overlay(entries_with_result, max_entries=top_n)
        st.plotly_chart(equity_fig, use_container_width=True)

    st.divider()

    # --- YAMLエクスポート ---
    section_header("💾", "Export Best Strategy")

    if best:
        yaml_str = yaml.dump(best.config, default_flow_style=False, allow_unicode=True)

        col_yaml, col_dl = st.columns([3, 1])
        with col_yaml:
            st.code(yaml_str, language="yaml")
        with col_dl:
            st.download_button(
                "📥 Download YAML",
                data=yaml_str,
                file_name=f"best_strategy_{best.template_name}.yaml",
                mime="text/yaml",
                use_container_width=True,
            )
