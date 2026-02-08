"""
Optimizerページ

戦略テンプレート×パラメータのグリッドサーチ自動最適化。
設定 → 結果の2ビュー構成。最適化完了後は自動で結果表示に切り替え。
"""

import base64
import json
import os
import time
from collections import Counter
import numpy as np
import streamlit as st
import yaml
import pandas as pd

from data.base import Timeframe
from analysis.trend import TrendDetector, TrendRegime
from optimizer.templates import BUILTIN_TEMPLATES, ParameterRange
from optimizer.scoring import ScoringWeights, detect_overfitting_warnings
from optimizer.grid import GridSearchOptimizer
from optimizer.genetic import GeneticOptimizer, GAConfig, GAResult
from optimizer.validation import (
    DataSplitConfig,
    ValidatedResultSet,
    run_validated_optimization,
)
from ui.components.styles import section_header, template_tag
from ui.views.batch_optimizer import render_batch_view, render_batch_load_view


REGIME_OPTIONS = {
    "all": "All (レジーム無視)",
    "uptrend": "Uptrend",
    "downtrend": "Downtrend",
    "range": "Range",
}

REGIME_ICONS = {
    "all": "🌐",
    "uptrend": "📈",
    "downtrend": "📉",
    "range": "↔️",
}


def render_optimizer_page():
    """Optimizerページを描画"""
    st.header("⚡ 戦略オプティマイザー")

    # セッション初期化（ガードより先に実行）
    if "optimization_result" not in st.session_state:
        st.session_state.optimization_result = None
    if "optimizer_view" not in st.session_state:
        st.session_state.optimizer_view = "config"
    if "comparison_results" not in st.session_state:
        st.session_state.comparison_results = []
    if "regime_switching_result" not in st.session_state:
        st.session_state.regime_switching_result = None
    if "validated_result" not in st.session_state:
        st.session_state.validated_result = None
    if "wfa_result" not in st.session_state:
        st.session_state.wfa_result = None

    has_results = st.session_state.optimization_result is not None
    has_data = bool(st.session_state.get("datasets"))
    n_compare = len(st.session_state.comparison_results)

    # ビュー切り替え（データ有無によらず表示）
    col_nav1, col_nav2, col_nav3, col_nav4, col_nav5, col_nav6 = st.columns(6)
    with col_nav1:
        if st.button(
            "⚙️ 設定",
            type="primary" if st.session_state.optimizer_view == "config" else "secondary",
            disabled=not has_data,
            use_container_width=True,
        ):
            st.session_state.optimizer_view = "config"
            st.rerun()
    with col_nav2:
        btn_label = f"📊 結果 ({st.session_state.optimization_result.total_combinations})" if has_results else "📊 結果"
        if st.button(
            btn_label,
            type="primary" if st.session_state.optimizer_view == "results" else "secondary",
            disabled=not has_results,
            use_container_width=True,
        ):
            st.session_state.optimizer_view = "results"
            st.rerun()
    with col_nav3:
        if st.button(
            "📁 読込",
            type="primary" if st.session_state.optimizer_view == "load" else "secondary",
            use_container_width=True,
        ):
            st.session_state.optimizer_view = "load"
            st.rerun()
    with col_nav4:
        compare_label = f"🔀 比較 ({n_compare})" if n_compare >= 2 else "🔀 比較"
        if st.button(
            compare_label,
            type="primary" if st.session_state.optimizer_view == "compare" else "secondary",
            disabled=n_compare < 2,
            use_container_width=True,
        ):
            st.session_state.optimizer_view = "compare"
            st.rerun()
    with col_nav5:
        if st.button(
            "🚀 バッチ",
            type="primary" if st.session_state.optimizer_view == "batch" else "secondary",
            use_container_width=True,
        ):
            st.session_state.optimizer_view = "batch"
            st.rerun()
    with col_nav6:
        if st.button(
            "📂 バッチ結果",
            type="primary" if st.session_state.optimizer_view == "batch_load" else "secondary",
            use_container_width=True,
        ):
            st.session_state.optimizer_view = "batch_load"
            st.rerun()

    st.divider()

    if st.session_state.optimizer_view == "config":
        if not has_data:
            st.info(
                "📂 **データが読み込まれていません**\n\n"
                "最適化にはOHLCVデータが必要です（2つ以上のタイムフレーム推奨）。\n\n"
                "サイドバーの **📂 データ** ページでCSVファイルを読み込んでください。"
            )
            return
        _render_config_view()
    elif st.session_state.optimizer_view == "load":
        _render_load_view()
    elif st.session_state.optimizer_view == "compare":
        _render_compare_view()
    elif st.session_state.optimizer_view == "batch":
        render_batch_view()
    elif st.session_state.optimizer_view == "batch_load":
        render_batch_load_view()
    else:
        _render_results_view()


def _render_config_view():
    """設定ビュー"""

    datasets = st.session_state.datasets
    trimmed_list = st.session_state.get("trimmed_datasets", [])

    # --- 0. データセット選択 ---
    section_header("📦", "データセット", "最適化に使用するデータ")

    symbols = list(datasets.keys())
    selected_symbol = st.selectbox(
        "シンボル",
        options=symbols,
        index=0,
        key="opt_symbol",
    )

    # データソース選択（オリジナル or 切り出し）
    source_options = ["original"]
    source_labels = {"original": f"📦 オリジナル（全期間）"}
    sym_trimmed = [e for e in trimmed_list if e["symbol"] == selected_symbol]
    for entry in sym_trimmed:
        source_options.append(entry["id"])
        source_labels[entry["id"]] = f"✂️ {entry['label']}"

    if len(source_options) > 1:
        selected_source = st.selectbox(
            "データソース",
            options=source_options,
            format_func=lambda x: source_labels[x],
            key="opt_data_source",
            help="オリジナルの全期間データ、または切り出した期間データを選択",
        )
    else:
        selected_source = "original"

    # 選択したデータソースのTF辞書を取得
    if selected_source == "original":
        active_tf_dict = datasets[selected_symbol]
        # データの実際の期間を取得
        _any_ohlcv = next(iter(active_tf_dict.values()))
        _ps = str(_any_ohlcv.start_time)[:10] if _any_ohlcv.start_time else ""
        _pe = str(_any_ohlcv.end_time)[:10] if _any_ohlcv.end_time else ""
        st.session_state.opt_data_source_info = {
            "source": "original",
            "period_start": _ps,
            "period_end": _pe,
        }
    else:
        trimmed_entry = next(
            (e for e in trimmed_list if e["id"] == selected_source), None
        )
        if trimmed_entry:
            active_tf_dict = trimmed_entry["data"]
            st.session_state.opt_data_source_info = {
                "source": "trimmed",
                "period_start": str(trimmed_entry["start_dt"])[:10],
                "period_end": str(trimmed_entry["end_dt"])[:10],
            }
        else:
            active_tf_dict = datasets[selected_symbol]
            _any_ohlcv = next(iter(active_tf_dict.values()))
            _ps = str(_any_ohlcv.start_time)[:10] if _any_ohlcv.start_time else ""
            _pe = str(_any_ohlcv.end_time)[:10] if _any_ohlcv.end_time else ""
            st.session_state.opt_data_source_info = {
                "source": "original",
                "period_start": _ps,
                "period_end": _pe,
            }

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
    section_header("📐", "トレンド検出", "トレンド判定の設定")

    col1, col2 = st.columns(2)
    with col1:
        exec_default = loaded_tfs.index("15m") if "15m" in loaded_tfs else 0
        exec_tf = st.selectbox(
            "実行タイムフレーム",
            options=loaded_tfs,
            index=exec_default,
            key="opt_exec_tf",
            help="バックテストを実行するタイムフレーム",
        )
    with col2:
        htf_options = [tf for tf in loaded_tfs if tf != exec_tf]
        htf_default = htf_options.index("1h") if "1h" in htf_options else 0
        if htf_options:
            htf = st.selectbox(
                "上位タイムフレーム",
                options=htf_options,
                index=htf_default,
                key="opt_htf",
                help="トレンド判定に使用する上位タイムフレーム",
            )
        else:
            htf = None
            st.info("トレンド検出には2つ以上のTFデータが必要です")

    col3, col4 = st.columns(2)
    with col3:
        trend_method = st.selectbox(
            "検出方法",
            options=["ma_cross", "adx", "combined", "dual_tf_ema"],
            format_func=lambda x: {
                "ma_cross": "MA Cross（移動平均クロス）",
                "adx": "ADX（トレンド強度）",
                "combined": "MA Cross + ADX（複合）",
                "dual_tf_ema": "Dual-TF EMA（4h+1h合意）",
            }[x],
            key="opt_trend_method",
            help="トレンド/レンジを判定するアルゴリズム",
        )
    with col4:
        target_regimes = st.multiselect(
            "対象レジーム",
            options=list(REGIME_OPTIONS.keys()),
            default=["uptrend", "downtrend", "range"],
            format_func=lambda x: f"{REGIME_ICONS.get(x, '')} {REGIME_OPTIONS[x]}",
            key="opt_regimes",
            help="「All」= レジーム分割なしで全バーを対象",
        )

    if trend_method == "dual_tf_ema":
        super_htf_options = [tf for tf in loaded_tfs if tf != exec_tf and tf != htf]
        if super_htf_options:
            super_htf = st.selectbox(
                "Super HTF（上位TF）",
                options=super_htf_options,
                index=super_htf_options.index("4h") if "4h" in super_htf_options else 0,
                key="opt_super_htf",
                help="Dual-TF EMAの上位タイムフレーム（通常4h）",
            )
        else:
            super_htf = None
            st.warning("Dual-TF EMAには3つ以上のTFデータが必要です（例: 15m, 1h, 4h）")

    with st.expander("トレンド検出パラメータ", expanded=False):
        tcol1, tcol2, tcol3 = st.columns(3)
        with tcol1:
            ma_fast = st.number_input("MA 短期", value=20, min_value=5, key="opt_ma_fast", help="短期移動平均の期間")
            ma_slow = st.number_input("MA 長期", value=50, min_value=10, key="opt_ma_slow", help="長期移動平均の期間")
        with tcol2:
            adx_period = st.number_input("ADX 期間", value=14, min_value=5, key="opt_adx_period", help="ADX算出の期間")
            adx_trend_th = st.number_input("ADX トレンド閾値", value=25.0, key="opt_adx_trend_th", help="この値以上でトレンドと判定")
        with tcol3:
            adx_range_th = st.number_input("ADX レンジ閾値", value=20.0, key="opt_adx_range_th", help="この値以下でレンジと判定")

    st.divider()

    # --- 2. テンプレート選択（Long/Short分類） ---
    section_header("🧩", "戦略テンプレート", "テスト対象のテンプレート")

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
    section_header("🎯", "スコア重み", "複合スコアの重み配分")

    wcol1, wcol2, wcol3, wcol4, wcol5 = st.columns(5)
    with wcol1:
        w_pf = st.slider("損益比率", 0.0, 1.0, 0.2, 0.05, key="opt_w_pf", help="総利益÷総損失の重み")
    with wcol2:
        w_wr = st.slider("勝率", 0.0, 1.0, 0.2, 0.05, key="opt_w_wr", help="勝ちトレード割合の重み")
    with wcol3:
        w_dd = st.slider("最大DD (逆)", 0.0, 1.0, 0.2, 0.05, key="opt_w_dd", help="ドローダウンが小さいほど高評価")
    with wcol4:
        w_sh = st.slider("シャープ比", 0.0, 1.0, 0.2, 0.05, key="opt_w_sh", help="リスクあたりリターンの重み")
    with wcol5:
        w_ret = st.slider("合計損益", 0.0, 1.0, 0.2, 0.05, key="opt_w_ret", help="合計リターンの重み")

    weight_sum = w_pf + w_wr + w_dd + w_sh + w_ret
    if abs(weight_sum - 1.0) > 0.01:
        st.warning(f"⚠️ 重みの合計 = {weight_sum:.2f}（1.0にしてください）")
    else:
        st.caption(f"✓ 重みの合計 = {weight_sum:.2f}")

    st.divider()

    # --- 4. バックテスト設定 ---
    section_header("⚙️", "バックテスト設定", "実行パラメータ")

    bcol1, bcol2, bcol3, bcol4 = st.columns(4)
    with bcol1:
        initial_capital = st.number_input(
            "初期資金", value=10000.0, min_value=100.0, key="opt_capital", help="バックテスト開始時の資金 (USDT)"
        )
    with bcol2:
        commission = st.number_input(
            "手数料 (%)", value=0.04, min_value=0.0, step=0.01, key="opt_commission", help="1トレードあたりの取引手数料率"
        )
    with bcol3:
        slippage = st.number_input(
            "スリッページ (%)", value=0.0, min_value=0.0, step=0.01, key="opt_slippage", help="注文時の価格ずれを想定"
        )
    with bcol4:
        max_workers = os.cpu_count() or 4
        n_workers = st.number_input(
            "Workers (並列数)",
            value=1,
            min_value=1,
            max_value=max_workers,
            step=1,
            key="opt_n_workers",
            help=f"CPU: {max_workers}コア",
        )

    st.divider()

    # --- 5. 探索方法 ---
    section_header("🔍", "探索方法", "パラメータ探索のアルゴリズム")

    search_method = st.radio(
        "探索方法",
        options=["grid", "ga"],
        format_func=lambda x: {
            "grid": "グリッドサーチ（全組み合わせ）",
            "ga": "遺伝的アルゴリズム（効率的探索）",
        }[x],
        horizontal=True,
        key="opt_search_method",
        index=0,
    )

    # GA設定
    ga_population = 20
    ga_generations = 10
    ga_elite_ratio = 0.25
    ga_mutation_rate = 0.15

    if search_method == "ga":
        st.info("💡 GAは全パラメータ組み合わせを試さず、進化的に良い解を探索します。グリッドサーチの1/3〜1/5の評価回数で済みます。")
        ga_col1, ga_col2, ga_col3, ga_col4 = st.columns(4)
        with ga_col1:
            ga_population = st.number_input(
                "集団サイズ",
                value=20,
                min_value=10,
                max_value=100,
                step=5,
                key="opt_ga_population",
                help="1世代あたりの個体数（多いほど探索範囲が広い）",
            )
        with ga_col2:
            ga_generations = st.number_input(
                "最大世代数",
                value=10,
                min_value=5,
                max_value=50,
                step=5,
                key="opt_ga_generations",
                help="進化を繰り返す最大回数（収束すれば早期終了）",
            )
        with ga_col3:
            ga_elite_ratio = st.slider(
                "エリート率",
                0.1, 0.5, 0.25, 0.05,
                key="opt_ga_elite",
                help="次世代に引き継ぐ上位個体の割合",
            )
        with ga_col4:
            ga_mutation_rate = st.slider(
                "突然変異率",
                0.05, 0.3, 0.15, 0.05,
                key="opt_ga_mutation",
                help="パラメータをランダムに変化させる確率",
            )

    st.divider()

    # --- 6. 検証モード ---
    section_header("🛡️", "検証設定", "過学習防止のための検証方式")

    validation_mode = st.radio(
        "検証モード",
        options=["off", "oos_3split", "wfa"],
        format_func=lambda x: {
            "off": "OFF（検証なし）",
            "oos_3split": "OOS 3分割（Train/Val/Test）",
            "wfa": "Walk-Forward Analysis",
        }[x],
        horizontal=True,
        key="opt_validation_mode",
        index=1,
    )

    # デフォルト値
    oos_train_pct = 60
    oos_val_pct = 20
    oos_test_pct = 20
    oos_top_n = 10
    oos_min_trades = 30
    wfa_n_folds = 5
    wfa_min_is_pct = 40
    wfa_use_val = True
    wfa_val_pct = 20
    wfa_top_n = 10
    wfa_min_trades = 30

    if validation_mode == "oos_3split":
        oos_col1, oos_col2, oos_col3, oos_col4 = st.columns(4)
        with oos_col1:
            oos_train_pct = st.slider(
                "Train %", 40, 80, 60, 5,
                key="opt_oos_train",
                help="グリッドサーチに使う期間の割合",
            )
        with oos_col2:
            oos_val_pct = st.slider(
                "Validation %", 10, 30, 20, 5,
                key="opt_oos_val",
                help="Train上位のパラメータを再評価する期間",
            )
        with oos_col3:
            oos_test_pct = 100 - oos_train_pct - oos_val_pct
            st.metric("Test %", f"{oos_test_pct}%")
            if oos_test_pct < 10:
                st.warning("Test期間が短すぎます")
        with oos_col4:
            oos_top_n = st.number_input(
                "Val Top-N",
                value=10,
                min_value=3,
                max_value=50,
                key="opt_oos_top_n",
                help="Train上位N件をValidationで再評価",
            )
        oos_min_trades = st.number_input(
            "Val最低トレード数",
            value=30,
            min_value=0,
            max_value=200,
            step=5,
            key="opt_oos_min_trades",
            help="Trainで最低この回数以上トレードしたエントリーのみValに通す（0=フィルタなし）",
        )

    elif validation_mode == "wfa":
        wfa_col1, wfa_col2, wfa_col3 = st.columns(3)
        with wfa_col1:
            wfa_n_folds = st.number_input(
                "フォールド数", value=5, min_value=3, max_value=10,
                key="opt_wfa_n_folds",
                help="データを何分割してWFAを実行するか",
            )
        with wfa_col2:
            wfa_min_is_pct = st.slider(
                "最小IS割合 (%)", 30, 60, 40, 5,
                key="opt_wfa_min_is",
                help="最初のフォールドのIn-Sample最小割合",
            )
        with wfa_col3:
            wfa_use_val = st.checkbox(
                "IS内Val分割",
                value=True,
                key="opt_wfa_use_val",
                help="IS期間内でさらにTrain/Valに分割してベスト選択",
            )
        if wfa_use_val:
            wfa_vcol1, wfa_vcol2, wfa_vcol3 = st.columns(3)
            with wfa_vcol1:
                wfa_val_pct = st.slider(
                    "IS内Val (%)", 10, 30, 20, 5,
                    key="opt_wfa_val_pct",
                )
            with wfa_vcol2:
                wfa_top_n = st.number_input(
                    "Val Top-N", value=10, min_value=3, max_value=50,
                    key="opt_wfa_top_n",
                )
            with wfa_vcol3:
                wfa_min_trades = st.number_input(
                    "Val最低トレード数", value=30, min_value=0, max_value=200,
                    step=5, key="opt_wfa_min_trades",
                )

    st.divider()

    # --- 実行 ---
    total_runs = total_combinations * len(target_regimes)

    # サマリーカード
    scol1, scol2, scol3, scol4 = st.columns(4)
    with scol1:
        st.metric("テンプレート", f"{len(selected_templates)}")
    with scol2:
        st.metric("レジーム", f"{len(target_regimes)}")
    with scol3:
        if search_method == "ga":
            # GAの場合は推定評価回数を表示
            ga_est_evals = ga_population * ga_generations * len(target_regimes)
            st.metric("推定評価数", f"~{ga_est_evals:,}")
        else:
            st.metric("組合せ数", f"{total_combinations:,}")
    with scol4:
        if search_method == "ga":
            st.metric("探索方法", "GA")
        else:
            st.metric("総実行数", f"{total_runs:,}")

    if st.button("🚀 最適化実行", type="primary", use_container_width=True):
        if not selected_templates:
            st.error("テンプレートを1つ以上選択してください")
            return
        if not target_regimes:
            st.error("レジームを1つ以上選択してください")
            return

        if search_method == "ga":
            # GA実行
            _run_ga_optimization(
                exec_tf=exec_tf,
                htf=htf,
                trend_method=trend_method,
                target_regimes=target_regimes,
                selected_templates=selected_templates,
                scoring_weights=ScoringWeights(w_pf, w_wr, w_dd, w_sh, w_ret),
                initial_capital=initial_capital,
                commission=commission,
                slippage=slippage,
                ma_fast=int(ma_fast),
                ma_slow=int(ma_slow),
                adx_period=int(adx_period),
                adx_trend_th=float(adx_trend_th),
                adx_range_th=float(adx_range_th),
                ga_population=int(ga_population),
                ga_generations=int(ga_generations),
                ga_elite_ratio=float(ga_elite_ratio),
                ga_mutation_rate=float(ga_mutation_rate),
            )
        else:
            # グリッドサーチ実行
            _run_optimization(
                exec_tf=exec_tf,
                htf=htf,
                trend_method=trend_method,
                target_regimes=target_regimes,
                selected_templates=selected_templates,
                custom_ranges=custom_ranges,
                scoring_weights=ScoringWeights(w_pf, w_wr, w_dd, w_sh, w_ret),
                initial_capital=initial_capital,
                commission=commission,
                slippage=slippage,
                ma_fast=int(ma_fast),
                ma_slow=int(ma_slow),
                adx_period=int(adx_period),
                adx_trend_th=float(adx_trend_th),
                adx_range_th=float(adx_range_th),
                n_workers=int(n_workers),
                validation_mode=validation_mode,
                oos_train_pct=oos_train_pct,
                oos_val_pct=oos_val_pct,
                oos_top_n=int(oos_top_n),
                oos_min_trades=int(oos_min_trades),
                wfa_n_folds=int(wfa_n_folds),
                wfa_min_is_pct=int(wfa_min_is_pct),
                wfa_use_val=wfa_use_val,
                wfa_val_pct=int(wfa_val_pct),
                wfa_top_n=int(wfa_top_n),
                wfa_min_trades=int(wfa_min_trades),
            )

    # --- バッチ実行 ---
    _render_batch_section(
        exec_tf=exec_tf,
        htf=htf,
        trend_method=trend_method,
        target_regimes=target_regimes,
        selected_templates=selected_templates,
        custom_ranges=custom_ranges,
        scoring_weights=ScoringWeights(w_pf, w_wr, w_dd, w_sh, w_ret),
        initial_capital=initial_capital,
        commission=commission,
        slippage=slippage,
        ma_fast=int(ma_fast),
        ma_slow=int(ma_slow),
        adx_period=int(adx_period),
        adx_trend_th=float(adx_trend_th),
        adx_range_th=float(adx_range_th),
        n_workers=int(n_workers),
        validation_mode=validation_mode,
        oos_train_pct=oos_train_pct,
        oos_val_pct=oos_val_pct,
        oos_top_n=int(oos_top_n),
        oos_min_trades=int(oos_min_trades),
        wfa_n_folds=int(wfa_n_folds),
        wfa_min_is_pct=int(wfa_min_is_pct),
        wfa_use_val=wfa_use_val,
        wfa_val_pct=int(wfa_val_pct),
        wfa_top_n=int(wfa_top_n),
        wfa_min_trades=int(wfa_min_trades),
    )


def _render_batch_section(
    exec_tf, htf, trend_method, target_regimes,
    selected_templates, custom_ranges, scoring_weights,
    initial_capital, commission, slippage,
    ma_fast, ma_slow, adx_period, adx_trend_th, adx_range_th,
    n_workers,
    validation_mode="off",
    oos_train_pct=60, oos_val_pct=20, oos_top_n=10, oos_min_trades=30,
    wfa_n_folds=5, wfa_min_is_pct=40, wfa_use_val=True,
    wfa_val_pct=20, wfa_top_n=10, wfa_min_trades=30,
):
    """バッチ実行セクション（複数銘柄×データソースを一括実行）"""
    datasets = st.session_state.get("datasets", {})
    trimmed_list = st.session_state.get("trimmed_datasets", [])

    if len(datasets) < 2 and not trimmed_list:
        return

    with st.expander("🔄 バッチ実行（複数銘柄を一括最適化）", expanded=False):
        # 全候補を列挙
        candidates = []
        for symbol, tf_dict in datasets.items():
            if exec_tf in tf_dict:
                candidates.append({
                    "id": f"orig_{symbol}",
                    "symbol": symbol,
                    "source": "original",
                    "label": f"{symbol} (📦 オリジナル)",
                    "tf_dict": tf_dict,
                    "period_start": "",
                    "period_end": "",
                })

        for entry in trimmed_list:
            if exec_tf in entry["data"]:
                candidates.append({
                    "id": f"trim_{entry['id']}_{entry['symbol']}",
                    "symbol": entry["symbol"],
                    "source": "trimmed",
                    "label": f"{entry['symbol']} (✂️ {entry['label']})",
                    "tf_dict": entry["data"],
                    "period_start": str(entry["start_dt"])[:10],
                    "period_end": str(entry["end_dt"])[:10],
                })

        if not candidates:
            st.caption(f"実行TF `{exec_tf}` を含むデータセットがありません。")
            return

        # multiselect で対象選択
        candidate_ids = [c["id"] for c in candidates]
        candidate_labels = {c["id"]: c["label"] for c in candidates}

        selected_ids = st.multiselect(
            "対象を選択",
            options=candidate_ids,
            default=[],
            format_func=lambda x: candidate_labels[x],
            key="batch_targets",
        )

        n_selected = len(selected_ids)

        if n_selected < 1:
            st.caption("バッチ実行する対象を1つ以上選択してください。")
            return

        runs_per_target = sum(
            BUILTIN_TEMPLATES[t].combination_count(custom_ranges.get(t, {}))
            for t in selected_templates
        ) * len(target_regimes)
        st.caption(
            f"**{n_selected}件** 選択中 / "
            f"1件あたり {runs_per_target:,} runs / "
            f"合計 {runs_per_target * n_selected:,} runs"
        )

        if st.button(
            f"🔄 {n_selected}件 バッチ実行",
            type="primary",
            use_container_width=True,
            disabled=n_selected < 1 or not selected_templates or not target_regimes,
        ):
            targets = [c for c in candidates if c["id"] in selected_ids]
            _run_batch_optimization(
                targets=targets,
                exec_tf=exec_tf,
                htf=htf,
                trend_method=trend_method,
                target_regimes=target_regimes,
                selected_templates=selected_templates,
                custom_ranges=custom_ranges,
                scoring_weights=scoring_weights,
                initial_capital=initial_capital,
                commission=commission,
                slippage=slippage,
                ma_fast=ma_fast,
                ma_slow=ma_slow,
                adx_period=adx_period,
                adx_trend_th=adx_trend_th,
                adx_range_th=adx_range_th,
                n_workers=n_workers,
                validation_mode=validation_mode,
                oos_train_pct=oos_train_pct,
                oos_val_pct=oos_val_pct,
                oos_top_n=oos_top_n,
                oos_min_trades=oos_min_trades,
                wfa_n_folds=wfa_n_folds,
                wfa_min_is_pct=wfa_min_is_pct,
                wfa_use_val=wfa_use_val,
                wfa_val_pct=wfa_val_pct,
                wfa_top_n=wfa_top_n,
                wfa_min_trades=wfa_min_trades,
            )


def _run_batch_optimization(
    targets, exec_tf, htf, trend_method, target_regimes,
    selected_templates, custom_ranges, scoring_weights,
    initial_capital, commission, slippage,
    ma_fast, ma_slow, adx_period, adx_trend_th, adx_range_th,
    n_workers,
    validation_mode="off",
    oos_train_pct=60, oos_val_pct=20, oos_top_n=10, oos_min_trades=30,
    wfa_n_folds=5, wfa_min_is_pct=40, wfa_use_val=True,
    wfa_val_pct=20, wfa_top_n=10, wfa_min_trades=30,
):
    """バッチ最適化を順次実行（検証モード対応）"""
    all_results = []
    batch_validated = {}  # symbol -> ValidatedResultSet
    batch_wfa = {}  # symbol -> WFAResultSet
    n_total = len(targets)

    mode_label = {
        "off": "検証なし",
        "oos_3split": "OOS 3分割",
        "wfa": "Walk-Forward",
    }.get(validation_mode, validation_mode)

    overall_progress = st.progress(0, text=f"バッチ実行準備中... ({mode_label})")
    status_text = st.empty()

    batch_start = time.time()

    for i, target in enumerate(targets):
        label = target["label"]
        status_text.markdown(f"**[{i+1}/{n_total}]** {label} ({mode_label})")
        overall_progress.progress(i / n_total, text=f"[{i+1}/{n_total}] {label}")

        item_progress = st.progress(0, text=f"{label}: 開始...")
        item_start = time.time()

        def on_item_progress(current, total, desc):
            elapsed = time.time() - item_start
            speed = current / elapsed if elapsed > 0 else 0
            item_progress.progress(
                current / total if total > 0 else 0,
                text=f"{label}: {current}/{total} ({speed:.0f} runs/s)",
            )

        common_kwargs = dict(
            tf_dict=target["tf_dict"],
            exec_tf=exec_tf,
            htf=htf,
            trend_method=trend_method,
            target_regimes=target_regimes,
            selected_templates=selected_templates,
            custom_ranges=custom_ranges,
            scoring_weights=scoring_weights,
            initial_capital=initial_capital,
            commission=commission,
            slippage=slippage,
            ma_fast=ma_fast,
            ma_slow=ma_slow,
            adx_period=adx_period,
            adx_trend_th=adx_trend_th,
            adx_range_th=adx_range_th,
            n_workers=n_workers,
            progress_callback=on_item_progress,
            data_source=target["source"],
            data_period_start=target.get("period_start", ""),
            data_period_end=target.get("period_end", ""),
        )

        if validation_mode == "oos_3split":
            validated = _execute_validated_optimization(
                **common_kwargs,
                oos_train_pct=oos_train_pct,
                oos_val_pct=oos_val_pct,
                oos_top_n=oos_top_n,
                oos_min_trades=oos_min_trades,
            )
            result_set = validated.train_results
            # シンボル情報の付与（_execute_validated_optimizationでは付与済みだがtrain_resultsに転写）
            tf_dict = target["tf_dict"]
            if exec_tf in tf_dict:
                result_set.symbol = tf_dict[exec_tf].symbol
            result_set.execution_tf = exec_tf
            result_set.data_source = target["source"]
            result_set.data_period_start = target.get("period_start", "")
            result_set.data_period_end = target.get("period_end", "")
            batch_validated[result_set.symbol] = validated

        elif validation_mode == "wfa":
            wfa_result = _execute_wfa_optimization(
                **common_kwargs,
                wfa_n_folds=wfa_n_folds,
                wfa_min_is_pct=wfa_min_is_pct,
                wfa_use_val=wfa_use_val,
                wfa_val_pct=wfa_val_pct,
                wfa_top_n=wfa_top_n,
                wfa_min_trades=wfa_min_trades,
            )
            result_set = wfa_result.final_train_results
            if result_set is None:
                from optimizer.results import OptimizationResultSet
                result_set = OptimizationResultSet(entries=[])
            tf_dict = target["tf_dict"]
            if exec_tf in tf_dict:
                result_set.symbol = tf_dict[exec_tf].symbol
            result_set.execution_tf = exec_tf
            result_set.data_source = target["source"]
            result_set.data_period_start = target.get("period_start", "")
            result_set.data_period_end = target.get("period_end", "")
            batch_wfa[result_set.symbol] = wfa_result

        else:
            result_set = _execute_single_optimization(**common_kwargs)

        _save_results(result_set)
        all_results.append(result_set)

        item_elapsed = time.time() - item_start
        item_progress.progress(1.0, text=f"{label}: ✅ 完了 [{item_elapsed:.1f}s]")

    batch_elapsed = time.time() - batch_start
    overall_progress.progress(1.0, text=f"✅ 全{n_total}件完了 [{batch_elapsed:.1f}s]")
    status_text.empty()

    st.success(f"バッチ完了: {n_total}件 / {batch_elapsed:.1f}s ({mode_label})")

    # 比較ビューへ遷移
    st.session_state.comparison_results = all_results
    st.session_state.batch_validated = batch_validated if batch_validated else None
    st.session_state.batch_wfa = batch_wfa if batch_wfa else None
    st.session_state.optimizer_view = "compare"
    st.rerun()


def _save_results(result_set):
    """最適化結果をCSV・JSONでファイルに保存"""
    import json
    from pathlib import Path
    from datetime import datetime

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # ファイル名: BTCUSDT_exec15m_htf4h_20260131_143000
    # 切り出し: BTCUSDT_exec15m_htf4h_trim0115-0125_20260131_143000
    sym = result_set.symbol or "UNKNOWN"
    etf = result_set.execution_tf or "?"
    htf = result_set.htf or "none"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if result_set.data_source == "trimmed" and result_set.data_period_start:
        # 期間を MMDD-MMDD 形式で付加
        ps = result_set.data_period_start.replace("-", "")[4:]  # "20250115" -> "0115"
        pe = result_set.data_period_end.replace("-", "")[4:]
        base_name = f"{sym}_exec{etf}_htf{htf}_trim{ps}-{pe}_{ts}"
    else:
        base_name = f"{sym}_exec{etf}_htf{htf}_{ts}"

    # CSV保存
    df = result_set.to_dataframe()
    csv_path = results_dir / f"{base_name}.csv"
    df.to_csv(csv_path, index=False)

    # JSON保存（configも含む完全版）
    json_rows = []
    for e in result_set.ranked():
        json_rows.append({
            "template": e.template_name,
            "params": e.params,
            "regime": e.trend_regime,
            "score": round(e.composite_score, 4),
            "metrics": {
                "trades": e.metrics.total_trades,
                "win_rate": round(e.metrics.win_rate, 1),
                "profit_factor": round(e.metrics.profit_factor, 2),
                "total_pnl": round(e.metrics.total_profit_pct, 2),
                "max_dd": round(e.metrics.max_drawdown_pct, 2),
                "sharpe": round(e.metrics.sharpe_ratio, 2),
            },
            "config": e.config,
        })

    json_path = results_dir / f"{base_name}.json"
    json_meta = {
        "symbol": sym,
        "execution_tf": etf,
        "htf": htf,
        "data_source": result_set.data_source,
        "total_combinations": result_set.total_combinations,
        "timestamp": ts,
    }
    if result_set.data_period_start:
        json_meta["data_period"] = {
            "start": result_set.data_period_start,
            "end": result_set.data_period_end,
        }
    json_meta["results"] = json_rows
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_meta, f, ensure_ascii=False, indent=2)

    return str(csv_path)


def _execute_single_optimization(
    tf_dict, exec_tf, htf, trend_method, target_regimes,
    selected_templates, custom_ranges, scoring_weights,
    initial_capital, commission, slippage,
    ma_fast, ma_slow, adx_period, adx_trend_th, adx_range_th,
    n_workers=1, progress_callback=None,
    data_source="original", data_period_start="", data_period_end="",
):
    """1銘柄分の最適化コア処理（UI非依存）"""
    exec_ohlcv = tf_dict[exec_tf]
    exec_df = exec_ohlcv.df.copy()

    # トレンドラベル付与
    if htf and htf in tf_dict:
        htf_ohlcv = tf_dict[htf]
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
        elif trend_method == "dual_tf_ema":
            _super_htf = st.session_state.get("opt_super_htf")
            if _super_htf and _super_htf in tf_dict:
                super_htf_df = tf_dict[_super_htf].df.copy()
                htf_df = detector.detect_dual_tf_ema(
                    htf_df, super_htf_df, fast_period=ma_fast, slow_period=ma_slow
                )
            else:
                htf_df = detector.detect_ma_cross(
                    htf_df, fast_period=ma_fast, slow_period=ma_slow
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

    result_set = optimizer.run(
        df=exec_df,
        configs=all_configs,
        target_regimes=target_regimes,
        progress_callback=progress_callback,
        n_workers=n_workers,
    )

    result_set.symbol = exec_ohlcv.symbol
    result_set.execution_tf = exec_tf
    result_set.htf = htf or ""
    result_set.data_source = data_source
    result_set.data_period_start = data_period_start
    result_set.data_period_end = data_period_end

    return result_set


def _execute_validated_optimization(
    tf_dict, exec_tf, htf, trend_method, target_regimes,
    selected_templates, custom_ranges, scoring_weights,
    initial_capital, commission, slippage,
    ma_fast, ma_slow, adx_period, adx_trend_th, adx_range_th,
    n_workers=1, progress_callback=None,
    data_source="original", data_period_start="", data_period_end="",
    oos_train_pct=60, oos_val_pct=20, oos_top_n=10, oos_min_trades=30,
):
    """OOS検証付き最適化（Train/Val/Test 3分割）"""
    exec_ohlcv = tf_dict[exec_tf]
    exec_df = exec_ohlcv.df.copy()

    # トレンドラベル付与（全データに対して）
    if htf and htf in tf_dict:
        htf_ohlcv = tf_dict[htf]
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
        elif trend_method == "dual_tf_ema":
            _super_htf = st.session_state.get("opt_super_htf")
            if _super_htf and _super_htf in tf_dict:
                super_htf_df = tf_dict[_super_htf].df.copy()
                htf_df = detector.detect_dual_tf_ema(
                    htf_df, super_htf_df, fast_period=ma_fast, slow_period=ma_slow
                )
            else:
                htf_df = detector.detect_ma_cross(
                    htf_df, fast_period=ma_fast, slow_period=ma_slow
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

    # オプティマイザ
    optimizer = GridSearchOptimizer(
        initial_capital=initial_capital,
        commission_pct=commission,
        slippage_pct=slippage,
        scoring_weights=scoring_weights,
    )

    # 分割設定
    split_config = DataSplitConfig(
        train_pct=oos_train_pct / 100.0,
        val_pct=oos_val_pct / 100.0,
        top_n_for_val=oos_top_n,
        min_trades_for_val=oos_min_trades,
    )

    # OOS検証実行
    validated = run_validated_optimization(
        df=exec_df,
        all_configs=all_configs,
        target_regimes=target_regimes,
        split_config=split_config,
        optimizer=optimizer,
        progress_callback=progress_callback,
        n_workers=n_workers,
    )

    # メタ情報をtrain_resultsに付与
    validated.train_results.symbol = exec_ohlcv.symbol
    validated.train_results.execution_tf = exec_tf
    validated.train_results.htf = htf or ""
    validated.train_results.data_source = data_source
    validated.train_results.data_period_start = data_period_start
    validated.train_results.data_period_end = data_period_end

    # Train結果に警告を付与
    for entry in validated.train_results.entries:
        entry.warnings = detect_overfitting_warnings(entry.metrics)

    return validated


def _execute_wfa_optimization(
    tf_dict, exec_tf, htf, trend_method, target_regimes,
    selected_templates, custom_ranges, scoring_weights,
    initial_capital, commission, slippage,
    ma_fast, ma_slow, adx_period, adx_trend_th, adx_range_th,
    n_workers=1, progress_callback=None,
    data_source="original", data_period_start="", data_period_end="",
    wfa_n_folds=5, wfa_min_is_pct=40, wfa_use_val=True,
    wfa_val_pct=20, wfa_top_n=10, wfa_min_trades=30,
):
    """Walk-Forward Analysis 付き最適化"""
    from optimizer.walk_forward import WFAConfig, run_walk_forward_analysis

    exec_ohlcv = tf_dict[exec_tf]
    exec_df = exec_ohlcv.df.copy()

    # トレンドラベル付与（全データに対して）
    if htf and htf in tf_dict:
        htf_ohlcv = tf_dict[htf]
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
        elif trend_method == "dual_tf_ema":
            _super_htf = st.session_state.get("opt_super_htf")
            if _super_htf and _super_htf in tf_dict:
                super_htf_df = tf_dict[_super_htf].df.copy()
                htf_df = detector.detect_dual_tf_ema(
                    htf_df, super_htf_df, fast_period=ma_fast, slow_period=ma_slow
                )
            else:
                htf_df = detector.detect_ma_cross(
                    htf_df, fast_period=ma_fast, slow_period=ma_slow
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

    # オプティマイザ
    optimizer = GridSearchOptimizer(
        initial_capital=initial_capital,
        commission_pct=commission,
        slippage_pct=slippage,
        scoring_weights=scoring_weights,
    )

    # WFA設定
    wfa_config = WFAConfig(
        n_folds=wfa_n_folds,
        min_is_pct=wfa_min_is_pct / 100.0,
        use_validation=wfa_use_val,
        val_pct_within_is=wfa_val_pct / 100.0,
        top_n_for_val=wfa_top_n,
        min_trades_for_val=wfa_min_trades,
    )

    # WFA実行
    wfa_result = run_walk_forward_analysis(
        df=exec_df,
        all_configs=all_configs,
        target_regimes=target_regimes,
        wfa_config=wfa_config,
        optimizer=optimizer,
        progress_callback=progress_callback,
        n_workers=n_workers,
    )

    # メタ情報
    wfa_result.symbol = exec_ohlcv.symbol
    wfa_result.execution_tf = exec_tf

    # final_train_results にもメタ情報を付与
    if wfa_result.final_train_results:
        wfa_result.final_train_results.symbol = exec_ohlcv.symbol
        wfa_result.final_train_results.execution_tf = exec_tf
        wfa_result.final_train_results.htf = htf or ""
        wfa_result.final_train_results.data_source = data_source
        wfa_result.final_train_results.data_period_start = data_period_start
        wfa_result.final_train_results.data_period_end = data_period_end

        for entry in wfa_result.final_train_results.entries:
            entry.warnings = detect_overfitting_warnings(entry.metrics)

    return wfa_result


def _run_ga_optimization(
    exec_tf, htf, trend_method, target_regimes,
    selected_templates, scoring_weights,
    initial_capital, commission, slippage,
    ma_fast, ma_slow, adx_period, adx_trend_th, adx_range_th,
    ga_population=20,
    ga_generations=10,
    ga_elite_ratio=0.25,
    ga_mutation_rate=0.15,
):
    """GA最適化実行（UIラッパー）"""
    progress_bar = st.progress(0, text="GA最適化を開始...")
    start_time = time.time()

    # データ準備
    tf_dict = st.session_state.ohlcv_dict
    exec_ohlcv = tf_dict[exec_tf]
    df = exec_ohlcv.df.copy()

    # 選択中のシンボル取得
    selected_symbol = st.session_state.get("opt_symbol", "UNKNOWN")

    # トレンドラベル追加
    progress_bar.progress(0.05, text="トレンドラベル追加中...")
    if htf and htf in tf_dict:
        htf_ohlcv = tf_dict[htf]
        htf_df = htf_ohlcv.df.copy()
        detector = TrendDetector()

        if trend_method == "ma_cross":
            htf_df = detector.detect_ma_cross(
                htf_df, fast_period=ma_fast, slow_period=ma_slow
            )
        elif trend_method == "adx":
            htf_df = detector.detect_adx(
                htf_df, adx_period=adx_period,
                trend_threshold=adx_trend_th, range_threshold=adx_range_th
            )
        elif trend_method == "dual_tf_ema":
            _super_htf = st.session_state.get("opt_super_htf")
            if _super_htf and _super_htf in tf_dict:
                super_htf_df = tf_dict[_super_htf].df.copy()
                htf_df = detector.detect_dual_tf_ema(
                    htf_df, super_htf_df, fast_period=ma_fast, slow_period=ma_slow
                )
            else:
                htf_df = detector.detect_ma_cross(
                    htf_df, fast_period=ma_fast, slow_period=ma_slow
                )
        else:
            htf_df = detector.detect_combined(
                htf_df, ma_fast=ma_fast, ma_slow=ma_slow,
                adx_period=adx_period, adx_trend_threshold=adx_trend_th
            )

        # 上位TFラベルを実行TFにマージ
        df = detector.merge_htf_labels(df, htf_df, "trend_regime")
    else:
        # 単一TF
        detector = TrendDetector()
        if trend_method == "ma_cross":
            df = detector.detect_ma_cross(df, fast_period=ma_fast, slow_period=ma_slow)
        elif trend_method == "adx":
            df = detector.detect_adx(
                df, adx_period=adx_period,
                trend_threshold=adx_trend_th, range_threshold=adx_range_th
            )
        else:
            df = detector.detect_combined(
                df, ma_fast=ma_fast, ma_slow=ma_slow,
                adx_period=adx_period, adx_trend_threshold=adx_trend_th
            )

    # trend_regime → trend_label にリネーム（GAが期待するカラム名）
    if "trend_regime" in df.columns:
        df["trend_label"] = df["trend_regime"]

    # GA設定
    ga_config = GAConfig(
        population_size=ga_population,
        max_generations=ga_generations,
        elite_ratio=ga_elite_ratio,
        mutation_rate=ga_mutation_rate,
        convergence_threshold=0.01,
        convergence_generations=4,
    )

    # GAオプティマイザー作成
    optimizer = GeneticOptimizer(
        templates=selected_templates,
        ga_config=ga_config,
        scoring_weights=scoring_weights,
        initial_capital=initial_capital,
        commission_pct=commission,
        slippage_pct=slippage,
    )

    # 各レジームでGA実行
    all_ga_results = []
    total_regimes = len(target_regimes)

    for i, regime in enumerate(target_regimes):
        def on_progress(gen, max_gen, desc):
            base_pct = (i / total_regimes) * 0.9 + 0.1
            gen_pct = (gen / max_gen) * (0.9 / total_regimes)
            elapsed = time.time() - start_time
            progress_bar.progress(
                min(base_pct + gen_pct, 1.0),
                text=f"🧬 GA [{regime}] 世代 {gen}/{max_gen} ({elapsed:.1f}s)",
            )

        ga_result = optimizer.run(
            df=df,
            target_regime=regime,
            symbol=selected_symbol,
            progress_callback=on_progress,
        )
        all_ga_results.append(ga_result)

        # 結果をJSON保存
        ga_result.save("results")

    progress_bar.progress(1.0, text="GA最適化完了!")

    # 結果サマリーを表示
    elapsed = time.time() - start_time
    st.success(f"✅ GA最適化完了 ({elapsed:.1f}秒)")

    # 結果テーブル
    st.subheader("🏆 GA最適化結果")
    result_data = []
    for ga_result in all_ga_results:
        w = ga_result.final_winner
        result_data.append({
            "レジーム": ga_result.regime,
            "テンプレート": w["template"],
            "パラメータ": str(w["params"]),
            "PnL (%)": f"{w['pnl']:+.2f}",
            "勝率 (%)": f"{w['win_rate']:.1f}",
            "トレード数": w["trades"],
            "Sharpe": f"{w['sharpe']:.2f}",
            "スコア": f"{w['score']:.4f}",
            "評価回数": ga_result.total_evaluations,
        })

    st.dataframe(pd.DataFrame(result_data), use_container_width=True)

    # 世代推移チャート
    with st.expander("📈 世代ごとのスコア推移", expanded=False):
        for ga_result in all_ga_results:
            st.caption(f"**{ga_result.regime}**")
            gen_data = []
            for gen in ga_result.generations:
                gen_data.append({
                    "世代": gen.generation,
                    "ベストスコア": gen.best_score,
                    "平均スコア": gen.avg_score,
                })
            if gen_data:
                chart_df = pd.DataFrame(gen_data)
                st.line_chart(chart_df.set_index("世代"))

    # GA結果をセッションに保存（後で比較等に使用可能）
    st.session_state.ga_results = all_ga_results


def _run_optimization(
    exec_tf, htf, trend_method, target_regimes,
    selected_templates, custom_ranges, scoring_weights,
    initial_capital, commission, slippage,
    ma_fast, ma_slow, adx_period, adx_trend_th, adx_range_th,
    n_workers=1,
    validation_mode="off",
    oos_train_pct=60,
    oos_val_pct=20,
    oos_top_n=10,
    oos_min_trades=30,
    wfa_n_folds=5,
    wfa_min_is_pct=40,
    wfa_use_val=True,
    wfa_val_pct=20,
    wfa_top_n=10,
    wfa_min_trades=30,
):
    """単一銘柄の最適化実行（UIラッパー）"""
    progress_bar = st.progress(0, text="Starting optimization...")
    start_time = time.time()

    def on_progress(current, total, desc):
        elapsed = time.time() - start_time
        speed = current / elapsed if elapsed > 0 else 0
        pct = min(current / total, 1.0) if total > 0 else 0
        progress_bar.progress(
            pct,
            text=f"⚡ {current}/{total} ({speed:.0f} runs/s) [{elapsed:.1f}s] {desc}",
        )

    ds_info = st.session_state.get("opt_data_source_info", {})

    if validation_mode == "oos_3split":
        # OOS 検証付き最適化
        validated_result = _execute_validated_optimization(
            tf_dict=st.session_state.ohlcv_dict,
            exec_tf=exec_tf,
            htf=htf,
            trend_method=trend_method,
            target_regimes=target_regimes,
            selected_templates=selected_templates,
            custom_ranges=custom_ranges,
            scoring_weights=scoring_weights,
            initial_capital=initial_capital,
            commission=commission,
            slippage=slippage,
            ma_fast=ma_fast,
            ma_slow=ma_slow,
            adx_period=adx_period,
            adx_trend_th=adx_trend_th,
            adx_range_th=adx_range_th,
            n_workers=n_workers,
            progress_callback=on_progress,
            data_source=ds_info.get("source", "original"),
            data_period_start=ds_info.get("period_start", ""),
            data_period_end=ds_info.get("period_end", ""),
            oos_train_pct=oos_train_pct,
            oos_val_pct=oos_val_pct,
            oos_top_n=oos_top_n,
            oos_min_trades=oos_min_trades,
        )

        result_set = validated_result.train_results
        st.session_state.optimization_result = result_set
        st.session_state.validated_result = validated_result
        st.session_state.wfa_result = None

    elif validation_mode == "wfa":
        # Walk-Forward Analysis
        wfa_result = _execute_wfa_optimization(
            tf_dict=st.session_state.ohlcv_dict,
            exec_tf=exec_tf,
            htf=htf,
            trend_method=trend_method,
            target_regimes=target_regimes,
            selected_templates=selected_templates,
            custom_ranges=custom_ranges,
            scoring_weights=scoring_weights,
            initial_capital=initial_capital,
            commission=commission,
            slippage=slippage,
            ma_fast=ma_fast,
            ma_slow=ma_slow,
            adx_period=adx_period,
            adx_trend_th=adx_trend_th,
            adx_range_th=adx_range_th,
            n_workers=n_workers,
            progress_callback=on_progress,
            data_source=ds_info.get("source", "original"),
            data_period_start=ds_info.get("period_start", ""),
            data_period_end=ds_info.get("period_end", ""),
            wfa_n_folds=wfa_n_folds,
            wfa_min_is_pct=wfa_min_is_pct,
            wfa_use_val=wfa_use_val,
            wfa_val_pct=wfa_val_pct,
            wfa_top_n=wfa_top_n,
            wfa_min_trades=wfa_min_trades,
        )

        # final_train_results をメインの結果として使う
        result_set = wfa_result.final_train_results
        if result_set is None:
            from optimizer.results import OptimizationResultSet
            result_set = OptimizationResultSet(entries=[])

        st.session_state.optimization_result = result_set
        st.session_state.wfa_result = wfa_result
        st.session_state.validated_result = None

    else:
        # 従来の最適化（検証なし）
        result_set = _execute_single_optimization(
            tf_dict=st.session_state.ohlcv_dict,
            exec_tf=exec_tf,
            htf=htf,
            trend_method=trend_method,
            target_regimes=target_regimes,
            selected_templates=selected_templates,
            custom_ranges=custom_ranges,
            scoring_weights=scoring_weights,
            initial_capital=initial_capital,
            commission=commission,
            slippage=slippage,
            ma_fast=ma_fast,
            ma_slow=ma_slow,
            adx_period=adx_period,
            adx_trend_th=adx_trend_th,
            adx_range_th=adx_range_th,
            n_workers=n_workers,
            progress_callback=on_progress,
            data_source=ds_info.get("source", "original"),
            data_period_start=ds_info.get("period_start", ""),
            data_period_end=ds_info.get("period_end", ""),
        )

        # 警告を付与（OOSなしでもPF/Sharpe/Trade数の警告は出す）
        for entry in result_set.entries:
            entry.warnings = detect_overfitting_warnings(entry.metrics)

        st.session_state.optimization_result = result_set
        st.session_state.validated_result = None
        st.session_state.wfa_result = None

    elapsed = time.time() - start_time

    progress_bar.progress(1.0, text=f"✅ Done! [{elapsed:.1f}s]")

    saved_path = _save_results(result_set)

    st.success(
        f"**{result_set.total_combinations}** results in **{elapsed:.1f}s** "
        f"(Workers: {n_workers})"
    )
    if saved_path:
        st.caption(f"💾 保存先: `{saved_path}`")

    st.session_state.optimizer_view = "results"
    st.rerun()


def _render_regime_switching_section(result_set, viable_strategies):
    """レジーム切替バックテスト（通しテスト）セクション"""
    section_header("🔄", "レジーム切替テスト", "採用戦略を全期間で統合検証")

    st.caption("チェックしたレジームの戦略でレジーム切替バックテストを実行")

    # レジーム選択チェックボックス
    regime_cols = st.columns(len(viable_strategies))
    selected_regimes = {}
    for i, (regime, entry) in enumerate(viable_strategies.items()):
        with regime_cols[i]:
            icon = REGIME_ICONS.get(regime, "")
            label = REGIME_OPTIONS.get(regime, regime)
            side_label = entry.config.get("side", "?")
            pnl = entry.metrics.total_profit_pct
            checked = st.checkbox(
                f"{icon} {label}: {entry.template_name} ({side_label}) / {pnl:+.1f}%",
                value=True,
                key=f"rs_regime_{regime}",
            )
            if checked:
                selected_regimes[regime] = entry

    # データソース選択（デフォルト: 全期間データ）
    data_option = st.radio(
        "テストデータ",
        ["original", "same"],
        format_func=lambda x: "全期間データ（original）" if x == "original" else "最適化と同じデータ",
        horizontal=True,
        key="rs_data_option",
    )

    if st.button("🔄 通しテスト実行", type="primary", use_container_width=True):
        if not selected_regimes:
            st.warning("少なくとも1つのレジームを選択してください")
        else:
            _run_regime_switching_test(
                result_set, selected_regimes, use_original=(data_option == "original"),
            )

    # 結果表示
    rs_result = st.session_state.get("regime_switching_result")
    if rs_result is not None:
        _render_regime_switching_results(rs_result, selected_regimes)


def _run_regime_switching_test(result_set, viable_strategies, use_original=False):
    """レジーム切替テストを実行"""
    from optimizer.regime_switching import run_regime_switching_backtest

    symbol = result_set.symbol
    exec_tf = result_set.execution_tf
    htf = result_set.htf

    # --- OHLCVデータ取得 ---
    ohlcv_dict = st.session_state.get("ohlcv_dict")

    if use_original:
        datasets = st.session_state.get("datasets", {})
        if symbol in datasets and exec_tf in datasets[symbol]:
            exec_df = datasets[symbol][exec_tf].df.copy()
        else:
            exec_df = _load_ohlcv_from_disk(symbol, exec_tf)
            if exec_df is None:
                st.error(f"データが見つかりません: {symbol} {exec_tf}")
                return
            exec_df = exec_df.copy()

        # HTFデータも取得
        htf_df = None
        if htf:
            if symbol in datasets and htf in datasets[symbol]:
                htf_df = datasets[symbol][htf].df.copy()
            else:
                htf_raw = _load_ohlcv_from_disk(symbol, htf)
                if htf_raw is not None:
                    htf_df = htf_raw.copy()
    else:
        if ohlcv_dict is None or exec_tf not in ohlcv_dict:
            st.error("最適化データが見つかりません。設定タブから最適化を実行してください。")
            return
        exec_df = ohlcv_dict[exec_tf].df.copy()
        htf_df = None
        if htf and htf in ohlcv_dict:
            htf_df = ohlcv_dict[htf].df.copy()

    # --- トレンドラベル付与 ---
    if htf_df is not None:
        detector = TrendDetector()
        trend_method = st.session_state.get("opt_trend_method", "ma_cross")
        ma_fast = st.session_state.get("opt_ma_fast", 20)
        ma_slow = st.session_state.get("opt_ma_slow", 50)
        adx_period = st.session_state.get("opt_adx_period", 14)
        adx_trend_th = st.session_state.get("opt_adx_trend_th", 25.0)
        adx_range_th = st.session_state.get("opt_adx_range_th", 20.0)

        if trend_method == "ma_cross":
            htf_df = detector.detect_ma_cross(htf_df, fast_period=ma_fast, slow_period=ma_slow)
        elif trend_method == "adx":
            htf_df = detector.detect_adx(
                htf_df, adx_period=adx_period,
                trend_threshold=adx_trend_th, range_threshold=adx_range_th,
            )
        elif trend_method == "dual_tf_ema":
            _super_htf = st.session_state.get("opt_super_htf")
            ohlcv_dict = st.session_state.get("ohlcv_dict", {})
            if _super_htf and _super_htf in ohlcv_dict:
                super_htf_df = ohlcv_dict[_super_htf].df.copy()
                htf_df = detector.detect_dual_tf_ema(
                    htf_df, super_htf_df, fast_period=ma_fast, slow_period=ma_slow
                )
            else:
                htf_df = detector.detect_ma_cross(
                    htf_df, fast_period=ma_fast, slow_period=ma_slow
                )
        else:
            htf_df = detector.detect_combined(
                htf_df, ma_fast=ma_fast, ma_slow=ma_slow,
                adx_period=adx_period,
                adx_trend_threshold=adx_trend_th,
                adx_range_threshold=adx_range_th,
            )
        exec_df = TrendDetector.label_execution_tf(exec_df, htf_df)
    else:
        exec_df["trend_regime"] = TrendRegime.RANGE.value

    # --- 実行 ---
    commission = st.session_state.get("opt_commission", 0.04)
    slippage = st.session_state.get("opt_slippage", 0.0)
    capital = st.session_state.get("opt_initial_capital", 10000.0)

    with st.spinner("レジーム切替テスト実行中..."):
        rs_result = run_regime_switching_backtest(
            df=exec_df,
            regime_strategies=viable_strategies,
            commission_pct=commission,
            slippage_pct=slippage,
            initial_capital=capital,
        )

    st.session_state.regime_switching_result = rs_result
    st.success("通しテスト完了")


def _render_regime_switching_results(rs_result, viable_strategies):
    """レジーム切替テスト結果を表示"""
    from ui.components.optimizer_charts import create_regime_switching_equity_chart

    m = rs_result.overall_metrics

    st.markdown("#### 統合バックテスト結果")

    # 全体メトリクス
    mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
    mc1.metric("取引数", m.total_trades)
    mc2.metric("勝率", f"{m.win_rate:.1f}%")
    mc3.metric("損益比率", f"{m.profit_factor:.2f}")
    mc4.metric("合計損益", f"{m.total_profit_pct:+.2f}%")
    mc5.metric("最大DD", f"{m.max_drawdown_pct:.2f}%")
    mc6.metric("シャープ", f"{m.sharpe_ratio:.2f}")

    # 個別最適化合算 vs 統合テスト 比較
    sum_pnl = sum(e.metrics.total_profit_pct for e in viable_strategies.values())
    diff = m.total_profit_pct - sum_pnl
    st.markdown(
        f"個別最適化合算PnL: **{sum_pnl:+.2f}%** → "
        f"統合テスト実績: **{m.total_profit_pct:+.2f}%** "
        f"(差: {diff:+.2f}%)"
    )

    # レジーム別分解テーブル
    st.markdown("#### レジーム別内訳")
    regime_rows = []
    for regime in ["uptrend", "downtrend", "range"]:
        rm = rs_result.regime_metrics.get(regime)
        tc = rs_result.regime_trade_counts.get(regime, 0)
        if rm and tc > 0:
            regime_rows.append({
                "レジーム": f"{REGIME_ICONS.get(regime, '')} {REGIME_OPTIONS.get(regime, regime)}",
                "取引数": tc,
                "勝率": f"{rm.win_rate:.1f}%",
                "合計損益": f"{rm.total_profit_pct:+.2f}%",
                "損益比率": f"{rm.profit_factor:.2f}",
                "最大DD": f"{rm.max_drawdown_pct:.2f}%",
            })
        else:
            regime_rows.append({
                "レジーム": f"{REGIME_ICONS.get(regime, '')} {REGIME_OPTIONS.get(regime, regime)}",
                "取引数": 0,
                "勝率": "-",
                "合計損益": "-",
                "損益比率": "-",
                "最大DD": "-",
            })

    st.dataframe(
        pd.DataFrame(regime_rows),
        use_container_width=True,
        hide_index=True,
    )

    # エクイティカーブ
    if len(rs_result.equity_curve) > 1:
        fig = create_regime_switching_equity_chart(rs_result.equity_curve)
        st.plotly_chart(fig, use_container_width=True, key="rs_equity_chart")


def _render_regime_best_summary(result_set, oos_passed=None):
    """レジーム別ベスト戦略サマリーを描画。採用可能な戦略のdictを返す。

    Args:
        result_set: 最適化結果セット
        oos_passed: OOS検証でPASSしたレジーム→テストエントリーのdict。
                    指定時はPASSレジームのみ表示。
    """
    section_header("🏆", "Best per Regime", "レジーム別トップ戦略")

    if oos_passed is not None:
        # OOS有効: PASSしたレジームのみ表示
        regimes_in_results = sorted(oos_passed.keys())
    else:
        regimes_in_results = sorted(set(e.trend_regime for e in result_set.entries))

    if not regimes_in_results:
        st.info("OOS検証をPASSしたレジームがありません" if oos_passed is not None else "結果がありません")
        return {}

    viable = {}
    cols = st.columns(len(regimes_in_results))

    for i, regime in enumerate(regimes_in_results):
        with cols[i]:
            if oos_passed is not None:
                # OOS PASS: Test期間のエントリーを使用
                best = oos_passed[regime]
            else:
                regime_set = result_set.filter_regime(regime)
                best = regime_set.best_by_pnl

            if not best:
                st.caption(f"{REGIME_ICONS.get(regime, '')} {REGIME_OPTIONS.get(regime, regime)}: データなし")
                continue

            pf = best.metrics.profit_factor
            pnl = best.metrics.total_profit_pct
            wr = best.metrics.win_rate
            trades = best.metrics.total_trades
            sharpe = best.metrics.sharpe_ratio
            dd = best.metrics.max_drawdown_pct
            score = best.composite_score

            # OOS PASSの場合は既にフィルタ済みなので常にviable
            if oos_passed is not None:
                is_viable = True
            else:
                # 採用基準: PF > 1.0 かつ P/L > 0 かつ trades >= 5
                is_viable = pf > 1.0 and pnl > 0 and trades >= 5

            if is_viable:
                viable[regime] = best

            icon = REGIME_ICONS.get(regime, "")
            label = REGIME_OPTIONS.get(regime, regime)
            card_cls = "viable" if is_viable else "not-viable"
            pnl_cls = "positive" if pnl > 0 else "negative"
            pf_cls = "positive" if pf > 1.0 else "negative"
            sharpe_cls = "positive" if sharpe > 0 else "negative"
            verdict_cls = "pass" if is_viable else "fail"
            verdict_text = "✅ 採用可" if is_viable else "❌ 不採用"

            st.markdown(f"""
            <div class="regime-best-card {card_cls}">
                <div class="regime-title">{icon} {label}</div>
                <div class="template-name">{best.template_name}</div>
                <div class="param-text">{best.param_str}</div>
                <div class="metric-row" title="各指標を重み付けした複合スコア（0〜1）">
                    <span class="metric-label">総合スコア</span>
                    <span class="metric-value">{score:.4f}</span>
                </div>
                <div class="metric-row" title="総利益÷総損失。1.0以上で利益＞損失。1.5以上が目安">
                    <span class="metric-label">損益比率</span>
                    <span class="metric-value {pf_cls}">{pf:.2f}</span>
                </div>
                <div class="metric-row" title="バックテスト期間の累計損益率">
                    <span class="metric-label">合計損益</span>
                    <span class="metric-value {pnl_cls}">{pnl:+.2f}%</span>
                </div>
                <div class="metric-row" title="勝ちトレードの割合。50%以上なら半分以上で利益">
                    <span class="metric-label">勝率</span>
                    <span class="metric-value">{wr:.1f}%</span>
                </div>
                <div class="metric-row" title="リスクあたりのリターン。1.0以上が良い、2.0以上は優秀">
                    <span class="metric-label">シャープ比</span>
                    <span class="metric-value {sharpe_cls}">{sharpe:.2f}</span>
                </div>
                <div class="metric-row" title="最高値から最も下がった幅。小さいほどリスクが低い">
                    <span class="metric-label">最大DD</span>
                    <span class="metric-value negative">{dd:.2f}%</span>
                </div>
                <div class="metric-row" title="バックテスト期間中の総トレード回数">
                    <span class="metric-label">取引数</span>
                    <span class="metric-value">{trades}</span>
                </div>
                <div class="verdict {verdict_cls}">{verdict_text}</div>
            </div>
            """, unsafe_allow_html=True)

    # 採用サマリー
    total_regimes = len(regimes_in_results)
    viable_count = len(viable)
    if viable_count == total_regimes:
        st.success(f"全{total_regimes}レジームで採用可能な戦略あり")
    elif viable_count > 0:
        st.warning(f"{total_regimes}レジーム中 {viable_count} で採用可能")
    else:
        st.error("全レジームで採用基準を満たす戦略なし")

    return viable


def _generate_results_markdown(
    result_set,
    validated=None,
    wfa_result=None,
) -> str:
    """結果をMarkdownテキストとして生成"""
    from datetime import datetime

    lines = []
    lines.append(f"# {result_set.symbol} 最適化結果")
    lines.append(f"")
    lines.append(f"- 実行TF: {result_set.execution_tf}")
    lines.append(f"- HTF: {result_set.htf}")
    lines.append(f"- 総組合せ: {result_set.total_combinations}")
    lines.append(f"- 出力日時: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # OOS セクション
    if validated:
        lines.append(f"")
        lines.append(f"## OOS検証")
        lines.append(f"")
        lines.append(
            f"- Train[:{validated.train_end}] / "
            f"Val[{validated.train_end}:{validated.val_end}] / "
            f"Test[{validated.val_end}:{validated.total_bars}]"
        )

        # 判定
        for regime in validated.val_best.keys():
            test_entry = validated.test_results.get(regime)
            has_warnings = any(f"[{regime}]" in w for w in validated.overfitting_warnings)
            test_positive = test_entry and test_entry.metrics.total_profit_pct > 0

            if test_positive and not has_warnings:
                verdict = "PASS"
            elif test_positive and has_warnings:
                verdict = "CAUTION"
            else:
                verdict = "FAIL"
            lines.append(f"- **{regime}: {verdict}**")

        # 警告
        if validated.overfitting_warnings:
            lines.append(f"")
            lines.append(f"### 警告")
            for w in validated.overfitting_warnings:
                lines.append(f"- {w}")

        # 比較テーブル（PASSのみ）
        lines.append(f"")
        lines.append(f"### PASS戦略 Train/Val/Test 比較")
        lines.append(f"")
        lines.append(f"| レジーム | 戦略 | Train PnL | Train Trades | Val PnL | Test PnL | 劣化 |")
        lines.append(f"|---------|------|-----------|-------------|---------|---------|------|")

        for regime in validated.val_best.keys():
            val_entry = validated.val_best.get(regime)
            test_entry = validated.test_results.get(regime)
            if not val_entry:
                continue

            # PASSしたレジームのみ
            _has_w = any(f"[{regime}]" in w for w in validated.overfitting_warnings)
            _test_pos = test_entry and test_entry.metrics.total_profit_pct > 0
            if not (_test_pos and not _has_w):
                continue

            train_entry = None
            for te in validated.train_results.filter_regime(regime).entries:
                if te.template_name == val_entry.template_name and te.params == val_entry.params:
                    train_entry = te
                    break

            t_pnl = f"{train_entry.metrics.total_profit_pct:+.2f}%" if train_entry else "-"
            t_trades = str(train_entry.metrics.total_trades) if train_entry else "-"
            v_pnl = f"{val_entry.metrics.total_profit_pct:+.2f}%"
            te_pnl = f"{test_entry.metrics.total_profit_pct:+.2f}%" if test_entry else "-"

            decay_str = "-"
            if test_entry and train_entry and train_entry.metrics.total_profit_pct > 0:
                decay = (
                    (train_entry.metrics.total_profit_pct - test_entry.metrics.total_profit_pct)
                    / abs(train_entry.metrics.total_profit_pct) * 100
                )
                decay_str = _format_decay(decay)

            strategy = f"{val_entry.template_name} ({val_entry.param_str})"
            lines.append(f"| {regime} | {strategy} | {t_pnl} | {t_trades} | {v_pnl} | {te_pnl} | {decay_str} |")

    # WFA セクション
    if wfa_result is not None:
        lines.append(f"")
        lines.append(f"## Walk-Forward Analysis")
        lines.append(f"")
        lines.append(f"- フォールド数: {len(wfa_result.folds)}")
        lines.append(f"- データ本数: {wfa_result.total_bars}")
        lines.append(f"")

        for regime in wfa_result.wfe.keys():
            cr = wfa_result.consistency_ratio.get(regime, 0)
            wfe = wfa_result.wfe.get(regime, 0)
            stitched = wfa_result.stitched_oos_pnl.get(regime, 0)
            stability = wfa_result.strategy_stability.get(regime, 0)

            if cr >= 0.6 and wfe > 0.3:
                verdict = "PASS"
            elif cr >= 0.4:
                verdict = "CAUTION"
            else:
                verdict = "FAIL"

            lines.append(f"### {regime}: {verdict}")
            lines.append(f"")
            lines.append(f"| 指標 | 値 |")
            lines.append(f"|------|-----|")
            lines.append(f"| WFE | {wfe:.3f} |")
            lines.append(f"| Consistency Ratio | {cr:.0%} |")
            lines.append(f"| Stitched OOS PnL | {stitched:+.2f}% |")
            lines.append(f"| Strategy Stability | {stability:.0%} |")
            lines.append(f"")

        # フォールド別テーブル
        lines.append(f"### フォールド別詳細")
        lines.append(f"")
        lines.append(f"| Fold | IS区間 | OOS区間 | 戦略 | IS PnL | OOS PnL |")
        lines.append(f"|------|--------|---------|------|--------|---------|")

        for fold in wfa_result.folds:
            for regime in wfa_result.wfe.keys():
                selected = fold.selected_strategy.get(regime)
                oos_entry = fold.oos_results.get(regime)
                strategy = f"{selected.template_name}" if selected else "-"
                is_pnl = f"{selected.metrics.total_profit_pct:+.2f}%" if selected else "-"
                oos_pnl = f"{oos_entry.metrics.total_profit_pct:+.2f}%" if oos_entry else "-"
                lines.append(
                    f"| {fold.fold_index + 1} ({regime}) "
                    f"| [{fold.is_range[0]}:{fold.is_range[1]}] "
                    f"| [{fold.oos_range[0]}:{fold.oos_range[1]}] "
                    f"| {strategy} | {is_pnl} | {oos_pnl} |"
                )

    # ランキング上位20
    lines.append(f"")
    lines.append(f"## ランキング Top 20")
    lines.append(f"")
    lines.append(f"| # | テンプレート | パラメータ | レジーム | スコア | Trades | WR% | PF | PnL% | Sharpe | 警告 |")
    lines.append(f"|---|------------|----------|---------|-------|--------|-----|-----|------|--------|------|")

    ranked = result_set.ranked()[:20]
    for i, e in enumerate(ranked):
        m = e.metrics
        warns = " / ".join(e.warnings) if e.warnings else ""
        lines.append(
            f"| {i} | {e.template_name} | {e.param_str} | {e.trend_regime} "
            f"| {e.composite_score:.4f} | {m.total_trades} | {m.win_rate:.1f} "
            f"| {m.profit_factor:.2f} | {m.total_profit_pct:+.2f} "
            f"| {m.sharpe_ratio:.2f} | {warns} |"
        )

    lines.append(f"")
    return "\n".join(lines)


def _render_md_copy_button(md_text: str, button_id: str = "cpbtn"):
    """Markdownテキストをクリップボードにコピーするボタンを描画"""
    _b64 = base64.b64encode(md_text.encode("utf-8")).decode("ascii")
    _copy_html = f"""<script>
function _copyMd_{button_id}() {{
    var bytes = Uint8Array.from(atob('{_b64}'), function(c) {{ return c.charCodeAt(0); }});
    var text = new TextDecoder('utf-8').decode(bytes);
    navigator.clipboard.writeText(text).then(function() {{
        document.getElementById('{button_id}').innerText = 'Copied!';
        setTimeout(function() {{ document.getElementById('{button_id}').innerText = 'Markdown'; }}, 2000);
    }}).catch(function() {{
        var ta = document.createElement('textarea');
        ta.value = text;
        ta.style.position = 'fixed';
        ta.style.left = '-9999px';
        document.body.appendChild(ta);
        ta.select();
        document.execCommand('copy');
        document.body.removeChild(ta);
        document.getElementById('{button_id}').innerText = 'Copied!';
        setTimeout(function() {{ document.getElementById('{button_id}').innerText = 'Markdown'; }}, 2000);
    }});
}}
</script>
<button id="{button_id}" onclick="_copyMd_{button_id}()" style="
    background-color:transparent;border:1px solid rgba(250,250,250,0.2);
    color:rgba(250,250,250,0.85);padding:0.4rem 0.75rem;border-radius:0.5rem;
    cursor:pointer;font-size:14px;width:100%;
">Markdown</button>"""
    st.components.v1.html(_copy_html, height=42)


def _format_decay(decay: float) -> str:
    """劣化率を判定ラベル付きでフォーマット"""
    if decay <= 0:
        return f"改善 {abs(decay):.0f}%"
    elif decay <= 30:
        return f"{decay:.0f}% (良好)"
    elif decay <= 50:
        return f"{decay:.0f}% (許容)"
    else:
        return f"{decay:.0f}% (危険)"


def _render_oos_section(validated: ValidatedResultSet):
    """OOS検証結果セクション"""
    section_header(
        "🛡️", "OOS検証結果",
        f"Train[:{validated.train_end}] / "
        f"Val[{validated.train_end}:{validated.val_end}] / "
        f"Test[{validated.val_end}:{validated.total_bars}]"
    )

    # ========== 判定バナー ==========
    for regime in validated.val_best.keys():
        test_entry = validated.test_results.get(regime)
        val_entry = validated.val_best.get(regime)
        stats = validated.filter_stats.get(regime, {})

        if not val_entry:
            continue

        # 判定ロジック
        has_warnings = any(f"[{regime}]" in w for w in validated.overfitting_warnings)
        test_positive = test_entry and test_entry.metrics.total_profit_pct > 0

        regime_label = f"{REGIME_ICONS.get(regime, '')} {regime}"

        if test_positive and not has_warnings:
            st.success(f"**{regime_label}: PASS** — Test期間でプラス、警告なし")
        elif test_positive and has_warnings:
            st.warning(f"**{regime_label}: CAUTION** — Test期間プラスだが警告あり")
        else:
            st.error(f"**{regime_label}: FAIL** — Test期間でマイナス（実運用非推奨）")

    # ========== 警告表示 ==========
    if validated.overfitting_warnings:
        for w in validated.overfitting_warnings:
            st.warning(f"{w}")

    # ========== Val-best の Train vs Val vs Test 比較テーブル（PASSのみ） ==========
    comparison_rows = []
    for regime in validated.val_best.keys():
        val_entry = validated.val_best.get(regime)
        test_entry = validated.test_results.get(regime)

        if not val_entry:
            continue

        # PASSしたレジームのみ表示
        has_warnings = any(f"[{regime}]" in w for w in validated.overfitting_warnings)
        test_positive = test_entry and test_entry.metrics.total_profit_pct > 0
        if not (test_positive and not has_warnings):
            continue

        # Val-best に対応する Train エントリーを探す
        train_entry = None
        for te in validated.train_results.filter_regime(regime).entries:
            if te.template_name == val_entry.template_name and te.params == val_entry.params:
                train_entry = te
                break

        if not train_entry:
            train_entry = validated.train_results.filter_regime(regime).best

        if not train_entry:
            continue

        row = {
            "レジーム": f"{REGIME_ICONS.get(regime, '')} {regime}",
            "戦略": f"{val_entry.template_name} ({val_entry.param_str})",
        }

        row["Train PnL"] = f"{train_entry.metrics.total_profit_pct:+.2f}%"
        row["Train Sharpe"] = f"{train_entry.metrics.sharpe_ratio:.2f}"
        row["Train Trades"] = train_entry.metrics.total_trades
        row["Val PnL"] = f"{val_entry.metrics.total_profit_pct:+.2f}%"
        row["Val Sharpe"] = f"{val_entry.metrics.sharpe_ratio:.2f}"

        if test_entry:
            row["Test PnL"] = f"{test_entry.metrics.total_profit_pct:+.2f}%"
            row["Test Sharpe"] = f"{test_entry.metrics.sharpe_ratio:.2f}"

            if train_entry.metrics.total_profit_pct > 0:
                decay = (
                    (train_entry.metrics.total_profit_pct - test_entry.metrics.total_profit_pct)
                    / abs(train_entry.metrics.total_profit_pct) * 100
                )
                row["劣化"] = _format_decay(decay)
            else:
                row["劣化"] = "-"
        else:
            row["Test PnL"] = "-"
            row["Test Sharpe"] = "-"
            row["劣化"] = "-"

        comparison_rows.append(row)

    if comparison_rows:
        st.dataframe(
            pd.DataFrame(comparison_rows),
            use_container_width=True,
            hide_index=True,
        )

    else:
        st.info("OOS検証結果がありません（Train期間でトレードが発生しなかった可能性）")


def _render_wfa_section(wfa_result):
    """Walk-Forward Analysis 結果セクション"""
    from optimizer.walk_forward import WFAResultSet

    n_folds = len(wfa_result.folds)
    section_header(
        "🔄", "Walk-Forward Analysis",
        f"{n_folds}フォールド | Anchored WFA | {wfa_result.total_bars}本"
    )

    # ========== レジーム別 PASS/CAUTION/FAIL バナー ==========
    for regime in wfa_result.wfe.keys():
        cr = wfa_result.consistency_ratio.get(regime, 0)
        wfe = wfa_result.wfe.get(regime, 0)
        stitched = wfa_result.stitched_oos_pnl.get(regime, 0)

        regime_label = f"{REGIME_ICONS.get(regime, '')} {regime}"

        if cr >= 0.6 and wfe > 0.3:
            st.success(
                f"**{regime_label}: PASS** — "
                f"CR={cr:.0%}, WFE={wfe:.2f}, OOS合計={stitched:+.2f}%"
            )
        elif cr >= 0.4:
            st.warning(
                f"**{regime_label}: CAUTION** — "
                f"CR={cr:.0%}, WFE={wfe:.2f}, OOS合計={stitched:+.2f}%"
            )
        else:
            st.error(
                f"**{regime_label}: FAIL** — "
                f"CR={cr:.0%}, WFE={wfe:.2f}, OOS合計={stitched:+.2f}%"
            )

    # ========== 集約メトリクスカード ==========
    regimes = list(wfa_result.wfe.keys())
    if regimes:
        st.markdown("#### 集約メトリクス")
        for regime in regimes:
            regime_label = f"{REGIME_ICONS.get(regime, '')} {regime}"
            cr = wfa_result.consistency_ratio.get(regime, 0)
            wfe = wfa_result.wfe.get(regime, 0)
            stitched = wfa_result.stitched_oos_pnl.get(regime, 0)
            stability = wfa_result.strategy_stability.get(regime, 0)

            st.caption(f"**{regime_label}**")
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric(
                "WFE",
                f"{wfe:.3f}",
                help="Walk-Forward Efficiency: mean(OOS PnL)/mean(IS PnL)。0.3以上が目安",
            )
            mc2.metric(
                "Consistency Ratio",
                f"{cr:.0%}",
                help="OOS期間でプラスだったフォールドの割合。60%以上が目安",
            )
            mc3.metric(
                "Stitched OOS PnL",
                f"{stitched:+.2f}%",
                help="全フォールドのOOS PnLを合算した値",
            )
            mc4.metric(
                "Strategy Stability",
                f"{stability:.0%}",
                help="各フォールドで同一戦略が選ばれた割合。高いほど安定",
            )

    # ========== フォールド別詳細テーブル（展開式） ==========
    with st.expander("フォールド別詳細を表示"):
        for regime in regimes:
            regime_label = f"{REGIME_ICONS.get(regime, '')} {regime}"
            st.caption(f"**{regime_label}**")

            fold_rows = []
            for fold in wfa_result.folds:
                is_range = fold.is_range
                oos_range = fold.oos_range
                selected = fold.selected_strategy.get(regime)
                oos_entry = fold.oos_results.get(regime)

                row = {
                    "Fold": fold.fold_index + 1,
                    "IS区間": f"[{is_range[0]}:{is_range[1]}]",
                    "OOS区間": f"[{oos_range[0]}:{oos_range[1]}]",
                    "選択戦略": (
                        f"{selected.template_name} ({selected.param_str})"
                        if selected else "-"
                    ),
                    "IS PnL": (
                        f"{selected.metrics.total_profit_pct:+.2f}%"
                        if selected else "-"
                    ),
                    "OOS PnL": (
                        f"{oos_entry.metrics.total_profit_pct:+.2f}%"
                        if oos_entry else "-"
                    ),
                    "OOS Trades": (
                        oos_entry.metrics.total_trades
                        if oos_entry else 0
                    ),
                }
                fold_rows.append(row)

            if fold_rows:
                st.dataframe(
                    pd.DataFrame(fold_rows),
                    use_container_width=True,
                    hide_index=True,
                )


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

    # ヘッダー情報 + エクスポートボタン
    hcol1, hcol2 = st.columns([4, 1])
    with hcol1:
        st.markdown(
            f"**{result_set.symbol}** | "
            f"Exec: `{result_set.execution_tf}` | "
            f"HTF: `{result_set.htf}` | "
            f"Total: **{result_set.total_combinations}** runs"
        )
    with hcol2:
        validated = st.session_state.get("validated_result")
        wfa_result = st.session_state.get("wfa_result")
        md_text = _generate_results_markdown(result_set, validated, wfa_result)
        _render_md_copy_button(md_text, button_id="cpbtn_single")

    # --- WFA 結果 ---
    wfa_result = st.session_state.get("wfa_result")
    if wfa_result is not None:
        _render_wfa_section(wfa_result)
        st.divider()

    # --- OOS 検証結果 ---
    validated = st.session_state.get("validated_result")
    if validated is not None:
        _render_oos_section(validated)
        st.divider()

    # --- レジーム別ベスト戦略サマリー ---
    # OOS有効時: PASSしたレジームのみ表示
    oos_passed = None
    if validated is not None:
        oos_passed = {}
        for regime in validated.val_best.keys():
            test_entry = validated.test_results.get(regime)
            has_warnings = any(f"[{regime}]" in w for w in validated.overfitting_warnings)
            test_positive = test_entry and test_entry.metrics.total_profit_pct > 0
            if test_positive and not has_warnings:
                oos_passed[regime] = test_entry
        if not oos_passed:
            oos_passed = None  # 全FAIL時はNoneにしてメッセージ表示

    viable_strategies = _render_regime_best_summary(result_set, oos_passed=oos_passed)

    # --- レジーム切替通しテスト ---
    if len(viable_strategies) >= 1:
        _render_regime_switching_section(result_set, viable_strategies)

    st.divider()

    # --- 自動分析 ---
    from ui.components.optimizer_analysis import analyze_single_result
    insights = analyze_single_result(result_set, viable_strategies)
    _render_analysis_section(insights, title="自動分析", icon="📝")

    st.divider()

    # --- フィルター ---
    section_header("🔎", "絞り込み・ランキング")

    fcol1, fcol2, fcol3 = st.columns([1, 1, 2])
    with fcol1:
        filter_regime = st.selectbox(
            "レジーム",
            options=["all"] + list(REGIME_OPTIONS.keys()),
            format_func=lambda x: (
                "すべて" if x == "all"
                else f"{REGIME_ICONS.get(x, '')} {REGIME_OPTIONS.get(x, x)}"
            ),
            key="result_filter_regime",
            help="相場の状態で絞り込み",
        )
    with fcol2:
        templates_in_results = sorted(set(e.template_name for e in result_set.entries))
        filter_template = st.selectbox(
            "テンプレート",
            options=["all"] + templates_in_results,
            format_func=lambda x: "すべて" if x == "all" else x,
            key="result_filter_template",
            help="戦略テンプレートで絞り込み",
        )
    with fcol3:
        min_trades = st.slider(
            "最低取引数",
            min_value=0,
            max_value=50,
            value=5,
            key="result_min_trades",
            help="取引回数が少なすぎる結果を除外（デフォルト5）",
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

    st.caption(f"{len(filtered.entries)} 件表示中")

    # --- ランキングテーブル（スタイル付き） ---
    ranking_df = filtered.to_dataframe()

    # カラム設定（日本語ラベル + ヘルプ付き）
    column_config = {
        "template": st.column_config.TextColumn(
            "テンプレート",
            help="使用した戦略テンプレート名",
        ),
        "params": st.column_config.TextColumn(
            "パラメータ",
            help="テンプレートに適用したパラメータの組み合わせ",
        ),
        "regime": st.column_config.TextColumn(
            "レジーム",
            help="相場の状態（uptrend=上昇, downtrend=下降, range=レンジ）",
        ),
        "score": st.column_config.ProgressColumn(
            "総合スコア",
            help="各指標を重み付けした複合スコア（0〜1）。高いほど良い",
            min_value=0,
            max_value=1,
            format="%.4f",
        ),
        "trades": st.column_config.NumberColumn(
            "取引数",
            help="バックテスト期間中の総トレード回数。少なすぎると統計的に信頼できない",
            format="%d",
        ),
        "win_rate": st.column_config.NumberColumn(
            "勝率 %",
            help="勝ちトレードの割合。50%以上なら半分以上のトレードで利益",
            format="%.1f%%",
        ),
        "profit_factor": st.column_config.NumberColumn(
            "損益比率",
            help="総利益 ÷ 総損失。1.0以上で利益が損失を上回る。1.5以上が目安",
            format="%.2f",
        ),
        "total_pnl": st.column_config.NumberColumn(
            "合計損益 %",
            help="バックテスト期間の累計損益率。プラスなら利益、マイナスなら損失",
            format="%.2f%%",
        ),
        "max_dd": st.column_config.NumberColumn(
            "最大DD %",
            help="最大ドローダウン。最高値から最も下がった幅。小さいほどリスクが低い",
            format="%.2f%%",
        ),
        "sharpe": st.column_config.NumberColumn(
            "シャープ比",
            help="リスクあたりのリターン。1.0以上が良い、2.0以上は優秀",
            format="%.2f",
        ),
        "warnings": st.column_config.TextColumn(
            "⚠️ 警告",
            help="過学習の疑いがある場合に表示。PF>2.0, Sharpe>3.0, Trade<30",
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
    section_header("📊", "チャート")

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
        section_header("📈", "資産推移カーブ", f"上位 {min(len(entries_with_result), 10)} 件")

        top_n = st.slider(
            "表示件数", 1, min(20, len(entries_with_result)),
            min(5, len(entries_with_result)), key="equity_top_n"
        )
        equity_fig = create_equity_overlay(entries_with_result, max_entries=top_n)
        st.plotly_chart(equity_fig, use_container_width=True)

    st.divider()

    # --- レジーム切替型YAMLエクスポート ---
    section_header("💾", "戦略エクスポート")

    if viable_strategies:
        # レジーム切替型の設定を組み立て
        regime_config = {
            "symbol": result_set.symbol,
            "execution_tf": result_set.execution_tf,
            "htf": result_set.htf,
            "regime_strategies": {},
        }
        for regime, entry in viable_strategies.items():
            regime_config["regime_strategies"][regime] = {
                "template": entry.template_name,
                "params": entry.params,
                "config": entry.config,
                "metrics": {
                    "profit_factor": round(entry.metrics.profit_factor, 2),
                    "win_rate": round(entry.metrics.win_rate, 1),
                    "total_pnl": round(entry.metrics.total_profit_pct, 2),
                    "max_dd": round(entry.metrics.max_drawdown_pct, 2),
                    "sharpe": round(entry.metrics.sharpe_ratio, 2),
                    "trades": entry.metrics.total_trades,
                },
            }

        yaml_str = yaml.dump(regime_config, default_flow_style=False, allow_unicode=True)

        st.markdown(f"**{len(viable_strategies)}** レジームの採用戦略をエクスポート")
        col_yaml, col_dl = st.columns([3, 1])
        with col_yaml:
            st.code(yaml_str, language="yaml")
        with col_dl:
            st.download_button(
                "📥 Download YAML",
                data=yaml_str,
                file_name=f"regime_strategy_{result_set.symbol}.yaml",
                mime="text/yaml",
                use_container_width=True,
            )
    else:
        st.warning("採用可能な戦略がありません（全レジームで不採用）")


@st.cache_data(ttl=300, show_spinner="CSVデータ読込中...")
def _load_ohlcv_from_disk(symbol: str, tf: str) -> "pd.DataFrame | None":
    """inputdata/ ディレクトリからCSVをロードしてDataFrameを返す（キャッシュ付き）"""
    from pathlib import Path
    from data.binance_loader import BinanceCSVLoader

    inputdata_dir = Path("inputdata")
    if not inputdata_dir.exists():
        return None

    # ファイル名パターン: {SYMBOL}-{TF}-*.csv
    pattern = f"{symbol}-{tf}-*.csv"
    matches = list(inputdata_dir.glob(pattern))
    if not matches:
        return None

    loader = BinanceCSVLoader()
    ohlcv = loader.load(str(matches[0]))
    return ohlcv.df


def _resolve_ohlcv_df(symbol: str, exec_tf: str, data_source: str, data_period: dict) -> "pd.DataFrame | None":
    """
    OHLCVデータを解決する（セッション → ディスクの順で探索）

    1. session_state.datasets / trimmed_datasets を確認
    2. なければ inputdata/ のCSVから自動ロード
    3. trimmed の場合は日付範囲でフィルタ
    """
    datasets = st.session_state.get("datasets", {})
    trimmed_list = st.session_state.get("trimmed_datasets", [])

    # --- セッション内の切り出しデータを確認 ---
    if data_source == "trimmed" and data_period:
        period_start = data_period.get("start", "")
        period_end = data_period.get("end", "")
        for entry in trimmed_list:
            if (entry["symbol"] == symbol
                    and exec_tf in entry["data"]
                    and str(entry["start_dt"])[:10] == period_start
                    and str(entry["end_dt"])[:10] == period_end):
                return entry["data"][exec_tf].df

    # --- セッション内のオリジナルデータを確認 ---
    if symbol in datasets and exec_tf in datasets[symbol]:
        df = datasets[symbol][exec_tf].df
        if data_source == "trimmed" and data_period:
            return _trim_df(df, data_period)
        return df

    # --- ディスクからCSV自動ロード ---
    df = _load_ohlcv_from_disk(symbol, exec_tf)
    if df is not None:
        if data_source == "trimmed" and data_period:
            return _trim_df(df, data_period)
        return df

    return None


def _trim_df(df: "pd.DataFrame", data_period: dict) -> "pd.DataFrame":
    """DataFrameを日付範囲でフィルタ"""
    period_start = data_period.get("start", "")
    period_end = data_period.get("end", "")
    if period_start:
        df = df[df["datetime"] >= pd.Timestamp(period_start)]
    if period_end:
        df = df[df["datetime"] <= pd.Timestamp(period_end + " 23:59:59")]
    return df


def _parse_result_filename(stem: str) -> dict:
    """結果ファイル名からメタ情報をパース（JSONを開かずに高速判定）"""
    import re
    m = re.match(
        r'^([A-Z0-9]+)_exec([^_]+)_htf([^_]+?)(?:_trim([^_]+))?_(\d{8})_(\d{6})$',
        stem,
    )
    if not m:
        return {
            "symbol": "?", "exec_tf": "?", "htf": "?",
            "is_trimmed": False, "trim_label": "", "date_label": stem,
            "date_str": "", "year": "?",
        }

    symbol, exec_tf, htf, trim_raw, date_str, time_str = m.groups()

    # 日時ラベル: "01/31 15:52"
    date_label = f"{date_str[4:6]}/{date_str[6:8]} {time_str[:2]}:{time_str[2:4]}"

    # トリム期間ラベル: "02/01~06/14"
    trim_label = ""
    if trim_raw:
        parts = trim_raw.split("-")
        if len(parts) == 2 and len(parts[0]) == 4 and len(parts[1]) == 4:
            trim_label = f"{parts[0][:2]}/{parts[0][2:]}~{parts[1][:2]}/{parts[1][2:]}"
        else:
            trim_label = trim_raw

    return {
        "symbol": symbol,
        "exec_tf": exec_tf,
        "htf": htf,
        "is_trimmed": trim_raw is not None,
        "trim_label": trim_label,
        "date_label": date_label,
        "date_str": date_str,
        "year": date_str[:4],
    }


def _read_data_period_from_json(path) -> tuple:
    """JSONファイルからdata_periodのstart/end年を高速読み取り。

    新形式: data_periodはresultsの前に配置 → 先頭2KBで見つかる
    旧形式: data_periodはresultsの後（末尾） → 末尾500Bで見つかる

    Returns:
        (start_year, end_year) e.g. ("2025", "2026") or ("", "")
    """
    import re
    import os
    pattern = r'"data_period"\s*:\s*\{\s*"start"\s*:\s*"(\d{4})-'
    pattern_end = r'"data_period"\s*:\s*\{[^}]*"end"\s*:\s*"(\d{4})-'
    try:
        file_size = os.path.getsize(path)
        with open(path, "r", encoding="utf-8") as f:
            # 先頭2KBを読む（新形式: data_periodがresultsの前）
            head = f.read(2048)
            m = re.search(pattern, head)
            if not m and file_size > 2048:
                # 末尾500Bを読む（旧形式: data_periodがresultsの後）
                f.seek(max(0, file_size - 500))
                tail = f.read()
                m = re.search(pattern, tail)
                if m:
                    m2 = re.search(pattern_end, tail)
                    end_year = m2.group(1) if m2 else m.group(1)
                    return m.group(1), end_year
            elif m:
                m2 = re.search(pattern_end, head)
                end_year = m2.group(1) if m2 else m.group(1)
                return m.group(1), end_year
    except Exception:
        pass
    return "", ""


def _render_load_view():
    """保存済み結果の読み込みビュー（左: 選択、右: プレビュー）"""
    import json
    from pathlib import Path
    from optimizer.results import OptimizationResultSet

    section_header("📁", "保存済み結果の読み込み", "results/ フォルダのJSONファイル")

    results_dir = Path("results")
    if not results_dir.exists():
        st.warning("results/ フォルダが見つかりません。")
        return

    json_files = sorted(
        results_dir.glob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not json_files:
        st.info("保存済みの結果ファイルがありません。")
        return

    # 全ファイルのメタ情報をファイル名からパース + JSONからdata_period読取
    file_meta = {}
    for fp in json_files:
        parsed = _parse_result_filename(fp.stem)
        # JSONからデータ期間の年を取得（ファイル名の実行日ではなく）
        dp_start, dp_end = _read_data_period_from_json(fp)
        if dp_start:
            # データ期間が跨る年を全て含める（例: 2025-02 ~ 2026-01 → "2025"）
            parsed["data_year"] = dp_start
        else:
            # data_period未保存のファイル → 実行年にフォールバック
            parsed["data_year"] = parsed["year"]
        file_meta[fp.stem] = {**parsed, "path": fp}

    all_symbols = sorted(set(m["symbol"] for m in file_meta.values()))
    all_exec_tfs = sorted(set(m["exec_tf"] for m in file_meta.values()))
    all_years = sorted(
        set(m["data_year"] for m in file_meta.values() if m["data_year"] != "?"),
        reverse=True,
    )

    # === 2カラムレイアウト: 左=選択 / 右=プレビュー ===
    left_col, right_col = st.columns([2, 3])

    with left_col:
        # --- フィルター ---
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            symbol_filter = st.selectbox(
                "銘柄",
                options=["すべて"] + all_symbols,
                key="load_symbol_filter",
            )
        with fc2:
            tf_filter = st.selectbox(
                "実行TF",
                options=["すべて"] + all_exec_tfs,
                key="load_tf_filter",
            )
        with fc3:
            year_filter = st.selectbox(
                "年度",
                options=["すべて"] + all_years,
                key="load_year_filter",
            )

        source_options = ["すべて", "📦 オリジナル", "✂️ 切り出し"]
        source_filter = st.radio(
            "データソース",
            options=source_options,
            horizontal=True,
            key="load_source_filter",
            label_visibility="collapsed",
        )

        # フィルタ適用
        filtered_stems = []
        for stem, meta in file_meta.items():
            if symbol_filter != "すべて" and meta["symbol"] != symbol_filter:
                continue
            if tf_filter != "すべて" and meta["exec_tf"] != tf_filter:
                continue
            if year_filter != "すべて" and meta["data_year"] != year_filter:
                continue
            if source_filter == "📦 オリジナル" and meta["is_trimmed"]:
                continue
            if source_filter == "✂️ 切り出し" and not meta["is_trimmed"]:
                continue
            filtered_stems.append(stem)

        file_options = {s: file_meta[s]["path"] for s in filtered_stems}

        # 読みやすいラベル生成
        def _format_label(stem):
            m = file_meta[stem]
            parts = [f"{m['symbol']} | {m['exec_tf']}→{m['htf']}"]
            if m["is_trimmed"] and m["trim_label"]:
                parts.append(f"✂️{m['trim_label']}")
            parts.append(m["date_label"])
            return " | ".join(parts)

        display_labels = {s: _format_label(s) for s in filtered_stems}

        st.caption(f"{len(filtered_stems)}件")

        selected_names = st.multiselect(
            "ファイルを選択",
            options=filtered_stems,
            default=[],
            format_func=lambda x: display_labels.get(x, x),
            key="load_file_select_multi",
        )

        if not selected_names:
            st.caption("ファイルを1つ以上選択してください。")

        # メタ情報プレビュー
        if selected_names and len(selected_names) == 1:
            selected_path = file_options[selected_names[0]]
            with open(selected_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            st.metric("シンボル", meta.get("symbol", "?"))
            mc1, mc2 = st.columns(2)
            with mc1:
                st.metric("実行TF", meta.get("execution_tf", "?"))
            with mc2:
                st.metric("結果数", f"{len(meta.get('results', [])):,}")

            ds = meta.get("data_source", "original")
            dp = meta.get("data_period", {})
            ts = meta.get("timestamp", "")
            if len(ts) >= 15:
                display_ts = f"{ts[:4]}/{ts[4:6]}/{ts[6:8]} {ts[9:11]}:{ts[11:13]}:{ts[13:15]}"
            else:
                display_ts = ts
            if ds == "trimmed" and dp:
                st.caption(f"✂️ {dp.get('start', '?')} ~ {dp.get('end', '?')}")
            st.caption(f"保存: {display_ts}")

        elif selected_names and len(selected_names) >= 2:
            preview_rows = []
            for name in selected_names:
                fp = file_options[name]
                with open(fp, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                ds = meta.get("data_source", "original")
                dp = meta.get("data_period", {})
                period = f"{dp.get('start', '')}~{dp.get('end', '')}" if ds == "trimmed" and dp else "全期間"
                preview_rows.append({
                    "銘柄": meta.get("symbol", "?"),
                    "TF": meta.get("execution_tf", "?"),
                    "件数": len(meta.get("results", [])),
                    "期間": f"✂️{period}" if ds == "trimmed" else "全期間",
                })
            st.dataframe(pd.DataFrame(preview_rows), use_container_width=True, hide_index=True, height=150)

        # アクションボタン
        if selected_names:
            st.divider()
            btn_col1, btn_col2, btn_col3 = st.columns(3)
            with btn_col1:
                if st.button(
                    "📊 読み込む",
                    type="primary" if len(selected_names) == 1 else "secondary",
                    disabled=len(selected_names) != 1,
                    use_container_width=True,
                ):
                    selected_path = file_options[selected_names[0]]
                    try:
                        result_set = OptimizationResultSet.from_json(str(selected_path))
                        # 現在のスコア重みで再スコアリング
                        weights = ScoringWeights(
                            profit_factor=st.session_state.get("opt_w_pf", 0.2),
                            win_rate=st.session_state.get("opt_w_wr", 0.2),
                            max_drawdown=st.session_state.get("opt_w_dd", 0.2),
                            sharpe_ratio=st.session_state.get("opt_w_sh", 0.2),
                            total_return=st.session_state.get("opt_w_ret", 0.2),
                        )
                        result_set.rescore(weights)
                        st.session_state.optimization_result = result_set
                        st.session_state.optimizer_view = "results"
                        st.rerun()
                    except Exception as e:
                        st.error(f"読み込みエラー: {e}")
            with btn_col2:
                compare_label = f"🔀 {len(selected_names)}件比較" if len(selected_names) >= 2 else "🔀 2件以上"
                if st.button(
                    compare_label,
                    type="primary" if len(selected_names) >= 2 else "secondary",
                    disabled=len(selected_names) < 2,
                    use_container_width=True,
                ):
                    loaded = []
                    weights = ScoringWeights(
                        profit_factor=st.session_state.get("opt_w_pf", 0.2),
                        win_rate=st.session_state.get("opt_w_wr", 0.2),
                        max_drawdown=st.session_state.get("opt_w_dd", 0.2),
                        sharpe_ratio=st.session_state.get("opt_w_sh", 0.2),
                        total_return=st.session_state.get("opt_w_ret", 0.2),
                    )
                    for name in selected_names:
                        fp = file_options[name]
                        try:
                            rs = OptimizationResultSet.from_json(str(fp))
                            rs.rescore(weights)
                            loaded.append(rs)
                        except Exception as e:
                            st.error(f"読み込みエラー ({name}): {e}")
                    if len(loaded) >= 2:
                        st.session_state.comparison_results = loaded
                        st.session_state.optimizer_view = "compare"
                        st.rerun()
            with btn_col3:
                if st.button(
                    f"🗑️ {len(selected_names)}件削除",
                    use_container_width=True,
                ):
                    st.session_state.delete_confirm_files = selected_names.copy()

            # 削除確認
            if st.session_state.get("delete_confirm_files"):
                targets = st.session_state.delete_confirm_files
                st.warning(f"**{len(targets)}件**のファイルを削除しますか？（JSON + CSV）")
                conf_col1, conf_col2 = st.columns(2)
                with conf_col1:
                    if st.button("✅ 削除する", type="primary", use_container_width=True):
                        deleted = 0
                        for name in targets:
                            fp = file_options.get(name)
                            if fp and fp.exists():
                                fp.unlink()
                                deleted += 1
                                # 対応するCSVも削除
                                csv_fp = fp.with_suffix(".csv")
                                if csv_fp.exists():
                                    csv_fp.unlink()
                        st.session_state.delete_confirm_files = None
                        st.toast(f"🗑️ {deleted}件削除しました")
                        st.rerun()
                with conf_col2:
                    if st.button("❌ キャンセル", use_container_width=True):
                        st.session_state.delete_confirm_files = None
                        st.rerun()

    # === 右カラム: ローソク足プレビュー ===
    with right_col:
        if not selected_names:
            st.markdown(
                '<div style="display:flex; align-items:center; justify-content:center; '
                'height:400px; color:#484f58; font-size:0.9rem;">'
                '← ファイルを選択するとチャートが表示されます'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            from ui.components.optimizer_charts import create_ohlcv_preview_chart

            for idx, name in enumerate(selected_names):
                fp = file_options[name]
                with open(fp, "r", encoding="utf-8") as f:
                    meta = json.load(f)

                symbol = meta.get("symbol", "")
                exec_tf = meta.get("execution_tf", "")
                ds = meta.get("data_source", "original")
                dp = meta.get("data_period", {})

                if ds == "trimmed" and dp:
                    chart_title = f"{symbol} ✂️ {dp.get('start', '')} ~ {dp.get('end', '')}"
                else:
                    chart_title = f"{symbol} 📦 オリジナル"

                try:
                    with st.spinner(f"{chart_title} 読込中..."):
                        ohlcv_df = _resolve_ohlcv_df(symbol, exec_tf, ds, dp)

                    if ohlcv_df is not None and not ohlcv_df.empty:
                        fig = create_ohlcv_preview_chart(ohlcv_df, title=chart_title)
                        st.plotly_chart(fig, use_container_width=True, key=f"preview_chart_{name}_{idx}")
                    else:
                        st.warning(f"{chart_title}: CSVファイルが見つかりません（inputdata/ を確認）")
                except Exception as e:
                    st.error(f"{chart_title}: プレビューエラー: {e}")


# ============================================================
# 比較ビュー
# ============================================================

def _get_regime_best_with_viability(result_set):
    """レジーム別ベスト + 採否判定を返す"""
    regimes = sorted(set(e.trend_regime for e in result_set.entries))
    result = {}
    for regime in regimes:
        regime_set = result_set.filter_regime(regime)
        best = regime_set.best_by_pnl
        if best:
            pf = best.metrics.profit_factor
            pnl = best.metrics.total_profit_pct
            trades = best.metrics.total_trades
            is_viable = pf > 1.0 and pnl > 0 and trades >= 5
            result[regime] = {"entry": best, "is_viable": is_viable}
    return result


def _render_compare_card(symbol, entry):
    """比較用ベスト戦略カード（1銘柄分）"""
    pf = entry.metrics.profit_factor
    pnl = entry.metrics.total_profit_pct
    wr = entry.metrics.win_rate
    trades = entry.metrics.total_trades
    sharpe = entry.metrics.sharpe_ratio
    dd = entry.metrics.max_drawdown_pct
    score = entry.composite_score

    is_viable = pf > 1.0 and pnl > 0 and trades >= 5
    card_cls = "viable" if is_viable else "not-viable"
    pnl_cls = "positive" if pnl > 0 else "negative"
    pf_cls = "positive" if pf > 1.0 else "negative"
    sharpe_cls = "positive" if sharpe > 0 else "negative"
    verdict_cls = "pass" if is_viable else "fail"
    verdict_text = "✅ 採用可" if is_viable else "❌ 不採用"

    st.markdown(f"""
    <div class="regime-best-card {card_cls}">
        <div class="regime-title">{symbol}</div>
        <div class="template-name">{entry.template_name}</div>
        <div class="param-text">{entry.param_str}</div>
        <div class="metric-row">
            <span class="metric-label">総合スコア</span>
            <span class="metric-value">{score:.4f}</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">損益比率</span>
            <span class="metric-value {pf_cls}">{pf:.2f}</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">合計損益</span>
            <span class="metric-value {pnl_cls}">{pnl:+.2f}%</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">勝率</span>
            <span class="metric-value">{wr:.1f}%</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">シャープ比</span>
            <span class="metric-value {sharpe_cls}">{sharpe:.2f}</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">最大DD</span>
            <span class="metric-value negative">{dd:.2f}%</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">取引数</span>
            <span class="metric-value">{trades}</span>
        </div>
        <div class="verdict {verdict_cls}">{verdict_text}</div>
    </div>
    """, unsafe_allow_html=True)


def _compare_md_header(comparison_results):
    """比較Markdown共通ヘッダー"""
    symbols = [rs.symbol for rs in comparison_results]
    return f"# 比較レポート ({len(symbols)}銘柄)\n\n比較対象: {' / '.join(symbols)}\n"


def _compare_all_regimes(comparison_results):
    """比較結果から全レジームを取得"""
    return sorted(set(
        e.trend_regime
        for rs in comparison_results
        for e in rs.entries
    ))


def _generate_meta_markdown(comparison_results):
    """メタ分析タブ用Markdown"""
    from collections import Counter as MdCounter
    lines = [_compare_md_header(comparison_results)]
    all_regimes = _compare_all_regimes(comparison_results)

    # テンプレート採択分布
    lines.append(f"## テンプレート採択分布")
    for regime in all_regimes:
        regime_label = REGIME_OPTIONS.get(regime, regime)
        bests = {}
        for rs in comparison_results:
            regime_set = rs.filter_regime(regime)
            best = regime_set.best_by_pnl
            if best:
                bests[rs.symbol] = best.template_name
        if not bests:
            continue
        counter = MdCounter(bests.values())
        most_common, count = counter.most_common(1)[0]
        pct = count / len(bests) * 100
        lines.append(f"")
        lines.append(f"### {regime_label}")
        lines.append(f"- 最頻テンプレート: **{most_common}** ({pct:.0f}%, {count}/{len(bests)}銘柄)")
        lines.append(f"- 採択分布:")
        for tmpl, cnt in counter.most_common():
            lines.append(f"  - {tmpl}: {cnt}銘柄")

    # 採用率
    lines.append(f"")
    lines.append(f"## レジーム採用率")
    for regime in all_regimes:
        regime_label = REGIME_OPTIONS.get(regime, regime)
        viable_count = 0
        total = 0
        for rs in comparison_results:
            regime_set = rs.filter_regime(regime)
            best = regime_set.best_by_pnl
            if best:
                total += 1
                m = best.metrics
                if m.profit_factor > 1.0 and m.total_profit_pct > 0 and m.total_trades >= 5:
                    viable_count += 1
        if total > 0:
            lines.append(f"- {regime_label}: {viable_count}/{total}銘柄で採用可 ({viable_count/total:.0%})")

    lines.append(f"")
    return "\n".join(lines)


def _generate_detail_markdown(comparison_results):
    """詳細比較タブ用Markdown"""
    lines = [_compare_md_header(comparison_results)]
    all_regimes = _compare_all_regimes(comparison_results)

    # サマリーマトリクス
    lines.append(f"## サマリーマトリクス")
    lines.append(f"")
    header = "| 銘柄 | " + " | ".join(
        f"{REGIME_OPTIONS.get(r, r)}" for r in all_regimes
    ) + " |"
    sep = "|------|" + "|".join("------" for _ in all_regimes) + "|"
    lines.append(header)
    lines.append(sep)

    for rs in comparison_results:
        regime_bests = _get_regime_best_with_viability(rs)
        cells = [rs.symbol]
        for regime in all_regimes:
            verdict, test_entry = _get_oos_verdict_for_symbol(rs.symbol, regime)
            if regime in regime_bests:
                info = regime_bests[regime]
                entry = info["entry"]
                display_entry = test_entry if (verdict and test_entry) else entry
                if verdict:
                    cells.append(f"{verdict} {display_entry.template_name} ({display_entry.composite_score:.3f})")
                else:
                    viable = "OK" if info["is_viable"] else "NG"
                    cells.append(f"{viable} {entry.template_name} ({entry.composite_score:.3f})")
            else:
                cells.append("-")
        lines.append("| " + " | ".join(cells) + " |")

    # レジーム別ベスト
    lines.append(f"")
    lines.append(f"## レジーム別ベスト")
    for regime in all_regimes:
        regime_label = REGIME_OPTIONS.get(regime, regime)
        lines.append(f"")
        lines.append(f"### {regime_label}")
        lines.append(f"")
        lines.append(f"| 銘柄 | テンプレート | PnL | 勝率 | PF | Sharpe | DD | トレード数 |")
        lines.append(f"|------|------------|-----|------|-----|--------|-----|----------|")
        for rs in comparison_results:
            regime_set = rs.filter_regime(regime)
            best = regime_set.best_by_pnl
            if not best:
                continue
            m = best.metrics
            lines.append(
                f"| {rs.symbol} | {best.template_name} | "
                f"{m.total_profit_pct:+.2f}% | {m.win_rate:.1f}% | "
                f"{m.profit_factor:.2f} | {m.sharpe_ratio:.2f} | "
                f"{m.max_drawdown_pct:.1f}% | {m.total_trades} |"
            )

    # 共通性サマリー
    from collections import Counter as MdCounter
    lines.append(f"")
    lines.append(f"## 共通性サマリー")
    for regime in all_regimes:
        regime_label = REGIME_OPTIONS.get(regime, regime)
        bests = {}
        for rs in comparison_results:
            regime_set = rs.filter_regime(regime)
            best = regime_set.best_by_pnl
            if best:
                bests[rs.symbol] = best.template_name
        if not bests:
            continue
        counter = MdCounter(bests.values())
        most_common, count = counter.most_common(1)[0]
        pct = count / len(bests) * 100
        lines.append(f"")
        lines.append(f"### {regime_label}")
        lines.append(f"- 最頻テンプレート: **{most_common}** ({pct:.0f}%, {count}/{len(bests)}銘柄)")

    lines.append(f"")
    return "\n".join(lines)


def _build_cv_results(comparison_results):
    """CVに使うデータソースを決定する。

    OOS検証結果がある場合はOOS Test結果を使い、
    なければTrain結果にフォールバックする。

    Returns:
        (cv_results: list, cv_source: str)
        cv_source は "OOS Test" / "WFA OOS" / "Train"
    """
    from optimizer.results import OptimizationResultSet

    batch_validated = st.session_state.get("batch_validated")
    batch_wfa = st.session_state.get("batch_wfa")

    if batch_validated:
        oos_results = []
        for symbol, validated in batch_validated.items():
            entries = list(validated.test_results.values())
            if entries:
                rs = OptimizationResultSet(entries=entries)
                rs.symbol = symbol
                oos_results.append(rs)
        if oos_results:
            return oos_results, "OOS Test"

    if batch_wfa:
        oos_results = []
        for symbol, wfa_result in batch_wfa.items():
            # 最終フォールド（最大IS量）のOOS結果を使用
            if wfa_result.folds:
                last_fold = wfa_result.folds[-1]
                entries = list(last_fold.oos_results.values())
                if entries:
                    rs = OptimizationResultSet(entries=entries)
                    rs.symbol = symbol
                    oos_results.append(rs)
        if oos_results:
            return oos_results, "WFA OOS"

    return comparison_results, "Train"


def _generate_cv_markdown(comparison_results):
    """クロスバリデーションタブ用Markdown"""
    from optimizer.cross_validation import run_cross_validation

    cv_results, cv_source = _build_cv_results(comparison_results)
    lines = [_compare_md_header(comparison_results)]
    all_regimes = _compare_all_regimes(cv_results)

    cv_result = run_cross_validation(
        batch_results=cv_results,
        target_regimes=all_regimes,
    )
    lines.append(f"## クロスバリデーション ({cv_source})")
    for regime in all_regimes:
        regime_result = cv_result.regime_results.get(regime)
        if not regime_result or not regime_result.strategy_profiles:
            continue
        regime_label = REGIME_OPTIONS.get(regime, regime)
        lines.append(f"")
        lines.append(f"### {regime_label}")
        if regime_result.dominant_template:
            lines.append(
                f"- 最頻テンプレート: **{regime_result.dominant_template}** "
                f"(支配率 {regime_result.dominant_template_ratio:.0%})"
            )
        lines.append(f"")
        lines.append(f"| テンプレート | 判定 | 通過率 | 通過銘柄 | 不合格銘柄 | 平均PnL |")
        lines.append(f"|------------|------|--------|---------|----------|---------|")
        for profile in regime_result.strategy_profiles:
            passed = ", ".join(profile.symbols_passed) if profile.symbols_passed else "-"
            failed = ", ".join(profile.symbols_failed) if profile.symbols_failed else "-"
            lines.append(
                f"| {profile.template_name} | {profile.verdict} | "
                f"{profile.pass_rate:.0%} | {passed} | {failed} | "
                f"{profile.avg_pnl:+.2f}% |"
            )

    lines.append(f"")
    return "\n".join(lines)


def _render_compare_view():
    """比較ビュー（メタ分析 / 詳細比較 タブ分割）"""
    comparison_results = st.session_state.get("comparison_results", [])

    if len(comparison_results) < 2:
        st.info("比較には2つ以上の結果を読み込んでください。「📁 読込」から複数選択できます。")
        return

    symbols = [rs.symbol for rs in comparison_results]
    st.markdown(
        f"**比較対象**: {' / '.join(symbols)} "
        f"({len(comparison_results)}銘柄)"
    )

    # 3銘柄以上ならクロスバリデーションタブを追加
    if len(comparison_results) >= 3:
        tab_meta, tab_detail, tab_cv = st.tabs(
            ["📊 メタ分析", "🔀 詳細比較", "🔬 クロスバリデーション"]
        )
    else:
        tab_meta, tab_detail = st.tabs(["📊 メタ分析", "🔀 詳細比較"])
        tab_cv = None

    with tab_meta:
        _render_md_copy_button(
            _generate_meta_markdown(comparison_results),
            button_id="cpbtn_meta",
        )
        _render_meta_analysis_view(comparison_results)

    with tab_detail:
        _render_md_copy_button(
            _generate_detail_markdown(comparison_results),
            button_id="cpbtn_detail",
        )
        _render_detail_compare_view(comparison_results)

    if tab_cv is not None:
        with tab_cv:
            _render_md_copy_button(
                _generate_cv_markdown(comparison_results),
                button_id="cpbtn_cv",
            )
            _render_cross_validation_tab(comparison_results)


def _render_detail_compare_view(comparison_results):
    """詳細比較ビュー（既存の4セクション）"""
    # --- セクションA: サマリーマトリクス ---
    _render_compare_summary_matrix(comparison_results)
    st.divider()

    # --- セクションB: レジーム別横断比較カード ---
    _render_compare_regime_cards(comparison_results)
    st.divider()

    # --- セクションC: メトリクス比較チャート ---
    _render_compare_metrics_chart(comparison_results)
    st.divider()

    # --- 横断分析 ---
    from ui.components.optimizer_analysis import analyze_comparison
    cross_insights = analyze_comparison(comparison_results)
    _render_analysis_section(cross_insights, title="横断分析", icon="📝")
    st.divider()

    # --- セクションD: 共通性サマリー ---
    _render_compare_commonality(comparison_results)


def _render_meta_analysis_view(comparison_results):
    """メタ分析ビュー: 大量データの統計集約"""
    from collections import Counter as MetaCounter
    from ui.components.optimizer_charts import (
        create_template_adoption_chart,
        create_parameter_boxplot,
        create_symbol_regime_heatmap,
    )
    from ui.components.optimizer_analysis import analyze_meta

    n = len(comparison_results)

    if n < 3:
        st.info("メタ分析には3件以上の結果が必要です。「📁 読込」からさらにファイルを追加してください。")
        return

    all_regimes = sorted(set(
        e.trend_regime
        for rs in comparison_results
        for e in rs.entries
    ))

    # --- 1. 銘柄×レジーム ヒートマップ ---
    section_header("🗺️", "全体鳥瞰", f"{n}銘柄×{len(all_regimes)}レジーム")
    heatmap_fig = create_symbol_regime_heatmap(comparison_results)
    st.plotly_chart(heatmap_fig, use_container_width=True)
    st.divider()

    # --- 2. テンプレート採択分布 ---
    section_header("📊", "テンプレート採択分布", "レジーム別ベスト1位テンプレートの頻度")

    regime_tabs = st.tabs([
        f"{REGIME_ICONS.get(r, '')} {REGIME_OPTIONS.get(r, r)}" for r in all_regimes
    ])

    for tab, regime in zip(regime_tabs, all_regimes):
        with tab:
            adoption_fig = create_template_adoption_chart(comparison_results, regime)
            st.plotly_chart(adoption_fig, use_container_width=True)

            # 最頻テンプレートの情報
            templates = []
            for rs in comparison_results:
                regime_set = rs.filter_regime(regime)
                best = regime_set.best_by_pnl
                if best:
                    templates.append(best.template_name)
            if templates:
                counter = MetaCounter(templates)
                most_common_tpl, most_common_count = counter.most_common(1)[0]
                dominance = most_common_count / len(templates)
                st.caption(
                    f"最頻テンプレート: **{most_common_tpl}** "
                    f"({most_common_count}/{len(templates)}銘柄 = {dominance:.0%})"
                )

    st.divider()

    # --- 3. パラメータ収束 ---
    section_header("🔬", "パラメータ収束", "最頻テンプレートの各パラメータ分布")

    param_regime_tabs = st.tabs([
        f"{REGIME_ICONS.get(r, '')} {REGIME_OPTIONS.get(r, r)}" for r in all_regimes
    ])

    for tab, regime in zip(param_regime_tabs, all_regimes):
        with tab:
            # 最頻テンプレートを特定
            templates = []
            for rs in comparison_results:
                regime_set = rs.filter_regime(regime)
                best = regime_set.best_by_pnl
                if best:
                    templates.append(best.template_name)

            if not templates:
                st.caption("データなし")
                continue

            counter = MetaCounter(templates)
            most_common_tpl, most_common_count = counter.most_common(1)[0]

            if most_common_count < 2:
                st.caption(f"同一テンプレートが2銘柄以上で未使用（最頻: {most_common_tpl} = {most_common_count}件）")
                continue

            boxplot_fig = create_parameter_boxplot(comparison_results, regime, most_common_tpl)
            st.plotly_chart(boxplot_fig, use_container_width=True, key=f"meta_boxplot_{regime}")

            # パラメータ別CV計算
            param_stats = {}
            for rs in comparison_results:
                regime_set = rs.filter_regime(regime)
                best = regime_set.best_by_pnl
                if best and best.template_name == most_common_tpl:
                    for k, v in best.params.items():
                        try:
                            val = float(v)
                            if k not in param_stats:
                                param_stats[k] = []
                            param_stats[k].append(val)
                        except (ValueError, TypeError):
                            pass

            if param_stats:
                cv_rows = []
                for param_name, values in param_stats.items():
                    if len(values) < 2:
                        continue
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    cv = std_val / abs(mean_val) if abs(mean_val) > 1e-10 else 0
                    status = "✅ 収束" if cv <= 0.2 else ("⚠️ 発散" if cv >= 0.5 else "〜 中程度")
                    cv_rows.append({
                        "パラメータ": param_name,
                        "平均": f"{mean_val:.2f}",
                        "標準偏差": f"{std_val:.2f}",
                        "CV": f"{cv:.3f}",
                        "判定": status,
                    })
                if cv_rows:
                    st.dataframe(
                        pd.DataFrame(cv_rows),
                        use_container_width=True,
                        hide_index=True,
                    )

    st.divider()

    # --- 4. メタ自動分析 ---
    meta_insights = analyze_meta(comparison_results)
    _render_analysis_section(meta_insights, title="メタ自動分析", icon="🧠")


def _get_oos_verdict_for_symbol(symbol, regime):
    """バッチOOS検証結果からPASS/CAUTION/FAIL/Noneを返す"""
    batch_validated = st.session_state.get("batch_validated")
    batch_wfa = st.session_state.get("batch_wfa")

    if batch_validated and symbol in batch_validated:
        validated = batch_validated[symbol]
        test_entry = validated.test_results.get(regime)
        has_warnings = any(f"[{regime}]" in w for w in validated.overfitting_warnings)
        test_positive = test_entry and test_entry.metrics.total_profit_pct > 0
        if test_positive and not has_warnings:
            return "PASS", test_entry
        elif test_positive and has_warnings:
            return "CAUTION", test_entry
        else:
            return "FAIL", test_entry
    elif batch_wfa and symbol in batch_wfa:
        wfa = batch_wfa[symbol]
        cr = wfa.consistency_ratio.get(regime, 0)
        wfe = wfa.wfe.get(regime, 0)
        if cr >= 0.6 and wfe > 0.3:
            return "PASS", None
        elif cr >= 0.4:
            return "CAUTION", None
        else:
            return "FAIL", None

    return None, None


def _render_compare_summary_matrix(comparison_results):
    """セクションA: 銘柄×レジーム サマリーマトリクス"""
    batch_validated = st.session_state.get("batch_validated")
    batch_wfa = st.session_state.get("batch_wfa")
    has_validation = bool(batch_validated or batch_wfa)

    if has_validation:
        mode_label = "OOS 3分割" if batch_validated else "Walk-Forward"
        section_header("📋", "サマリーマトリクス", f"銘柄×レジーム ベスト戦略一覧（{mode_label}検証済み）")
    else:
        section_header("📋", "サマリーマトリクス", "銘柄×レジーム ベスト戦略一覧")

    all_regimes = sorted(set(
        e.trend_regime
        for rs in comparison_results
        for e in rs.entries
    ))

    rows = []
    for rs in comparison_results:
        regime_bests = _get_regime_best_with_viability(rs)
        row = {"銘柄": rs.symbol}
        for regime in all_regimes:
            icon = REGIME_ICONS.get(regime, "")
            label = REGIME_OPTIONS.get(regime, regime)
            col_name = f"{icon} {label}"
            if regime in regime_bests:
                info = regime_bests[regime]
                entry = info["entry"]

                # OOS検証結果があればそちらの判定を使う
                verdict, test_entry = _get_oos_verdict_for_symbol(rs.symbol, regime)
                if verdict is not None:
                    verdict_icon = {"PASS": "✅", "CAUTION": "⚠️", "FAIL": "❌"}[verdict]
                    # OOS PASS時はTest期間のエントリーを表示
                    display_entry = test_entry if test_entry else entry
                    row[col_name] = f"{verdict_icon} {display_entry.template_name} ({display_entry.composite_score:.3f})"
                else:
                    viable_icon = "✅" if info["is_viable"] else "❌"
                    row[col_name] = f"{viable_icon} {entry.template_name} ({entry.composite_score:.3f})"
            else:
                row[col_name] = "- データなし"
        rows.append(row)

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_compare_regime_cards(comparison_results):
    """セクションB: レジーム別 横断比較カード"""
    section_header("🏆", "レジーム別 横断比較", "レジームごとに全銘柄のベストを比較")

    all_regimes = sorted(set(
        e.trend_regime
        for rs in comparison_results
        for e in rs.entries
    ))

    tabs = st.tabs([
        f"{REGIME_ICONS.get(r, '')} {REGIME_OPTIONS.get(r, r)}" for r in all_regimes
    ])

    for tab, regime in zip(tabs, all_regimes):
        with tab:
            cols = st.columns(len(comparison_results))
            for col, rs in zip(cols, comparison_results):
                with col:
                    regime_set = rs.filter_regime(regime)
                    best = regime_set.best_by_pnl
                    if not best:
                        st.caption(f"**{rs.symbol}**: データなし")
                        continue
                    _render_compare_card(rs.symbol, best)


def _render_compare_metrics_chart(comparison_results):
    """セクションC: メトリクス比較チャート"""
    from ui.components.optimizer_charts import create_comparison_bar_chart, _METRIC_LABELS

    section_header("📊", "メトリクス比較チャート", "銘柄×レジーム グループ棒グラフ")

    selected_metric = st.selectbox(
        "表示メトリクス",
        options=list(_METRIC_LABELS.keys()),
        format_func=lambda x: _METRIC_LABELS[x],
        key="compare_metric_select",
    )

    fig = create_comparison_bar_chart(comparison_results, selected_metric)
    st.plotly_chart(fig, use_container_width=True)


def _render_compare_commonality(comparison_results):
    """セクションD: 共通性サマリー"""
    section_header("🔗", "共通性サマリー", "銘柄間の戦略共通性を分析")

    all_regimes = sorted(set(
        e.trend_regime
        for rs in comparison_results
        for e in rs.entries
    ))

    for regime in all_regimes:
        icon = REGIME_ICONS.get(regime, "")
        label = REGIME_OPTIONS.get(regime, regime)
        st.markdown(f"**{icon} {label}**")

        # 各銘柄のベストを収集
        bests = []
        for rs in comparison_results:
            regime_set = rs.filter_regime(regime)
            best = regime_set.best_by_pnl
            if best:
                bests.append({"symbol": rs.symbol, "entry": best})

        if len(bests) < 2:
            st.caption("比較可能なデータが不足しています。")
            st.divider()
            continue

        # テンプレート一致率
        templates_used = [b["entry"].template_name for b in bests]
        template_counts = Counter(templates_used)
        most_common_tpl, most_common_count = template_counts.most_common(1)[0]
        match_rate = most_common_count / len(bests) * 100

        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "最頻テンプレート",
                most_common_tpl,
                f"{match_rate:.0f}% 一致 ({most_common_count}/{len(bests)}銘柄)",
            )

        # パラメータ類似度（同テンプレートのもの同士で比較）
        with col2:
            same_tpl = [b for b in bests if b["entry"].template_name == most_common_tpl]
            if len(same_tpl) >= 2:
                _render_param_similarity(same_tpl)
            else:
                st.caption("同一テンプレートが2銘柄以上で未使用")

        # テンプレートが異なる銘柄をハイライト
        outliers = [b["symbol"] for b in bests if b["entry"].template_name != most_common_tpl]
        if outliers:
            st.caption(f"⚠️ 異なるテンプレート: {', '.join(outliers)}")

        st.divider()


def _render_param_similarity(bests_with_same_template):
    """同一テンプレート使用銘柄間のパラメータ類似度"""
    all_params = {}
    for b in bests_with_same_template:
        for k, v in b["entry"].params.items():
            if k not in all_params:
                all_params[k] = []
            try:
                all_params[k].append(float(v))
            except (ValueError, TypeError):
                pass

    rows = []
    for param_name, values in all_params.items():
        if len(values) < 2:
            continue
        mean_val = np.mean(values)
        std_val = np.std(values)
        cv = std_val / mean_val if mean_val != 0 else 0
        similarity_pct = max(0, (1 - cv)) * 100
        rows.append({
            "パラメータ": param_name,
            "値": ", ".join(f"{v:.0f}" for v in values),
            "平均": f"{mean_val:.1f}",
            "一致度": f"{similarity_pct:.0f}%",
        })

    if rows:
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# クロスバリデーションタブ
# ============================================================

def _render_cross_validation_tab(comparison_results):
    """クロスバリデーションタブ: 銘柄横断で戦略の汎用性を判定"""
    from optimizer.cross_validation import (
        run_cross_validation,
        CrossValidationVerdict,
    )

    cv_results, cv_source = _build_cv_results(comparison_results)

    section_header("🔬", "Multi-Symbol Cross-Validation",
                   f"{len(cv_results)}銘柄の横断検証")

    # データソース表示
    if cv_source == "Train":
        st.warning("⚠️ Train期間の生結果です（OOS検証なし → 過学習バイアスの可能性あり）")
    elif cv_source == "OOS Test":
        st.success("✅ OOS Test期間の結果でクロスバリデーション実行")
    elif cv_source == "WFA OOS":
        st.success("✅ WFA 最終フォールドOOS結果でクロスバリデーション実行")

    # 対象レジームを自動検出
    all_regimes = sorted(set(
        e.trend_regime
        for rs in cv_results
        for e in rs.entries
    ))

    if not all_regimes:
        st.info("レジーム情報が見つかりません。")
        return

    # クロスバリデーション実行
    cv_result = run_cross_validation(
        batch_results=cv_results,
        target_regimes=all_regimes,
    )

    # レジーム別の結果表示
    VERDICT_DISPLAY = {
        CrossValidationVerdict.ALL_PASS: ("✅ ALL PASS", "success"),
        CrossValidationVerdict.MAJORITY: ("⚠️ MAJORITY", "warning"),
        CrossValidationVerdict.MINORITY: ("❌ MINORITY", "error"),
        CrossValidationVerdict.FAIL: ("❌ FAIL", "error"),
    }

    for regime in all_regimes:
        regime_result = cv_result.regime_results.get(regime)
        if not regime_result:
            continue

        regime_label = f"{REGIME_ICONS.get(regime, '')} {REGIME_OPTIONS.get(regime, regime)}"
        st.markdown(f"### {regime_label}")

        # 最頻テンプレート情報
        if regime_result.dominant_template:
            st.caption(
                f"最頻テンプレート: **{regime_result.dominant_template}** "
                f"(支配率 {regime_result.dominant_template_ratio:.0%})"
            )

        # 戦略プロファイルテーブル
        profile_rows = []
        for profile in regime_result.strategy_profiles:
            verdict_text, verdict_type = VERDICT_DISPLAY.get(
                profile.verdict,
                (profile.verdict, "info"),
            )
            profile_rows.append({
                "テンプレート": profile.template_name,
                "判定": verdict_text,
                "通過率": f"{profile.pass_rate:.0%}",
                "通過銘柄": ", ".join(profile.symbols_passed) if profile.symbols_passed else "-",
                "不合格銘柄": ", ".join(profile.symbols_failed) if profile.symbols_failed else "-",
                "平均PnL": f"{profile.avg_pnl:+.2f}%",
                "PnL標準偏差": f"{profile.pnl_std:.2f}%",
                "平均Sharpe": f"{profile.avg_sharpe:.2f}",
                "平均勝率": f"{profile.avg_win_rate:.1f}%",
            })

        if profile_rows:
            st.dataframe(
                pd.DataFrame(profile_rows),
                use_container_width=True,
                hide_index=True,
            )

        # 各プロファイルの判定バナー
        for profile in regime_result.strategy_profiles:
            verdict_text, verdict_type = VERDICT_DISPLAY.get(
                profile.verdict,
                (profile.verdict, "info"),
            )
            msg = (
                f"**{profile.template_name}**: {verdict_text} "
                f"({len(profile.symbols_passed)}/{len(profile.symbols_passed) + len(profile.symbols_failed)}銘柄通過)"
            )
            if verdict_type == "success":
                st.success(msg)
            elif verdict_type == "warning":
                st.warning(msg)
            else:
                st.error(msg)

        st.divider()


# ============================================================
# 分析セクション描画
# ============================================================

def _render_analysis_section(insights, title="自動分析", icon="📝"):
    """AnalysisInsightリストをStreamlit UIとして描画"""
    from collections import OrderedDict
    from ui.components.optimizer_analysis import InsightLevel

    section_header(icon, title)

    if not insights:
        st.caption("分析対象のデータが不足しています。")
        return

    CATEGORY_LABELS = {
        "quality_warning": "⚠️ 品質警告",
        "strategy_quality": "📋 戦略品質評価",
        "action": "💡 アクション推奨",
        "cross_pattern": "🔍 パターン検出",
        "cross_risk": "⚠️ リスク評価",
        "cross_action": "💡 推奨アクション",
        "meta_dominance": "👑 テンプレート支配率",
        "meta_convergence": "🔬 パラメータ収束",
        "meta_viability": "📈 レジーム採用率",
        "meta_outlier": "🎯 外れ値銘柄",
    }

    grouped = OrderedDict()
    for insight in insights:
        if insight.category not in grouped:
            grouped[insight.category] = []
        grouped[insight.category].append(insight)

    for category, group_insights in grouped.items():
        st.markdown(f"**{CATEGORY_LABELS.get(category, category)}**")
        for insight in group_insights:
            if insight.level == InsightLevel.SUCCESS:
                st.success(insight.message)
            elif insight.level == InsightLevel.WARNING:
                st.warning(insight.message)
            elif insight.level == InsightLevel.ERROR:
                st.error(insight.message)
            else:
                st.info(insight.message)
