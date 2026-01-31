"""
Optimizerページ

戦略テンプレート×パラメータのグリッドサーチ自動最適化。
設定 → 結果の2ビュー構成。最適化完了後は自動で結果表示に切り替え。
"""

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
from optimizer.scoring import ScoringWeights
from optimizer.grid import GridSearchOptimizer
from ui.components.styles import section_header, template_tag


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
    st.header("⚡ 戦略オプティマイザー")

    # セッション初期化（ガードより先に実行）
    if "optimization_result" not in st.session_state:
        st.session_state.optimization_result = None
    if "optimizer_view" not in st.session_state:
        st.session_state.optimizer_view = "config"
    if "comparison_results" not in st.session_state:
        st.session_state.comparison_results = []

    has_results = st.session_state.optimization_result is not None
    has_data = bool(st.session_state.get("datasets"))
    n_compare = len(st.session_state.comparison_results)

    # ビュー切り替え（データ有無によらず表示）
    col_nav1, col_nav2, col_nav3, col_nav4, col_spacer = st.columns([1, 1, 1, 1, 2])
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
        st.session_state.opt_data_source_info = {
            "source": "original",
            "period_start": "",
            "period_end": "",
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
            st.session_state.opt_data_source_info = {
                "source": "original",
                "period_start": "",
                "period_end": "",
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
        exec_tf = st.selectbox(
            "実行タイムフレーム",
            options=loaded_tfs,
            index=0,
            key="opt_exec_tf",
            help="バックテストを実行するタイムフレーム",
        )
    with col2:
        htf_options = [tf for tf in loaded_tfs if tf != exec_tf]
        if htf_options:
            htf = st.selectbox(
                "上位タイムフレーム",
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
            "検出方法",
            options=["ma_cross", "adx", "combined"],
            format_func=lambda x: {
                "ma_cross": "MA Cross（移動平均クロス）",
                "adx": "ADX（トレンド強度）",
                "combined": "MA Cross + ADX（複合）",
            }[x],
            key="opt_trend_method",
            help="トレンド/レンジを判定するアルゴリズム",
        )
    with col4:
        target_regimes = st.multiselect(
            "対象レジーム",
            options=list(REGIME_OPTIONS.keys()),
            default=list(REGIME_OPTIONS.keys()),
            format_func=lambda x: f"{REGIME_ICONS.get(x, '')} {REGIME_OPTIONS[x]}",
            key="opt_regimes",
        )

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

    # --- 実行 ---
    total_runs = total_combinations * len(target_regimes)

    # サマリーカード
    scol1, scol2, scol3, scol4 = st.columns(4)
    with scol1:
        st.metric("テンプレート", f"{len(selected_templates)}")
    with scol2:
        st.metric("レジーム", f"{len(target_regimes)}")
    with scol3:
        st.metric("組合せ数", f"{total_combinations:,}")
    with scol4:
        st.metric("総実行数", f"{total_runs:,}")

    if st.button("🚀 最適化実行", type="primary", use_container_width=True):
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
    )


def _render_batch_section(
    exec_tf, htf, trend_method, target_regimes,
    selected_templates, custom_ranges, scoring_weights,
    initial_capital, commission, slippage,
    ma_fast, ma_slow, adx_period, adx_trend_th, adx_range_th,
    n_workers,
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
            )


def _run_batch_optimization(
    targets, exec_tf, htf, trend_method, target_regimes,
    selected_templates, custom_ranges, scoring_weights,
    initial_capital, commission, slippage,
    ma_fast, ma_slow, adx_period, adx_trend_th, adx_range_th,
    n_workers,
):
    """バッチ最適化を順次実行"""
    all_results = []
    n_total = len(targets)

    overall_progress = st.progress(0, text="バッチ実行準備中...")
    status_text = st.empty()

    batch_start = time.time()

    for i, target in enumerate(targets):
        label = target["label"]
        status_text.markdown(f"**[{i+1}/{n_total}]** {label}")
        overall_progress.progress(i / n_total, text=f"[{i+1}/{n_total}] {label}")

        item_progress = st.progress(0, text=f"{label}: 開始...")
        item_start = time.time()

        def on_item_progress(current, total, desc):
            elapsed = time.time() - item_start
            speed = current / elapsed if elapsed > 0 else 0
            item_progress.progress(
                current / total,
                text=f"{label}: {current}/{total} ({speed:.0f} runs/s)",
            )

        result_set = _execute_single_optimization(
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

        _save_results(result_set)
        all_results.append(result_set)

        item_elapsed = time.time() - item_start
        item_progress.progress(1.0, text=f"{label}: ✅ 完了 [{item_elapsed:.1f}s]")

    batch_elapsed = time.time() - batch_start
    overall_progress.progress(1.0, text=f"✅ 全{n_total}件完了 [{batch_elapsed:.1f}s]")
    status_text.empty()

    st.success(f"バッチ完了: {n_total}件 / {batch_elapsed:.1f}s")

    # 比較ビューへ遷移
    st.session_state.comparison_results = all_results
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
        "results": json_rows,
    }
    if result_set.data_source == "trimmed":
        json_meta["data_period"] = {
            "start": result_set.data_period_start,
            "end": result_set.data_period_end,
        }
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


def _run_optimization(
    exec_tf, htf, trend_method, target_regimes,
    selected_templates, custom_ranges, scoring_weights,
    initial_capital, commission, slippage,
    ma_fast, ma_slow, adx_period, adx_trend_th, adx_range_th,
    n_workers=1,
):
    """単一銘柄の最適化実行（UIラッパー）"""
    progress_bar = st.progress(0, text="Starting optimization...")
    start_time = time.time()

    def on_progress(current, total, desc):
        elapsed = time.time() - start_time
        speed = current / elapsed if elapsed > 0 else 0
        progress_bar.progress(
            current / total,
            text=f"⚡ {current}/{total} ({speed:.0f} runs/s) [{elapsed:.1f}s]",
        )

    ds_info = st.session_state.get("opt_data_source_info", {})

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

    elapsed = time.time() - start_time

    st.session_state.optimization_result = result_set
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


def _render_regime_best_summary(result_set):
    """レジーム別ベスト戦略サマリーを描画。採用可能な戦略のdictを返す。"""
    section_header("🏆", "Best per Regime", "レジーム別トップ戦略")

    regimes_in_results = sorted(set(e.trend_regime for e in result_set.entries))

    if not regimes_in_results:
        st.info("結果がありません")
        return {}

    viable = {}
    cols = st.columns(len(regimes_in_results))

    for i, regime in enumerate(regimes_in_results):
        with cols[i]:
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

    # --- レジーム別ベスト戦略サマリー ---
    viable_strategies = _render_regime_best_summary(result_set)

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
            value=0,
            key="result_min_trades",
            help="取引回数が少なすぎる結果を除外。5以上推奨",
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
    }


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

    # 全ファイルのメタ情報をファイル名からパース
    file_meta = {}
    for fp in json_files:
        parsed = _parse_result_filename(fp.stem)
        file_meta[fp.stem] = {**parsed, "path": fp}

    all_symbols = sorted(set(m["symbol"] for m in file_meta.values()))
    all_exec_tfs = sorted(set(m["exec_tf"] for m in file_meta.values()))

    # === 2カラムレイアウト: 左=選択 / 右=プレビュー ===
    left_col, right_col = st.columns([2, 3])

    with left_col:
        # --- フィルター ---
        fc1, fc2 = st.columns(2)
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
        best = regime_set.best
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

    tab_meta, tab_detail = st.tabs(["📊 メタ分析", "🔀 詳細比較"])

    with tab_meta:
        _render_meta_analysis_view(comparison_results)

    with tab_detail:
        _render_detail_compare_view(comparison_results)


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
                best = regime_set.best
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
                best = regime_set.best
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
            st.plotly_chart(boxplot_fig, use_container_width=True)

            # パラメータ別CV計算
            param_stats = {}
            for rs in comparison_results:
                regime_set = rs.filter_regime(regime)
                best = regime_set.best
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


def _render_compare_summary_matrix(comparison_results):
    """セクションA: 銘柄×レジーム サマリーマトリクス"""
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
                    best = regime_set.best
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
            best = regime_set.best
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
