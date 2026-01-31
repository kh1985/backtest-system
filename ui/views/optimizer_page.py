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

    has_results = st.session_state.optimization_result is not None
    has_data = bool(st.session_state.get("datasets"))

    # ビュー切り替え（データ有無によらず表示）
    col_nav1, col_nav2, col_nav3, col_spacer = st.columns([1, 1, 1, 3])
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
    else:
        trimmed_entry = next(
            (e for e in trimmed_list if e["id"] == selected_source), None
        )
        if trimmed_entry:
            active_tf_dict = trimmed_entry["data"]
        else:
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

    wcol1, wcol2, wcol3, wcol4 = st.columns(4)
    with wcol1:
        w_pf = st.slider("損益比率", 0.0, 1.0, 0.3, 0.05, key="opt_w_pf", help="総利益÷総損失の重み")
    with wcol2:
        w_wr = st.slider("勝率", 0.0, 1.0, 0.3, 0.05, key="opt_w_wr", help="勝ちトレード割合の重み")
    with wcol3:
        w_dd = st.slider("最大DD (逆)", 0.0, 1.0, 0.2, 0.05, key="opt_w_dd", help="ドローダウンが小さいほど高評価")
    with wcol4:
        w_sh = st.slider("シャープ比", 0.0, 1.0, 0.2, 0.05, key="opt_w_sh", help="リスクあたりリターンの重み")

    weight_sum = w_pf + w_wr + w_dd + w_sh
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


def _save_results(result_set):
    """最適化結果をCSV・JSONでファイルに保存"""
    import json
    from pathlib import Path
    from datetime import datetime

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # ファイル名: BTCUSDT_exec15m_htf4h_20260131_143000
    sym = result_set.symbol or "UNKNOWN"
    etf = result_set.execution_tf or "?"
    htf = result_set.htf or "none"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
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
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "symbol": sym,
            "execution_tf": etf,
            "htf": htf,
            "total_combinations": result_set.total_combinations,
            "timestamp": ts,
            "results": json_rows,
        }, f, ensure_ascii=False, indent=2)

    return str(csv_path)


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

    # 結果をファイルに自動保存
    saved_path = _save_results(result_set)

    st.success(
        f"**{result_set.total_combinations}** results in **{elapsed:.1f}s** "
        f"(Workers: {n_workers})"
    )
    if saved_path:
        st.caption(f"💾 保存先: `{saved_path}`")

    # 自動で結果ビューに切り替え
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
            best = regime_set.best
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


def _render_load_view():
    """保存済み結果の読み込みビュー"""
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

    file_options = {fp.stem: fp for fp in json_files}

    selected_name = st.selectbox(
        "ファイルを選択",
        options=list(file_options.keys()),
        format_func=lambda x: f"{x} ({file_options[x].stat().st_size / 1024:.0f} KB)",
        key="load_file_select",
    )

    if not selected_name:
        return

    selected_path = file_options[selected_name]

    # メタ情報プレビュー
    with open(selected_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    pcol1, pcol2, pcol3, pcol4 = st.columns(4)
    with pcol1:
        st.metric("シンボル", meta.get("symbol", "?"))
    with pcol2:
        st.metric("実行TF", meta.get("execution_tf", "?"))
    with pcol3:
        st.metric("上位TF", meta.get("htf") or "なし")
    with pcol4:
        st.metric("結果数", f"{len(meta.get('results', [])):,}")

    ts = meta.get("timestamp", "")
    if len(ts) >= 15:
        display_ts = f"{ts[:4]}/{ts[4:6]}/{ts[6:8]} {ts[9:11]}:{ts[11:13]}:{ts[13:15]}"
    else:
        display_ts = ts
    st.caption(f"保存日時: {display_ts}")

    if st.button("📊 この結果を読み込む", type="primary", use_container_width=True):
        try:
            result_set = OptimizationResultSet.from_json(str(selected_path))
            st.session_state.optimization_result = result_set
            st.session_state.optimizer_view = "results"
            st.rerun()
        except Exception as e:
            st.error(f"読み込みエラー: {e}")
