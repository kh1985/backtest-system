"""
データローダーページ

CSV読み込み / 取引所APIからのフェッチ / プレビュー表示
アップロードしたデータはシンボル別に datasets に蓄積。
"""

import streamlit as st
import pandas as pd
from pathlib import Path

from data.csv_loader import TradingViewCSVLoader
from data.binance_loader import BinanceCSVLoader
from data.base import Timeframe
from ui.components.chart import create_candlestick_chart


TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "1d"]


def _add_to_datasets(ohlcv):
    """OHLCVDataを datasets に追加（シンボル別に蓄積）"""
    sym = getattr(ohlcv, "symbol", "UNKNOWN")
    tf_str = ohlcv.timeframe.value if hasattr(ohlcv, "timeframe") else "?"

    if sym not in st.session_state.datasets:
        st.session_state.datasets[sym] = {}
    st.session_state.datasets[sym][tf_str] = ohlcv

    # 後方互換: ohlcv_dict と ohlcv_data も更新
    st.session_state.ohlcv_dict[tf_str] = ohlcv
    st.session_state.ohlcv_data = ohlcv


def render_data_loader_page():
    """データローダーページを描画"""
    st.header("📂 データ読み込み")
    st.caption("OHLCVデータのインポート・プレビュー")

    # ohlcv_dict の初期化 (後方互換)
    if "ohlcv_dict" not in st.session_state:
        st.session_state.ohlcv_dict = {}

    # 読み込み済みデータセットの一覧（ページ上部に表示）
    if st.session_state.datasets:
        _render_datasets_summary()
        st.divider()

    tab_binance, tab_csv = st.tabs(["Binance CSV", "TradingView CSV"])

    with tab_binance:
        _render_binance_tab()

    with tab_csv:
        _render_csv_tab()

    # 選択中データのプレビュー
    if st.session_state.datasets:
        _render_data_preview()


def _render_datasets_summary():
    """全データセットの一覧"""
    st.subheader("📦 読み込み済みデータ")

    datasets = st.session_state.datasets
    rows = []
    for sym, tf_dict in datasets.items():
        for tf_str, ohlcv in tf_dict.items():
            rows.append({
                "シンボル": sym,
                "時間足": tf_str,
                "データ数": f"{ohlcv.bars:,}",
                "開始": str(ohlcv.start_time)[:19] if ohlcv.start_time else "",
                "終了": str(ohlcv.end_time)[:19] if ohlcv.end_time else "",
                "ソース": getattr(ohlcv, "source", ""),
            })

    if rows:
        summary_df = pd.DataFrame(rows)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        # 個別削除
        col_del1, col_del2, col_del3 = st.columns([2, 2, 1])
        with col_del1:
            del_sym = st.selectbox(
                "データセット削除",
                options=[""] + list(datasets.keys()),
                format_func=lambda x: "選択..." if x == "" else x,
                key="del_dataset_sym",
            )
        with col_del2:
            if del_sym:
                st.caption(
                    f"{del_sym}: {', '.join(datasets[del_sym].keys())}"
                )
        with col_del3:
            if del_sym and st.button("🗑 削除", use_container_width=True):
                del st.session_state.datasets[del_sym]
                # ohlcv_dict からも該当シンボルのTFを削除
                for tf_str in list(st.session_state.ohlcv_dict.keys()):
                    ohlcv = st.session_state.ohlcv_dict[tf_str]
                    if getattr(ohlcv, "symbol", "") == del_sym:
                        del st.session_state.ohlcv_dict[tf_str]
                st.rerun()


def _render_csv_tab():
    """CSVインポートタブ（複数TF対応）"""
    st.subheader("TradingView CSV インポート")

    symbol = st.text_input("シンボル", value="", placeholder="WLDUSDT.P")

    st.markdown("**各タイムフレームのCSVを設定してください（不要なTFは空のままでOK）**")

    loader = TradingViewCSVLoader()
    entries = []

    for tf_str in TIMEFRAMES:
        with st.expander(f"{tf_str}", expanded=False):
            col1, col2 = st.columns([3, 1])
            with col1:
                csv_path = st.text_input(
                    "ファイルパス",
                    placeholder=rf"C:\path\to\data_{tf_str}.csv",
                    key=f"csv_path_{tf_str}",
                )
            with col2:
                uploaded = st.file_uploader(
                    "アップロード",
                    type=["csv"],
                    key=f"csv_upload_{tf_str}",
                )

            entries.append((tf_str, csv_path, uploaded))

    if st.button("CSV一括読み込み", type="primary", use_container_width=True):
        loaded_count = 0
        for tf_str, csv_path, uploaded in entries:
            try:
                ohlcv = _load_single_csv(
                    loader, tf_str, csv_path, uploaded, symbol
                )
                if ohlcv:
                    _add_to_datasets(ohlcv)
                    loaded_count += 1
            except Exception as e:
                st.error(f"[{tf_str}] Error: {e}")

        if loaded_count > 0:
            st.success(f"{loaded_count} タイムフレーム読み込み完了")
            st.rerun()
        else:
            st.warning("読み込むCSVがありません。パスを入力するかファイルをアップロードしてください。")


def _load_single_csv(loader, tf_str, csv_path, uploaded, symbol):
    """1つのTFのCSVを読み込む。入力がなければNone"""
    import tempfile
    import os

    tf = Timeframe.from_str(tf_str)

    if uploaded is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = tmp.name
        sym = symbol or uploaded.name.replace(".csv", "")
        ohlcv = loader.load(tmp_path, symbol=sym, timeframe=tf)
        os.unlink(tmp_path)
        return ohlcv

    elif csv_path:
        sym = symbol or loader.detect_symbol_from_filename(csv_path)
        detected_tf = loader.detect_timeframe_from_filename(csv_path)
        if detected_tf:
            tf = detected_tf
        return loader.load(csv_path, symbol=sym, timeframe=tf)

    return None


def _detect_tf_from_data(df: pd.DataFrame):
    """OHLCVデータのタイムスタンプ差分からTFを自動判定"""
    if "datetime" not in df.columns or len(df) < 3:
        return None
    diffs = df["datetime"].head(100).diff().dropna()
    if diffs.empty:
        return None
    mode_diff = diffs.mode().iloc[0]
    minutes = mode_diff.total_seconds() / 60
    tf_map = {
        1: Timeframe.M1, 5: Timeframe.M5, 15: Timeframe.M15,
        60: Timeframe.H1, 240: Timeframe.H4, 1440: Timeframe.D1,
    }
    closest = min(tf_map.keys(), key=lambda x: abs(x - minutes))
    return tf_map[closest]


def _render_binance_tab():
    """Binance Data CSV/ZIPインポート（一括読み込み対応）"""
    st.subheader("Binance CSV/ZIP インポート")
    st.caption("ディレクトリ指定 or ドラッグ＆ドロップで一括読み込み（TFはデータから自動判定）")

    loader = BinanceCSVLoader()

    # --- ディレクトリスキャン ---
    col_dir, col_scan = st.columns([5, 1])
    with col_dir:
        dir_path = st.text_input(
            "ディレクトリパス",
            value="inputdata",
            key="binance_dir",
            help="CSV/ZIPファイルが入っているフォルダのパス",
        )
    with col_scan:
        st.markdown("<br>", unsafe_allow_html=True)
        scan_clicked = st.button("スキャン")

    if scan_clicked and dir_path:
        p = Path(dir_path)
        if not p.is_absolute():
            p = Path.cwd() / p
        if p.exists() and p.is_dir():
            files = sorted(list(p.glob("*.csv")) + list(p.glob("*.zip")))
            if files:
                groups = {}
                for f in files:
                    sym = loader.detect_symbol(f.name)
                    if sym not in groups:
                        groups[sym] = []
                    groups[sym].append(str(f))
                st.session_state.binance_scan = groups
            else:
                st.warning("CSV/ZIPファイルが見つかりません")
        else:
            st.error(f"ディレクトリが見つかりません: {p}")

    # --- ファイルアップロード（複数対応）---
    uploaded = st.file_uploader(
        "またはドラッグ＆ドロップ（複数可）",
        type=["csv", "zip"],
        accept_multiple_files=True,
        key="binance_multi_upload",
    )

    # --- スキャン結果表示＆ロード ---
    scan = st.session_state.get("binance_scan")
    if scan:
        total = sum(len(v) for v in scan.values())
        st.markdown(f"**{total} ファイル / {len(scan)} シンボル検出**")

        selected = st.multiselect(
            "読み込むシンボル",
            options=list(scan.keys()),
            default=list(scan.keys()),
            key="binance_syms",
        )

        if selected:
            rows = []
            for sym in selected:
                for fp in scan[sym]:
                    fname = Path(fp).name
                    tf = loader.detect_timeframe(fname)
                    rows.append({
                        "シンボル": sym,
                        "ファイル": fname,
                        "時間足 (参考)": tf.value if tf else "?",
                    })
            st.dataframe(
                pd.DataFrame(rows), use_container_width=True, hide_index=True
            )
            st.caption("※ TFはCSVデータのタイムスタンプから自動判定します（上記は参考値）")

        col_load, col_clear = st.columns([3, 1])
        with col_load:
            if selected and st.button(
                "選択を読み込み", type="primary", use_container_width=True
            ):
                _bulk_load_paths(loader, scan, selected)
        with col_clear:
            if st.button("クリア", use_container_width=True):
                if "binance_scan" in st.session_state:
                    del st.session_state.binance_scan
                st.rerun()

    elif uploaded:
        st.markdown(f"**{len(uploaded)} ファイル選択済み**")
        if st.button(
            "アップロードを読み込み", type="primary", use_container_width=True
        ):
            _bulk_load_uploads(loader, uploaded)


def _bulk_load_paths(loader, scan, selected_symbols):
    """ディレクトリスキャン結果からの一括読み込み"""
    files = []
    for sym in selected_symbols:
        files.extend(scan[sym])

    progress = st.progress(0, text="読み込み中...")
    loaded = 0
    errors = []

    for i, fp in enumerate(files):
        try:
            ohlcv = loader.load(fp)
            detected_tf = _detect_tf_from_data(ohlcv.df)
            if detected_tf:
                ohlcv.timeframe = detected_tf
            _add_to_datasets(ohlcv)
            loaded += 1
        except Exception as e:
            errors.append(f"[{Path(fp).name}] {e}")
        progress.progress((i + 1) / len(files), text=f"{i+1}/{len(files)}")

    progress.empty()
    for err in errors:
        st.error(err)
    if loaded:
        st.success(f"{loaded} ファイル読み込み完了")
        if "binance_scan" in st.session_state:
            del st.session_state.binance_scan
        st.rerun()


def _bulk_load_uploads(loader, uploaded_files):
    """アップロードファイルからの一括読み込み"""
    import tempfile
    import os

    progress = st.progress(0, text="読み込み中...")
    loaded = 0
    errors = []

    for i, uf in enumerate(uploaded_files):
        try:
            suffix = ".zip" if uf.name.endswith(".zip") else ".csv"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uf.getvalue())
                tmp_path = tmp.name

            sym = loader.detect_symbol(uf.name)
            ohlcv = loader.load(tmp_path, symbol=sym)
            os.unlink(tmp_path)

            detected_tf = _detect_tf_from_data(ohlcv.df)
            if detected_tf:
                ohlcv.timeframe = detected_tf
            _add_to_datasets(ohlcv)
            loaded += 1
        except Exception as e:
            errors.append(f"[{uf.name}] {e}")
        progress.progress(
            (i + 1) / len(uploaded_files), text=f"{i+1}/{len(uploaded_files)}"
        )

    progress.empty()
    for err in errors:
        st.error(err)
    if loaded:
        st.success(f"{loaded} ファイル読み込み完了")
        st.rerun()


def _render_data_preview():
    """選択したデータセットのプレビュー"""
    st.divider()
    st.subheader("📊 データプレビュー")

    datasets = st.session_state.datasets
    symbols = list(datasets.keys())

    # シンボル選択
    selected_sym = st.selectbox(
        "シンボル",
        options=symbols,
        index=0,
        key="preview_symbol",
    )

    tf_dict = datasets[selected_sym]
    tfs = list(tf_dict.keys())

    # TF選択
    selected_tf = st.selectbox(
        "タイムフレーム",
        options=tfs,
        index=0,
        key="preview_tf",
    )

    ohlcv = tf_dict[selected_tf]

    # 基本情報
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("シンボル", ohlcv.symbol)
    with col2:
        st.metric("データ数", f"{ohlcv.bars:,}")
    with col3:
        st.caption(f"開始: {str(ohlcv.start_time)[:19]}" if ohlcv.start_time else "")
    with col4:
        st.caption(f"終了: {str(ohlcv.end_time)[:19]}" if ohlcv.end_time else "")

    # チャート
    max_bars = st.slider(
        "表示本数",
        min_value=50,
        max_value=ohlcv.bars,
        value=min(500, ohlcv.bars),
        key="preview_bars",
        help="チャートに表示するローソク足の本数",
    )
    display_df = ohlcv.df.tail(max_bars).copy()

    fig = create_candlestick_chart(
        display_df,
        title=f"{ohlcv.symbol} - {selected_tf}",
    )
    st.plotly_chart(fig, use_container_width=True)

    # データテーブル
    with st.expander("生データ"):
        st.dataframe(
            ohlcv.df.tail(100),
            use_container_width=True,
            hide_index=True,
        )
