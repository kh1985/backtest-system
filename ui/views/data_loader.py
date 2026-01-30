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
    st.header("📂 Data Loader")
    st.caption("OHLCVデータの読み込み・プレビュー")

    # ohlcv_dict の初期化 (後方互換)
    if "ohlcv_dict" not in st.session_state:
        st.session_state.ohlcv_dict = {}

    # 読み込み済みデータセットの一覧（ページ上部に表示）
    if st.session_state.datasets:
        _render_datasets_summary()
        st.divider()

    tab_csv, tab_binance, tab_exchange = st.tabs(
        ["TradingView CSV", "Binance CSV", "Exchange API"]
    )

    with tab_csv:
        _render_csv_tab()

    with tab_binance:
        _render_binance_tab()

    with tab_exchange:
        _render_exchange_tab()

    # 選択中データのプレビュー
    if st.session_state.datasets:
        _render_data_preview()


def _render_datasets_summary():
    """全データセットの一覧"""
    st.subheader("📦 Loaded Datasets")

    datasets = st.session_state.datasets
    rows = []
    for sym, tf_dict in datasets.items():
        for tf_str, ohlcv in tf_dict.items():
            rows.append({
                "Symbol": sym,
                "TF": tf_str,
                "Bars": f"{ohlcv.bars:,}",
                "Start": str(ohlcv.start_time)[:19] if ohlcv.start_time else "",
                "End": str(ohlcv.end_time)[:19] if ohlcv.end_time else "",
                "Source": getattr(ohlcv, "source", ""),
            })

    if rows:
        summary_df = pd.DataFrame(rows)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        # 個別削除
        col_del1, col_del2, col_del3 = st.columns([2, 2, 1])
        with col_del1:
            del_sym = st.selectbox(
                "Delete dataset",
                options=[""] + list(datasets.keys()),
                format_func=lambda x: "Select..." if x == "" else x,
                key="del_dataset_sym",
            )
        with col_del2:
            if del_sym:
                st.caption(
                    f"{del_sym}: {', '.join(datasets[del_sym].keys())}"
                )
        with col_del3:
            if del_sym and st.button("🗑 Delete", use_container_width=True):
                del st.session_state.datasets[del_sym]
                # ohlcv_dict からも該当シンボルのTFを削除
                for tf_str in list(st.session_state.ohlcv_dict.keys()):
                    ohlcv = st.session_state.ohlcv_dict[tf_str]
                    if getattr(ohlcv, "symbol", "") == del_sym:
                        del st.session_state.ohlcv_dict[tf_str]
                st.rerun()


def _render_csv_tab():
    """CSVインポートタブ（複数TF対応）"""
    st.subheader("TradingView CSV Import")

    symbol = st.text_input("Symbol", value="", placeholder="WLDUSDT.P")

    st.markdown("**各タイムフレームのCSVを設定してください（不要なTFは空のままでOK）**")

    loader = TradingViewCSVLoader()
    entries = []

    for tf_str in TIMEFRAMES:
        with st.expander(f"{tf_str}", expanded=False):
            col1, col2 = st.columns([3, 1])
            with col1:
                csv_path = st.text_input(
                    "File Path",
                    placeholder=rf"C:\path\to\data_{tf_str}.csv",
                    key=f"csv_path_{tf_str}",
                )
            with col2:
                uploaded = st.file_uploader(
                    "Upload",
                    type=["csv"],
                    key=f"csv_upload_{tf_str}",
                )

            entries.append((tf_str, csv_path, uploaded))

    if st.button("Load All CSV", type="primary", use_container_width=True):
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
            st.success(f"{loaded_count} timeframe(s) loaded.")
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


def _render_binance_tab():
    """Binance Data CSV/ZIPインポートタブ"""
    st.subheader("Binance Data CSV/ZIP Import")
    st.caption("data.binance.vision からダウンロードしたCSV/ZIPファイルを読み込みます")

    symbol = st.text_input(
        "Symbol",
        value="",
        placeholder="WLDUSDT",
        key="binance_symbol",
    )

    st.markdown("**各タイムフレームのファイルを設定してください**")

    loader = BinanceCSVLoader()
    entries = []

    for tf_str in TIMEFRAMES:
        with st.expander(f"{tf_str}", expanded=False):
            col1, col2 = st.columns([3, 1])
            with col1:
                file_path = st.text_input(
                    "File Path (CSV or ZIP)",
                    placeholder=rf"C:\path\to\WLDUSDT-{tf_str}-2025-01.csv",
                    key=f"binance_path_{tf_str}",
                )
            with col2:
                uploaded = st.file_uploader(
                    "Upload",
                    type=["csv", "zip"],
                    key=f"binance_upload_{tf_str}",
                )

            entries.append((tf_str, file_path, uploaded))

    if st.button(
        "Load Binance CSV", type="primary", use_container_width=True
    ):
        loaded_count = 0
        for tf_str, file_path, uploaded in entries:
            try:
                ohlcv = _load_single_binance(
                    loader, tf_str, file_path, uploaded, symbol
                )
                if ohlcv:
                    _add_to_datasets(ohlcv)
                    loaded_count += 1
            except Exception as e:
                st.error(f"[{tf_str}] Error: {e}")

        if loaded_count > 0:
            st.success(f"{loaded_count} timeframe(s) loaded.")
            st.rerun()
        else:
            st.warning(
                "読み込むファイルがありません。"
                "パスを入力するかファイルをアップロードしてください。"
            )


def _load_single_binance(loader, tf_str, file_path, uploaded, symbol):
    """1つのTFのBinance CSVを読み込む。入力がなければNone"""
    import tempfile
    import os

    tf = Timeframe.from_str(tf_str)

    if uploaded is not None:
        suffix = ".zip" if uploaded.name.endswith(".zip") else ".csv"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = tmp.name
        sym = symbol or loader.detect_symbol(uploaded.name)
        ohlcv = loader.load(tmp_path, symbol=sym, timeframe=tf)
        os.unlink(tmp_path)
        return ohlcv

    elif file_path:
        sym = symbol or loader.detect_symbol(Path(file_path).name)
        return loader.load(file_path, symbol=sym, timeframe=tf)

    return None


def _render_exchange_tab():
    """取引所APIタブ"""
    st.subheader("Exchange API (ccxt)")

    col1, col2 = st.columns(2)
    with col1:
        exchange = st.selectbox(
            "Exchange",
            options=["mexc", "binance", "bybit"],
            index=0,
        )
        symbol = st.text_input(
            "Symbol (Exchange format)",
            value="WLD/USDT",
            placeholder="BTC/USDT",
        )
    with col2:
        timeframe = st.selectbox(
            "Timeframe ",
            options=TIMEFRAMES,
            index=0,
            key="exchange_tf",
        )
        limit = st.number_input(
            "Bars to fetch",
            min_value=100,
            max_value=10000,
            value=1000,
            step=100,
        )

    col3, col4 = st.columns(2)
    with col3:
        start_date = st.date_input("Start Date")
    with col4:
        end_date = st.date_input("End Date")

    if st.button("Fetch Data", type="primary"):
        try:
            from data.exchange_fetcher import ExchangeFetcher
            from config.settings import API_KEY, API_SECRET

            fetcher = ExchangeFetcher(
                exchange_id=exchange,
                api_key=API_KEY if API_KEY else None,
                api_secret=API_SECRET if API_SECRET else None,
            )
            tf = Timeframe.from_str(timeframe)
            ohlcv = fetcher.load(
                symbol=symbol,
                timeframe=tf,
                limit=limit,
            )

            _add_to_datasets(ohlcv)
            st.success(
                f"Fetched: {ohlcv.symbol} ({ohlcv.timeframe.value}) "
                f"- {ohlcv.bars} bars"
            )
            st.rerun()

        except ImportError:
            st.error("ccxt is not installed. Run: pip install ccxt")
        except Exception as e:
            st.error(f"Error: {e}")


def _render_data_preview():
    """選択したデータセットのプレビュー"""
    st.divider()
    st.subheader("📊 Data Preview")

    datasets = st.session_state.datasets
    symbols = list(datasets.keys())

    # シンボル選択
    selected_sym = st.selectbox(
        "Symbol",
        options=symbols,
        index=0,
        key="preview_symbol",
    )

    tf_dict = datasets[selected_sym]
    tfs = list(tf_dict.keys())

    # TF選択
    selected_tf = st.selectbox(
        "Timeframe",
        options=tfs,
        index=0,
        key="preview_tf",
    )

    ohlcv = tf_dict[selected_tf]

    # 基本情報
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Symbol", ohlcv.symbol)
    with col2:
        st.metric("Bars", f"{ohlcv.bars:,}")
    with col3:
        st.caption(f"Start: {str(ohlcv.start_time)[:19]}" if ohlcv.start_time else "")
    with col4:
        st.caption(f"End: {str(ohlcv.end_time)[:19]}" if ohlcv.end_time else "")

    # チャート
    max_bars = st.slider(
        "Display bars",
        min_value=50,
        max_value=ohlcv.bars,
        value=min(500, ohlcv.bars),
        key="preview_bars",
    )
    display_df = ohlcv.df.tail(max_bars).copy()

    fig = create_candlestick_chart(
        display_df,
        title=f"{ohlcv.symbol} - {selected_tf}",
    )
    st.plotly_chart(fig, use_container_width=True)

    # データテーブル
    with st.expander("Raw Data"):
        st.dataframe(
            ohlcv.df.tail(100),
            use_container_width=True,
            hide_index=True,
        )
