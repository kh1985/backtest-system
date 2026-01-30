"""
トレードテーブルコンポーネント

トレード一覧をDataFrame形式で表示する。
"""

from typing import List

import pandas as pd
import streamlit as st

from engine.position import Trade


def render_trade_table(trades: List[Trade]) -> None:
    """トレード一覧をテーブル表示"""
    if not trades:
        st.info("No trades to display.")
        return

    data = []
    for i, t in enumerate(trades, 1):
        data.append({
            "#": i,
            "Entry Time": t.entry_time,
            "Exit Time": t.exit_time,
            "Side": t.side.upper(),
            "Entry Price": f"{t.entry_price:.6f}",
            "Exit Price": f"{t.exit_price:.6f}",
            "P/L (%)": f"{t.profit_pct:+.2f}%",
            "Duration": f"{t.duration_bars} bars",
            "Exit Type": t.exit_type,
            "Reason": t.reason,
        })

    df = pd.DataFrame(data)

    # P/Lの色付け
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "#": st.column_config.NumberColumn(width="small"),
            "P/L (%)": st.column_config.TextColumn(width="small"),
            "Exit Type": st.column_config.TextColumn(width="small"),
            "Side": st.column_config.TextColumn(width="small"),
        },
    )

    # サマリー
    wins = sum(1 for t in trades if t.profit_pct > 0)
    losses = sum(1 for t in trades if t.profit_pct <= 0)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption(f"Wins: {wins}")
    with col2:
        st.caption(f"Losses: {losses}")
    with col3:
        st.caption(f"Total: {len(trades)}")
