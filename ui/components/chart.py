"""
チャートコンポーネント

Plotlyでインタラクティブなローソク足チャートを描画する。
"""

from typing import Dict, List, Optional

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd


def create_candlestick_chart(
    df: pd.DataFrame,
    title: str = "",
    overlays: Optional[Dict[str, List[str]]] = None,
    show_volume: bool = True,
    trades: Optional[list] = None,
    height: int = 700,
) -> go.Figure:
    """
    ローソク足チャートを生成

    Args:
        df: OHLCVデータ（datetime, open, high, low, close, volume）
        title: チャートタイトル
        overlays: 価格チャートに重ねるカラム名 {"SMA": ["sma_20", "sma_50"]}
        show_volume: 出来高バーを表示するか
        trades: トレードリスト（エントリー/イグジットのマーカー表示用）
        height: チャートの高さ

    Returns:
        plotly Figure
    """
    rows = 2 if show_volume else 1
    row_heights = [0.75, 0.25] if show_volume else [1.0]

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
    )

    # ローソク足
    fig.add_trace(
        go.Candlestick(
            x=df["datetime"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="OHLC",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        ),
        row=1,
        col=1,
    )

    # オーバーレイ（SMA, EMA, BB等）
    colors = [
        "#ff9800", "#2196f3", "#9c27b0", "#4caf50",
        "#f44336", "#00bcd4", "#ffeb3b", "#e91e63",
    ]

    if overlays:
        color_idx = 0
        for group_name, columns in overlays.items():
            for col in columns:
                if col in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df["datetime"],
                            y=df[col],
                            name=col,
                            line=dict(
                                color=colors[color_idx % len(colors)],
                                width=1,
                            ),
                        ),
                        row=1,
                        col=1,
                    )
                    color_idx += 1

    # 出来高
    if show_volume and "volume" in df.columns:
        vol_colors = [
            "#26a69a" if c >= o else "#ef5350"
            for o, c in zip(df["open"], df["close"])
        ]
        fig.add_trace(
            go.Bar(
                x=df["datetime"],
                y=df["volume"],
                name="Volume",
                marker_color=vol_colors,
                opacity=0.5,
            ),
            row=2,
            col=1,
        )

    # トレードマーカー
    if trades:
        _add_trade_markers(fig, trades)

    fig.update_layout(
        title=title,
        height=height,
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    fig.update_xaxes(type="date")

    return fig


def create_indicator_subplot(
    df: pd.DataFrame,
    indicator_columns: List[str],
    title: str = "",
    height: int = 200,
) -> go.Figure:
    """
    インジケーター用サブチャート（RSI, MACD等）

    Args:
        df: インジケーター付きDataFrame
        indicator_columns: 表示するカラム名のリスト
        title: チャートタイトル
        height: チャートの高さ
    """
    fig = go.Figure()

    colors = ["#ff9800", "#2196f3", "#9c27b0", "#4caf50"]

    for i, col in enumerate(indicator_columns):
        if col in df.columns:
            # ヒストグラムっぽいカラム名ならBar、それ以外はLine
            if "histogram" in col:
                fig.add_trace(
                    go.Bar(
                        x=df["datetime"],
                        y=df[col],
                        name=col,
                        marker_color=colors[i % len(colors)],
                        opacity=0.6,
                    )
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=df["datetime"],
                        y=df[col],
                        name=col,
                        line=dict(
                            color=colors[i % len(colors)], width=1.5
                        ),
                    )
                )

    fig.update_layout(
        title=title,
        height=height,
        template="plotly_dark",
        showlegend=True,
        margin=dict(t=30, b=20),
    )

    return fig


def create_equity_curve(
    equity_curve: List[float],
    title: str = "Equity Curve",
    height: int = 300,
) -> go.Figure:
    """エクイティカーブを描画"""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            y=equity_curve,
            mode="lines",
            name="Equity",
            line=dict(color="#2196f3", width=2),
            fill="tozeroy",
            fillcolor="rgba(33, 150, 243, 0.1)",
        )
    )

    # 初期資金ライン
    if equity_curve:
        fig.add_hline(
            y=equity_curve[0],
            line_dash="dash",
            line_color="gray",
            annotation_text="Initial Capital",
        )

    fig.update_layout(
        title=title,
        height=height,
        template="plotly_dark",
        yaxis_title="Equity (USDT)",
        xaxis_title="Trade #",
    )

    return fig


def create_drawdown_chart(
    drawdown_series: List[float],
    title: str = "Drawdown",
    height: int = 200,
) -> go.Figure:
    """ドローダウンチャートを描画"""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            y=[-d for d in drawdown_series],
            mode="lines",
            name="Drawdown",
            line=dict(color="#ef5350", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(239, 83, 80, 0.2)",
        )
    )

    fig.update_layout(
        title=title,
        height=height,
        template="plotly_dark",
        yaxis_title="Drawdown (%)",
    )

    return fig


def _add_trade_markers(fig: go.Figure, trades: list) -> None:
    """チャートにエントリー/イグジットマーカーを追加"""
    entries_x, entries_y = [], []
    exits_win_x, exits_win_y = [], []
    exits_loss_x, exits_loss_y = [], []

    for trade in trades:
        entries_x.append(trade.entry_time)
        entries_y.append(trade.entry_price)

        if trade.profit_pct > 0:
            exits_win_x.append(trade.exit_time)
            exits_win_y.append(trade.exit_price)
        else:
            exits_loss_x.append(trade.exit_time)
            exits_loss_y.append(trade.exit_price)

    # エントリーマーカー
    if entries_x:
        fig.add_trace(
            go.Scatter(
                x=entries_x,
                y=entries_y,
                mode="markers",
                name="Entry",
                marker=dict(
                    symbol="triangle-up",
                    size=10,
                    color="#2196f3",
                    line=dict(width=1, color="white"),
                ),
            ),
            row=1,
            col=1,
        )

    # 勝ちイグジット
    if exits_win_x:
        fig.add_trace(
            go.Scatter(
                x=exits_win_x,
                y=exits_win_y,
                mode="markers",
                name="Exit (Win)",
                marker=dict(
                    symbol="triangle-down",
                    size=10,
                    color="#26a69a",
                    line=dict(width=1, color="white"),
                ),
            ),
            row=1,
            col=1,
        )

    # 負けイグジット
    if exits_loss_x:
        fig.add_trace(
            go.Scatter(
                x=exits_loss_x,
                y=exits_loss_y,
                mode="markers",
                name="Exit (Loss)",
                marker=dict(
                    symbol="triangle-down",
                    size=10,
                    color="#ef5350",
                    line=dict(width=1, color="white"),
                ),
            ),
            row=1,
            col=1,
        )
