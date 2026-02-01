"""
オプティマイザ用チャートコンポーネント

散布図（PF vs 勝率）、エクイティカーブオーバーレイ、比較チャート等。
"""

from collections import Counter
from typing import List, Optional, Dict

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from optimizer.results import OptimizationResultSet, OptimizationEntry


# レジーム別カラー
REGIME_COLORS = {
    "uptrend": "#26a69a",
    "downtrend": "#ef5350",
    "range": "#42a5f5",
    "all": "#ab47bc",
}


def create_scatter_chart(
    result_set: OptimizationResultSet,
    x_metric: str = "profit_factor",
    y_metric: str = "win_rate",
) -> go.Figure:
    """
    散布図を作成（X軸: PF, Y軸: 勝率、色: レジーム、サイズ: スコア）

    Args:
        result_set: 最適化結果セット
        x_metric: X軸メトリクス名
        y_metric: Y軸メトリクス名
    """
    df = result_set.to_dataframe()
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data", showarrow=False)
        return fig

    fig = px.scatter(
        df,
        x=x_metric,
        y=y_metric,
        color="regime",
        size="score",
        hover_data=["template", "params", "trades", "total_pnl", "max_dd"],
        color_discrete_map=REGIME_COLORS,
        title=f"{y_metric} vs {x_metric}",
    )

    fig.update_layout(
        template="plotly_dark",
        height=500,
        xaxis_title=x_metric.replace("_", " ").title(),
        yaxis_title=y_metric.replace("_", " ").title(),
    )

    return fig


def create_equity_overlay(
    entries: List[OptimizationEntry],
    max_entries: int = 10,
) -> go.Figure:
    """
    上位N件のエクイティカーブをオーバーレイ表示

    Args:
        entries: OptimizationEntryリスト（スコア順）
        max_entries: 表示する最大数
    """
    fig = go.Figure()

    for i, entry in enumerate(entries[:max_entries]):
        if entry.backtest_result is None:
            continue

        equity = entry.backtest_result.portfolio.equity_curve
        label = f"#{i+1} {entry.template_name} ({entry.trend_regime})"

        fig.add_trace(go.Scatter(
            y=equity,
            mode="lines",
            name=label,
            line=dict(width=1.5),
            hovertemplate=(
                f"{label}<br>"
                f"Score: {entry.composite_score:.4f}<br>"
                f"Equity: %{{y:.2f}}<br>"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        template="plotly_dark",
        title="Equity Curve Overlay (Top N)",
        height=400,
        xaxis_title="Bar",
        yaxis_title="Equity",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            font=dict(size=10),
        ),
    )

    return fig


def create_regime_distribution_chart(
    result_set: OptimizationResultSet,
) -> go.Figure:
    """レジーム別のスコア分布をボックスプロットで表示"""
    df = result_set.to_dataframe()
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data", showarrow=False)
        return fig

    fig = px.box(
        df,
        x="regime",
        y="score",
        color="regime",
        color_discrete_map=REGIME_COLORS,
        title="Score Distribution by Regime",
    )

    fig.update_layout(
        template="plotly_dark",
        height=350,
        showlegend=False,
    )

    return fig


# メトリクス取得マッピング
_METRIC_GETTERS = {
    "score": lambda e: e.composite_score,
    "profit_factor": lambda e: e.metrics.profit_factor,
    "total_pnl": lambda e: e.metrics.total_profit_pct,
    "win_rate": lambda e: e.metrics.win_rate,
    "sharpe": lambda e: e.metrics.sharpe_ratio,
    "max_dd": lambda e: e.metrics.max_drawdown_pct,
    "trades": lambda e: e.metrics.total_trades,
}

_METRIC_LABELS = {
    "score": "総合スコア",
    "profit_factor": "損益比率 (PF)",
    "total_pnl": "合計損益 (%)",
    "win_rate": "勝率 (%)",
    "sharpe": "シャープ比",
    "max_dd": "最大DD (%)",
    "trades": "取引数",
}


def create_comparison_bar_chart(
    comparison_results: List[OptimizationResultSet],
    metric: str = "score",
) -> go.Figure:
    """
    銘柄×レジーム グループ棒グラフ（比較ビュー用）

    各銘柄のレジーム別ベスト戦略のメトリクスを比較。
    X軸=銘柄, 色=レジーム, Y軸=選択メトリクス。
    """
    getter = _METRIC_GETTERS.get(metric, lambda e: e.composite_score)

    all_regimes = sorted(set(
        e.trend_regime
        for rs in comparison_results
        for e in rs.entries
    ))

    fig = go.Figure()

    for regime in all_regimes:
        symbols = []
        values = []
        for rs in comparison_results:
            regime_set = rs.filter_regime(regime)
            best = regime_set.best
            symbols.append(rs.symbol)
            values.append(getter(best) if best else 0)

        fig.add_trace(go.Bar(
            name=regime,
            x=symbols,
            y=values,
            marker_color=REGIME_COLORS.get(regime, "#888"),
        ))

    fig.update_layout(
        template="plotly_dark",
        barmode="group",
        title=f"銘柄別 {_METRIC_LABELS.get(metric, metric)} 比較",
        xaxis_title="銘柄",
        yaxis_title=_METRIC_LABELS.get(metric, metric),
        height=500,
        legend=dict(title="レジーム"),
    )

    return fig


# ============================================================
# メタ分析チャート
# ============================================================

def create_template_adoption_chart(
    comparison_results: List[OptimizationResultSet],
    regime: str,
) -> go.Figure:
    """
    テンプレート採択分布 棒グラフ

    指定レジームで各銘柄のベスト戦略テンプレートを集計し、
    「どのテンプレートが何回ベスト1位に選ばれたか」を棒グラフで表示。
    """
    templates = []
    for rs in comparison_results:
        regime_set = rs.filter_regime(regime)
        best = regime_set.best
        if best:
            templates.append(best.template_name)

    if not templates:
        fig = go.Figure()
        fig.add_annotation(text="No data", showarrow=False)
        return fig

    counter = Counter(templates)
    names = list(counter.keys())
    counts = list(counter.values())

    # 頻度の多い順でソート
    sorted_pairs = sorted(zip(names, counts), key=lambda x: x[1], reverse=True)
    names = [p[0] for p in sorted_pairs]
    counts = [p[1] for p in sorted_pairs]

    colors = []
    max_count = max(counts)
    for c in counts:
        if c == max_count:
            colors.append(REGIME_COLORS.get(regime, "#58a6ff"))
        else:
            colors.append("#30363d")

    fig = go.Figure(data=[
        go.Bar(
            x=names,
            y=counts,
            marker_color=colors,
            text=counts,
            textposition="auto",
        )
    ])

    regime_label = {"uptrend": "Uptrend", "downtrend": "Downtrend", "range": "Range"}.get(regime, regime)
    fig.update_layout(
        template="plotly_dark",
        title=f"{regime_label}: テンプレート採択分布 ({len(templates)}銘柄)",
        xaxis_title="テンプレート",
        yaxis_title="ベスト1位回数",
        height=400,
        yaxis=dict(dtick=1),
    )

    return fig


def create_parameter_boxplot(
    comparison_results: List[OptimizationResultSet],
    regime: str,
    template_name: str,
) -> go.Figure:
    """
    パラメータ収束 箱ひげ図

    指定レジーム・テンプレートで各銘柄のベストパラメータを収集し、
    各パラメータの分布を箱ひげ図で表示。収束しているか一目でわかる。
    """
    param_values: Dict[str, List[float]] = {}
    param_symbols: Dict[str, List[str]] = {}

    for rs in comparison_results:
        regime_set = rs.filter_regime(regime)
        best = regime_set.best
        if best and best.template_name == template_name:
            for k, v in best.params.items():
                try:
                    val = float(v)
                    if k not in param_values:
                        param_values[k] = []
                        param_symbols[k] = []
                    param_values[k].append(val)
                    param_symbols[k].append(rs.symbol)
                except (ValueError, TypeError):
                    pass

    if not param_values:
        fig = go.Figure()
        fig.add_annotation(text="No data", showarrow=False)
        return fig

    fig = go.Figure()

    for param_name, values in param_values.items():
        symbols = param_symbols[param_name]
        fig.add_trace(go.Box(
            y=values,
            name=param_name,
            boxpoints="all",
            jitter=0.3,
            pointpos=-1.5,
            text=symbols,
            hovertemplate=(
                "%{text}<br>"
                f"{param_name}: %{{y}}<br>"
                "<extra></extra>"
            ),
            marker=dict(color=REGIME_COLORS.get(regime, "#58a6ff")),
        ))

    regime_label = {"uptrend": "Uptrend", "downtrend": "Downtrend", "range": "Range"}.get(regime, regime)
    fig.update_layout(
        template="plotly_dark",
        title=f"{regime_label} / {template_name}: パラメータ分布",
        yaxis_title="値",
        height=400,
    )

    return fig


def create_symbol_regime_heatmap(
    comparison_results: List[OptimizationResultSet],
) -> go.Figure:
    """
    銘柄×レジーム ヒートマップ

    行=銘柄, 列=レジーム, セルの色=ベストスコア。
    全体像を鳥瞰するための annotated heatmap。
    """
    all_regimes = sorted(set(
        e.trend_regime
        for rs in comparison_results
        for e in rs.entries
    ))

    symbols = [rs.symbol for rs in comparison_results]
    z = []
    annotations = []

    for i, rs in enumerate(comparison_results):
        row = []
        for j, regime in enumerate(all_regimes):
            regime_set = rs.filter_regime(regime)
            best = regime_set.best
            if best:
                score = best.composite_score
                row.append(score)
                annotations.append(dict(
                    x=j, y=i,
                    text=f"{score:.3f}",
                    showarrow=False,
                    font=dict(color="white" if score < 0.6 else "black", size=11),
                ))
            else:
                row.append(None)
                annotations.append(dict(
                    x=j, y=i,
                    text="-",
                    showarrow=False,
                    font=dict(color="#484f58", size=11),
                ))
        z.append(row)

    regime_labels = [
        {"uptrend": "Uptrend", "downtrend": "Downtrend", "range": "Range"}.get(r, r)
        for r in all_regimes
    ]

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=regime_labels,
        y=symbols,
        colorscale=[
            [0, "#161b22"],
            [0.3, "#1f6feb"],
            [0.6, "#238636"],
            [1.0, "#f0883e"],
        ],
        zmin=0,
        zmax=1,
        colorbar=dict(title="Score"),
        hoverongaps=False,
    ))

    fig.update_layout(
        template="plotly_dark",
        title="銘柄×レジーム ベストスコア",
        height=max(300, 50 * len(symbols) + 100),
        annotations=annotations,
        xaxis=dict(side="top"),
    )

    return fig


def create_load_preview_chart(
    result_set: "OptimizationResultSet",
) -> go.Figure:
    """
    読込ビュー用プレビューチャート

    レジーム別のスコア散布図（X=PF, Y=勝率, 色=レジーム）を
    コンパクトに表示する。
    """
    df = result_set.to_dataframe()
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data", showarrow=False)
        fig.update_layout(template="plotly_dark", height=300)
        return fig

    fig = px.scatter(
        df,
        x="profit_factor",
        y="win_rate",
        color="regime",
        size="score",
        hover_data=["template", "total_pnl", "trades"],
        color_discrete_map=REGIME_COLORS,
    )

    fig.update_layout(
        template="plotly_dark",
        height=320,
        margin=dict(l=40, r=20, t=30, b=40),
        xaxis_title="PF",
        yaxis_title="勝率 %",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(size=10),
        ),
    )

    return fig


def create_load_preview_score_bars(
    result_set: "OptimizationResultSet",
) -> go.Figure:
    """
    読込ビュー用レジーム別ベストスコア棒グラフ
    """
    regimes = sorted(set(e.trend_regime for e in result_set.entries))
    if not regimes:
        fig = go.Figure()
        fig.add_annotation(text="No data", showarrow=False)
        fig.update_layout(template="plotly_dark", height=200)
        return fig

    labels = []
    scores = []
    colors = []

    for regime in regimes:
        regime_set = result_set.filter_regime(regime)
        best = regime_set.best
        regime_label = {"uptrend": "Uptrend", "downtrend": "Downtrend", "range": "Range"}.get(regime, regime)
        labels.append(regime_label)
        scores.append(best.composite_score if best else 0)
        colors.append(REGIME_COLORS.get(regime, "#888"))

    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=scores,
            marker_color=colors,
            text=[f"{s:.3f}" for s in scores],
            textposition="auto",
        )
    ])

    fig.update_layout(
        template="plotly_dark",
        height=220,
        margin=dict(l=40, r=20, t=10, b=40),
        yaxis_title="Best Score",
        yaxis=dict(range=[0, 1]),
    )

    return fig


MAX_PREVIEW_CANDLES = 3000  # プレビュー表示の最大ローソク足数


def _resample_ohlcv(df: pd.DataFrame, max_candles: int = MAX_PREVIEW_CANDLES) -> tuple:
    """
    ローソク足が多すぎる場合にリサンプリングする。

    Returns:
        (resampled_df, resample_label)  resample_label は "" or "→ 1H" 等
    """
    n = len(df)
    if n <= max_candles:
        return df, ""

    # 適切なリサンプル間隔を決定
    ratio = n / max_candles
    if ratio <= 5:
        rule = "5min"
        label = "→ 5m"
    elif ratio <= 15:
        rule = "15min"
        label = "→ 15m"
    elif ratio <= 60:
        rule = "1h"
        label = "→ 1H"
    elif ratio <= 240:
        rule = "4h"
        label = "→ 4H"
    else:
        rule = "1D"
        label = "→ 1D"

    resampled = (
        df.set_index("datetime")
        .resample(rule)
        .agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        })
        .dropna(subset=["open"])
        .reset_index()
    )

    return resampled, label


def create_ohlcv_preview_chart(
    df: pd.DataFrame,
    title: str = "",
) -> go.Figure:
    """
    読込ビュー用ローソク足 + 出来高プレビュー

    Args:
        df: OHLCVData.df（datetime, open, high, low, close, volume列）
        title: チャートタイトル
    """
    from plotly.subplots import make_subplots

    # 大量データの場合はリサンプリング
    plot_df, resample_label = _resample_ohlcv(df)
    if resample_label:
        title = f"{title}  ({len(df):,}本 {resample_label})"

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25],
    )

    fig.add_trace(
        go.Candlestick(
            x=plot_df["datetime"],
            open=plot_df["open"],
            high=plot_df["high"],
            low=plot_df["low"],
            close=plot_df["close"],
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
            name="Price",
        ),
        row=1, col=1,
    )

    # ベクトル化された色計算（iterrows不使用）
    vol_colors = np.where(
        plot_df["close"].values >= plot_df["open"].values,
        "#26a69a",
        "#ef5350",
    )
    fig.add_trace(
        go.Bar(
            x=plot_df["datetime"],
            y=plot_df["volume"],
            marker_color=vol_colors.tolist(),
            name="Volume",
            opacity=0.6,
        ),
        row=2, col=1,
    )

    fig.update_layout(
        template="plotly_dark",
        height=480,
        margin=dict(l=40, r=20, t=30, b=30),
        title=dict(text=title, font=dict(size=13)) if title else None,
        xaxis_rangeslider_visible=False,
        showlegend=False,
    )

    fig.update_yaxes(title_text="", row=1, col=1)
    fig.update_yaxes(title_text="Vol", row=2, col=1)

    return fig


def create_regime_switching_equity_chart(
    equity_curve: np.ndarray,
    title: str = "レジーム切替バックテスト - エクイティカーブ",
) -> go.Figure:
    """レジーム切替バックテストのエクイティカーブを描画"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(equity_curve))),
        y=equity_curve,
        mode="lines",
        name="Equity",
        line=dict(color="#42a5f5", width=2),
    ))
    fig.update_layout(
        template="plotly_dark",
        title=dict(text=title, font=dict(size=13)),
        xaxis_title="Trade #",
        yaxis_title="Equity",
        height=350,
        margin=dict(l=50, r=20, t=40, b=40),
    )
    return fig
