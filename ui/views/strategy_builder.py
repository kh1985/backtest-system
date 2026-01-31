"""
戦略ビルダーページ

UIでインジケーター・条件・決済ルールを設定して戦略を構築する。
"""

from pathlib import Path
from typing import Any, Dict, List

import streamlit as st
import yaml

from indicators.registry import INDICATOR_INFO, INDICATOR_REGISTRY
from strategy.builder import ConfigStrategy, load_strategy_from_yaml


EXAMPLES_DIR = Path(__file__).resolve().parent.parent.parent / "strategy" / "examples"


def render_strategy_builder_page():
    """戦略ビルダーページを描画"""
    st.header("🧩 戦略ビルダー")
    st.caption("インジケーター・条件・決済ルールの設定")

    # セッション初期化
    if "strategy_config" not in st.session_state:
        st.session_state.strategy_config = _default_config()

    col_main, col_side = st.columns([3, 1])

    with col_side:
        _render_strategy_sidebar()

    with col_main:
        _render_strategy_form()


def _render_strategy_sidebar():
    """サイドバー: 保存済み戦略の読み込み"""
    st.subheader("保存済み戦略")

    # YAMLファイル一覧
    if EXAMPLES_DIR.exists():
        yaml_files = list(EXAMPLES_DIR.glob("*.yaml"))
        if yaml_files:
            for f in yaml_files:
                if st.button(f.stem, key=f"load_{f.stem}", use_container_width=True):
                    try:
                        with open(f, "r", encoding="utf-8") as fh:
                            config = yaml.safe_load(fh)
                        st.session_state.strategy_config = config
                        st.rerun()
                    except Exception as e:
                        st.error(f"読み込みエラー: {e}")

    st.divider()

    # YAML直接入力
    with st.expander("YAML読み込み"):
        yaml_text = st.text_area(
            "YAMLを貼り付け",
            height=200,
            placeholder="name: My Strategy\nside: long\n...",
        )
        if st.button("YAMLから読み込み"):
            try:
                config = yaml.safe_load(yaml_text)
                st.session_state.strategy_config = config
                st.rerun()
            except Exception as e:
                st.error(f"不正なYAML: {e}")


def _render_strategy_form():
    """メイン: 戦略設定フォーム"""
    config = st.session_state.strategy_config

    # 基本設定
    col1, col2 = st.columns(2)
    with col1:
        config["name"] = st.text_input(
            "戦略名",
            value=config.get("name", ""),
            placeholder="My Strategy",
        )
    with col2:
        side_options = ["long", "short"]
        current_side = config.get("side", "long")
        config["side"] = st.selectbox(
            "売買方向",
            help="long=買い, short=売り",
            options=side_options,
            index=side_options.index(current_side) if current_side in side_options else 0,
        )

    st.divider()

    # インジケーター設定
    st.subheader("インジケーター")
    _render_indicators_section(config)

    st.divider()

    # エントリー条件
    st.subheader("エントリー条件")
    _render_conditions_section(config)

    st.divider()

    # 決済ルール
    st.subheader("決済ルール")
    _render_exit_section(config)

    st.divider()

    # アクションボタン
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("保存", type="primary"):
            try:
                strategy = ConfigStrategy(config)
                st.session_state.strategy = strategy
                st.session_state.strategy_config = config
                st.success(f"戦略 '{config['name']}' を保存しました")
            except Exception as e:
                st.error(f"エラー: {e}")

    with col2:
        if st.button("YAMLエクスポート"):
            yaml_str = yaml.dump(
                config, default_flow_style=False, allow_unicode=True
            )
            st.code(yaml_str, language="yaml")

    with col3:
        if st.button("リセット"):
            st.session_state.strategy_config = _default_config()
            st.rerun()


def _render_indicators_section(config: dict):
    """インジケーター設定セクション"""
    indicators = config.get("indicators", [])

    for i, ind in enumerate(indicators):
        with st.container():
            cols = st.columns([2, 3, 1])
            with cols[0]:
                ind_type = st.selectbox(
                    "種類",
                    options=list(INDICATOR_INFO.keys()),
                    index=list(INDICATOR_INFO.keys()).index(ind.get("type", "sma"))
                    if ind.get("type") in INDICATOR_INFO else 0,
                    key=f"ind_type_{i}",
                )
                ind["type"] = ind_type

            with cols[1]:
                # パラメータ入力
                info = INDICATOR_INFO.get(ind_type, {})
                params = info.get("params", {})
                for param_name, default_val in params.items():
                    if isinstance(default_val, int):
                        ind[param_name] = st.number_input(
                            param_name,
                            value=ind.get(param_name, default_val),
                            step=1,
                            key=f"ind_{i}_{param_name}",
                        )
                    elif isinstance(default_val, float):
                        ind[param_name] = st.number_input(
                            param_name,
                            value=float(ind.get(param_name, default_val)),
                            step=0.1,
                            key=f"ind_{i}_{param_name}",
                        )
                    elif isinstance(default_val, str):
                        ind[param_name] = st.text_input(
                            param_name,
                            value=ind.get(param_name, default_val),
                            key=f"ind_{i}_{param_name}",
                        )

            with cols[2]:
                if st.button("X", key=f"del_ind_{i}"):
                    indicators.pop(i)
                    st.rerun()

    if st.button("+ インジケーター追加"):
        indicators.append({"type": "sma", "period": 20})
        st.rerun()

    config["indicators"] = indicators


def _render_conditions_section(config: dict):
    """エントリー条件セクション"""
    conditions = config.get("entry_conditions", [])

    logic = st.selectbox(
        "論理結合",
        options=["and", "or"],
        help="and=すべて満たす, or=いずれか満たす",
        index=0 if config.get("entry_logic", "and") == "and" else 1,
    )
    config["entry_logic"] = logic

    for i, cond in enumerate(conditions):
        with st.container():
            cols = st.columns([2, 3, 1])
            with cols[0]:
                cond_type = st.selectbox(
                    "条件タイプ",
                    options=["threshold", "crossover", "candle"],
                    index=["threshold", "crossover", "candle"].index(
                        cond.get("type", "threshold")
                    ),
                    key=f"cond_type_{i}",
                )
                cond["type"] = cond_type

            with cols[1]:
                if cond_type == "threshold":
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        cond["column"] = st.text_input(
                            "カラム",
                            value=cond.get("column", ""),
                            placeholder="rsi_14",
                            key=f"cond_col_{i}",
                            help="インジケーターの出力カラム名",
                        )
                    with c2:
                        cond["operator"] = st.selectbox(
                            "演算子",
                            options=[">", "<", ">=", "<=", "=="],
                            index=[">", "<", ">=", "<=", "=="].index(
                                cond.get("operator", ">")
                            ),
                            key=f"cond_op_{i}",
                        )
                    with c3:
                        cond["value"] = st.number_input(
                            "値",
                            value=float(cond.get("value", 0)),
                            step=0.1,
                            key=f"cond_val_{i}",
                        )

                elif cond_type == "crossover":
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        cond["fast"] = st.text_input(
                            "短期",
                            value=cond.get("fast", ""),
                            placeholder="sma_20",
                            key=f"cond_fast_{i}",
                            help="速い（短期）インジケーター",
                        )
                    with c2:
                        cond["slow"] = st.text_input(
                            "長期",
                            value=cond.get("slow", ""),
                            placeholder="sma_50",
                            key=f"cond_slow_{i}",
                            help="遅い（長期）インジケーター",
                        )
                    with c3:
                        cond["direction"] = st.selectbox(
                            "方向",
                            options=["above", "below"],
                            help="above=上抜け, below=下抜け",
                            key=f"cond_dir_{i}",
                        )

                elif cond_type == "candle":
                    cond["pattern"] = st.selectbox(
                        "パターン",
                        options=["bearish", "bullish"],
                        help="bullish=陽線, bearish=陰線",
                        key=f"cond_pattern_{i}",
                    )

            with cols[2]:
                if st.button("X", key=f"del_cond_{i}"):
                    conditions.pop(i)
                    st.rerun()

    if st.button("+ 条件追加"):
        conditions.append({"type": "threshold", "column": "", "operator": ">", "value": 0})
        st.rerun()

    config["entry_conditions"] = conditions


def _render_exit_section(config: dict):
    """決済ルール設定"""
    exit_config = config.get("exit", {})

    col1, col2, col3 = st.columns(3)
    with col1:
        exit_config["take_profit_pct"] = st.number_input(
            "利確 (%)",
            value=float(exit_config.get("take_profit_pct", 1.0)),
            min_value=0.01,
            step=0.1,
            help="この利益率に達したら利確決済",
        )
    with col2:
        exit_config["stop_loss_pct"] = st.number_input(
            "損切り (%)",
            value=float(exit_config.get("stop_loss_pct", 0.5)),
            min_value=0.01,
            step=0.1,
            help="この損失率に達したら損切り決済",
        )
    with col3:
        timeout = st.number_input(
            "タイムアウト (本数, 0=無効)",
            value=int(exit_config.get("timeout_bars") or 0),
            min_value=0,
            step=10,
            help="指定本数経過後に強制決済。0で無効",
        )
        exit_config["timeout_bars"] = timeout if timeout > 0 else None

    # トレーリングストップ
    trailing = st.number_input(
        "トレーリングストップ (%, 0=無効)",
        value=float(exit_config.get("trailing_stop_pct") or 0),
        min_value=0.0,
        step=0.1,
        help="最高値から指定%下がったら決済。利益を伸ばしつつ損失を限定",
    )
    exit_config["trailing_stop_pct"] = trailing if trailing > 0 else None

    config["exit"] = exit_config


def _default_config() -> dict:
    """デフォルト戦略設定"""
    return {
        "name": "New Strategy",
        "side": "long",
        "indicators": [],
        "entry_conditions": [],
        "entry_logic": "and",
        "exit": {
            "take_profit_pct": 1.0,
            "stop_loss_pct": 0.5,
            "timeout_bars": None,
            "trailing_stop_pct": None,
        },
    }
