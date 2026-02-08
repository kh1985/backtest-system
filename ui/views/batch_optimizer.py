"""
バッチ最適化ビュー

inputdataディレクトリから利用可能なデータをスキャンし、
全銘柄 × 全TF × 全期間のバッチ最適化を実行する。
"""

import copy
import json
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import streamlit as st
import pandas as pd

from data.binance_loader import BinanceCSVLoader
from data.base import OHLCVData
from analysis.trend import TrendDetector, TrendRegime
from optimizer.grid import GridSearchOptimizer
from optimizer.genetic import GeneticOptimizer, GAConfig, GAResult
from optimizer.templates import BUILTIN_TEMPLATES
from optimizer.blacklist import get_blacklist, add_fail_to_blacklist
from optimizer.results import OptimizationResultSet, OptimizationEntry
from optimizer.validation import (
    DataSplitConfig,
    ValidatedResultSet,
    run_validated_optimization,
)
from optimizer.exit_profiles import (
    ATR_PROFILES,
    ATR_COMPACT_PROFILES,
    ATR_TRAILING_PROFILES,
    ATR_HYBRID_PROFILES,
    VWAP_PROFILES,
    BB_PROFILES,
    VWAP_TREND_PROFILES,
    VWAP_RANGE_PROFILES,
    ALL_PROFILES,
)


# =============================================================================
# 定数
# =============================================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
INPUTDATA_DIR = PROJECT_DIR / "inputdata"
RESULTS_DIR = PROJECT_DIR / "results" / "batch"

# トレンド検出設定
TREND_METHOD = "ma_cross"
MA_FAST = 20
MA_SLOW = 50

# バックテスト設定
INITIAL_CAPITAL = 10000.0
COMMISSION_PCT = 0.04
SLIPPAGE_PCT = 0.0
TOP_N_RESULTS = 20

# OOS設定
OOS_TRAIN_PCT = 0.6
OOS_VAL_PCT = 0.2
OOS_TOP_N_FOR_VAL = 10

TARGET_REGIMES = ["uptrend", "downtrend", "range"]


# =============================================================================
# テンプレート日本語名マッピング
# =============================================================================

TEMPLATE_LABELS: Dict[str, Tuple[str, str]] = {
    # (日本語名, 説明)
    # ロング系
    "ma_crossover": ("MAクロス", "SMA fast/slow クロスでロング"),
    "rsi_reversal": ("RSI反発", "RSI売られすぎからの反発"),
    "bb_bounce": ("BBバウンス", "ボリンジャー下限タッチ + RSI反発"),
    "macd_signal": ("MACDシグナル", "MACDラインがシグナルを上抜け"),
    "volume_spike": ("出来高急増", "出来高急増 + 陰線からの反転"),
    "stochastic_reversal": ("ストキャス反転", "Stochastic K/D 売られすぎクロス"),
    "trend_pullback_long": ("押し目買い", "上昇トレンドの押し目でロング"),
    "vwap_touch_long": ("VWAPタッチ", "VWAPラインへの押し目でロング"),
    "vwap_upper1_long": ("VWAP+1σ順張り", "VWAP+1σ到達で順張りロング"),
    "vwap_2sigma_long": ("VWAP-2σ逆張り", "VWAP-2σで逆張りロング"),
    # ショート系
    "ma_crossover_short": ("MAクロス", "SMA fast/slow 下抜けでショート"),
    "rsi_reversal_short": ("RSI反落", "RSI買われすぎからの反落"),
    "bb_bounce_short": ("BBバウンス", "ボリンジャー上限タッチ + RSI高"),
    "macd_signal_short": ("MACDシグナル", "MACDラインがシグナルを下抜け"),
    "volume_spike_short": ("出来高急増", "出来高急増 + 陽線からの反落"),
    "stochastic_reversal_short": ("ストキャス反転", "Stochastic K/D 買われすぎクロス"),
    "trend_pullback_short": ("戻り売り", "下落トレンドの戻りでショート"),
    "vwap_touch_short": ("VWAPタッチ", "VWAPラインへの戻りでショート"),
    "vwap_lower1_short": ("VWAP-1σ順張り", "VWAP-1σ到達で順張りショート"),
    "vwap_2sigma_short": ("VWAP+2σ逆張り", "VWAP+2σで逆張りショート"),
}


def get_template_label(name: str) -> str:
    """テンプレート名から日本語ラベルを取得"""
    if name in TEMPLATE_LABELS:
        label, desc = TEMPLATE_LABELS[name]
        return f"{label} - {desc}"
    return name


# =============================================================================
# Exit戦略カテゴリ定義
# =============================================================================

EXIT_CATEGORIES: Dict[str, Dict[str, Any]] = {
    "atr_compact": {
        "label": "ATRコンパクト（推奨）",
        "description": "SL=ATR×2.0固定、TP 3択（1.5/2.0/3.0）",
        "profiles": ATR_COMPACT_PROFILES,
    },
    "atr": {
        "label": "ATR固定",
        "description": "ATR倍率でTP/SL固定",
        "profiles": ATR_PROFILES,
    },
    "atr_trailing": {
        "label": "ATRトレーリング",
        "description": "ATRベーストレーリング + タイムアウト",
        "profiles": ATR_TRAILING_PROFILES,
    },
    "atr_hybrid": {
        "label": "ATRハイブリッド",
        "description": "ATRトレーリング + ATR SL",
        "profiles": ATR_HYBRID_PROFILES,
    },
    "vwap": {
        "label": "VWAP系",
        "description": "VWAPバンドでTP",
        "profiles": VWAP_PROFILES + VWAP_TREND_PROFILES + VWAP_RANGE_PROFILES,
    },
    "bb": {
        "label": "BB系",
        "description": "BBバンドでTP",
        "profiles": BB_PROFILES,
    },
}


def get_exit_profile_label(profile: Dict[str, Any]) -> str:
    """Exit profileから日本語ラベルを取得"""
    name = profile.get("name", "unknown")

    # ATR固定系
    if "atr_tp" in name and "trail" not in name:
        tp = profile.get("atr_tp_mult", 0)
        sl = profile.get("atr_sl_mult", 0)
        return f"TP={tp}×ATR, SL={sl}×ATR"

    # ATRトレーリング系（タイムアウト付き）
    if "trail" in name and "to" in name:
        trail = profile.get("atr_trailing_mult", 0)
        timeout = profile.get("timeout_bars", 0)
        return f"トレーリング{trail}×ATR, タイムアウト{timeout}bar"

    # ATRハイブリッド系
    if "trail" in name and "sl" in name and "to" not in name:
        trail = profile.get("atr_trailing_mult", 0)
        sl = profile.get("atr_sl_mult", 0)
        return f"トレーリング{trail}×ATR + SL={sl}×ATR"

    # VWAP系
    if "vwap" in name:
        if "trend" in name:
            trail = profile.get("atr_trailing_mult", 0)
            sl = profile.get("atr_sl_mult", 0)
            return f"VWAPトレンド: Trail{trail}×ATR, SL={sl}×ATR"
        elif "range" in name:
            band = profile.get("vwap_band", 0)
            sl = profile.get("atr_sl_mult", 0)
            tp_label = "VWAPライン" if band == 0 else f"±{band}σ"
            return f"VWAPレンジ: TP={tp_label}, SL={sl}×ATR"
        else:
            band = profile.get("vwap_band", 1)
            sl = profile.get("atr_sl_mult", 0)
            return f"TP=VWAP±{band}σ, SL={sl}×ATR"

    # BB系
    if "bb_exit" in name:
        sl = profile.get("atr_sl_mult", 0)
        return f"TP=BBバンド, SL={sl}×ATR"

    return name


# =============================================================================
# データスキャン
# =============================================================================

def scan_inputdata() -> Dict[str, Any]:
    """
    inputdataディレクトリをスキャンして利用可能なデータを取得

    Returns:
        {
            "symbols": ["BTCUSDT", "ETHUSDT", ...],
            "timeframes": ["1m", "15m", "1h", "4h"],
            "periods": ["20240201-20250131", "20250201-20260130"],
            "files": {(symbol, tf, period): path, ...}
        }
    """
    if not INPUTDATA_DIR.exists():
        return {"symbols": [], "timeframes": [], "periods": [], "files": {}}

    symbols: Set[str] = set()
    timeframes: Set[str] = set()
    periods: Set[str] = set()
    files: Dict[Tuple[str, str, str], Path] = {}

    pattern = re.compile(r"^(.+?)-(\d+[mhd])-(\d{8}-\d{8})-merged\.csv$")

    for f in INPUTDATA_DIR.glob("*-merged.csv"):
        match = pattern.match(f.name)
        if match:
            symbol, tf, period = match.groups()
            symbols.add(symbol)
            timeframes.add(tf)
            periods.add(period)
            files[(symbol, tf, period)] = f

    # TFを順序付きでソート
    tf_order = {"1m": 0, "5m": 1, "15m": 2, "30m": 3, "1h": 4, "4h": 5, "1d": 6}
    sorted_tfs = sorted(timeframes, key=lambda x: tf_order.get(x, 99))

    return {
        "symbols": sorted(symbols),
        "timeframes": sorted_tfs,
        "periods": sorted(periods),
        "files": files,
    }


def get_valid_tf_combos(
    selected_tfs: List[str],
    files: Dict[Tuple[str, str, str], Path],
    selected_symbols: List[str],
    selected_periods: List[str],
) -> List[Tuple[str, str]]:
    """
    選択されたTFから有効なTFコンボ（実行TF, HTF）を生成

    条件:
    - 実行TF < HTF（時間軸が小さい方が実行TF）
    - 両方のTFのデータが存在する銘柄・期間の組み合わせがある
    """
    tf_order = {"1m": 0, "5m": 1, "15m": 2, "30m": 3, "1h": 4, "4h": 5, "1d": 6}

    combos = []
    for exec_tf in selected_tfs:
        for htf in selected_tfs:
            if tf_order.get(exec_tf, 99) < tf_order.get(htf, 99):
                # 少なくとも1つの銘柄・期間でデータが揃っているか確認
                has_data = False
                for symbol in selected_symbols:
                    for period in selected_periods:
                        if (symbol, exec_tf, period) in files and (symbol, htf, period) in files:
                            has_data = True
                            break
                    if has_data:
                        break
                if has_data:
                    combos.append((exec_tf, htf))

    return combos


def count_jobs(
    selected_symbols: List[str],
    selected_periods: List[str],
    tf_combos: List[Tuple[str, str]],
    files: Dict[Tuple[str, str, str], Path],
) -> int:
    """実行ジョブ数をカウント"""
    count = 0
    for symbol in selected_symbols:
        for period in selected_periods:
            for exec_tf, htf in tf_combos:
                if (symbol, exec_tf, period) in files and (symbol, htf, period) in files:
                    count += 1
    return count


# =============================================================================
# バッチ実行ロジック
# =============================================================================

def load_symbol_data(
    symbol: str,
    period: str,
    exec_tf: str,
    htf: str,
    files: Dict[Tuple[str, str, str], Path],
) -> Dict[str, OHLCVData]:
    """銘柄データをロード"""
    loader = BinanceCSVLoader()
    tf_dict = {}

    exec_path = files.get((symbol, exec_tf, period))
    htf_path = files.get((symbol, htf, period))

    if exec_path:
        tf_dict[exec_tf] = loader.load(str(exec_path), symbol=symbol)
    if htf_path:
        tf_dict[htf] = loader.load(str(htf_path), symbol=symbol)

    return tf_dict


def prepare_exec_df(
    tf_dict: Dict[str, OHLCVData],
    exec_tf: str,
    htf: str,
    trend_method: str = "ma_cross",
    super_htf_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """実行TFのDataFrameにトレンドラベルを付与"""
    exec_ohlcv = tf_dict[exec_tf]
    exec_df = exec_ohlcv.df.copy()

    if htf and htf in tf_dict:
        htf_ohlcv = tf_dict[htf]
        htf_df = htf_ohlcv.df.copy()
        detector = TrendDetector()

        if trend_method == "dual_tf_ema" and super_htf_df is not None:
            htf_df = detector.detect_dual_tf_ema(
                htf_df, super_htf_df, fast_period=MA_FAST, slow_period=MA_SLOW
            )
        else:
            htf_df = detector.detect_ma_cross(
                htf_df, fast_period=MA_FAST, slow_period=MA_SLOW
            )
        exec_df = TrendDetector.label_execution_tf(exec_df, htf_df)
    else:
        exec_df["trend_regime"] = TrendRegime.RANGE.value

    return exec_df


def generate_all_configs(
    selected_templates: List[str],
    selected_exit_profiles: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """選択されたテンプレート × exit profiles の config リストを生成"""
    all_configs = []

    for tname in selected_templates:
        if tname not in BUILTIN_TEMPLATES:
            continue
        template = BUILTIN_TEMPLATES[tname]

        if selected_exit_profiles:
            configs = template.generate_configs(exit_profiles=selected_exit_profiles)
        else:
            configs = template.generate_configs()

        all_configs.extend(configs)

    return all_configs


def run_single_ga_optimization(
    exec_df: pd.DataFrame,
    templates: List[str],
    target_regimes: List[str],
    ga_config: GAConfig,
    symbol: str = "UNKNOWN",
    progress_callback=None,
    use_oos: bool = False,
    train_ratio: float = 0.8,
    use_blacklist: bool = True,
) -> List[GAResult]:
    """
    GA最適化を実行（OOS検証オプション付き）

    Args:
        exec_df: トレンドラベル付きのDataFrame
        templates: 使用するテンプレート名リスト
        target_regimes: 対象レジームリスト
        ga_config: GA設定
        symbol: 銘柄名
        progress_callback: 進捗コールバック
        use_oos: OOS検証を行うかどうか
        train_ratio: Train期間の割合（デフォルト80%）
        use_blacklist: ブラックリスト参照するかどうか

    Returns:
        List[GAResult]: 各レジームのGA結果リスト
    """
    from optimizer.scoring import ScoringWeights

    # OOS用にデータ分割
    n_rows = len(exec_df)
    if use_oos:
        split_idx = int(n_rows * train_ratio)
        train_df = exec_df.iloc[:split_idx].copy()
        test_df = exec_df.iloc[split_idx:].copy()
    else:
        train_df = exec_df
        test_df = None

    optimizer = GeneticOptimizer(
        templates=templates,
        ga_config=ga_config,
        scoring_weights=ScoringWeights(),
        initial_capital=INITIAL_CAPITAL,
        commission_pct=COMMISSION_PCT,
        slippage_pct=SLIPPAGE_PCT,
    )

    results = []
    total_regimes = len(target_regimes)

    for i, regime in enumerate(target_regimes):
        # ブラックリストフィルタリング（有効な場合のみ）
        if use_blacklist:
            blacklist = get_blacklist()
            filtered_templates = blacklist.filter_templates(templates, symbol, regime)

            if not filtered_templates:
                # 全テンプレートがブラックリスト済みの場合はスキップ
                continue

            # テンプレートを更新
            optimizer.templates = filtered_templates
        else:
            optimizer.templates = templates

        def ga_progress(gen, max_gen, desc):
            if progress_callback:
                base = i * ga_config.max_generations
                current = base + gen
                total = total_regimes * ga_config.max_generations
                progress_callback(current, total, f"GA [{regime}] 世代 {gen}/{max_gen}")

        # TrainデータでGA探索
        ga_result = optimizer.run(
            df=train_df,
            target_regime=regime,
            symbol=symbol,
            progress_callback=ga_progress,
        )

        # OOS検証: 最終勝者をTestデータで評価
        if use_oos and test_df is not None and ga_result.final_winner:
            winner = ga_result.final_winner
            train_pnl = winner.get("pnl", 0)

            # Test期間で同じ戦略を評価
            from optimizer.genetic import Individual
            test_ind = Individual(
                template_name=winner["template"],
                params=winner["params"],
                generation=0,
            )

            try:
                config = test_ind.to_config()
                entry = optimizer._grid_optimizer._run_single(
                    df=test_df,
                    config=config,
                    template_name=test_ind.template_name,
                    params=test_ind.params,
                    target_regime=regime,
                    trend_column="trend_label",
                )
                test_pnl = entry.metrics.total_profit_pct
                test_trades = entry.metrics.total_trades
            except Exception:
                test_pnl = 0.0
                test_trades = 0

            # 判定: Test PnL > 0 かつ Train PnLの30%以上維持でPASS
            if test_pnl > 0 and (train_pnl <= 0 or test_pnl >= train_pnl * 0.3):
                verdict = "PASS"
            else:
                verdict = "FAIL"
                # FAILをブラックリストに自動登録
                template_name = ga_result.final_winner.get("template", "")
                if template_name:
                    add_fail_to_blacklist(
                        symbol=symbol,
                        regime=regime,
                        template=template_name,
                        test_pnl=test_pnl,
                        train_pnl=train_pnl,
                    )

            # GA結果にOOS情報を追加
            ga_result.final_winner["train_pnl"] = train_pnl
            ga_result.final_winner["test_pnl"] = test_pnl
            ga_result.final_winner["test_trades"] = test_trades
            ga_result.final_winner["verdict"] = verdict

        results.append(ga_result)

    return results


def run_single_optimization(
    exec_df: pd.DataFrame,
    all_configs: List[Dict[str, Any]],
    use_oos: bool,
    n_workers: int,
    target_regimes: List[str] = None,
    progress_callback=None,
    adaptive_config: Optional["AdaptiveSearchConfig"] = None,
) -> Any:
    """
    単一銘柄の最適化を実行

    Args:
        adaptive_config: 適応型探索の設定（Noneの場合は従来の総当たり）
    """
    from optimizer.adaptive import AdaptiveSearchConfig

    if target_regimes is None:
        target_regimes = TARGET_REGIMES

    optimizer = GridSearchOptimizer(
        initial_capital=INITIAL_CAPITAL,
        commission_pct=COMMISSION_PCT,
        slippage_pct=SLIPPAGE_PCT,
        top_n_results=TOP_N_RESULTS,
    )

    configs = copy.deepcopy(all_configs)

    # 適応型探索が有効な場合
    if adaptive_config is not None:
        # 適応型探索はOOS検証と組み合わせない（Scoutで十分なフィルタリング）
        adaptive_result = optimizer.run_adaptive(
            df=exec_df,
            configs=configs,
            target_regimes=target_regimes,
            adaptive_config=adaptive_config,
            n_workers=n_workers,
            progress_callback=progress_callback,
        )
        # 互換性のためOptimizationResultSetを返す
        return adaptive_result.final_result

    # 従来の総当たり探索
    if use_oos:
        split_config = DataSplitConfig(
            train_pct=OOS_TRAIN_PCT,
            val_pct=OOS_VAL_PCT,
            top_n_for_val=OOS_TOP_N_FOR_VAL,
        )
        result = run_validated_optimization(
            df=exec_df,
            all_configs=configs,
            target_regimes=target_regimes,
            split_config=split_config,
            optimizer=optimizer,
            n_workers=n_workers,
            progress_callback=progress_callback,
        )
    else:
        result = optimizer.run(
            df=exec_df,
            configs=configs,
            target_regimes=target_regimes,
            n_workers=n_workers,
            progress_callback=progress_callback,
        )

    return result


def save_batch_result(
    result: Any,
    symbol: str,
    period: str,
    exec_tf: str,
    htf: str,
    use_oos: bool,
    output_dir: Path,
) -> Path:
    """最適化結果をJSONで保存"""
    def _safe_float(v: float) -> float:
        import math
        if math.isinf(v) or math.isnan(v):
            return 9999.0 if v > 0 else -9999.0 if v < 0 else 0.0
        return v

    def _entry_to_dict(e: OptimizationEntry) -> Dict[str, Any]:
        return {
            "template": e.template_name,
            "params": e.params,
            "regime": e.trend_regime,
            "exit_profile": e.config.get("_exit_profile", "default"),
            "score": round(_safe_float(e.composite_score), 4),
            "metrics": {
                "trades": e.metrics.total_trades,
                "win_rate": round(_safe_float(e.metrics.win_rate), 1),
                "profit_factor": round(_safe_float(e.metrics.profit_factor), 2),
                "total_pnl": round(_safe_float(e.metrics.total_profit_pct), 2),
                "max_dd": round(_safe_float(e.metrics.max_drawdown_pct), 2),
                "sharpe": round(_safe_float(e.metrics.sharpe_ratio), 2),
            },
        }

    data: Dict[str, Any] = {
        "symbol": symbol,
        "period": period,
        "execution_tf": exec_tf,
        "htf": htf,
        "oos": use_oos,
    }

    if use_oos and isinstance(result, ValidatedResultSet):
        data["train_results"] = [
            _entry_to_dict(e)
            for e in result.train_results.ranked()[:TOP_N_RESULTS]
        ]
        data["test_results"] = {}
        for regime, entry in result.test_results.items():
            data["test_results"][regime] = _entry_to_dict(entry)
        data["val_best"] = {}
        for regime, entry in result.val_best.items():
            data["val_best"][regime] = _entry_to_dict(entry)
        data["warnings"] = result.overfitting_warnings
        data["total_combinations"] = result.train_results.total_combinations
    else:
        result_set = result
        data["total_combinations"] = result_set.total_combinations
        data["results"] = [
            _entry_to_dict(e)
            for e in result_set.ranked()[:TOP_N_RESULTS]
        ]

    fname = f"{symbol}_{period}_{exec_tf}_{htf}.json"
    json_path = output_dir / fname
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return json_path


# =============================================================================
# バッチビューUI
# =============================================================================

def render_batch_view():
    """バッチ最適化ビューを描画"""
    st.subheader("🚀 バッチ最適化")
    st.caption("inputdataから全銘柄×全TF×全期間を一括最適化")

    # GA採用候補からの設定がある場合は通知
    if st.session_state.get("batch_from_ga_candidates"):
        candidates = st.session_state.get("ga_adopted_candidates", {})
        source = candidates.get("source", "")
        symbols = candidates.get("symbols", [])
        templates = candidates.get("templates", [])

        st.info(f"🎯 **GA採用候補からの設定** (ソース: {source})\n\n"
                f"銘柄: {len(symbols)}種 | テンプレート: {len(templates)}種")

        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("❌ 設定をクリア", key="clear_ga_candidates"):
                st.session_state.batch_from_ga_candidates = False
                st.session_state.ga_adopted_candidates = {}
                st.rerun()

    # データスキャン
    scan_result = scan_inputdata()

    if not scan_result["symbols"]:
        st.warning("📂 inputdataディレクトリにデータがありません")
        st.info(f"パス: `{INPUTDATA_DIR}`")
        return

    # セッション初期化
    if "batch_running" not in st.session_state:
        st.session_state.batch_running = False
    if "batch_progress" not in st.session_state:
        st.session_state.batch_progress = {"current": 0, "total": 0, "status": ""}

    # 実行中の場合は進捗表示
    if st.session_state.batch_running:
        _render_batch_progress()
        return

    # --- データ選択 ---
    with st.expander("📦 データ選択", expanded=True):
        _render_data_selection(scan_result)

    # --- タイムフレーム ---
    with st.expander("⏱️ タイムフレーム", expanded=True):
        _render_timeframe_selection(scan_result)

    # --- 戦略テンプレート ---
    with st.expander("🎯 戦略テンプレート", expanded=True):
        _render_template_selection()

    # --- Exit戦略 ---
    with st.expander("🚪 Exit戦略", expanded=True):
        _render_exit_selection()

    # --- 実行設定 ---
    with st.expander("⚙️ 実行設定", expanded=True):
        _render_execution_settings()

    # --- 実行サマリー ---
    _render_execution_summary(scan_result)

    # --- 自動実行チェック（GA候補からの遷移時） ---
    if st.session_state.get("batch_auto_start"):
        st.session_state.batch_auto_start = False  # フラグクリア
        st.info("🚀 GA採用候補のグリッドサーチを自動開始します...")
        _start_batch_optimization(scan_result)
        return

    # --- 実行ボタン ---
    st.divider()

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        is_running = st.session_state.get("batch_running", False)
        btn_label = "⏳ 実行中..." if is_running else "🚀 バッチ最適化を実行"
        if st.button(
            btn_label,
            type="primary",
            use_container_width=True,
            disabled=is_running,
        ):
            _start_batch_optimization(scan_result)


def _render_data_selection(scan_result: Dict[str, Any]):
    """データ選択UIを描画"""
    # 期間選択
    st.markdown("**期間**")
    periods = scan_result["periods"]

    # 期間から年を抽出してラベル化
    period_labels = {}
    for p in periods:
        start_year = p[:4]
        end_year = p[9:13]
        if start_year == end_year:
            period_labels[p] = f"{start_year}年"
        else:
            period_labels[p] = f"{start_year}-{end_year}年"

    cols = st.columns(len(periods))
    selected_periods = []
    for i, period in enumerate(periods):
        with cols[i]:
            if st.checkbox(
                period_labels[period],
                value=True,
                key=f"batch_period_{period}",
            ):
                selected_periods.append(period)

    st.session_state.batch_selected_periods = selected_periods

    # 銘柄選択
    st.markdown("**銘柄**")
    symbols = scan_result["symbols"]

    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        if st.button("全選択", key="batch_select_all_symbols", use_container_width=True):
            for sym in symbols:
                st.session_state[f"batch_symbol_{sym}"] = True
            st.rerun()
    with col2:
        if st.button("全解除", key="batch_deselect_all_symbols", use_container_width=True):
            for sym in symbols:
                st.session_state[f"batch_symbol_{sym}"] = False
            st.rerun()

    # 銘柄チェックボックス（グリッド表示）
    n_cols = 6
    selected_symbols = []
    rows = [symbols[i:i + n_cols] for i in range(0, len(symbols), n_cols)]

    # GA採用候補からの銘柄リスト
    ga_candidates = st.session_state.get("ga_adopted_candidates", {})
    ga_symbols = ga_candidates.get("symbols", [])
    from_ga = st.session_state.get("batch_from_ga_candidates", False)

    for row in rows:
        cols = st.columns(n_cols)
        for i, sym in enumerate(row):
            with cols[i]:
                # GA採用候補モードの場合は、候補の銘柄だけをデフォルトON
                if from_ga and ga_symbols:
                    default_val = sym in ga_symbols
                else:
                    default_val = st.session_state.get(f"batch_symbol_{sym}", True)
                if st.checkbox(
                    sym.replace("USDT", ""),
                    value=default_val,
                    key=f"batch_symbol_{sym}",
                ):
                    selected_symbols.append(sym)

    st.session_state.batch_selected_symbols = selected_symbols
    ga_info = f" (GA候補: {len(ga_symbols)})" if from_ga and ga_symbols else ""
    st.caption(f"選択中: {len(selected_symbols)} / {len(symbols)} 銘柄{ga_info}")


def _render_timeframe_selection(scan_result: Dict[str, Any]):
    """タイムフレーム選択UIを描画"""
    st.markdown("**実行タイムフレーム** (小さい方が実行TF、大きい方がHTFになります)")
    st.caption("⚠️ 1mは処理が重いため、必要な場合のみ選択してください")

    timeframes = scan_result["timeframes"]

    cols = st.columns(len(timeframes))
    selected_tfs = []
    for i, tf in enumerate(timeframes):
        with cols[i]:
            # 1m, 4hはデフォルトOFF
            default_val = tf not in ("1m", "4h")
            if st.checkbox(
                tf,
                value=st.session_state.get(f"batch_tf_{tf}", default_val),
                key=f"batch_tf_{tf}",
            ):
                selected_tfs.append(tf)

    st.session_state.batch_selected_tfs = selected_tfs

    # 有効なTFコンボを計算して表示
    files = scan_result["files"]
    selected_symbols = st.session_state.get("batch_selected_symbols", [])
    selected_periods = st.session_state.get("batch_selected_periods", [])

    tf_combos = get_valid_tf_combos(selected_tfs, files, selected_symbols, selected_periods)
    st.session_state.batch_tf_combos = tf_combos

    if tf_combos:
        combo_strs = [f"{e}/{h}" for e, h in tf_combos]
        st.success(f"✅ 有効なTFコンボ: {', '.join(combo_strs)}")
    else:
        st.warning("⚠️ 有効なTFコンボがありません（2つ以上のTFを選択してください）")


def _render_template_selection():
    """テンプレート選択UIを描画"""
    # ボタン行
    col1, col2, col3, col4 = st.columns([1, 1, 1, 3])
    with col1:
        if st.button("全選択", key="batch_select_all_templates", use_container_width=True):
            for tname in BUILTIN_TEMPLATES:
                st.session_state[f"batch_template_{tname}"] = True
            st.rerun()
    with col2:
        if st.button("全解除", key="batch_deselect_all_templates", use_container_width=True):
            for tname in BUILTIN_TEMPLATES:
                st.session_state[f"batch_template_{tname}"] = False
            st.rerun()
    with col3:
        if st.button("ロングのみ", key="batch_long_only", use_container_width=True):
            for tname in BUILTIN_TEMPLATES:
                st.session_state[f"batch_template_{tname}"] = "short" not in tname
            st.rerun()
    with col4:
        if st.button("ショートのみ", key="batch_short_only", use_container_width=True):
            for tname in BUILTIN_TEMPLATES:
                st.session_state[f"batch_template_{tname}"] = "short" in tname
            st.rerun()

    # ブラックリストからテンプレートごとの件数を取得
    blacklist = get_blacklist()
    bl_summary = blacklist.summary()
    bl_by_template = bl_summary.get("by_template", {})

    # GA採用候補からのテンプレートリスト
    ga_candidates = st.session_state.get("ga_adopted_candidates", {})
    ga_templates = ga_candidates.get("templates", [])
    from_ga = st.session_state.get("batch_from_ga_candidates", False)

    # ロング系
    st.markdown("**📈 ロング系**")
    long_templates = [t for t in BUILTIN_TEMPLATES if "short" not in t]
    selected_templates = []

    for tname in long_templates:
        label = get_template_label(tname)
        bl_count = bl_by_template.get(tname, 0)
        if bl_count > 0:
            label = f"{label} (BL: {bl_count})"
        # GA採用候補モードの場合は、候補のテンプレートだけをデフォルトON
        if from_ga and ga_templates:
            default_val = tname in ga_templates
        else:
            default_val = st.session_state.get(f"batch_template_{tname}", True)
        if st.checkbox(
            label,
            value=default_val,
            key=f"batch_template_{tname}",
        ):
            selected_templates.append(tname)

    # ショート系
    st.markdown("**📉 ショート系**")
    short_templates = [t for t in BUILTIN_TEMPLATES if "short" in t]

    for tname in short_templates:
        label = get_template_label(tname)
        bl_count = bl_by_template.get(tname, 0)
        if bl_count > 0:
            label = f"{label} (BL: {bl_count})"
        # GA採用候補モードの場合は、候補のテンプレートだけをデフォルトON
        if from_ga and ga_templates:
            default_val = tname in ga_templates
        else:
            default_val = st.session_state.get(f"batch_template_{tname}", True)
        if st.checkbox(
            label,
            value=default_val,
            key=f"batch_template_{tname}",
        ):
            selected_templates.append(tname)

    st.session_state.batch_selected_templates = selected_templates

    # ブラックリスト合計も表示
    total_bl = bl_summary.get("total", 0)
    bl_info = f" | BL合計: {total_bl}件" if total_bl > 0 else ""
    ga_info = f" | GA候補: {len(ga_templates)}" if from_ga and ga_templates else ""
    st.caption(f"選択中: {len(selected_templates)} / {len(BUILTIN_TEMPLATES)} テンプレート{bl_info}{ga_info}")


def _render_exit_selection():
    """Exit戦略選択UIを描画"""
    selected_profiles = []

    for cat_key, cat_info in EXIT_CATEGORIES.items():
        profiles = cat_info["profiles"]
        label = cat_info["label"]
        desc = cat_info["description"]

        # カテゴリヘッダー
        col1, col2, col3, col4 = st.columns([3, 1, 1, 2])
        with col1:
            st.markdown(f"**{label}** ({len(profiles)}) - {desc}")
        with col2:
            if st.button("全ON", key=f"batch_exit_cat_on_{cat_key}", use_container_width=True):
                for p in profiles:
                    st.session_state[f"batch_exit_{p['name']}"] = True
                st.rerun()
        with col3:
            if st.button("全OFF", key=f"batch_exit_cat_off_{cat_key}", use_container_width=True):
                for p in profiles:
                    st.session_state[f"batch_exit_{p['name']}"] = False
                st.rerun()

        # プロファイル一覧（最大5行表示、それ以上は折りたたみ）
        show_all = len(profiles) <= 5
        display_profiles = profiles if show_all else profiles[:3]

        for p in display_profiles:
            pname = p["name"]
            plabel = get_exit_profile_label(p)
            default_val = st.session_state.get(f"batch_exit_{pname}", True)
            if st.checkbox(
                plabel,
                value=default_val,
                key=f"batch_exit_{pname}",
            ):
                selected_profiles.append(p)

        if not show_all:
            with st.expander(f"さらに {len(profiles) - 3} 件を表示"):
                for p in profiles[3:]:
                    pname = p["name"]
                    plabel = get_exit_profile_label(p)
                    default_val = st.session_state.get(f"batch_exit_{pname}", True)
                    if st.checkbox(
                        plabel,
                        value=default_val,
                        key=f"batch_exit_{pname}",
                    ):
                        selected_profiles.append(p)

        st.divider()

    st.session_state.batch_selected_exit_profiles = selected_profiles
    total_profiles = sum(len(c["profiles"]) for c in EXIT_CATEGORIES.values())
    st.caption(f"選択中: {len(selected_profiles)} / {total_profiles} Exit戦略")


def _render_execution_settings():
    """実行設定UIを描画"""
    # トレンド検出方法
    st.markdown("**トレンド検出方法**")
    trend_col1, trend_col2 = st.columns(2)
    with trend_col1:
        batch_trend_method = st.selectbox(
            "検出方法",
            options=["ma_cross", "dual_tf_ema", "adx", "combined"],
            format_func=lambda x: {
                "ma_cross": "MA Cross（移動平均クロス）",
                "dual_tf_ema": "Dual-TF EMA（4h+1h合意）★推奨",
                "adx": "ADX（トレンド強度）",
                "combined": "MA Cross + ADX（複合）",
            }[x],
            index=1,
            key="batch_trend_method",
            help="トレンド/レンジを判定するアルゴリズム",
        )
    with trend_col2:
        if batch_trend_method == "dual_tf_ema":
            batch_super_htf = st.selectbox(
                "Super HTF",
                options=["4h", "1d"],
                index=0,
                key="batch_super_htf",
                help="Dual-TF EMAの上位タイムフレーム（通常4h）",
            )
        else:
            batch_super_htf = None

    st.session_state.batch_trend_method = batch_trend_method
    st.session_state.batch_super_htf = batch_super_htf

    st.divider()

    # レジーム選択
    st.markdown("**対象レジーム**")
    regime_cols = st.columns(3)
    selected_regimes = []

    with regime_cols[0]:
        if st.checkbox(
            "📈 Uptrend",
            value=st.session_state.get("batch_regime_uptrend", True),
            key="batch_regime_uptrend",
        ):
            selected_regimes.append("uptrend")

    with regime_cols[1]:
        if st.checkbox(
            "📉 Downtrend",
            value=st.session_state.get("batch_regime_downtrend", True),
            key="batch_regime_downtrend",
        ):
            selected_regimes.append("downtrend")

    with regime_cols[2]:
        if st.checkbox(
            "↔️ Range",
            value=st.session_state.get("batch_regime_range", True),
            key="batch_regime_range",
        ):
            selected_regimes.append("range")

    st.session_state.batch_selected_regimes = selected_regimes

    st.divider()

    # その他の設定
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        use_oos = st.checkbox(
            "OOS検証",
            value=st.session_state.get("batch_use_oos", True),
            key="batch_use_oos",
            help="Train 60% / Val 20% / Test 20% の3分割検証",
        )

    with col2:
        use_blacklist = st.checkbox(
            "ブラックリスト参照",
            value=st.session_state.get("batch_use_blacklist", True),
            key="batch_use_blacklist",
            help="過去にFAILした組み合わせをスキップ",
        )

    with col3:
        n_workers = st.selectbox(
            "並列数",
            options=[1, 2, 4, 8],
            index=2,
            key="batch_n_workers",
        )

    with col4:
        reuse_existing = st.checkbox(
            "既存結果を再利用",
            value=st.session_state.get("batch_reuse_existing", True),
            key="batch_reuse_existing",
            help="同じ条件の結果が存在する場合はスキップ",
        )

    # 探索方法の選択
    st.divider()
    st.markdown("### 🔍 探索方法")

    search_method = st.radio(
        "探索アルゴリズム",
        ["grid", "ga"],
        format_func=lambda x: {
            "grid": "グリッドサーチ（全組み合わせ）",
            "ga": "🧬 遺伝的アルゴリズム（効率的探索）",
        }[x],
        index=st.session_state.get("batch_search_method_idx", 0),
        key="batch_search_method",
        horizontal=True,
    )
    st.session_state["batch_search_method_idx"] = 0 if search_method == "grid" else 1

    if search_method == "ga":
        st.info("💡 GAは全パラメータ組み合わせを試さず、進化的に良い解を探索。グリッドサーチの1/3〜1/5の評価回数で済みます。")

        ga_col1, ga_col2, ga_col3, ga_col4 = st.columns(4)
        with ga_col1:
            ga_population = st.number_input(
                "集団サイズ",
                value=st.session_state.get("batch_ga_population", 20),
                min_value=10,
                max_value=100,
                step=5,
                key="batch_ga_population",
                help="1世代あたりの個体数",
            )
        with ga_col2:
            ga_generations = st.number_input(
                "最大世代数",
                value=st.session_state.get("batch_ga_generations", 10),
                min_value=5,
                max_value=50,
                step=5,
                key="batch_ga_generations",
                help="進化を繰り返す最大回数",
            )
        with ga_col3:
            ga_elite_ratio = st.slider(
                "エリート率",
                0.1, 0.5, st.session_state.get("batch_ga_elite", 0.25), 0.05,
                key="batch_ga_elite",
                help="次世代に引き継ぐ上位個体の割合",
            )
        with ga_col4:
            ga_mutation_rate = st.slider(
                "突然変異率",
                0.05, 0.3, st.session_state.get("batch_ga_mutation", 0.15), 0.05,
                key="batch_ga_mutation",
                help="パラメータをランダムに変化させる確率",
            )


def _render_execution_summary(scan_result: Dict[str, Any]):
    """実行サマリーを描画"""
    st.markdown("### 📊 実行サマリー")

    selected_symbols = st.session_state.get("batch_selected_symbols", [])
    selected_periods = st.session_state.get("batch_selected_periods", [])
    tf_combos = st.session_state.get("batch_tf_combos", [])
    selected_templates = st.session_state.get("batch_selected_templates", [])
    selected_exit_profiles = st.session_state.get("batch_selected_exit_profiles", [])

    files = scan_result["files"]
    n_jobs = count_jobs(selected_symbols, selected_periods, tf_combos, files)

    # config数の計算
    n_templates = len(selected_templates)
    n_exits = len(selected_exit_profiles) if selected_exit_profiles else 1
    n_configs = n_templates * n_exits * 3  # 3 = パラメータ組み合わせの概算

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("銘柄", len(selected_symbols))
    with col2:
        st.metric("TFコンボ", len(tf_combos))
    with col3:
        st.metric("テンプレート", n_templates)
    with col4:
        st.metric("Exit戦略", n_exits)

    st.info(f"📌 **推定ジョブ数: {n_jobs}** ({len(selected_symbols)}銘柄 × {len(tf_combos)}TF × {len(selected_periods)}期間)")


def _render_batch_progress():
    """バッチ実行中の進捗表示"""
    progress = st.session_state.batch_progress

    st.markdown("### 🚀 バッチ最適化実行中")

    # 全体進捗
    total = progress.get("total", 1)
    current = progress.get("current", 0)
    pct = current / max(total, 1)

    st.progress(pct, text=f"全体進捗: {current}/{total} ({pct:.0%})")

    # 現在の処理
    status = progress.get("status", "")
    if status:
        st.markdown(f"**現在処理中:** {status}")

    # 詳細進捗
    grid_current = progress.get("grid_current", 0)
    grid_total = progress.get("grid_total", 0)
    search_method = st.session_state.get("batch_search_method", "grid")
    search_label = "🧬 GA" if search_method == "ga" else "グリッドサーチ"
    if grid_total > 0:
        grid_pct = grid_current / grid_total
        st.progress(grid_pct, text=f"{search_label}: {grid_current:,}/{grid_total:,}")

    # 統計
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("✅ 完了", progress.get("completed", 0))
    with col2:
        st.metric("⏭️ スキップ", progress.get("skipped", 0))
    with col3:
        st.metric("❌ エラー", progress.get("errors", 0))

    # 中止ボタン
    if st.button("⏹️ 中止", type="secondary"):
        st.session_state.batch_running = False
        st.session_state.batch_cancel_requested = True
        st.rerun()


def _start_batch_optimization(scan_result: Dict[str, Any]):
    """バッチ最適化を開始"""
    from optimizer.adaptive import (
        AdaptiveSearchConfig, ScoutConfig, PlateauConfig, RoundConfig,
    )

    selected_symbols = st.session_state.get("batch_selected_symbols", [])
    selected_periods = st.session_state.get("batch_selected_periods", [])
    tf_combos = st.session_state.get("batch_tf_combos", [])
    selected_templates = st.session_state.get("batch_selected_templates", [])
    selected_exit_profiles = st.session_state.get("batch_selected_exit_profiles", [])
    selected_regimes = st.session_state.get("batch_selected_regimes", TARGET_REGIMES)
    use_oos = st.session_state.get("batch_use_oos", True)
    use_blacklist = st.session_state.get("batch_use_blacklist", True)
    n_workers = st.session_state.get("batch_n_workers", 4)
    reuse_existing = st.session_state.get("batch_reuse_existing", True)

    # 探索方法を取得
    search_method = st.session_state.get("batch_search_method", "grid")

    # GA設定
    ga_config = None
    if search_method == "ga":
        ga_config = GAConfig(
            population_size=st.session_state.get("batch_ga_population", 20),
            max_generations=st.session_state.get("batch_ga_generations", 10),
            elite_ratio=st.session_state.get("batch_ga_elite", 0.25),
            mutation_rate=st.session_state.get("batch_ga_mutation", 0.15),
            convergence_threshold=0.01,
            convergence_generations=4,
        )

    # 適応型探索の設定を取得（グリッドサーチの場合のみ）
    adaptive_config = None
    enable_adaptive = st.session_state.get("batch_enable_adaptive", False) if search_method == "grid" else False

    if enable_adaptive:
        adaptive_config = AdaptiveSearchConfig(
            scout=ScoutConfig(
                sample_ratio=st.session_state.get("batch_scout_ratio", 0.2),
                top_n_to_scale=st.session_state.get("batch_scout_top_n", 20),
            ),
            plateau=PlateauConfig(
                min_improvement=st.session_state.get("batch_plateau_threshold", 0.001),
                consecutive_rounds=st.session_state.get("batch_plateau_rounds", 2),
            ),
            round=RoundConfig(
                max_rounds=st.session_state.get("batch_max_rounds", 5),
                top_n_survivors=st.session_state.get("batch_top_n_survivors", 10),
                exploration_ratio=st.session_state.get("batch_exploration_ratio", 0.2),
            ),
            enable_scout=True,
            enable_plateau=True,
            enable_rounds=True,
            verbose=True,
        )

    files = scan_result["files"]

    # バリデーション
    if not selected_symbols:
        st.error("銘柄を選択してください")
        return
    if not tf_combos:
        st.error("有効なTFコンボがありません")
        return
    if not selected_templates:
        st.error("テンプレートを選択してください")
        return
    if not selected_regimes:
        st.error("レジームを選択してください")
        return

    # 出力ディレクトリ
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = RESULTS_DIR / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # ジョブリスト作成
    jobs = []
    for symbol in selected_symbols:
        for period in selected_periods:
            for exec_tf, htf in tf_combos:
                if (symbol, exec_tf, period) in files and (symbol, htf, period) in files:
                    jobs.append({
                        "symbol": symbol,
                        "period": period,
                        "exec_tf": exec_tf,
                        "htf": htf,
                    })

    # 進捗初期化
    st.session_state.batch_running = True
    st.session_state.batch_cancel_requested = False
    st.session_state.batch_progress = {
        "current": 0,
        "total": len(jobs),
        "completed": 0,
        "skipped": 0,
        "errors": 0,
        "status": "",
        "grid_current": 0,
        "grid_total": 0,
    }
    st.session_state.batch_output_dir = str(output_dir)

    # config生成
    all_configs = generate_all_configs(selected_templates, selected_exit_profiles)

    # プレースホルダー
    progress_placeholder = st.empty()
    grid_progress_placeholder = st.empty()
    status_placeholder = st.empty()
    stats_placeholder = st.empty()
    cancel_placeholder = st.empty()

    # 進捗ラベル（探索方法によって変更）
    search_label = "🧬 GA" if search_method == "ga" else "グリッドサーチ"

    # 初期状態を表示
    with progress_placeholder.container():
        st.progress(0.0, text=f"全体進捗: 0/{len(jobs)} (0%)")
    with grid_progress_placeholder.container():
        st.progress(0.0, text=f"{search_label}: データ読み込み中...")
    with status_placeholder.container():
        st.markdown(f"**🚀 バッチ最適化開始** - {len(jobs)}件のジョブを実行します ({search_label})")
    with stats_placeholder.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("✅ 完了", 0)
        with col2:
            st.metric("⏭️ スキップ", 0)
        with col3:
            st.metric("❌ エラー", 0)
    with cancel_placeholder.container():
        if st.button("⏹️ 中止", type="secondary", key="batch_cancel_btn"):
            st.session_state.batch_running = False
            st.session_state.batch_cancel_requested = True
            st.rerun()

    def update_progress(completed: int, total: int, desc: str):
        """進捗コールバック"""
        st.session_state.batch_progress["grid_current"] = completed
        st.session_state.batch_progress["grid_total"] = total
        # リアルタイムでプレースホルダーを更新
        if total > 0:
            grid_pct = completed / total
            grid_progress_placeholder.progress(grid_pct, text=f"{search_label}: {completed:,}/{total:,}")

    # 実行ループ
    for i, job in enumerate(jobs):
        if st.session_state.get("batch_cancel_requested", False):
            break

        symbol = job["symbol"]
        period = job["period"]
        exec_tf = job["exec_tf"]
        htf = job["htf"]

        st.session_state.batch_progress["current"] = i + 1
        st.session_state.batch_progress["status"] = f"{symbol} | {exec_tf}/{htf} | {period}"
        st.session_state.batch_progress["grid_current"] = 0
        st.session_state.batch_progress["grid_total"] = 0

        # 既存結果チェック
        existing_file = output_dir / f"{symbol}_{period}_{exec_tf}_{htf}.json"
        if reuse_existing and existing_file.exists():
            st.session_state.batch_progress["skipped"] += 1
            continue

        try:
            # データロード
            tf_dict = load_symbol_data(symbol, period, exec_tf, htf, files)

            # Dual-TF EMA用のsuper_htfデータ
            _trend_method = st.session_state.get("batch_trend_method", "ma_cross")
            _super_htf = st.session_state.get("batch_super_htf")
            _super_htf_df = None
            if _trend_method == "dual_tf_ema" and _super_htf:
                super_htf_path = files.get((symbol, _super_htf, period))
                if super_htf_path:
                    loader = BinanceCSVLoader()
                    _super_htf_df = loader.load(str(super_htf_path), symbol=symbol).df.copy()

            exec_df = prepare_exec_df(
                tf_dict, exec_tf, htf,
                trend_method=_trend_method,
                super_htf_df=_super_htf_df,
            )

            if search_method == "ga":
                # GA最適化実行（OOS検証付き）
                ga_results = run_single_ga_optimization(
                    exec_df=exec_df,
                    templates=selected_templates,
                    target_regimes=selected_regimes,
                    ga_config=ga_config,
                    symbol=symbol,
                    progress_callback=update_progress,
                    use_oos=use_oos,
                    use_blacklist=use_blacklist,
                )

                # GA結果保存
                for ga_result in ga_results:
                    ga_result.save(str(output_dir))

                # GA進捗を完了状態に更新
                total_gens = len(selected_regimes) * ga_config.max_generations
                st.session_state.batch_progress["grid_current"] = total_gens
                st.session_state.batch_progress["grid_total"] = total_gens
                grid_progress_placeholder.progress(1.0, text=f"{search_label}: 完了")

                result = None  # GAは別形式で保存

            else:
                # グリッドサーチ最適化実行
                result = run_single_optimization(
                    exec_df=exec_df,
                    all_configs=all_configs,
                    use_oos=use_oos,
                    n_workers=n_workers,
                    target_regimes=selected_regimes,
                    progress_callback=update_progress,
                    adaptive_config=adaptive_config,
                )

                # 結果保存
                save_batch_result(
                    result=result,
                    symbol=symbol,
                    period=period,
                    exec_tf=exec_tf,
                    htf=htf,
                    use_oos=use_oos,
                    output_dir=output_dir,
                )

            st.session_state.batch_progress["completed"] += 1

        except Exception as e:
            st.session_state.batch_progress["errors"] += 1
            st.error(f"エラー: {symbol} | {exec_tf}/{htf} | {period}: {e}")

        # UI更新（進捗表示）
        with progress_placeholder.container():
            progress = st.session_state.batch_progress
            pct = progress["current"] / max(progress["total"], 1)
            st.progress(pct, text=f"全体進捗: {progress['current']}/{progress['total']} ({pct:.0%})")

        with grid_progress_placeholder.container():
            grid_current = st.session_state.batch_progress.get("grid_current", 0)
            grid_total = st.session_state.batch_progress.get("grid_total", 0)
            if grid_total > 0:
                grid_pct = grid_current / grid_total
                st.progress(grid_pct, text=f"{search_label}: {grid_current:,}/{grid_total:,}")
            else:
                st.progress(0.0, text=f"{search_label}: 準備中...")

        with status_placeholder.container():
            st.markdown(f"**現在処理中:** {st.session_state.batch_progress['status']}")

        with stats_placeholder.container():
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("✅ 完了", st.session_state.batch_progress["completed"])
            with col2:
                st.metric("⏭️ スキップ", st.session_state.batch_progress["skipped"])
            with col3:
                st.metric("❌ エラー", st.session_state.batch_progress["errors"])

    # 完了
    st.session_state.batch_running = False

    progress = st.session_state.batch_progress
    st.success(
        f"🎉 バッチ最適化完了！\n\n"
        f"- 完了: {progress['completed']} 件\n"
        f"- スキップ: {progress['skipped']} 件\n"
        f"- エラー: {progress['errors']} 件\n\n"
        f"結果: `{output_dir}`"
    )


# =============================================================================
# バッチ結果読み込み・表示
# =============================================================================

def list_batch_result_dirs() -> List[Dict[str, Any]]:
    """results/batchディレクトリから利用可能なバッチ結果一覧を取得"""
    result_dirs = []

    if not RESULTS_DIR.exists():
        return result_dirs

    for d in sorted(RESULTS_DIR.iterdir(), reverse=True):
        if not d.is_dir():
            continue

        # ディレクトリ名がタイムスタンプ形式か確認
        if not re.match(r"^\d{8}_\d{6}$", d.name):
            continue

        # JSON数をカウント
        json_files = list(d.glob("*.json"))
        if not json_files:
            continue

        # GA結果かどうかを判定（ga_で始まるファイルがあるか）
        ga_files = [f for f in json_files if f.name.startswith("ga_")]
        is_ga = len(ga_files) > 0

        # 最初のJSONから情報を取得
        try:
            with open(json_files[0], "r", encoding="utf-8") as f:
                sample = json.load(f)
        except Exception:
            continue

        # ファイルからシンボル一覧を抽出
        symbols = set()
        for jf in json_files:
            if jf.name.startswith("ga_"):
                # GA形式: ga_BTCUSDT_uptrend_20260205_180454.json
                parts = jf.stem.split("_")
                if len(parts) >= 2:
                    symbols.add(parts[1])
            else:
                # グリッドサーチ形式: BTCUSDT_20250201-20260130_15m_1h.json
                parts = jf.stem.split("_")
                if parts:
                    symbols.add(parts[0])

        # 日時をパース
        dt_str = d.name  # 20260204_174904
        dt_display = f"{dt_str[:4]}/{dt_str[4:6]}/{dt_str[6:8]} {dt_str[9:11]}:{dt_str[11:13]}"

        result_dirs.append({
            "path": d,
            "name": d.name,
            "datetime": dt_display,
            "file_count": len(json_files),
            "symbols": sorted(symbols),
            "oos": sample.get("oos", False),
            "is_ga": is_ga,  # GA結果かどうか
        })

    return result_dirs


def load_batch_result_files(dir_path: Path) -> List[Dict[str, Any]]:
    """指定ディレクトリの全JSONを読み込み"""
    results = []

    for jf in sorted(dir_path.glob("*.json")):
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["_file"] = jf.name
            results.append(data)
        except Exception:
            continue

    return results


def _render_blacklist_management():
    """ブラックリスト管理UI"""
    st.markdown("### 🚫 ブラックリスト管理")
    st.caption("OOS検証でFAILした組み合わせ（銘柄 × レジーム × テンプレート）を管理")

    blacklist = get_blacklist()

    if not blacklist.entries:
        st.info("ブラックリストは空です。OOS検証でFAILすると自動的に追加されます。")
        return

    # サマリー表示
    summary = blacklist.summary()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("合計", summary["total"])
    with col2:
        regime_str = ", ".join(f"{k}: {v}" for k, v in summary["by_regime"].items())
        st.metric("レジーム別", regime_str or "—")
    with col3:
        top_templates = sorted(summary["by_template"].items(), key=lambda x: -x[1])[:3]
        template_str = ", ".join(f"{k}: {v}" for k, v in top_templates)
        st.metric("Top3テンプレート", template_str or "—")

    # 一覧表示
    st.markdown("---")
    df_blacklist = pd.DataFrame([
        {
            "銘柄": e.symbol,
            "レジーム": e.regime,
            "テンプレート": e.template,
            "Test PnL": f"{e.test_pnl:.2f}%" if e.test_pnl is not None else "—",
            "Train PnL": f"{e.train_pnl:.2f}%" if e.train_pnl is not None else "—",
            "追加日": e.added_at,
            "理由": e.reason,
        }
        for e in blacklist.entries
    ])

    # フィルタ
    col1, col2, col3 = st.columns(3)
    with col1:
        filter_regime = st.selectbox(
            "レジームでフィルタ",
            options=["すべて"] + list(set(e.regime for e in blacklist.entries)),
            key="blacklist_filter_regime",
        )
    with col2:
        filter_symbol = st.selectbox(
            "銘柄でフィルタ",
            options=["すべて"] + sorted(set(e.symbol for e in blacklist.entries)),
            key="blacklist_filter_symbol",
        )
    with col3:
        filter_template = st.selectbox(
            "テンプレートでフィルタ",
            options=["すべて"] + sorted(set(e.template for e in blacklist.entries)),
            key="blacklist_filter_template",
        )

    # フィルタ適用
    df_filtered = df_blacklist.copy()
    if filter_regime != "すべて":
        df_filtered = df_filtered[df_filtered["レジーム"] == filter_regime]
    if filter_symbol != "すべて":
        df_filtered = df_filtered[df_filtered["銘柄"] == filter_symbol]
    if filter_template != "すべて":
        df_filtered = df_filtered[df_filtered["テンプレート"] == filter_template]

    st.dataframe(df_filtered, use_container_width=True, hide_index=True)

    # 削除機能
    st.markdown("---")
    st.markdown("#### ⚠️ エントリ削除")
    col1, col2 = st.columns([3, 1])
    with col1:
        delete_options = [
            f"{e.symbol} / {e.regime} / {e.template}"
            for e in blacklist.entries
        ]
        selected_delete = st.multiselect(
            "削除するエントリを選択",
            options=delete_options,
            key="blacklist_delete_select",
        )
    with col2:
        if st.button("🗑️ 選択を削除", type="secondary", disabled=not selected_delete):
            for item in selected_delete:
                parts = item.split(" / ")
                if len(parts) == 3:
                    blacklist.remove(parts[0], parts[1], parts[2])
            blacklist.save()
            st.success(f"{len(selected_delete)}件を削除しました")
            st.rerun()

    # 全削除
    with st.expander("⚠️ 全削除（注意）"):
        if st.button("🗑️ ブラックリストを全削除", type="secondary"):
            count = blacklist.clear()
            blacklist.save()
            st.success(f"{count}件を削除しました")
            st.rerun()


def render_batch_load_view():
    """バッチ結果読み込み・表示ビューを描画"""
    st.subheader("📁 バッチ結果読み込み")

    # サブビュー切り替え
    sub_view = st.radio(
        "表示",
        options=["📂 結果一覧", "🚫 ブラックリスト"],
        horizontal=True,
        key="batch_load_sub_view",
    )

    st.markdown("---")

    if sub_view == "🚫 ブラックリスト":
        _render_blacklist_management()
        return

    # --- 結果一覧ビュー ---
    st.caption("保存されたバッチ最適化結果を閲覧")

    # 結果ディレクトリ一覧を取得
    result_dirs = list_batch_result_dirs()

    if not result_dirs:
        st.info(
            "📂 バッチ結果がありません\n\n"
            f"パス: `{RESULTS_DIR}`\n\n"
            "「🚀 バッチ」ビューから最適化を実行してください。"
        )
        return

    # セッション初期化
    if "batch_load_selected" not in st.session_state:
        st.session_state.batch_load_selected = None
    if "batch_load_data" not in st.session_state:
        st.session_state.batch_load_data = None

    # --- 結果一覧 ---
    st.markdown("### 保存済みバッチ結果")

    # 一覧をデータフレームで表示
    df_dirs = pd.DataFrame([
        {
            "選択": False,
            "日時": d["datetime"],
            "件数": d["file_count"],
            "銘柄": ", ".join(d["symbols"][:5]) + ("..." if len(d["symbols"]) > 5 else ""),
            "OOS": "✅" if d["oos"] else "—",
            "パス": d["name"],
        }
        for d in result_dirs
    ])

    # 選択
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_name = st.selectbox(
            "結果を選択",
            options=[d["name"] for d in result_dirs],
            format_func=lambda x: next(
                f"{d['datetime']} | {d['file_count']}件 | {'ga' if d.get('is_ga') else 'grid'}"
                for d in result_dirs if d["name"] == x
            ),
            key="batch_load_selector",
        )
    with col2:
        if st.button("📂 読み込み", type="primary", use_container_width=True):
            selected_dir = next(d for d in result_dirs if d["name"] == selected_name)
            st.session_state.batch_load_selected = selected_dir
            st.session_state.batch_load_data = load_batch_result_files(selected_dir["path"])
            st.rerun()

    # --- 読み込み済みデータの表示 ---
    if st.session_state.batch_load_data:
        selected = st.session_state.batch_load_selected
        if selected.get("is_ga"):
            _render_ga_results_detail()
        else:
            _render_batch_results_detail()


def _render_ga_results_detail():
    """GA結果の詳細を表示"""
    data_list = st.session_state.batch_load_data
    selected = st.session_state.batch_load_selected

    st.divider()
    st.markdown(f"### 🧬 GA結果: {selected['datetime']}")
    st.caption(f"{len(data_list)}件のGA結果")

    # GA結果をテーブルに整形
    table_data = []
    has_oos = False  # OOS検証結果があるかどうか

    for data in data_list:
        if "final_winner" not in data:
            continue  # GA結果でない場合はスキップ

        winner = data.get("final_winner", {})

        # OOS検証結果の有無を確認
        has_verdict = "verdict" in winner
        if has_verdict:
            has_oos = True

        row = {
            "銘柄": data.get("symbol", "?"),
            "レジーム": data.get("regime", "?"),
            "テンプレート": winner.get("template", "?"),
        }

        trades = winner.get("trades", 0)

        # OOS検証結果がある場合
        if has_verdict:
            verdict = winner.get("verdict", "?")
            train_pnl = winner.get("train_pnl", 0)
            test_pnl = winner.get("test_pnl", 0)

            # 安定度 (Test/Train比率)
            if train_pnl > 0:
                stability = test_pnl / train_pnl
            elif train_pnl < 0 and test_pnl > 0:
                stability = 2.0  # Train負け→Test勝ちは高評価
            else:
                stability = 0.0

            # 採用スコア計算
            # Test PnL × 0.4 + 安定度 × 0.3 + トレード数正規化 × 0.3
            test_pnl_score = max(0, min(test_pnl / 30, 1.0))  # 30%で満点
            stability_score = max(0, min(stability, 1.0))  # 1.0で満点
            trades_score = min(trades / 100, 1.0)  # 100回で満点
            adoption_score = test_pnl_score * 0.4 + stability_score * 0.3 + trades_score * 0.3

            row["Test PnL"] = f"{test_pnl:+.2f}%"
            row["安定度"] = f"{stability:.2f}"
            row["トレード数"] = trades
            row["採用スコア"] = round(adoption_score, 2)
            row["採用"] = "⭐" if verdict == "PASS" and adoption_score >= 0.5 else "—"
            row["判定"] = verdict
            row["_test_pnl"] = test_pnl  # ソート用（非表示）
            row["_adoption_score"] = adoption_score  # ソート用（非表示）
        else:
            row["PnL (%)"] = f"{winner.get('pnl', 0):+.2f}%"
            row["トレード数"] = trades
            row["_test_pnl"] = winner.get("pnl", 0)
            row["_adoption_score"] = 0

        table_data.append(row)

    if not table_data:
        st.warning("該当する結果がありません")
        return

    df = pd.DataFrame(table_data)

    # フィルタリング
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        symbols = ["すべて"] + sorted(df["銘柄"].unique().tolist())
        filter_symbol = st.selectbox("銘柄", options=symbols, key="ga_filter_symbol")
    with col2:
        regimes = ["すべて"] + sorted(df["レジーム"].unique().tolist())
        filter_regime = st.selectbox("レジーム", options=regimes, key="ga_filter_regime")
    with col3:
        # 判定フィルタ（OOS結果がある場合のみ）
        if has_oos:
            verdict_options = ["すべて", "PASSのみ", "FAILのみ"]
            filter_verdict = st.selectbox("判定", options=verdict_options, key="ga_filter_verdict")
        else:
            filter_verdict = "すべて"
    with col4:
        min_trades = st.number_input(
            "最小トレード数",
            min_value=0,
            max_value=100,
            value=0,
            step=5,
            key="ga_filter_min_trades",
        )

    # フィルタ適用
    filtered = df.copy()
    if filter_symbol != "すべて":
        filtered = filtered[filtered["銘柄"] == filter_symbol]
    if filter_regime != "すべて":
        filtered = filtered[filtered["レジーム"] == filter_regime]
    if has_oos and filter_verdict == "PASSのみ":
        filtered = filtered[filtered["判定"] == "PASS"]
    elif has_oos and filter_verdict == "FAILのみ":
        filtered = filtered[filtered["判定"] == "FAIL"]
    if min_trades > 0:
        filtered = filtered[filtered["トレード数"] >= min_trades]

    # 採用スコア順でソート（OOS結果がある場合）
    if has_oos and "_adoption_score" in filtered.columns:
        filtered = filtered.sort_values("_adoption_score", ascending=False)

    # 非表示列を除外
    display_cols = [c for c in filtered.columns if not c.startswith("_")]
    display_df = filtered[display_cols]

    # 件数表示
    st.caption(f"表示: {len(display_df)}件 / 全{len(df)}件")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # 世代推移チャート
    with st.expander("📈 世代ごとのスコア推移", expanded=False):
        for data in data_list:
            if "generations" not in data:
                continue
            symbol = data.get("symbol", "?")
            regime = data.get("regime", "?")
            if filter_symbol != "すべて" and symbol != filter_symbol:
                continue
            if filter_regime != "すべて" and regime != filter_regime:
                continue

            st.caption(f"**{symbol} - {regime}**")
            gen_data = []
            for gen in data.get("generations", []):
                gen_data.append({
                    "世代": gen.get("generation", 0),
                    "ベストスコア": gen.get("best_score", 0),
                    "平均スコア": gen.get("avg_score", 0),
                })
            if gen_data:
                chart_df = pd.DataFrame(gen_data)
                st.line_chart(chart_df.set_index("世代"))

    # 採用候補をグリッドサーチボタン
    if has_oos:
        st.divider()
        st.markdown("### 🔬 採用候補をグリッドサーチ")

        # 採用候補を抽出（⭐のもの）
        adopted_rows = filtered[filtered["採用"] == "⭐"] if "採用" in filtered.columns else pd.DataFrame()
        n_adopted = len(adopted_rows)

        if n_adopted == 0:
            st.info("採用候補（⭐）がありません。フィルタを調整してください。")
        else:
            # 採用候補の銘柄とテンプレートを抽出
            adopted_symbols = adopted_rows["銘柄"].unique().tolist()
            adopted_templates = adopted_rows["テンプレート"].unique().tolist()

            st.caption(f"採用候補: {n_adopted}件 | 銘柄: {len(adopted_symbols)}種 | テンプレート: {len(adopted_templates)}種")

            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"**銘柄:** {', '.join(adopted_symbols[:10])}{'...' if len(adopted_symbols) > 10 else ''}")
                st.markdown(f"**テンプレート:** {', '.join(adopted_templates[:5])}{'...' if len(adopted_templates) > 5 else ''}")

            with col2:
                if st.button("🚀 グリッドサーチ開始", type="primary", use_container_width=True):
                    # 採用候補リストをsession_stateに保存
                    st.session_state.ga_adopted_candidates = {
                        "symbols": adopted_symbols,
                        "templates": adopted_templates,
                        "source": selected["datetime"],
                    }
                    # グリッドサーチモードをセット
                    st.session_state.batch_search_method = "grid"
                    st.session_state.batch_from_ga_candidates = True
                    # 自動実行フラグ
                    st.session_state.batch_auto_start = True

                    # 既存のチェックボックス状態をクリア（GA候補のdefault値を有効にするため）
                    keys_to_delete = [k for k in st.session_state.keys()
                                      if k.startswith("batch_symbol_") or k.startswith("batch_template_")]
                    for k in keys_to_delete:
                        del st.session_state[k]

                    # バッチタブに自動遷移
                    st.session_state.optimizer_view = "batch"
                    st.rerun()


def _render_batch_results_detail():
    """読み込んだバッチ結果の詳細を表示"""
    data_list = st.session_state.batch_load_data
    selected = st.session_state.batch_load_selected

    st.divider()
    st.markdown(f"### 📊 結果詳細: {selected['datetime']}")
    st.caption(f"{len(data_list)}件のデータ | OOS: {'有効' if selected['oos'] else '無効'}")

    # フィルタリング
    col1, col2, col3 = st.columns(3)

    # 銘柄フィルター
    all_symbols = sorted(set(d["symbol"] for d in data_list))
    with col1:
        filter_symbol = st.selectbox(
            "銘柄",
            options=["すべて"] + all_symbols,
            key="batch_result_filter_symbol",
        )

    # レジームフィルター
    with col2:
        filter_regime = st.selectbox(
            "レジーム",
            options=["すべて", "uptrend", "downtrend", "range"],
            key="batch_result_filter_regime",
        )

    # 表示モード
    with col3:
        view_mode = st.radio(
            "表示モード",
            options=["サマリー", "詳細"],
            horizontal=True,
            key="batch_result_view_mode",
        )

    # フィルタ適用
    filtered_data = data_list
    if filter_symbol != "すべて":
        filtered_data = [d for d in filtered_data if d["symbol"] == filter_symbol]

    if view_mode == "サマリー":
        _render_batch_summary_view(filtered_data, filter_regime)
    else:
        _render_batch_detail_view(filtered_data, filter_regime)


def _render_batch_summary_view(data_list: List[Dict], filter_regime: str):
    """サマリービュー: レジーム×銘柄ごとのベスト戦略"""

    # 各データからレジームごとのベストを抽出
    summary_rows = []

    for data in data_list:
        symbol = data["symbol"]
        period = data.get("period", "")
        exec_tf = data.get("execution_tf", "")
        htf = data.get("htf", "")
        is_oos = data.get("oos", False)

        if is_oos and "test_results" in data:
            # OOS結果（test_results）を使用
            test_results = data["test_results"]
            for regime, entry in test_results.items():
                if filter_regime != "すべて" and regime != filter_regime:
                    continue

                pnl = entry.get("metrics", {}).get("total_pnl", 0)
                trades = entry.get("metrics", {}).get("trades", 0)

                # 採用スコア計算
                pnl_score = max(0, min(pnl / 30, 1.0))  # 30%で満点
                trades_score = min(trades / 100, 1.0)  # 100回で満点
                adoption_score = pnl_score * 0.5 + trades_score * 0.5

                # 採用判定: PnL > 0 かつ 取引数 >= 30 かつ 採用スコア >= 0.3
                is_adopted = pnl > 0 and trades >= 30 and adoption_score >= 0.3

                summary_rows.append({
                    "銘柄": symbol,
                    "期間": period,
                    "TF": f"{exec_tf}/{htf}",
                    "レジーム": _regime_label(regime),
                    "テンプレート": _template_label(entry.get("template", "")),
                    "Exit": entry.get("exit_profile", ""),
                    "スコア": entry.get("score", 0),
                    "勝率": entry.get("metrics", {}).get("win_rate", 0),
                    "PF": entry.get("metrics", {}).get("profit_factor", 0),
                    "PnL%": pnl,
                    "DD%": entry.get("metrics", {}).get("max_dd", 0),
                    "取引数": trades,
                    "採用スコア": round(adoption_score, 2),
                    "採用": "⭐" if is_adopted else "—",
                })
        elif "results" in data:
            # 通常結果
            for entry in data["results"][:3]:  # 上位3件
                regime = entry.get("regime", "all")
                if filter_regime != "すべて" and regime != filter_regime:
                    continue

                pnl = entry.get("metrics", {}).get("total_pnl", 0)
                trades = entry.get("metrics", {}).get("trades", 0)

                # 採用スコア計算（OOSなしの場合は参考値）
                pnl_score = max(0, min(pnl / 30, 1.0))
                trades_score = min(trades / 100, 1.0)
                adoption_score = pnl_score * 0.5 + trades_score * 0.5

                summary_rows.append({
                    "銘柄": symbol,
                    "期間": period,
                    "TF": f"{exec_tf}/{htf}",
                    "レジーム": _regime_label(regime),
                    "テンプレート": _template_label(entry.get("template", "")),
                    "Exit": entry.get("exit_profile", ""),
                    "スコア": entry.get("score", 0),
                    "勝率": entry.get("metrics", {}).get("win_rate", 0),
                    "PF": entry.get("metrics", {}).get("profit_factor", 0),
                    "PnL%": pnl,
                    "DD%": entry.get("metrics", {}).get("max_dd", 0),
                    "取引数": trades,
                    "採用スコア": round(adoption_score, 2),
                    "採用": "—",  # OOSなしは採用判定しない
                })

    if not summary_rows:
        st.info("該当する結果がありません")
        return

    df = pd.DataFrame(summary_rows)

    # 採用スコア順でソート
    df = df.sort_values("採用スコア", ascending=False)

    # フィルターオプション
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        filter_adopted = st.checkbox("採用候補のみ表示", value=False, key="batch_result_filter_adopted")
    with col_f2:
        min_trades = st.number_input("最小取引数", min_value=0, value=0, step=10, key="batch_result_min_trades")

    # フィルタ適用
    filtered_df = df.copy()
    if filter_adopted:
        filtered_df = filtered_df[filtered_df["採用"] == "⭐"]
    if min_trades > 0:
        filtered_df = filtered_df[filtered_df["取引数"] >= min_trades]

    # 採用候補の件数を表示
    n_adopted = len(df[df["採用"] == "⭐"])
    st.caption(f"表示: {len(filtered_df)}件 / 全{len(df)}件 | 採用候補: {n_adopted}件")

    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "スコア": st.column_config.NumberColumn(format="%.3f"),
            "勝率": st.column_config.NumberColumn(format="%.1f%%"),
            "PF": st.column_config.NumberColumn(format="%.2f"),
            "PnL%": st.column_config.NumberColumn(format="%.2f%%"),
            "DD%": st.column_config.NumberColumn(format="%.2f%%"),
            "採用スコア": st.column_config.NumberColumn(format="%.2f"),
        },
    )

    # レジーム別サマリー
    st.markdown("#### レジーム別ベスト戦略")

    for regime in ["uptrend", "downtrend", "range"]:
        if filter_regime != "すべて" and regime != filter_regime:
            continue

        regime_df = df[df["レジーム"] == _regime_label(regime)]
        if regime_df.empty:
            continue

        best = regime_df.iloc[0]
        with st.expander(f"{_regime_label(regime)} - {best['銘柄']} | {best['テンプレート']}", expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("スコア", f"{best['スコア']:.3f}")
            with col2:
                st.metric("勝率", f"{best['勝率']:.1f}%")
            with col3:
                st.metric("PF", f"{best['PF']:.2f}")
            with col4:
                st.metric("PnL", f"{best['PnL%']:.2f}%")

            st.markdown(f"**Exit:** `{best['Exit']}`")
            st.markdown(f"**期間:** {best['期間']} | **TF:** {best['TF']} | **取引数:** {best['取引数']}")


def _render_batch_detail_view(data_list: List[Dict], filter_regime: str):
    """詳細ビュー: 各データの全結果を表示"""

    for data in data_list:
        symbol = data["symbol"]
        period = data.get("period", "")
        exec_tf = data.get("execution_tf", "")
        htf = data.get("htf", "")
        is_oos = data.get("oos", False)

        with st.expander(f"**{symbol}** | {exec_tf}/{htf} | {period}", expanded=False):
            if is_oos:
                # OOS結果
                st.markdown("##### 🏆 テスト結果 (OOS)")
                if "test_results" in data:
                    for regime, entry in data["test_results"].items():
                        if filter_regime != "すべて" and regime != filter_regime:
                            continue

                        st.markdown(f"**{_regime_label(regime)}**")
                        _render_entry_card(entry)

                st.markdown("##### 📈 訓練ベスト (Top 5)")
                if "train_results" in data:
                    for entry in data["train_results"][:5]:
                        regime = entry.get("regime", "all")
                        if filter_regime != "すべて" and regime != filter_regime:
                            continue
                        _render_entry_card(entry)

                if "warnings" in data and data["warnings"]:
                    st.markdown("##### ⚠️ オーバーフィッティング警告")
                    for w in data["warnings"]:
                        st.warning(w)
            else:
                # 通常結果
                st.markdown("##### 🏆 Top 10")
                if "results" in data:
                    for entry in data["results"][:10]:
                        regime = entry.get("regime", "all")
                        if filter_regime != "すべて" and regime != filter_regime:
                            continue
                        _render_entry_card(entry)


def _render_entry_card(entry: Dict):
    """1エントリのカード表示"""
    metrics = entry.get("metrics", {})
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        template = _template_label(entry.get("template", ""))
        st.markdown(f"**{template}**")
    with col2:
        st.metric("スコア", f"{entry.get('score', 0):.3f}")
    with col3:
        st.metric("勝率", f"{metrics.get('win_rate', 0):.1f}%")
    with col4:
        st.metric("PF", f"{metrics.get('profit_factor', 0):.2f}")
    with col5:
        st.metric("PnL", f"{metrics.get('total_pnl', 0):.2f}%")

    st.caption(f"Exit: `{entry.get('exit_profile', '')}` | 取引数: {metrics.get('trades', 0)} | DD: {metrics.get('max_dd', 0):.2f}%")


def _regime_label(regime: str) -> str:
    """レジームのラベル"""
    labels = {
        "uptrend": "📈 Uptrend",
        "downtrend": "📉 Downtrend",
        "range": "↔️ Range",
        "all": "🌐 All",
    }
    return labels.get(regime, regime)


def _template_label(template: str) -> str:
    """テンプレートのラベル（短縮形）"""
    if template in TEMPLATE_LABELS:
        return TEMPLATE_LABELS[template][0]
    return template
