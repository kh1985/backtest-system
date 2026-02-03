"""
Exit戦略プロファイル定義

エントリーテンプレートとは独立した exit 条件のセット。
テンプレートの generate_configs() に渡すことで、
エントリー条件 x exit条件 の直積でグリッドサーチを実行する。

5つのモード:
- fixed: 固定% TP/SL
- atr: ATR倍率 TP/SL（ボラティリティ適応）
- no_sl: SLなし（トレーリング + タイムアウトで利益延伸）
- hybrid: トレーリング + SL のハイブリッド
- bb: BB帯動的exit（ロング→上限バンド、ショート→下限バンドで決済）
"""

from typing import Any, Dict, List

# 各プロファイルは name + exit パラメータの辞書
ExitProfile = Dict[str, Any]


# --- 固定%モード ---
FIXED_PROFILES: List[ExitProfile] = [
    {"name": "fixed_2_1", "take_profit_pct": 2.0, "stop_loss_pct": 1.0},
    {"name": "fixed_3_1.5", "take_profit_pct": 3.0, "stop_loss_pct": 1.5},
    {"name": "fixed_1.5_0.5", "take_profit_pct": 1.5, "stop_loss_pct": 0.5},
]

# --- ATRモード ---
ATR_PROFILES: List[ExitProfile] = [
    {
        "name": "atr_3_1.5",
        "use_atr_exit": True,
        "atr_tp_mult": 3.0,
        "atr_sl_mult": 1.5,
        "atr_period": 14,
    },
    {
        "name": "atr_4_2",
        "use_atr_exit": True,
        "atr_tp_mult": 4.0,
        "atr_sl_mult": 2.0,
        "atr_period": 14,
    },
    {
        "name": "atr_2_1",
        "use_atr_exit": True,
        "atr_tp_mult": 2.0,
        "atr_sl_mult": 1.0,
        "atr_period": 14,
    },
]

# --- SLなしモード（トレーリング + タイムアウト） ---
NO_SL_PROFILES: List[ExitProfile] = [
    {
        "name": "no_sl_trail2",
        "take_profit_pct": 0,
        "stop_loss_pct": 0,
        "trailing_stop_pct": 2.0,
        "timeout_bars": 50,
    },
    {
        "name": "no_sl_trail3",
        "take_profit_pct": 0,
        "stop_loss_pct": 0,
        "trailing_stop_pct": 3.0,
        "timeout_bars": 80,
    },
]

# --- トレーリング + SL ハイブリッド ---
HYBRID_PROFILES: List[ExitProfile] = [
    {
        "name": "trail2_sl1",
        "take_profit_pct": 0,
        "stop_loss_pct": 1.0,
        "trailing_stop_pct": 2.0,
    },
    {
        "name": "atr_trail",
        "use_atr_exit": True,
        "atr_sl_mult": 1.5,
        "atr_tp_mult": 0,
        "atr_period": 14,
        "trailing_stop_pct": 2.0,
    },
]

# --- BB帯exit（動的TP: ロング→上限バンド、ショート→下限バンドで決済） ---
BB_PROFILES: List[ExitProfile] = [
    {
        "name": "bb_exit_sl1",
        "use_bb_exit": True,
        "bb_period": 20,
        "take_profit_pct": 0,
        "stop_loss_pct": 1.0,
    },
    {
        "name": "bb_exit_sl0.5",
        "use_bb_exit": True,
        "bb_period": 20,
        "take_profit_pct": 0,
        "stop_loss_pct": 0.5,
    },
    {
        "name": "bb_exit_atr_sl",
        "use_bb_exit": True,
        "bb_period": 20,
        "use_atr_exit": True,
        "atr_sl_mult": 1.5,
        "atr_tp_mult": 0,
        "atr_period": 14,
    },
]


# 全プロファイル
ALL_PROFILES: List[ExitProfile] = (
    FIXED_PROFILES + ATR_PROFILES + NO_SL_PROFILES + HYBRID_PROFILES + BB_PROFILES
)


def get_profiles(mode: str = "all") -> List[ExitProfile]:
    """
    指定モードの exit profiles を取得

    Args:
        mode: "all", "fixed", "atr", "no_sl", "hybrid", "bb"

    Returns:
        ExitProfile のリスト
    """
    modes = {
        "fixed": FIXED_PROFILES,
        "atr": ATR_PROFILES,
        "no_sl": NO_SL_PROFILES,
        "hybrid": HYBRID_PROFILES,
        "bb": BB_PROFILES,
        "all": ALL_PROFILES,
    }
    return modes.get(mode, ALL_PROFILES)


def profile_to_exit_config(profile: ExitProfile) -> Dict[str, Any]:
    """ExitProfile 辞書から exit config を生成（name を除外）"""
    return {k: v for k, v in profile.items() if k != "name"}
