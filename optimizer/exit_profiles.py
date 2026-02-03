"""
Exit戦略プロファイル定義（ATRベース版）

エントリーテンプレートとは独立した exit 条件のセット。
テンプレートの generate_configs() に渡すことで、
エントリー条件 x exit条件 の直積でグリッドサーチを実行する。

5つのカテゴリ（全33パターン）:
- atr: ATR倍率 TP/SL（12パターン）
- atr_trailing: ATRベーストレーリング + タイムアウト（6パターン）
- atr_hybrid: ATRベーストレーリング + ATR SL（6パターン）
- vwap: VWAPバンドTP + ATR SL（6パターン）
- bb: BBバンドTP + ATR SL（3パターン）

※固定%は削除（ボラティリティ無視のため）
"""

from typing import Any, Dict, List

# 各プロファイルは name + exit パラメータの辞書
ExitProfile = Dict[str, Any]


# =============================================================================
# ATR系: 純粋なATR TP/SL（12パターン）
# SL: ATR×1, 1.5, 2 / TP: ATR×1.5, 2, 3, 4
# =============================================================================
ATR_PROFILES: List[ExitProfile] = []

_atr_sl_mults = [1.0, 1.5, 2.0]
_atr_tp_mults = [1.5, 2.0, 3.0, 4.0]

for sl in _atr_sl_mults:
    for tp in _atr_tp_mults:
        ATR_PROFILES.append({
            "name": f"atr_tp{tp}_sl{sl}".replace(".", ""),
            "use_atr_exit": True,
            "atr_tp_mult": tp,
            "atr_sl_mult": sl,
            "atr_period": 14,
        })


# =============================================================================
# ATRトレーリング系: SLなし、ATRベーストレーリング + タイムアウト（36パターン）
# トレーリング幅: ATR×1, 1.5, 2 / タイムアウト: 2〜24（2刻み）
# =============================================================================
ATR_TRAILING_PROFILES: List[ExitProfile] = []

_atr_trail_mults = [1.0, 1.5, 2.0]
_timeout_bars = list(range(2, 25, 2))  # 2, 4, 6, ..., 24

for trail in _atr_trail_mults:
    for timeout in _timeout_bars:
        ATR_TRAILING_PROFILES.append({
            "name": f"atr_trail{trail}_to{timeout}".replace(".", ""),
            "use_atr_exit": True,
            "atr_tp_mult": 0,  # TPなし（トレーリングで決済）
            "atr_sl_mult": 0,  # SLなし
            "atr_period": 14,
            "use_atr_trailing": True,
            "atr_trailing_mult": trail,
            "timeout_bars": timeout,
        })


# =============================================================================
# ATRハイブリッド系: ATRベーストレーリング + ATR SL（6パターン）
# トレーリング幅: ATR×1, 1.5, 2 / SL: ATR×1.5, 2
# =============================================================================
ATR_HYBRID_PROFILES: List[ExitProfile] = []

_hybrid_sl_mults = [1.5, 2.0]

for trail in _atr_trail_mults:
    for sl in _hybrid_sl_mults:
        ATR_HYBRID_PROFILES.append({
            "name": f"atr_trail{trail}_sl{sl}".replace(".", ""),
            "use_atr_exit": True,
            "atr_tp_mult": 0,  # TPなし（トレーリングで決済）
            "atr_sl_mult": sl,
            "atr_period": 14,
            "use_atr_trailing": True,
            "atr_trailing_mult": trail,
        })


# =============================================================================
# VWAP系: VWAPバンドTP + ATR SL（6パターン）
# バンド: 1σ, 2σ / SL: ATR×1, 1.5, 2
# ロング→上限バンドでTP、ショート→下限バンドでTP
# =============================================================================
VWAP_PROFILES: List[ExitProfile] = []

_vwap_bands = [1, 2]  # 1σ, 2σ

for band in _vwap_bands:
    for sl in _atr_sl_mults:
        VWAP_PROFILES.append({
            "name": f"vwap_{band}sigma_atr_sl{sl}".replace(".", ""),
            "use_vwap_exit": True,
            "vwap_band": band,
            "use_atr_exit": True,
            "atr_tp_mult": 0,  # VWAPバンドでTP
            "atr_sl_mult": sl,
            "atr_period": 14,
        })


# =============================================================================
# BB系: BBバンドTP + ATR SL（3パターン）
# SL: ATR×1, 1.5, 2
# ロング→上限バンドでTP、ショート→下限バンドでTP
# =============================================================================
BB_PROFILES: List[ExitProfile] = []

for sl in _atr_sl_mults:
    BB_PROFILES.append({
        "name": f"bb_exit_atr_sl{sl}".replace(".", ""),
        "use_bb_exit": True,
        "bb_period": 20,
        "use_atr_exit": True,
        "atr_tp_mult": 0,  # BBバンドでTP
        "atr_sl_mult": sl,
        "atr_period": 14,
    })


# =============================================================================
# VWAPエントリー専用: トレンド用（ATRトレーリング + ATR SL、タイムアウトなし）
# Trail×(1, 1.5, 2) × SL×(1.5, 2) = 6パターン
# 用途: vwap_touch, vwap_upper1/lower1 でエントリー後、トレンドを伸ばす
# =============================================================================
VWAP_TREND_PROFILES: List[ExitProfile] = []

_vwap_trend_sl_mults = [1.5, 2.0]

for trail in _atr_trail_mults:
    for sl in _vwap_trend_sl_mults:
        VWAP_TREND_PROFILES.append({
            "name": f"vwap_trend_trail{trail}_sl{sl}".replace(".", ""),
            "use_atr_exit": True,
            "atr_tp_mult": 0,  # TPなし（トレーリングで決済）
            "atr_sl_mult": sl,
            "atr_period": 14,
            "use_atr_trailing": True,
            "atr_trailing_mult": trail,
            # タイムアウトなし
        })


# =============================================================================
# VWAPエントリー専用: レンジ用（VWAP TP + ATR SL）
# TP: VWAPライン(0) or 反対側2σ(2) × SL×(1.5, 2) = 4パターン
# 用途: vwap_2sigma_long/short でエントリー後、中央回帰または反対側バンドで利確
# =============================================================================
VWAP_RANGE_PROFILES: List[ExitProfile] = []

_vwap_range_tp_bands = [0, 2]  # 0=VWAPライン, 2=反対側2σ

for tp_band in _vwap_range_tp_bands:
    for sl in _vwap_trend_sl_mults:
        band_name = "vwap" if tp_band == 0 else "2sigma"
        VWAP_RANGE_PROFILES.append({
            "name": f"vwap_range_tp{band_name}_sl{sl}".replace(".", ""),
            "use_vwap_exit": True,
            "vwap_band": tp_band,
            "use_atr_exit": True,
            "atr_tp_mult": 0,  # VWAPバンドでTP
            "atr_sl_mult": sl,
            "atr_period": 14,
        })


# =============================================================================
# 全プロファイル
# =============================================================================
ALL_PROFILES: List[ExitProfile] = (
    ATR_PROFILES
    + ATR_TRAILING_PROFILES
    + ATR_HYBRID_PROFILES
    + VWAP_PROFILES
    + BB_PROFILES
    + VWAP_TREND_PROFILES
    + VWAP_RANGE_PROFILES
)


def get_profiles(mode: str = "all") -> List[ExitProfile]:
    """
    指定モードの exit profiles を取得

    Args:
        mode: "all", "atr", "atr_trailing", "atr_hybrid", "vwap", "bb",
              "vwap_trend", "vwap_range", "vwap_entry"

    Returns:
        ExitProfile のリスト
    """
    modes = {
        "atr": ATR_PROFILES,
        "atr_trailing": ATR_TRAILING_PROFILES,
        "atr_hybrid": ATR_HYBRID_PROFILES,
        "vwap": VWAP_PROFILES,
        "bb": BB_PROFILES,
        "vwap_trend": VWAP_TREND_PROFILES,
        "vwap_range": VWAP_RANGE_PROFILES,
        "vwap_entry": VWAP_TREND_PROFILES + VWAP_RANGE_PROFILES,  # VWAPエントリー専用全て
        "all": ALL_PROFILES,
    }
    return modes.get(mode, ALL_PROFILES)


def profile_to_exit_config(profile: ExitProfile) -> Dict[str, Any]:
    """ExitProfile 辞書から exit config を生成（name を除外）"""
    return {k: v for k, v in profile.items() if k != "name"}


def list_profiles() -> None:
    """全プロファイル一覧を表示（デバッグ用）"""
    print(f"=== Exit Profiles ({len(ALL_PROFILES)}パターン) ===\n")

    print(f"ATR系 ({len(ATR_PROFILES)}パターン):")
    for p in ATR_PROFILES:
        print(f"  {p['name']}: TP=ATR×{p['atr_tp_mult']}, SL=ATR×{p['atr_sl_mult']}")

    print(f"\nATRトレーリング系 ({len(ATR_TRAILING_PROFILES)}パターン):")
    for p in ATR_TRAILING_PROFILES:
        print(f"  {p['name']}: Trail=ATR×{p['atr_trailing_mult']}, Timeout={p['timeout_bars']}bars")

    print(f"\nATRハイブリッド系 ({len(ATR_HYBRID_PROFILES)}パターン):")
    for p in ATR_HYBRID_PROFILES:
        print(f"  {p['name']}: Trail=ATR×{p['atr_trailing_mult']}, SL=ATR×{p['atr_sl_mult']}")

    print(f"\nVWAP系 ({len(VWAP_PROFILES)}パターン):")
    for p in VWAP_PROFILES:
        print(f"  {p['name']}: TP=VWAP±{p['vwap_band']}σ, SL=ATR×{p['atr_sl_mult']}")

    print(f"\nBB系 ({len(BB_PROFILES)}パターン):")
    for p in BB_PROFILES:
        print(f"  {p['name']}: TP=BBバンド, SL=ATR×{p['atr_sl_mult']}")


if __name__ == "__main__":
    list_profiles()
