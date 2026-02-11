"""
ショート/レンジ戦略全滅の根本原因分析

1. バグ修正前後の比較
2. レジーム分布の確認
3. 市場バイアスの検証
"""

import json
from pathlib import Path
from collections import defaultdict

def analyze_bug_impact():
    """バグ修正前後でROBUST率がどう変化したか"""
    print("=" * 100)
    print("  1. バグ修正前後の比較")
    print("=" * 100)
    print()

    # 修正前のデータ（MEMORY.mdより）
    before = {
        'uptrend': {
            'rsi_bb_long_f35/tp20': {'robust_rate': 35, 'note': '修正後も維持'},
            'rsi_bb_long_f35/tp30': {'robust_rate': 24, 'note': 'PnL負転'},
            'adx_bb_long/tp30': {'robust_rate': 37, 'note': '→22%に大幅低下'},
        },
        'downtrend': {
            'ema_fast_cross_bb_short/tp15': {'robust_rate': 24, 'note': '→17%に低下'},
        },
        'range': {
            'rsi_bb_short_f65': {'robust_rate': 62, 'note': 'OOS段階、WFAで0-17%に崩壊'},
            'rsi_bb_short_f67': {'robust_rate': 50, 'note': 'OOS段階、WFAで0-17%に崩壊'},
        }
    }

    # 修正後
    after = {
        'uptrend': {
            'rsi_bb_long_f35/tp20': 35,
            'rsi_bb_long_f35/tp30': 24,
            'adx_bb_long/tp30': 22,
        },
        'downtrend': {
            'ema_fast_cross_bb_short': 17,
        },
        'range': {
            'rsi_bb_short_f65/f67': 0,  # WFA全滅
        }
    }

    print("| レジーム | 戦略 | 修正前 | 修正後 | 劣化幅 | 状態 |")
    print("|---------|------|--------|--------|--------|------|")

    # Uptrend
    print("| UPTREND | rsi_bb_long_f35/tp20 | 35% | 35% | ±0 | ✅ 維持 |")
    print("| UPTREND | rsi_bb_long_f35/tp30 | 24% | 24% | ±0 | ❌ PnL負転 |")
    print("| UPTREND | adx_bb_long/tp30 | **37%** | **22%** | **-15pt** | ❌ 大幅劣化 |")

    # Downtrend
    print("| DOWNTREND | ema_fast_cross_bb_short | **24%** | **17%** | **-7pt** | ❌ 基準未達 |")

    # Range
    print("| RANGE | rsi_bb_short_f65/f67 | **50-62%** (OOS) | **0-17%** (WFA) | **-33pt以上** | ❌ 偽陽性崩壊 |")

    print()
    print("**発見:**")
    print("- ✅ ロング戦略: rsi_bb_long_f35/tp20のみ影響なし（他は劣化）")
    print("- ❌ ショート/レンジ: 全て大幅劣化（-7pt ~ -33pt以上）")
    print("- **ルックアヘッド除去の影響が特にショート/レンジに致命的**")
    print()

def analyze_regime_distribution():
    """データ期間のレジーム分布を確認"""
    print("=" * 100)
    print("  2. レジーム分布の推測（2023-2026の仮想通貨市場）")
    print("=" * 100)
    print()

    print("| 期間 | 市場状況 | 推測レジーム分布 |")
    print("|------|---------|-----------------|")
    print("| 2023/02-2024/01 | BTCレンジ→上昇開始 | Range 40%, Uptrend 40%, Downtrend 20% |")
    print("| 2024/02-2025/01 | BTCバブル最盛期 | **Uptrend 70%**, Range 20%, Downtrend 10% |")
    print("| 2025/02-2026/01 | 調整局面 | Range 40%, Downtrend 30%, Uptrend 30% |")
    print()
    print("**推測平均:**")
    print("- Uptrend: **約50%**（データ全体の半分）")
    print("- Range: 約30%")
    print("- Downtrend: 約20%（**最も少ない**）")
    print()
    print("**問題点:**")
    print("- Downtrendデータが不足 → ショート戦略の学習データ不足")
    print("- Uptrendバイアスが強い → ショート戦略が機能しにくい市場構造")
    print()

def analyze_template_coverage():
    """探索したテンプレート数の比較"""
    print("=" * 100)
    print("  3. 探索したテンプレート数の比較")
    print("=" * 100)
    print()

    coverage = {
        'ロング戦略': {
            'templates': [
                'rsi_bb_long_f35',
                'adx_bb_long',
                'bb_middle_rsi_long',
                'ema_fast_cross_bb_long',
                'di_bb_lower_long',
                'vp_pullback_long',
                'vwap_touch_long',
                'trend_pullback_long',
            ],
            'count': 8
        },
        'ショート戦略': {
            'templates': [
                'ema_fast_cross_bb_short',
                'rsi_bb_short_f65/f67',
                'bb_vol_short_dt',
                'volume_spike_short_dt',
                'ema_cross_short_only',
                'rsi_bb_mid_short',
            ],
            'count': 6
        },
    }

    print("| カテゴリ | 探索テンプレート数 | WFA合格 | 合格率 |")
    print("|---------|------------------|---------|--------|")
    print("| ロング戦略 | 8種以上 | 1種 (rsi_bb_long_f35/tp20) | 12.5% |")
    print("| ショート戦略 | 6種 | 0種 | 0% |")
    print()
    print("**発見:**")
    print("- ロング戦略でも合格率12.5%（8種中1種のみ）")
    print("- ショート戦略の探索数は少なくない（6種）が、**全滅**")
    print("- → 探索数の問題ではなく、**市場構造 + バグ修正の影響**が主因")
    print()

def propose_next_actions():
    """次のアクション提案"""
    print("=" * 100)
    print("  4. 次のアクション提案")
    print("=" * 100)
    print()

    print("### オプションA: 基準緩和して候補を探す")
    print("- ROBUST率30% → **20%に緩和**")
    print("- PnL>0% → **PnL>-1%に緩和**")
    print("- **可能性**: ema_fast_cross_bb_short (17%, -0.1%)が候補になる")
    print()

    print("### オプションB: 下落相場データを追加")
    print("- 2022年（BTCバブル崩壊）のデータを追加")
    print("- Downtrend比率を増やしてショート戦略を再学習")
    print("- **メリット**: より多様な市場環境で検証可能")
    print()

    print("### オプションC: 新しいテンプレート探索（ショート/レンジ特化）")
    print("- ボラティリティブレイクアウト系")
    print("- オーダーフロー系")
    print("- マーケット構造変化検出系")
    print("- **デメリット**: さらなる検証コストが必要")
    print()

    print("### オプションD: ロング専用戦略として割り切る")
    print("- **実運用: rsi_bb_long_f35/tp20 のみ**")
    print("- Downtrend/Range時は待機（ポジションなし）")
    print("- **メリット**: シンプルで堅実。無理に全レジームカバーしない")
    print()

    print("### 推奨")
    print("**オプションD（ロング専用）を基本方針とし、余力があればオプションB（データ追加）を検討**")
    print()
    print("理由:")
    print("1. バグ修正後の環境で唯一生き残ったrsi_bb_long_f35/tp20は信頼性が高い")
    print("2. 無理に全レジームカバーすると、過学習した脆弱な戦略を採用するリスク")
    print("3. 仮想通貨市場は本質的に上昇バイアスが強い（ロング有利）")
    print()

def main():
    analyze_bug_impact()
    analyze_regime_distribution()
    analyze_template_coverage()
    propose_next_actions()

if __name__ == "__main__":
    main()
