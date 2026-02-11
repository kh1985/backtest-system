# 戦略研究 統合レポート

**作成日**: 2026-02-11
**チーム**: strategy-research
**目的**: 下落トレンド対応・急変動キャッチ戦略の実装優先順位決定

---

## エグゼクティブサマリー

### 調査結果の統合

3つの観点から戦略を調査:
1. **下落トレンド対応**（ショート戦略）
2. **急変動キャッチ**（ブレイクアウト戦略）
3. **レンジ対応**（トラリピ戦略）

### 核心的発見

**現状の課題:**
- UT（uptrend）戦略: **50% ROBUST**達成（11銘柄）← 実用可
- DT（downtrend）戦略: **27% ROBUST**（6銘柄）← 実用不可
- Range戦略: 未検証（既存VP戦略は全滅）

**根本原因:**
- ショート戦略は**銘柄横断で機能しにくい**（個別最適化が必要）
- 既存DT戦略（ema_fast_cross_bb_short）はexit最適化でも改善不十分
- **新しいシグナル・指標が必要**

---

## 実装優先順位（ユーザー要求順）

### 優先度1: 下落トレンド対応（最優先）

#### 候補1-A: BB Squeeze Breakout Short ⭐推奨

**ロジック:**
```
エントリー条件:
- BB Squeeze検出（bandwidth < 閾値）
- close < BB lower（下方ブレイク）
- ADX >= 25（トレンド強度確認）
- volume >= avg_volume * 2.0（需給確認）
- EMA(5) < EMA(13)（モメンタム方向確認）

決済:
- TP: ATR × 1.5 または固定1.5%
- SL: ATR × 2.0
- Timeout: 100 bars
```

**期待効果:**
- **既存DT戦略（EMA+BB）との違い**: ボラティリティ圧縮フィルター + 出来高フィルター
- GogoJungle実績: Break Scal System（勝率90%超）
- 暗号資産最新トレンド: Sharpe 1.0-1.2達成例あり

**実装難易度:** 🟢 低
- 既存指標のみ（BB, ADX, EMA, Volume）
- パラメータ固定推奨（過学習回避）
- `bb_squeeze_breakout_short` テンプレート追加のみ

**リスク:**
- トレード数不足の可能性（Squeeze頻度次第）
- 仮想通貨は常時高ボラ → FXよりSqueeze検出困難かも

**推奨アクション:**
1. テンプレート追加（`optimizer/templates.py`）
2. OOS検証（26銘柄 × 3期間）
3. ROBUST率 >= 30%ならWFA実行

---

#### 候補1-B: ADX+DMI+BB Short

**ロジック:**
```
エントリー条件:
- ADX >= 25
- -DI > +DI（下降トレンド確認）
- close > BB middle（BB中間以上からのショート）

決済:
- TP: ATR × 1.5
- SL: ATR × 2.0
```

**期待効果:**
- DMI（+DI/-DI）でトレンド方向を直接判定
- ADXでトレンド強度を確認（ダマシ軽減）

**実装難易度:** 🟡 中
- **DMI指標の実装が必要**（indicators/trend.py に追加）
- テンプレート追加: `adx_dmi_bb_short`

**推奨アクション:**
1. DMI指標実装（Wilder's Directional Movement Index）
2. 候補1-Aの結果次第で検討

---

#### 候補1-C: 既存戦略の再評価（低優先度）

**対象:**
- `ema_fast_cross_bb_short`（現在ROBUST 27%）
- `rsi_bb_short_f67`（Range用、WFA不合格）

**アプローチ:**
- パラメータ微調整（EMA期間、BB σ等）
- Exit profileの再最適化（Priority B試行）

**評価:**
- Priority A exit検証で改善不十分（27%）
- 構造的な問題の可能性（新シグナルが必要）

**推奨:** 優先度低（候補1-A/1-Bを先行）

---

### 優先度2: 急変動キャッチ（Guardian Bot）

#### 候補2-A: BB Squeeze Breakout Long/Short（両方向）⭐推奨

**ロジック:**
```
# 候補1-Aのロング版を追加
エントリー条件（ロング）:
- BB Squeeze検出
- close > BB upper（上方ブレイク）
- ADX >= 25
- volume >= avg_volume * 2.0
- EMA(5) > EMA(13)

決済:
- TP: ATR × 1.5-2.0
- SL: ATR × 2.0
```

**期待効果:**
- ロング/ショート両方向で急変動をキャッチ
- 既存UT戦略（RSI+BB）との**相互補完**
  - RSI+BB = 逆張り（mean reversion）
  - BB Squeeze Breakout = 順張り（momentum）

**統合戦略:**
| 市場状況 | 戦略 | 期待効果 |
|---------|------|---------|
| **通常Uptrend** | RSI+BB Long（既存） | コツコツ利益 |
| **急騰** | BB Squeeze Breakout Long（新） | 大きく取る |
| **Downtrend** | BB Squeeze Breakout Short（新） | 下落でも利益 |
| **急落** | BB Squeeze Breakout Short（新） | 急落をキャッチ |

**実装難易度:** 🟢 低（候補1-Aと同じ）

**推奨アクション:**
1. 候補1-Aと同時実装（long/short両方）
2. OOS検証で両方向の性能を確認
3. 既存UT戦略（50% ROBUST）との組み合わせ効果を分析

---

#### 候補2-B: 価格速度検知（将来拡張）

**ロジック:**
```
# N分でX%動いたらエントリー
エントリー条件:
- ROC（Rate of Change）> 閾値
- volume > avg * 3.0
- ATR急上昇検出
```

**評価:**
- シンプルだがパラメータ調整が難しい（N, X の最適値）
- トレード数不足のリスク
- 候補2-Aで不十分な場合の次の選択肢

---

### 優先度3: レンジ対応（トラリピ）

#### 候補3-A: トラリピ（簡易版）

**設計詳細:** `/tmp/trapreate_design.md` 参照

**ロジック:**
```
エントリー条件:
- レジーム == "range"（Dual-TF EMA判定）
- BB範囲内にトラップ配置（trap_interval刻み）
- 価格がトラップに接触

決済:
- TP: profit_width（固定利幅）
- Timeout: 100 bars
```

**実装アプローチ:**
1. **Phase 1**: 簡易版（1ポジションモデル）
   - `TrapGridCondition` 追加
   - 既存エンジン改修不要
2. **Phase 2**: OOS/WFA検証（Range期間のみ）
3. **Phase 3**: 完全版（複数ポジション管理）

**期待効果:**
- Range期間での収益化（現在は休む）
- レンジ相場専用（トレンド期間は自動停止）

**実装難易度:** 🟡 中
- Phase 1は容易（Condition追加のみ）
- Phase 3は大規模（エンジン改修）

**リスク:**
- Range検出精度次第
- トレード数不足の可能性（Range期間が少ない）
- レンジ逸脱時の含み損

**推奨アクション:**
1. 候補1-A/2-Aの結果を見てから判断
2. DT/急変動対応が優先

---

## 実装ロードマップ

### Phase 1: BB Squeeze Breakout 戦略（即座）⭐

**実装対象:**
- `bb_squeeze_breakout_long`
- `bb_squeeze_breakout_short`

**実装手順:**
1. テンプレート追加（`optimizer/templates.py`）
   ```python
   ParameterRange("adx_threshold", 20, 30, 5, "int")  # 20, 25, 30
   ParameterRange("volume_mult", 1.5, 2.5, 0.5, "float")  # 1.5, 2.0, 2.5
   ParameterRange("squeeze_threshold", 0.05, 0.15, 0.05, "float")  # BB bandwidth閾値
   # 探索空間: 3 × 3 × 3 = 27 configs（小さい→過学習リスク低）
   ```

2. BB Squeeze検出ロジック追加（`strategy/conditions.py`）
   ```python
   class BBSqueezeCondition(Condition):
       def evaluate(self, row, prev_row):
           bandwidth = (row["bb_upper"] - row["bb_lower"]) / row["bb_middle"]
           return bandwidth < self.squeeze_threshold
   ```

3. Modal OOS検証
   ```bash
   python3 -m modal run scripts/modal_optimize.py \
     --templates bb_squeeze_breakout_long bb_squeeze_breakout_short \
     --regimes uptrend downtrend \
     --exit-profiles atr_tp15_sl15
   ```

4. WFA検証（ROBUST率 >= 30%が目標）

**期待成果:**
- DT戦略のROBUST率改善（27% → 35%以上を目標）
- UT戦略との相互補完（急騰時のパフォーマンス向上）

---

### Phase 2: トラリピ（条件付き）

**実装条件:**
- Phase 1でDT ROBUST率 < 30%の場合に検討
- またはRange戦略への需要が高い場合

**実装手順:**
1. `TrapGridCondition` 実装
2. `trap_repeat_long/short` テンプレート追加
3. Range期間のみでOOS/WFA検証

---

### Phase 3: DMI/Parabolic SAR（低優先度）

**実装条件:**
- Phase 1で不十分な場合
- 新しい指標の必要性が確認された場合

**実装対象:**
- DMI（Directional Movement Index）
- Parabolic SAR

---

## ROI分析

| 戦略 | 実装工数 | 期待ROBUST率 | リスク | ROI |
|------|---------|------------|-------|-----|
| **BB Squeeze Breakout** | 🟢 1日 | 35-40% | 🟡 中 | **⭐⭐⭐ 最高** |
| トラリピ（簡易版） | 🟡 2-3日 | 30-35% | 🟡 中 | ⭐⭐ 中 |
| ADX+DMI Short | 🟡 2日 | 30-35% | 🟡 中 | ⭐⭐ 中 |
| 既存戦略再評価 | 🟢 0.5日 | 25-30% | 🔴 高 | ⭐ 低 |

---

## リスク評価

### BB Squeeze Breakout戦略のリスク

| リスク | 対策 |
|--------|------|
| **トレード数不足** | Squeeze閾値を調整（0.05-0.15で探索） |
| **仮想通貨の高ボラ** | BB bandwidth正規化（ATR考慮） |
| **過学習** | パラメータ固定推奨（ADX=25, volume=2.0） |
| **ダマシ** | 複数フィルター（ADX + Volume + EMA） |

### トラリピ戦略のリスク

| リスク | 対策 |
|--------|------|
| **レンジ逸脱** | Dual-TF EMAでトレンド期間は停止 |
| **含み損積み重ね** | max_positions制限、SL設定 |
| **Range期間が少ない** | 事前にレジーム分布を分析 |
| **トレード数不足** | trap_interval調整、BB幅確認 |

---

## 最終推奨

### ユーザー要求の優先順位

1. **下落トレンドでも利益**（最優先）→ **BB Squeeze Breakout Short**
2. **急落・急騰をカバー**（次）→ **BB Squeeze Breakout Long/Short**

### 推奨実装順序

**即座に実装:**
1. **BB Squeeze Breakout Long/Short**（両方向）
   - 実装工数: 1日
   - 期待ROBUST率: 35-40%（DT）、40-50%（UT急騰時）
   - 既存UT戦略（50%）との相互補完

**条件付き実装:**
2. **トラリピ**（Phase 1結果次第）
   - 実装工数: 2-3日
   - Range期間専用

**低優先度:**
3. DMI/Parabolic SAR
4. 既存戦略再評価

---

## 次のアクション

### ユーザー承認が必要

**質問:**
1. **BB Squeeze Breakout Long/Short を実装開始しますか？**
   - Yes → Phase 1実装開始（1日で完了予定）
   - No → 他の候補を検討

2. **トラリピは検討しますか？**
   - Yes → Phase 1後に実装
   - No → スキップ

3. **他に調査・検討すべき戦略はありますか？**

---

## 参考資料

### 調査レポート
- `/tmp/downtrend_strategies.md`（下落トレンド対応）
- `/tmp/breakout_strategies.md`（急変動キャッチ）
- `/tmp/trapreate_design.md`（トラリピ設計）

### GogoJungle参考戦略
- Break Scal System（勝率90%超）
- BB Squeeze + ADX + Volume
- Volatility Momentum Breakout Strategy（Sharpe 1.0-1.2）

---

**作成者**: strategy-research team (team-lead)
**承認待ち**: ユーザー判断
