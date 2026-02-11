# 汎用戦略探索ロードマップ

## ゴール
3銘柄以上のOOSで機能する実運用可能な汎用戦略を見つける

## 固定済みの前提
- レジーム: Dual-TF EMA (4h+1h)
- Exit: atr_compact (SL=ATR×2.0, TP 3択)
- OOS: train60/val20/test20, min_oos_trades=20
- データ: 30銘柄 × 3年(2023-02〜2026-01)
- Modal Volume: 30銘柄×3期間×3TF = 約285ファイル

---

## 現在の状態: Step 16 — 次の一手検討中

### 確定した実運用推奨戦略（2026-02-11 バグ修正後WFA再検証）

| # | 戦略 | レジーム | Exit | ROBUST率 | 平均PnL | 状態 |
|---|------|---------|------|---------|--------|------|
| 1 | **rsi_bb_long_f35** | **uptrend** | **tp20固定** | **35% (27/78)** | **+6.1%** | **唯一の実運用候補** |
| 2 | rsi_bb_long_f35 | uptrend | tp30固定 | 24% (19/78) | -2.3% | ❌ PnL負転で除外 |
| 3 | adx_bb_long | uptrend | tp30固定 | 22% (17/78) | +2.1% | ❌ ROBUST率不足で除外 |
| 4 | ema_fast_cross_bb_short | downtrend | tp15固定 | 17% (13/78) | -0.1% | ❌ PnL負転で除外 |

### UT ROBUST銘柄（rsi_bb_long_f35/tp20, CR>=0.6）
| 銘柄 | ROBUST/3 | 平均PnL |
|------|----------|---------|
| **NEARUSDT** | **3/3** | **+29.1%** |
| SUIUSDT | 2/3 | +44.2% |
| FILUSDT | 2/3 | +29.9% |
| UNIUSDT | 2/3 | +29.6% |
| ADAUSDT | 2/3 | +28.8% |
| SOLUSDT | 2/3 | +17.3% |

### DT ROBUST銘柄（ema_fast_cross_bb_short/tp15）
**Tier 1（2期間）:** FIL(+4.5%), OP(+6.2%), ADA, ETH
**Tier 2（1期間）:** APT, ARB, BNB, LINK, NEAR

---

## Step 14: DT探索の核心的発見

- **EMAモメンタム確認 > RSI閾値**: EMA death cross(5/13)の方がRSI>60-67より安定
- **BB middle必須**: 無し=13% → 有り=24%
- **高速EMA(5/13) > 標準EMA(9/21)**: 24% > 20%
- **トリプル条件は過剰**: EMA+RSI+BB→トレード0件
- **tp15最適**: DT=小さい利幅で確実に

---

## Step 14b: VP Deep Dive（2026-02-09完了）

ユーザーのVP押し目仮説を徹底検証。6種のVP複合テンプレートを作成し、26銘柄×3期間でOOS検証。

### 追加テンプレート（#56-#61）
| # | テンプレート | 概要 | 結果 |
|---|---|---|---|
| 56 | vp_pullback_wide | パラメータ緩和版（n_bins小, tolerance大） | 全滅（max 4銘柄） |
| 57 | vp_rsi_pullback_long | LVN first_touch + RSI<40 | 全滅（1銘柄のみ） |
| 58 | vp_bb_pullback_long | LVN first_touch + BB下半分 | 全滅（1銘柄のみ） |
| 59 | vp_poc_rsi_long | POC接近 + RSI<45 | 27% OOS（PnLマイナス） |
| 60 | vp_hvn_rsi_long | HVN接近 + RSI<45 | **25% OOS（最高）** |
| 61 | vp_pullback_short | LVN ショート版 | 全滅（3銘柄のみ） |

### 結論
- **first_touch系はトレード数不足が根本的に解決不能**（パラメータ緩和しても年20件未達）
- **POC/HVN系はトレード数は確保できたが25-27%止まり**（50%閾値未達）
- **VP戦略は実運用不可** → WFA検証不要
- Run ID: 20260209_102855

---

## Step 15: 自律探索Iteration 1（2026-02-09）

### ADX+BB Long 発見
- 8テンプレート作成、4回スカウト、1回フル検証、2回WFA検証
- **adx_bb_long / uptrend / tp30 = WFA ROBUST 37%（既存と同率！）**
- ADX（トレンド強度） + BB lower = RSI+BBとは異なるシグナルで相互補完可能

### 不採用テンプレート（#62-#69）
- ema_fast_cross_bb_long/ema_fast_bb_lower_long → crossover条件でトレード不足
- ema_state_bb_lower/mid_long → UT向けPnLマイナス
- di_bb_lower_long/di_bb_upper_short → 全滅

### 核心的教訓
- **crossover（瞬間）条件はロング側で機能しにくい**（DT shortでは有効）
- **EMAモメンタム確認はUT側では逆効果**
- **ADX（トレンド強度フィルタ）はBB lowerとの組み合わせでUT向けに有効**
- **tp30がADX戦略の最適Exit**（OOS: tp30=30% > tp20=27%、WFA: tp30=37% > tp20=28%）

---

## Step 15: 致命的バグ修正 + WFA再検証（2026-02-11）✅

### Codex MCPで全モジュール総点検 → 致命的バグ6件発見・修正

| # | バグ内容 | 影響 | 修正内容 |
|---|---------|------|---------|
| 1 | **レジーム判定ルックアヘッドバイアス** | **最大要因** | HTF closeを1バーシフト（`analysis/trend.py`） |
| 2 | インジケータキャッシュのtrain/test漏洩 | 中 | キーにデータ範囲追加（`indicators/base.py`） |
| 3 | WFA CR分母不正 | 中 | 全フォールド数基準に修正（`scripts/local_wfa_test.py`） |
| 4 | WFA OOS PnL単純加算 | 中 | 複利計算化（`scripts/local_wfa_test.py`） |
| 5 | RSI adjust=True→False | 小 | Wilder's smoothing統一（`indicators/rsi.py`） |
| 6 | Timeout判定がTP/SLより優先 | 小 | 優先順位修正（`engine/backtest.py`） |

### WFA再実行結果（2026-02-11）

| 戦略 | レジーム | Exit | ROBUST率（変化） | 平均PnL（変化） | 結論 |
|------|---------|------|---------------|--------------|------|
| rsi_bb_long_f35 | uptrend | tp20 | 35% (37%→35%, -2pt) | +6.1% (+9.9%→+6.1%, -3.8pt) | **唯一の実運用候補** |
| rsi_bb_long_f35 | uptrend | tp30 | 24% (39%→24%, **-15pt**) | -2.3% (+14.1%→-2.3%, **-16.4pt**) | ❌ PnL負転で除外 |
| adx_bb_long | uptrend | tp30 | 22% (37%→22%, **-15pt**) | +2.1% (+10.4%→+2.1%, -8.3pt) | ❌ ROBUST率不足で除外 |
| ema_fast_cross_bb_short | downtrend | tp15 | 17% (24%→17%, -7pt) | -0.1% (+2.3%→-0.1%, -2.4pt) | ❌ PnL負転で除外 |

### 核心的発見

- **tp30系の大幅劣化（-15pt）= ルックアヘッド依存の証拠**
  - レジーム判定が1バー先の情報を使っていたため、tp30のような大きな利幅でも高ROBUST率を記録
  - 修正後は現実的な情報のみでの判定となり、tp30は機能不全に
- **tp20は堅牢性が高い**（-2pt）
  - 小さい利幅 = レジーム判定の誤差の影響を受けにくい
  - 唯一実運用可能な水準（35%）を維持
- **DT戦略もルックアヘッド依存**
  - ema_fast_cross_bb_short も-7pt低下し、PnL負転

### 次のステップ（候補）

1. **rsi_bb_long_f35/tp20の銘柄選別強化**（ROBUST銘柄に絞った運用）
2. **DT戦略の再探索**（tp10等のより小さい利幅で再検証）
3. **レジーム検出精度の改善**（1バーシフト後も機能する新手法の探索）
4. **Guardian Bot開発に移行**（急変動キャッチ戦略）

---

## Phase 2: Guardian Bot（急変動キャッチ戦略）★ユーザー発案

**コンセプト**: 急騰急落を「守る」のではなく「利益にする」。Prismのエッジ戦略とは逆相関の全天候型ポートフォリオ構築。

### 設計思想
- **Prism（Layer 1）**: 通常相場でコツコツ利益を積む（mean reversion）
- **Guardian（Layer 2）**: 急変動を検知してその方向にポジションを取る（momentum/breakout）
- 2層を組み合わせることで**負ける環境がほぼない構造**を目指す

### 実装ステップ（案）
| # | タスク | 内容 | 依存 |
|---|--------|------|------|
| G1 | 閾値バックテスト | 過去3年で「N分でX%」の最適閾値を検証。空振り回数 vs 大暴落キャッチ回数のトレードオフ分析 | Prismの既存データで可能 |
| G2 | ペイオフ分析 | 空振り損失の合計 vs 大暴落利益の合計を算出。期待値がプラスか検証 | G1 |
| G3 | Prism+Guardian複合シミュレーション | Layer 1の損益 + Layer 2の損益 = 全天候ポートフォリオのパフォーマンス推定 | G2 + Step候補4 |
| G4 | Guardian Bot実装 | 独立プロセスとして実装。5秒間隔の価格監視 + 上下ストップ注文管理 | G3承認後 |
| G5 | 統合テスト | Prism Bot + Guardian Bot の同時稼働テスト（testnet） | G4 + Step候補5 |

### 監視シグナル候補（優先度順）
| Lv | シグナル | 内容 | 実装難易度 |
|----|---------|------|-----------|
| 1 | 価格速度検知 | N分でX%動いたらエントリー | 超簡単（OHLCV） |
| 2 | ボラ急拡大 | 実現ボラがNσ超え | 簡単 |
| 3 | 出来高異常 | 通常の5倍+方向性 | 簡単 |
| 4 | Funding Rate | 0.1%超え=過熱 | 中（API） |
| 5 | OI急変 | 建玉1hで10%減 | 中（API） |

### 期待される効果
| 市場環境 | Prismのみ | Prism+Guardian |
|---------|----------|---------------|
| 通常上昇 | +コツコツ | +コツコツ - 空振り小 = **+プラス** |
| 通常下落(DT) | +EMA Short | +EMA Short - 空振り小 = **+プラス** |
| 大暴落 | **-損切り** | -損切り + **Guardian利益 = ±相殺〜プラス** |
| 大暴騰 | +利益 | +利益 + **Guardian利益 = ++大プラス** |

---

## 設計原則（確定）

- インジケーター期間は**固定**（RSI=14, BB=20/2σ, EMA=5/13）
- BB σ = **2.0固定**（σ拡張は逆効果）
- **3重複合は不要**（条件過剰でトレード数激減）
- **ロング: RSI<35固定 / tp20 が最も堅牢**（ROBUST 37%）
- **ショート: EMA(5)<EMA(13) + BB中間 / tp15 がDT最強**（ROBUST 24%）
- **exit固定がWFA安定性の鍵**
- **モメンタム確認(EMA cross) > 閾値条件(RSI>X)**
- **探索空間を小さくする = PASS率改善の鍵**
- **CR>=0.6が推奨閾値**（3/5フォールド黒字 = 合理的基準）

---

> Step 1-13の詳細記録: `.claude/memory/roadmap_archive.md`
