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

## 現在の状態: Step 15進行中（自律探索）

### 確定した実運用推奨戦略

| # | 戦略 | レジーム | Exit | ROBUST率 | 平均PnL | 用途 |
|---|------|---------|------|---------|--------|------|
| 1 | **rsi_bb_long_f35** | **uptrend** | **tp20固定** | **37% (20/54)** | **+9.9%** | **UT安定重視（推奨）** |
| 2 | rsi_bb_long_f35 | uptrend | tp30固定 | 39% (21/54) | +14.1% | UT利益重視 |
| 3 | **adx_bb_long** | **uptrend** | **tp30固定** | **37% (20/54)** | **+10.4%** | **UT第2戦略（新発見★）** |
| 4 | ema_fast_cross_bb_short | downtrend | tp15固定 | 24% (13/54) | +2.3% | DT向け |

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

### 次のステップ（候補）

1. **adx_bb_long確定戦略YAMLを作成**
2. **rsi_bb_long_f35 vs adx_bb_long の相互補完分析**（銘柄重複度、シグナル非相関性）
3. **DT戦略YAML作成**: ema_fast_cross_bb_short / downtrend / tp15
4. **3戦略複合運用シミュレーション**: rsi_bb + adx_bb + ema_fast_cross_bb の年間パフォーマンス推定
5. **自動売買bot開発**: 確定3戦略をリアルタイム自動売買に実装

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
