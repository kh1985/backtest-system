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

## 現在の状態: Step 19 — UT+DT ポートフォリオ確定

### 確定した実運用推奨戦略（2026-02-11 Step 17完了）

**★ 新最強戦略（Step 17で確定）:**
| 戦略 | レジーム | Exit | ROBUST率 | 平均PnL | Stability | 状態 |
|------|---------|------|---------|--------|----------|------|
| **rsi_bb_long_f35** | **uptrend** | **atr_tp15_sl15** | **50% (11/22)** | **+12.6%** | **100%** | **✅ 実運用最優先** |

**旧戦略（比較用）:**
| # | 戦略 | レジーム | Exit | ROBUST率 | 平均PnL | 状態 |
|---|------|---------|------|---------|--------|------|
| 1 | rsi_bb_long_f35 | uptrend | tp20固定 | 35% (27/78) | +6.1% | ⚠️ 旧戦略（新戦略に劣る） |
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

## Step 17: Priority A Exit Profiles 検証（2026-02-11）✅

### 背景
これまでexit profileは73種定義されていたが、実際に検証されたのは1種（atr_tp15_sl20）のみ。
2人のOpusチームで全73種を分類し、Priority A（29種）を優先検証することに決定。

### 検証内容
- **テンプレート**: rsi_bb_long_f35（UT最強戦略）
- **レジーム**: uptrend
- **Exit profiles**: Priority A 29種（ATR固定8種 + ATRトレーリング21種）
- **銘柄**: 22銘柄（データあり）
- **実行時間**: 36秒（Modal並列）
- **Run ID**: 20260211_174446_wfa

### 結果（Top 3 Exit Profiles）

| Exit Profile | ROBUST率 | 選択回数 | 平均PnL | 評価 |
|-------------|----------|---------|--------|------|
| **atr_tp20_sl15** | **100% (2/2)** | 12回 | **+35.2%** | ⭐最強（サンプル少） |
| **atr_tp15_sl15** | **62.5% (5/8)** | 26回 | **+6.5%** | ⭐実用的（サンプル多） |
| atr_trail20_to6 | 50% (1/2) | 15回 | +4.7% | 検討 |

### 核心的発見

1. **SL=ATR×1.5が最適**
   - これまでSL=2.0固定で検証していたが、SL=1.5の方がROBUST
   - atr_tp20_sl15: 100% ROBUST（ただし2銘柄のみ）
   - atr_tp15_sl15: 62.5% ROBUST（8銘柄で検証）

2. **TP/SL固定 > トレーリング**
   - 上位2つはATR固定（TP/SL明示）
   - トレーリングは3位以下（50%）

3. **tp15_sl15が最もバランス良い**
   - 8銘柄で検証済み、62.5% ROBUST
   - 平均PnL +6.5%（安定的）

### PASS銘柄（9/22 = 41%）

トップ3: NEAR (100% CR, +35.1%), SUI (80% CR, +61.2%), FIL (80% CR, +9.1%)

### ★ atr_tp15_sl15固定で全銘柄WFA再実行（完了）

**結果**:
- **ROBUST率**: **50% (11/22)** ← 既存35%から**+15pt改善**
- **平均PnL**: +12.6% ← 既存+6.1%の**2倍**
- **Stability**: **100%（全銘柄）** ← 大幅改善
- **Run ID**: 20260211_175749_wfa

**PASS銘柄（11/22）**:
- **Tier 1（CR 80-100%）**: NEAR(100%, +43.3%), OP(80%, +42.1%), SUI(80%, +31.4%), FIL(80%, +13.8%)
- **Tier 2（CR 60%）**: UNI(+23.2%), ARB(+16.6%), ATOM(+11.2%), ADA(+10.5%), AAVE(+7.3%), LTC(+4.9%), LINK(-8.3%)

**結論**: **rsi_bb_long_f35 / atr_tp15_sl15が新最強戦略として確定**

### 次のステップ候補

1. ~~atr_tp15_sl15でWFA再実行~~（完了・50% ROBUST達成）
2. **Priority B（21種）の検証**（Timeout極端、VWAP exit等）
3. **SL=1.5固定での再探索**（他のテンプレートでも検証）
4. **実運用フェーズ移行**（11銘柄ポートフォリオで運用開始）

---

## Step 18: DT/Range 新アプローチ調査（2026-02-11）✅

### 背景
Priority A exit検証でDT戦略が改善不十分（27% ROBUST）。FR/OI等の新データなしで別アプローチを模索。strategy-researchチーム（3人のSonnetエージェント）でGogoJungle調査・トラリピ設計を実施。

### 調査内容
- **GogoJungle**: FX/仮想通貨EA販売サイト（4,364種以上）
- **調査対象**: 下落トレンド対応、急変動キャッチ、レンジ戦略（トラリピ）
- **制約**: OHLCVデータのみ（FR/OI/CVD不使用）

### 最優先推奨: BB Squeeze Breakout Long/Short ⭐

**核心ロジック:**
```
エントリー条件:
- BB Squeeze検出（bandwidth < 閾値）← 新要素
- close > BB upper (Long) / < BB lower (Short)
- ADX >= 25（トレンド強度）
- volume >= avg × 2.0（需給確認）← 新要素
- EMA(5) vs EMA(13)（モメンタム方向）

決済:
- TP: ATR × 1.5、SL: ATR × 2.0
```

**既存DT戦略との違い:**
| 項目 | 既存（EMA+BB） | 新（BB Squeeze Breakout） |
|------|--------------|------------------------|
| エントリー | EMA crossのみ | ボラティリティ圧縮 + 出来高フィルター |
| ROBUST率 | 27% | **35-40%（目標）** |
| 実績 | Prism独自 | GogoJungle: 勝率90%超、Sharpe 1.0-1.2 |

**期待効果:**
- **DT**: 27% → 35-40% ROBUST
- **UT急騰時**: 既存50%戦略と相互補完（順張り vs 逆張り）

**実装難易度:** 🟢 低（1日で完了）
- 既存指標のみ（BB, ADX, EMA, Volume）
- 新Condition: `BBSqueezeCondition`
- パラメータ: 27 configs（過学習リスク低）

### トラリピ（レンジ対応）

**設計詳細:** `/tmp/trapreate_design.md`

**アプローチ:**
- Phase 1: 簡易版（1ポジションモデル、既存エンジン改修不要）
- Phase 2: OOS/WFA検証（Range期間のみ、ROBUST率30%目標）
- Phase 3: 完全版（複数ポジション管理、エンジン改修）

**レンジ検出:** Dual-TF EMA Range判定 + BB範囲トラップ

**優先度:** 条件付き（BB Squeeze結果次第）

### 調査成果物
- `/tmp/integrated_strategy_report.md`（統合レポート）
- `/tmp/downtrend_strategies.md`（下落トレンド対応）
- `/tmp/breakout_strategies.md`（急変動キャッチ）
- `/tmp/trapreate_design.md`（トラリピ設計）

### 次のアクション候補

1. ~~BB Squeeze Breakout 実装~~ → 不要（pullback_sma_shortで十分）
2. ~~トラリピ実装~~ → 不要（Range待機方針に決定）
3. ~~Priority B exit検証~~ → 不要（SL最適化で解決）
4. **実運用Bot設計・実装** ← 最優先
5. **Guardian Bot（G1）** ← 次優先

---

## Step 19: DT戦略確立 + UT+DTポートフォリオ確定（2026-02-13）✅

### やがみ式パターン探索
- 5テンプレート実装（reversal_high, bearish_engulfing, wick_fill, ema_crossdown, bearish_engulfing_dt）
- WFA結果: 20-30% PASS（銘柄限定Go）
- 15m足ではDT全滅→1h足で改善（DT構造は時間足レベルで機能する発見）

### pullback_sma_short 発見・最適化
- SMAタッチ拒否→ショート（high>=SMA AND close<SMA）
- 既存column_compare条件で実装、パラメータ3つ（sma_period=15/20/25）
- **SL最適化: SL=2.0→SL=1.0で大幅改善**（AVAX PnL 2倍、ADA 3.3倍）
- DT WFA: **14/30 (47%) PASS** — 中小型アルトが主戦場
- トレーリングストップ: DT全般で全滅確定

### SL最適化の知見
- DT: SL=1.0×ATR（狭い）が最適 — エントリー根拠が明確なので即損切り
- UT: SL=1.5×ATR が最適（Step 17で確定済み）
- レジームごとに最適SLが異なる

### 30銘柄WFA拡大
- DT: 10銘柄(3 PASS) → 30銘柄(**14 PASS**)。中小型アルトで大幅改善
- UT: 22銘柄(11 PASS) → 30銘柄(10 PASS)。UTは大型銘柄寄りで安定

### UT+DT ポートフォリオ確定
| レジーム | 戦略 | TF | Exit | PASS |
|---|---|---|---|---|
| **UT** | rsi_bb_long_f35 | 15m:1h | atr_tp15_sl15 | 10/30 (33%) |
| **DT** | pullback_sma_short | 1h:4h | atr_tp15_sl10 | 14/30 (47%) |
| **Range** | 待機（ノーポジ） | - | - | - |

### 全天候型銘柄（UT+DT両方PASS）
PEPE(+161%), WIF(+77%), OP(+51%), APT(+47%), JUP(+41%), ADA(+26%)

### 理論APY
- 保守: 年10-12%、標準: 年15-20%、集中: 年25-35%

### 確定事項
- Range戦略は不要（UT+DTでカバー、Range期間はノーポジ待機）
- BB Squeeze Breakout不要（既存DT戦略で十分なPASS率）
- 年間トレード数: DT中に約170回（月14回、週3-4回）

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
- **ロング: RSI<35 + BB lower / tp15_sl15 が最強**（ROBUST 50%→33% @30銘柄）
- **ショート: SMAタッチ拒否 / tp15_sl10 がDT最強**（ROBUST 47% @30銘柄）
- **SL最適化はレジーム依存**（UT: SL=1.5, DT: SL=1.0）
- **exit固定がWFA安定性の鍵**
- **モメンタム確認(EMA cross) > 閾値条件(RSI>X)**
- **探索空間を小さくする = PASS率改善の鍵**
- **CR>=0.6が推奨閾値**（3/5フォールド黒字 = 合理的基準）

---

> Step 1-13の詳細記録: `.claude/memory/roadmap_archive.md`
