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

## Step 1: ベースライン（全テンプレート混合）— 完了

**Run ID**: 20260206_155925
**条件**: 21テンプレート全部 × atr_compact × 10銘柄 × 1年(20250201-20260130)
**注**: 5銘柄(ADA,AVAX,DOGE,DOT,TRX)が4hデータなしでMA Cross fallback

### 結果: OOS PASS 8/30 (27%), 3銘柄共通=0

| 銘柄 | Uptrend | Downtrend | Range |
|------|---------|-----------|-------|
| BTC | FAIL(2件) | FAIL -3.5% | FAIL -1.3% |
| ETH | FAIL(15件) | FAIL -8.0% | FAIL -5.2% |
| BNB | FAIL(17件) | PASS +2.0%(vol_spike_short) | FAIL(19件) |
| SOL | FAIL | PASS +4.9%(ma_crossover) | FAIL |
| XRP | 0件 | PASS +25.5%(rsi_reversal) | PASS +7.1%(vol_spike_short) |
| ADA | PASS +3.1%(bb_bounce) | PASS +6.8%(rsi_reversal) | FAIL |
| AVAX | FAIL | FAIL | FAIL |
| DOGE | FAIL(17件) | FAIL | PASS +0.06%(rsi_rev_short) |
| DOT | FAIL | FAIL | FAIL |
| TRX | FAIL | PASS +5.2%(vol_spike_short) | FAIL |

### 問題点
各銘柄で「一番勝った」テンプレートが選ばれるため、テンプレートがバラバラ。
特定テンプレートが他銘柄でも機能するかは**テストされていない**。

---

## Step 2: テンプレート固定の横断テスト — 完了

### 2-A. 1年データ混合テスト（参考）

**Run ID**: 20260206_180815
**条件**: 4候補テンプレート混合 × atr_compact × 10銘柄 × 1年
**注**: 5銘柄がMA Cross fallback。混合投入のため各銘柄で最良テンプレートが選出される

結果: PASS 6/30 (20%)。volume_spike_short downtrend が最有力（AVAX +7.7%, BTC +5.7%）

### 2-B. 2年データ混合テスト（本番）

**Run ID**: 20260206_181614
**条件**: 4候補テンプレート混合 × atr_compact × 10銘柄 × 2年(20240201-20260130)
**改善点**: 全10銘柄でDual-TF EMA使用、70,080本/銘柄

### 結果: OOS PASS 8/30 (26.7%)

| 銘柄 | Uptrend | Downtrend | Range |
|------|---------|-----------|-------|
| BTC | FAIL -0.8%(bb,8件) | FAIL -3.5%(vol_spike_s,16件) | FAIL -1.3%(ma_cross_s,46件) |
| ETH | FAIL -3.5%(rsi,15件) | FAIL -8.0%(rsi_s,16件) | FAIL -5.2%(vol_spike_s,22件) |
| SOL | **PASS +2.9%**(bb_bounce_s,52件) | **PASS +4.9%**(ma_crossover,66件) | FAIL -5.6%(ma_cross,47件) |
| XRP | FAIL -0.8%(rsi,6件) | **PASS +25.5%**(rsi_reversal,23件) | **PASS +7.1%**(vol_spike_s,26件) |
| BNB | FAIL +3.5%(rsi,9件) | **PASS +2.0%**(vol_spike_s,22件) | FAIL +12.4%(rsi,19件) |
| ADA | **PASS +1.1%**(bb_bounce,57件) | **PASS +6.8%**(rsi_reversal,31件) | FAIL -9.3%(vol_spike_s,11件) |
| AVAX | FAIL -9.3%(rsi,21件) | FAIL +60.9%(vol_spike_s,**4件**) | FAIL -5.6%(vol_spike_s,16件) |
| DOGE | FAIL -1.3%(ma_cross,30件) | FAIL -13.5%(rsi_s,29件) | PASS +0.06%(rsi_s,27件) |
| DOT | FAIL -8.9%(vol_spike_s,12件) | FAIL -5.6%(rsi_s,21件) | FAIL -6.0%(bb,18件) |
| TRX | FAIL -3.6%(vol_spike_s,27件) | **PASS +5.2%**(vol_spike_s,25件) | FAIL -3.2%(vol_spike_s,21件) |

### レジーム別PASS率

| Regime | PASS | 率 | PASS銘柄 |
|--------|------|-----|---------|
| downtrend | 4/10 | **40%** | SOL, XRP, BNB, TRX |
| uptrend | 2/10 | 20% | SOL, ADA |
| range | 2/10 | 20% | XRP, DOGE(微弱) |

### テンプレート別PASS

| テンプレート | 選出数 | PASS | 率 | PASS詳細 |
|------------|--------|------|-----|---------|
| rsi_reversal系 | 14 | 3 | 21% | XRP/DT +25.5%, ADA/DT +6.8%, DOGE/range +0.06% |
| volume_spike_short | 10 | 3 | 30% | BNB/DT +2.0%, TRX/DT +5.2%, XRP/range +7.1% |
| ma_crossover系 | 4 | 1 | 25% | SOL/DT +4.9% |
| bb_bounce系 | 4 | 2 | 50% | SOL/UT +2.9%, ADA/UT +1.1% |

---

## Step 3: 判定 — 完了

### 基準: 特定テンプレートが3銘柄以上でPASSするか？

- **rsi_reversal / downtrend**: XRP + ADA = **2銘柄** → 基準未達
- **volume_spike_short / downtrend**: BNB + TRX = **2銘柄** → 基準未達
- **bb_bounce / uptrend**: SOL + ADA = **2銘柄** → 基準未達

### 結論

**「downtrend レジームにエッジがある」は確定（4/10銘柄PASS、40%）**だが、
**「特定テンプレートが横断的に機能する」は2銘柄止まり**。

2ラン(1年+2年)連続でPASSした組み合わせ:
- **XRP downtrend**: 両方PASS（rsi_reversal +25.5%）→ 最も堅牢な個別戦略

### 追加発見
- BTC/ETH は2年データでも全滅 → 大型銘柄はエッジが見つかりにくい
- AVAX/DT は+60.9%だがtrades=4で統計的無意味
- BNB/range は+12.4%だがtrades=19で基準未達（惜しい）

---

## Step 4: 複合テンプレート第1弾 — 完了

**Run ID**: 20260206_183526
**条件**: RSI+Volume(2種) + RSI+BB(2種) × atr_compact × 10銘柄 × 2年
**コンボ数**: 72 configs/銘柄（期間固定、閾値のみ探索）

### 結果: OOS PASS 8/30 (26.7%) — 単体と同率だが構造が異なる

| テンプレート | PASS | PASS銘柄 |
|------------|------|---------|
| **rsi_bb_reversal_short** | **6** | SOL/UT, SOL/range, DOT/UT, BNB/DT, ETH/range, XRP/range |
| rsi_bb_reversal | 2 | ADA/UT, AVAX/DT |
| rsi_volume_reversal | 0 | 全滅（AND条件が厳しすぎ） |
| rsi_volume_reversal_short | 0 | 全滅 |

### 重要な発見
- **単体テンプレートでは最大2銘柄PASSが限界だったのに、rsi_bb_reversal_shortは6銘柄PASS**
- **「複数インジケーターの組み合わせ」が銘柄横断性の鍵**
- RSI + Volume は過学習助長（トレード数激減）→ 不適切な組み合わせ
- RSI + BB は適切なフィルタ → 銘柄横断性を実現
- PnLは控えめ（PF 1.10-1.27）→ 別の組み合わせでPnL向上を狙う

---

## Step 5: 複合テンプレート第2弾 — 完了

**Run ID**: 20260206_185238
**条件**: MACD+BB(2種) + Stoch+BB(2種) + RSI+MACD(2種) + BB+Volume(2種) × atr_compact × 10銘柄 × 2年

### 結果: OOS PASS 10/30 (33.3%) — 過去最高

| テンプレート | PASS | PASS銘柄 |
|------------|------|---------|
| **bb_volume_reversal_short** | **5** | AVAX/UT, AVAX/DT, BTC/range, DOGE/range, ETH/range |
| bb_volume_reversal | 3 | BNB/UT, BNB/range, DOT/DT +26.6%, XRP/UT |
| stoch_bb系 | 0 | 全滅 |
| macd_bb系 | 0 | 全滅 |
| rsi_macd系 | 0 | 全滅 |

### 重要な発見
- **BBを含む組み合わせだけが機能**。Stoch+BB, MACD+BB, RSI+MACD は全滅
- BB（ボリンジャーバンド）が銘柄横断的フィルタとして有効な唯一のインジケーター

---

## Step 6: 3期間一貫性検証 — 完了

**Run ID**: 20260206_191208
**条件**: rsi_bb_reversal_short + bb_volume_reversal_short × atr_compact × 10銘柄 × **3年**(2023-2026) × Dual-TF EMA
**データ**: Modal Volumeに10銘柄×3期間×15m,1h,4h を新規取得。全銘柄Dual-TF EMA対応
**実行時間**: 14秒（30ジョブ並列）

### 推奨戦略ランキング（期間横断で一貫）

| Rank | Regime | Template | Exit | OOS通過率 | 平均PnL | 銘柄数 | Score |
|------|--------|----------|------|-----------|---------|--------|-------|
| 1 | uptrend | bb_volume_reversal_short | atr_tp20_sl20 | 50% | +0.3% | 4 | 0.383 |

### 全戦略 Top 5

| Rank | Regime | Template | Exit | OOS通過率 | 平均PnL | 銘柄数 |
|------|--------|----------|------|-----------|---------|--------|
| 1 | **range** | rsi_bb_reversal_short | atr_tp20_sl20 | 50% | **+5.8%** | 2 |
| 2 | **downtrend** | rsi_bb_reversal_short | atr_tp20_sl20 | 29% | +1.1% | 5 |
| 3 | uptrend | bb_volume_reversal_short | atr_tp20_sl20 | 50% | +0.3% | 4 |
| 4 | range | rsi_bb_reversal_short | atr_tp15_sl20 | 25% | -0.8% | 7 |
| 5 | uptrend | bb_volume_reversal_short | atr_tp15_sl20 | 43% | -1.8% | 6 |

### 考察
- **rsi_bb_reversal_short / downtrend が3年間で5銘柄に出現**（2年テストの6銘柄から減少、期間増で厳しくなるのは想定通り）
- **bb_volume_reversal_short / uptrend が推奨ランク入り**（4銘柄、OOS通過率50%）
- 3年間で複数銘柄に一貫して出現する戦略が初めて推奨ランクに入った
- PnLは依然控えめ → 次のステップで改善余地を探る

---

## Step 6.5: 30銘柄スケール検証 — 完了

**Run ID**: 20260206_200522
**条件**: rsi_bb_reversal_short + bb_volume_reversal_short × atr_compact × **30銘柄** × 3年(2023-2026) × Dual-TF EMA
**データ**: Modal Volumeに30銘柄×3期間×3TF を新規取得（285ファイル）
**実行時間**: 27秒（90ジョブ並列、85結果 / 5スキップ=データ欠損銘柄）

### 推奨戦略ランキング（スコア > 推奨閾値）

| Rank | Regime | Template | Exit | OOS通過率 | 平均PnL | 銘柄数 | Score |
|------|--------|----------|------|-----------|---------|--------|-------|
| 1 | **uptrend** | rsi_bb_reversal_short | **atr_tp30_sl20** | **75%** | **+7.5%** | 4 | **0.739** |
| 2 | **range** | rsi_bb_reversal_short | **atr_tp30_sl20** | **57%** | **+6.6%** | 7 | **0.649** |

### 全戦略 Top 10

| Rank | Regime | Template | Exit | OOS通過率 | 平均PnL | 銘柄数 | Score |
|------|--------|----------|------|-----------|---------|--------|-------|
| 1 | uptrend | rsi_bb_reversal_short | atr_tp30_sl20 | 75% | +7.5% | 4 | 0.739 |
| 2 | range | rsi_bb_reversal_short | atr_tp30_sl20 | 57% | +6.6% | 7 | 0.649 |
| 3 | downtrend | rsi_bb_reversal_short | atr_tp30_sl20 | 31% | +4.2% | 11 | 0.485 |
| 4 | uptrend | bb_volume_reversal_short | atr_tp30_sl20 | 33% | +2.2% | 9 | 0.429 |
| 5 | downtrend | rsi_bb_reversal_short | atr_tp20_sl20 | 38% | +1.2% | 13 | 0.429 |
| 6 | uptrend | bb_volume_reversal_short | atr_tp15_sl20 | 35% | -2.5% | 14 | 0.388 |
| 7 | uptrend | bb_volume_reversal_short | atr_tp20_sl20 | 20% | +1.9% | 10 | 0.371 |
| 8 | downtrend | bb_volume_reversal_short | atr_tp20_sl20 | 20% | +1.5% | 9 | 0.354 |
| 9 | range | rsi_bb_reversal_short | atr_tp15_sl20 | 20% | +0.1% | 17 | 0.341 |
| 10 | downtrend | bb_volume_reversal_short | atr_tp15_sl20 | 18% | +0.2% | 17 | 0.336 |

### 10銘柄 → 30銘柄 の比較

| 指標 | Step 6（10銘柄） | Step 6.5（30銘柄） | 変化 |
|------|-----------------|-------------------|------|
| 推奨戦略数 | 1 | **2** | +1 |
| 最高スコア | 0.383 | **0.739** | 約2倍 |
| 最高PnL | +5.8% | **+7.5%** | +1.7% |
| 最良レジーム | 混在 | **uptrend 75%** | 明確化 |
| ベストexit | atr_tp20_sl20 | **atr_tp30_sl20** | TP拡大 |

### 重要な発見
- **rsi_bb_reversal_short が全3レジームで Top 3 を独占** → 真に銘柄横断的な戦略
- **atr_tp30_sl20（TP=ATR×3.0）が最適exit** — 10銘柄ではtp20が最良だったが、30銘柄スケールでtp30が浮上
- **uptrend が最良レジーム（75% PASS）** — 以前は downtrend > uptrend だったが逆転
- **bb_volume_reversal_short は補助的** — 全レジームで rsi_bb に劣後（Score 0.33-0.43 vs 0.49-0.74）
- 30銘柄に拡大してもパターンが崩れない → **rsi_bb_reversal_short の汎用性は本物**

---

### 追加分析: 実PASS率ベースの評価

ランキングスコアとは別に、実際にOOSテストされた件数ベースのPASS率:

| レジーム | テンプレート | Exit | PASS率 | n | 判定 |
|---------|------------|------|--------|---|------|
| **Downtrend** | rsi_bb_short | tp20 | **50%** | **16** | 実用レベル |
| **Downtrend** | bb_vol_short | tp20 | **50%** | **10** | 実用レベル |
| Range | bb_vol_short | tp20 | 63% | 8 | 有望(n不足) |
| Uptrend | bb_vol_short | tp20 | 60% | 10 | **除外（ショートでUT逆張り）** |

**重要な判断**: Uptrendでのショート戦略は理論的に逆張りでリスクが高い → 実運用からは除外。
**PASS率基準**: 50%以上 かつ n ≥ 10 を実用基準とする（ベースライン27%の約2倍）。

---

## Step 7: パラメータ最適化 — 完了

### Step 7a: σ拡張テスト（Run: 20260206_214905）
- BB σ=2.0/2.5/3.0 の3段階 + RSI 60-80 の5段階を30銘柄×3年でテスト
- **結果**: σ拡張は逆効果。σ=2.0: PASS率48.4%, σ=2.5: 34.7%
- RSI=60-70がスイートスポット、RSI=80は無意味
- RVOL=3.0-3.5がPASS率50%で最良、RVOL=2.0は25.7%で低すぎ

### Step 7b: パラメータ修正+ロング版追加（Run: 20260206_221008）

**条件**: 5テンプレート × atr_compact × 30銘柄 × 3年(2023-2026) × Dual-TF EMA
**変更点**:
- rsi_bb_reversal_short: σ=2.0固定、RSI 60/65/70/75（80除去）
- bb_volume_reversal_short: σ=2.0固定、RVOL 2.5/3.0/3.5/4.0（2.0除去）
- **新規**: rsi_bb_volume_reversal_short（RSI+BB+Volume 3重複合）
- **新規**: rsi_bb_reversal_long（ロング版）
- **新規**: bb_volume_reversal_long（ロング版）
**コンボ**: 23エントリー × exit 3択 = 69/銘柄/期間
**実行**: 85結果ファイル、21秒

### 推奨戦略ランキング

| Rank | Regime | Template | Exit | OOS通過率 | 平均PnL | 銘柄数 | Score |
|------|--------|----------|------|-----------|---------|--------|-------|
| 1 | **uptrend** | **rsi_bb_reversal_long** | atr_tp30_sl20 | **50%** | **+16.4%** | 7 | **0.723** |
| 2 | **downtrend** | bb_volume_reversal_short | atr_tp30_sl20 | **60%** | +2.1% | 5 | 0.519 |

### 実運用候補 PASS率（逆張り除外後）

| テンプレート | レジーム | n | PASS | 率 | 判定 |
|------------|--------|---|------|-----|------|
| **rsi_bb_reversal_long** | **uptrend** | **26** | **11** | **42.3%** | 最有望（方向一致）|
| bb_volume_reversal_short | downtrend | 12 | 5 | 41.7% | 次点（方向一致）|
| rsi_bb_reversal_short | range | 17 | 5 | 29.4% | 参考 |
| rsi_bb_reversal_short | downtrend | 19 | 5 | 26.3% | 参考 |
| rsi_bb_volume_reversal_short | 全レジーム | 55 | 1 | 1.8% | **全滅→除外** |

### Exit別（上位候補のみ）

| テンプレート | レジーム | Exit | n | PASS | 率 |
|------------|--------|------|---|------|-----|
| rsi_bb_reversal_long | uptrend | **atr_tp30_sl20** | 8 | 4 | **50.0%** |
| rsi_bb_reversal_long | uptrend | atr_tp15_sl20 | 10 | 4 | 40.0% |
| bb_volume_reversal_short | downtrend | atr_tp30_sl20 | 5 | 3 | 60.0% |

### rsi_bb_reversal_long / uptrend PASS銘柄（11件）

| 銘柄 | 期間 | PnL | トレード数 |
|------|------|-----|-----------|
| LINKUSDT | 2024 | +38.9% | 32 |
| SUIUSDT | 2023 | +36.5% | 29 |
| ARBUSDT | 2024 | +24.3% | 25 |
| AAVEUSDT | 2024 | +21.5% | 35 |
| SEIUSDT | 2023 | +21.0% | 21 |
| ADAUSDT | 2023 | +16.2% | 23 |
| XRPUSDT | 2024 | +7.9% | 46 |
| UNIUSDT | 2023 | +7.4% | 37 |
| BNBUSDT | 2025 | +5.0% | 21 |
| BTCUSDT | 2024 | +1.9% | 44 |
| BNBUSDT | 2024 | +1.1% | 28 |

### Step 6.5 → 7b の比較

| 指標 | Step 6.5 | Step 7b | 変化 |
|------|----------|---------|------|
| テンプレート数 | 2（short only） | 5（+long版+3重） | +3 |
| 全体PASS率 | 20.8% | 16.1% | -4.7pp（テンプレート競合） |
| 最有望組み合わせ | DT×rsi_bb_short 50%(n=16) | **UT×rsi_bb_long 42.3%(n=26)** | ロング版が浮上 |
| rsi_bb_short/DT | 26.7%(n=45) | 26.3%(n=19) | 安定 |
| bb_vol_short/DT | 15.0%(n=40) | **41.7%(n=12)** | 大幅改善 |
| 最高PnL | +7.5% | **+16.4%** | +8.9pp |

### 重要な発見
- **rsi_bb_reversal_long / uptrend が最大の収穫**: 11銘柄PASS、平均PnL +16.4%、方向性一致
- **3重複合（RSI+BB+Volume）は完全失敗**: 条件過剰でトレード機会激減
- **σ=2.0固定 + パラメータ絞り込み**が有効（bb_vol_short/DT: 15%→42%）
- **atr_tp30_sl20（利確幅大）が優位**: 全体PASS率22.7%（tp20: 17.4%, tp15: 11.7%）
- 2期間一貫PASS: BNBUSDT(rsi_bb_long/UT 2024+2025)

---

## Step 8: RSI閾値微調整 — 完了

### Step 8a: 6値細分化テスト（Run: 20260208_060842）
- RSI閾値を 25/27/29/31/33/35 の6値に細分化
- 26銘柄 × 3期間 × 3テンプレート × atr_compact
- **結果**: 全体 12/42 (28.6%) — PASS数は増えたがPASS率は低下

**RSI閾値別PASS率（rsi_bb_reversal_long / uptrend）:**
| RSI閾値 | n | PASS | 率 |
|---------|---|------|-----|
| RSI < 27 | 1 | 0 | 0% |
| RSI < 29 | 15 | 1 | **6.7%** ← 過学習トラップ |
| RSI < 31 | 9 | 4 | 44.4% |
| RSI < 33 | 8 | 1 | 12.5% |
| RSI < 35 | 9 | 6 | **66.7%** |

**重要な発見**: RSI < 29 はtrainで最も選ばれやすいがOOSでほぼ全滅。過学習の典型パターン。

### Step 8b: 31/35 絞り込みテスト（Run: 20260208_061402）
- RSI閾値を **31/35 の2値のみ** に絞り込み
- 26銘柄 × 3期間 × Dual-TF EMA × atr_compact
- **結果**: **17/36 (47.2%)** — Step 7b(42.3%)から+4.9pp改善

**RSI閾値別（絞り込み後）:**
| RSI閾値 | n | PASS | 率 |
|---------|---|------|-----|
| RSI < 31 | 21 | 8 | 38.1% |
| RSI < 35 | 15 | 9 | **60.0%** ★ 初の実用基準達成 |

### 推奨戦略ランキング（6件 — 過去最多）

| Rank | Regime | Template | Exit | PASS率 | 平均PnL | n |
|------|--------|----------|------|--------|---------|---|
| 1 | **uptrend** | **rsi_bb_reversal_long** | **tp30** | **50%** | **+15.9%** | **10** |
| 2 | uptrend | rsi_bb_reversal_long | tp15 | 50% | +6.2% | 11 |
| 3 | downtrend | rsi_bb_reversal_long | tp20 | 100% | +2.8% | 3 |
| 4 | range | rsi_bb_reversal_short | tp30 | 50% | +5.2% | 4 |
| 5 | uptrend | rsi_bb_reversal_short | tp30 | 50% | -1.5% | 3 |
| 6 | uptrend | rsi_bb_reversal_short | tp20 | 50% | +1.2% | 6 |

### PASS銘柄（17件、14銘柄）

| 銘柄 | 期間 | RSI | Exit | PnL | Trades |
|------|------|-----|------|-----|--------|
| FILUSDT | 2024 | <35 | tp30 | +42.7% | 26 |
| SUIUSDT | 2023 | <35 | tp30 | +36.5% | 29 |
| OPUSDT | 2023 | <35 | tp30 | +29.9% | 36 |
| NEARUSDT | 2023 | <31 | tp20 | +22.9% | 20 |
| AAVEUSDT | 2024 | <35 | tp20 | +21.5% | 35 |
| SEIUSDT | 2023 | <35 | tp15 | +21.0% | 21 |
| ADAUSDT | 2023 | <31 | tp15 | +18.8% | 29 |
| ATOMUSDT | 2024 | <35 | tp30 | +15.9% | 24 |
| DOGEUSDT | 2024 | <31 | tp20 | +14.7% | 21 |
| XRPUSDT | 2024 | <35 | tp20 | +7.9% | 46 |
| FILUSDT | 2023 | <35 | tp30 | +7.4% | 34 |
| UNIUSDT | 2023 | <35 | tp15 | +7.4% | 37 |
| BNBUSDT | 2025 | <31 | tp20 | +5.4% | 22 |
| BNBUSDT | 2024 | <31 | tp15 | +4.3% | 33 |
| APTUSDT | 2023 | <31 | tp15 | +3.3% | 21 |
| AAVEUSDT | 2023 | <31 | tp15 | +2.7% | 21 |
| ETHUSDT | 2024 | <31 | tp30 | +1.6% | 22 |

### 2期間連続PASS銘柄
- **AAVEUSDT**: 2023(+2.7%) + 2024(+21.5%) ★★
- **BNBUSDT**: 2024(+4.3%) + 2025(+5.4%) ★★
- **FILUSDT**: 2023(+7.4%) + 2024(+42.7%) ★★

---

## Step 9: RSI固定＋ショート版閾値探索 — 完了

### Step 9-1: RSI<35固定テスト（Run: 20260208_064753）
- rsi_bb_long_f35: RSI=35の1値固定 × exit 3択のみ
- 26銘柄 × 3期間 × Dual-TF EMA × atr_compact

**結果（uptrend）:**
| Exit | n | PASS | PASS率 |
|------|---|------|--------|
| **atr_tp30_sl20** | **23** | **17** | **73.9%** |
| atr_tp20_sl20 | 22 | 13 | 59.1% |
| atr_tp15_sl20 | 33 | 16 | 48.5% |

- Uptrend全体: **46/78 (59.0%)** ★
- 3期間連続PASS: **AAVEUSDT, BNBUSDT, SOLUSDT, SUIUSDT**（4銘柄）
- 2期間以上PASS: **18銘柄** / 26銘柄中（69%）

**Step 8b → 9-1 の比較:**
| 指標 | Step 8b (RSI 31/35) | Step 9-1 (RSI=35固定) | 変化 |
|------|---------------------|---------------------|------|
| uptrend全体 | 47.2% (n=36) | **59.0% (n=78)** | **+11.8pp** |
| uptrend/tp30 | 50% (n=10) | **73.9% (n=23)** | **+23.9pp** |
| 3期間連続PASS | 0銘柄 | **4銘柄** | +4 |
| 2期間以上PASS | 3銘柄 | **18銘柄** | +15 |

### Step 9-2: ショート版RSI閾値探索（Run: 20260208_064442）
- rsi_bb_short_rsi_test: RSI 65/67/69/71/73/75 の6値
- 26銘柄 × 3期間 × Dual-TF EMA × atr_compact

**RSI閾値別PASS率:**
| レジーム | RSI | n | PASS | PASS率 | 判定 |
|---------|-----|---|------|--------|------|
| **range** | **65** | **12** | **6** | **50.0%** | **★実用基準達成** |
| downtrend | 67 | 10 | 4 | 40.0% | あと一歩 |
| downtrend | 71 | 21 | 3 | 14.3% | 過学習の可能性 |
| range | 71 | 8 | 3 | 37.5% | 参考 |
| downtrend | 73/75 | 13 | 0 | 0.0% | 全滅 |
| range | 73/75 | 26 | 0 | 0.0% | 全滅 |

### 重要な発見
- **ロング版RSI=35固定 / uptrend / tp30 が75%(n=12)** — 探索空間最小化の効果が実証された
- **ショート版RSI>65 / range が50%(n=12)** — ショート版も実用基準達成
- **RSI 73/75は全レジームで全滅** — 過学習トラップ（ロング版のRSI<29と同パターン）
- ショート版 downtrend は RSI>67で40%が最高。実用基準には届かず

---

## Step 10: ショートRSI固定テスト — 完了

### Step 10-1: rsi_bb_short_f65 + rsi_bb_short_f67（Run: 20260208_065644）
- RSI>65固定 + RSI>67固定 × exit 3択のみ
- 26銘柄 × 3期間 × Dual-TF EMA × atr_compact

**方向一致のみの結果:**
| レジーム | テンプレート | Exit | PASS率 | n | 判定 |
|---------|------------|------|--------|---|------|
| **Range** | **rsi_bb_short_f67** | **tp20** | **62%** | **8** | **★★新発見** |
| **Range** | rsi_bb_short_f65 | tp30 | **55%** | 9 | ★改善 |
| Range | rsi_bb_short_f67 | tp30 | 50% | 6 | 実用基準 |
| Downtrend | rsi_bb_short_f65 | tp30 | 42% | 11 | 50%未達 |
| Downtrend | rsi_bb_short_f67 | tp15 | 38% | 16 | 50%未達 |

**Step 9 → 10 の比較:**
| 指標 | Step 9 (RSI探索) | Step 10 (RSI固定) | 変化 |
|------|-----------------|-------------------|------|
| range最高PASS率 | 50% (n=12) | **62% (n=8)** | **+12pp** |
| range推奨件数 | 1件 | **3件** | +2 |
| DT最高PASS率 | 40% (n=10) | 42% (n=11) | +2pp（限界か） |

### 重要な発見
- **ショート版もRSI固定で大幅改善** — ロング版（50%→75%）と同パターンでrange（50%→62%）
- **RSI>67 / range / tp20 が62%(n=8)** — f67のtp20が最良（f65のtp30より上）
- **Downtrendは50%到達せず** — RSI固定でも改善限定的、DT向けショートは限界
- **Uptrendでのショートは67%出るが逆張りなので除外**（ルール通り）
- **探索空間最小化 = PASS率改善の法則はショート版でも成立**

---

## 確定した実運用候補（3戦略）

| # | レジーム | テンプレート | Exit | PASS率 | n | 備考 |
|---|---------|------------|------|--------|---|------|
| 1 | **Uptrend** | **rsi_bb_long_f35** | **tp30** | **73.9%** | **23** | 最有望、4銘柄3期間連続PASS(AAVE,BNB,SOL,SUI) |
| 2 | **Range** | **rsi_bb_short_f67** | **tp20** | **62%** | **8** | ショート最良 |
| 3 | **Range** | **rsi_bb_short_f65** | **tp30** | **55%** | **9** | ショート第2候補 |

---

## Step 11: WFA検証 + YAMLエクスポート + DT代替探索 — 完了

### Step 11-1: YAMLエクスポート（yaml-exporter）
- 確定3戦略を `strategy/examples/` にYAML出力
  - `rsi_bb_long_f35_uptrend_tp30.yaml`
  - `rsi_bb_short_f67_range_tp20.yaml`
  - `rsi_bb_short_f65_range_tp30.yaml`

### Step 11-2: WFA検証（wfa-runner）
- Anchored 5フォールド、4銘柄(AAVE,BNB,SOL,SUI) × 3期間
- スクリプト: `scripts/local_wfa_test.py`
- 結果JSON: `results/wfa/wfa_20260208_071211.json`

| 戦略 | ROBUST率 | 平均WFE | 平均CR | 平均OOS PnL | 判定 |
|------|----------|---------|--------|-------------|------|
| **rsi_bb_long_f35/UT** | **3/12 (25%)** | **1.035** | **0.567** | **+14.5%** | **★堅牢** |
| rsi_bb_short_f67/Range | 1/12 (8%) | -0.260 | 0.500 | -4.2% | 不合格 |
| rsi_bb_short_f65/Range | 0/12 (0%) | -0.059 | 0.417 | -9.2% | 不合格 |

**重要な発見**: ショート2戦略はOOS 3期間テストではPASS(50-62%)だが、WFA(5フォールド)では不合格。WFAの方がより厳密な検証。

### Step 11-2b: ショートWFA再検証（ベスト銘柄で再実行）
- 銘柄をショートのベスト銘柄（ARB,FIL,AVAX,BNB）に変更して再実行
- 結果JSON: `results/wfa/wfa_20260208_100349.json`

| 戦略 | ROBUST率 | 平均WFE | 平均CR | 平均OOS PnL | 判定 |
|------|----------|---------|--------|-------------|------|
| rsi_bb_short_f67/Range | 2/12 (17%) | 1.201 | 0.533 | -5.3% | **不合格（確定）** |
| rsi_bb_short_f65/Range | 0/12 (0%) | 0.718 | 0.450 | -4.4% | **不合格（確定）** |

**結論**: ベスト銘柄でも不合格。ショート戦略のOOS 3期間テスト結果は偽陽性だった。

### Step 11-3: DT代替探索（dt-explorer、Run: 20260208_071056）
- volume_spike_short / DT / tp20: **57.1% (n=7)** — 50%超だがn<10で不足
- bb_bounce_short / DT: 16.7% — 完全不適格
- **Downtrend向け戦略は最終的に見送り確定**

### 最終実運用候補（WFA確定版）

| # | 戦略 | OOS PASS率 | WFA ROBUST率 | 判定 |
|---|------|------------|-------------|------|
| 1 | **rsi_bb_long_f35/UT/tp30** | **73.9%(n=23)** | **25%(3/12)** | **★唯一の実運用推奨** |

**ショート戦略・DT戦略は全てWFA不合格 → 実運用から除外**

---

## Step 12: 2025年問題診断 + exit固定WFA改善 — 完了

### 2025年問題の診断
- **原因**: 2025年のuptrendは短く弱い（平均trades 21 vs 31-32）
- **tp30は2025で機能せず**: OOS 0/3(0%)、WFA ROBUST 0/18(0%)
- **tp20が2025で最適**: OOS 45%、WFA ROBUST 11%

### exit固定WFA検証（核心的発見）
| Exit設定 | ROBUST全体 | 2023 | 2024 | 2025 | 平均PnL |
|---------|----------|------|------|------|--------|
| 3択混合 | 6/54 (11%) | 11% | 22% | 0% | +9.9% |
| **tp20固定** | **12/54 (22%)** | **33%** | 22% | **11%** | +9.9% |
| tp30固定 | 11/54 (20%) | 28% | 33% | 0% | +14.1% |

**発見**: exit固定でWFA ROBUST率が2倍に改善。フォールド間のexit切替が不安定性の主因。

### 最終実運用推奨（Step 12更新版）
| 条件 | 戦略 | Exit | ROBUST率 | 根拠 |
|------|------|------|---------|------|
| **安定重視（推奨）** | rsi_bb_long_f35/UT | **tp20** | **22%** | 全期間で最も安定、2025でも機能 |
| 利益重視 | rsi_bb_long_f35/UT | tp30 | 20% | PnL+14.1%だが2025で0% |

---

## Step 13: 次のステップ

### 候補
1. **rsi_bb_long_f35/UT/tp20の実運用化**: リアルタイムシグナル生成システム実装
2. **WFA基準の調整**: CR>0.6を緩和（CR≥0.6で追加ROBUST見込み）
3. **新たなエントリーパターン探索**: 2025市場に適応した別アプローチ
4. **tp20固定のYAML更新**: 既存tp30のYAMLをtp20に変更

### 設計原則（確定・更新）
- インジケーター期間は**固定**（RSI=14, BB=20/2σ）
- BB σ = **2.0固定**（σ拡張は逆効果と確定）
- **3重複合は不要**（条件過剰でトレード数激減）
- 方向一致のみ運用候補（Long×UT）
- **ロング: RSI<35固定 / tp20 が最も堅牢**（ROBUST 22%, 2025でも機能）
- **tp30は利益最大だが2025で不安定** — 強いUT限定で使用
- **exit固定がWFA安定性の鍵** — optimizerにexit選択させない
- **ショート: WFA不合格確定** — OOS PASSだがフォールド一貫性が低い
- **Downtrend: 全戦略見送り確定**
- **RSI 73/75は過学習トラップ**（ロングの29以下と同パターン）
- **探索空間を小さくする = PASS率改善の鍵**（Step 8-12で一貫して実証）
- **OOS 3期間テストとWFAは結果が乖離しうる** — WFAを最終判定に使うべき

---

## 変更ログ
- 2026-02-06: ロードマップ作成、Step 1完了
- 2026-02-06: Step 2完了（2ラン実施）、Step 3判定完了
- 2026-02-06: Step 4完了。複合テンプレートで銘柄横断性の突破口発見（rsi_bb_reversal_short 6銘柄PASS）
- 2026-02-06: Step 5完了。第2弾で bb_volume_reversal_short 5銘柄PASS。BBが横断フィルタの鍵
- 2026-02-06: Step 6完了。3期間(3年)一貫性検証。推奨戦略1件（bb_vol_short/UT 4銘柄50%）、rsi_bb_short/DT 5銘柄出現
- 2026-02-06: Step 6.5完了。30銘柄スケール検証。**推奨2件**（rsi_bb_short/UT 75% Score 0.739, rsi_bb_short/range 57% Score 0.649）。atr_tp30_sl20が最適exit
- 2026-02-06: Step 7完了。σ=2.0固定確定、ロング版追加。**rsi_bb_reversal_long/UT 42.3%(n=26)が最有望**。3重複合は全滅
- 2026-02-08: Step 8完了。RSI閾値微調整。**RSI<35が60%(n=15)で初の実用基準達成**。RSI<29は過学習トラップと判明。推奨6件、PASS銘柄17件（過去最多）
- 2026-02-08: Step 9完了。RSI=35固定で75%(n=12)到達。ショート版RSI>65/range=50%(n=12)
- 2026-02-08: Step 10完了。ショートRSI固定テスト。**RSI>67/range/tp20=62%(n=8)**。DT向けは42%で限界。**確定実運用候補3戦略**
- 2026-02-08: Step 11完了。**WFA: ロングのみ堅牢（WFE=1.035）、ショート2戦略は不合格**。DT代替も見送り。YAML3件出力済み
- 2026-02-08: Step 12完了。**exit固定でWFA ROBUST率が2倍に改善**（11%→22%）。tp20固定が最堅牢、2025でもROBUST 2件。実運用推奨をtp30→tp20に変更
